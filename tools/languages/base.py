"""Shared infrastructure for the per-language tool implementations.

Centralizes subprocess execution, toolchain-availability checks, dependency
auto-provisioning (run once per project+language), bundled-config lookup, and
Python interpreter resolution. The per-language modules (`python`, `javascript`,
`go`, `rust`, `java`) build their commands and call the helpers here.

Design notes
------------
* Only Python has a per-project "environment with the tools inside it"; every
  other language relies on a system-wide toolchain (on PATH) plus a shared
  global dependency cache the toolchain auto-fetches. So `_venv_python` /
  managed venvs are Python-only; the other languages just run from
  `cwd=project_path` and surface a clear message when their toolchain is absent.
* All Windows-vs-Unix executable resolution goes through `shutil.which`, which
  honors PATHEXT — calling bare ``npm``/``npx`` via subprocess would otherwise
  raise FileNotFoundError on Windows.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from tools._common import _truncate

# Bundled default lint configs (used only when the reviewed project has none).
CONFIG_DIR = Path(__file__).parent / "configs"

# Managed venvs we create for Python projects that have none of their own live
# here — under code_review_agent, never inside the reviewed project.
_TOOLENVS_DIR = Path(__file__).resolve().parents[2] / ".toolenvs"

# ── auto-provisioning guard (server-lifetime, thread-safe) ───────────────
# Provision a project's dependencies at most once per (project, language) per
# server run — same spirit as the snapshot store in tools/change_tracking.py.
_PROVISIONED: set[tuple[str, str]] = set()
_PROVISION_LOCK = threading.RLock()


def _env_msg(name: str) -> str:
    """Standard 'toolchain missing' message — an environment, not a code, problem."""
    return (f"[environment] '{name}' is not installed or not on PATH. This is an "
            f"environment problem, NOT a defect in the reviewed project. Do not "
            f"report it as a code finding.")


def _exe(name: str) -> "str | None":
    """Resolve an executable on PATH (honors PATHEXT on Windows), or None."""
    return shutil.which(name)


def _tool_or_msg(name: str) -> "tuple[str | None, str | None]":
    """(resolved_path, None) if `name` is on PATH, else (None, env-missing message)."""
    exe = shutil.which(name)
    return (exe, None) if exe else (None, _env_msg(name))


def _project_bin(project: str, *candidates: str) -> "str | None":
    """First existing project-local binary among `candidates` (relative to root)."""
    root = Path(project)
    for rel in candidates:
        p = root / rel
        if p.exists():
            return str(p)
    return None


def _run(cmd: "list[str]", cwd: "str | None" = None, timeout: int = 120,
         env: "dict | None" = None) -> str:
    """Run a subprocess; return truncated stdout+stderr, or a clear error string.

    A nonzero exit still returns the captured output (that IS the linter/test
    result). Only a missing binary, a timeout, or a launch failure short-circuit.
    """
    name = cmd[0] if cmd else "?"
    print(f"Running: {cmd} (cwd={cwd})")
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                                timeout=timeout, env=env)
    except FileNotFoundError:
        return _env_msg(name)
    except subprocess.TimeoutExpired:
        return (f"[tool timeout] '{name}' did not finish within {timeout}s. The scan "
                f"is likely too broad — narrow it to a subdirectory or specific files.")
    except Exception as e:  # noqa: BLE001
        return f"[tool error] could not launch '{name}': {e}"
    out = (result.stdout + result.stderr).strip()
    return _truncate(out) or f"'{name}' ran and produced no output."


def _run_python_tool(python_exe: str, module: str, args: "list[str]",
                     timeout: int = 120) -> str:
    """Run ``python -m <module> <args>`` and normalize the 'not installed' case."""
    print(f"Running python: {python_exe} with {module} with args: {args}")
    try:
        result = subprocess.run([python_exe, "-m", module, *args],
                                capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return (f"[tool timeout] '{module}' did not finish within {timeout}s. "
                f"The scan is likely too broad — narrow it to the source "
                f"directory or specific files rather than the whole project root.")
    except Exception as e:  # noqa: BLE001
        return f"[tool error] could not launch {module}: {e}"
    out = (result.stdout + result.stderr).strip()
    if f"No module named {module}" in out:
        return (f"[tool unavailable] '{module}' is not installed in the agent's "
                f"environment. This is an environment problem, NOT a defect in the "
                f"reviewed project. Do not report it as a code finding.")
    return _truncate(out) or f"{module} ran and produced no output."


def _provision_once(project: str, language: str, fn) -> None:
    """Run `fn(project)` (dependency provisioning) at most once per project+language.

    Best-effort: the cache key is recorded BEFORE running so a slow/failing
    install isn't retried on every tool call; real problems surface when the
    actual tool runs afterward.
    """
    key = (str(Path(project).resolve()), language)
    with _PROVISION_LOCK:
        if key in _PROVISIONED:
            return
        _PROVISIONED.add(key)
    try:
        fn(project)
    except Exception:  # noqa: BLE001
        pass


# ── Python interpreter resolution (project venv -> managed venv -> ours) ──

def _has_project_venv(project_path: str) -> bool:
    """True if the reviewed project (or a near parent) already has a venv."""
    return _find_project_venv(project_path) is not None


def _find_project_venv(project_path: str) -> "str | None":
    root = Path(project_path).resolve()
    for base in (root, *list(root.parents)[:2]):  # repo venv often sits a level up
        for name in (".venv", "venv"):
            for rel in ("Scripts/python.exe", "bin/python"):  # Windows, then Unix
                cand = base / name / rel
                if cand.exists():
                    return str(cand)
    return None


def _managed_venv_dir(project_path: str) -> Path:
    h = hashlib.sha1(str(Path(project_path).resolve()).encode()).hexdigest()[:16]
    return _TOOLENVS_DIR / h


def _managed_venv_python(project_path: str) -> "str | None":
    d = _managed_venv_dir(project_path)
    for rel in ("Scripts/python.exe", "bin/python"):
        cand = d / rel
        if cand.exists():
            return str(cand)
    return None


def _create_managed_venv(project_path: str) -> str:
    """Create (once) a managed venv for a project that has none, return its python."""
    existing = _managed_venv_python(project_path)
    if existing:
        return existing
    d = _managed_venv_dir(project_path)
    d.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "venv", str(d)],
                   capture_output=True, text=True, timeout=180)
    return _managed_venv_python(project_path) or sys.executable


def _venv_python(project_path: str) -> str:
    """Best interpreter for a Python project: its venv, else a managed one, else ours.

    Find-only — never creates anything (creation happens during provisioning).
    Falls back to our own interpreter, which ships pylint/mypy/pytest.
    """
    return (_find_project_venv(project_path)
            or _managed_venv_python(project_path)
            or sys.executable)
