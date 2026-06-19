"""Python language tools: pylint, pytest, mypy, and a static import check — all
run inside the `crav-python` container so the host needs no Python toolchain.

The image (see ``docker/crav-python.Dockerfile``) bakes in pylint, mypy, pytest
and pytest-cov. Project deps are pre-installed into a per-project derived image
by `_ensure_project_image` (in base.py): the first review of a project pays the
`pip install` cost once; every subsequent review reuses the cached Docker layer
and pays nothing. The derived image tag is keyed on the manifest *content*, so
it auto-invalidates when requirements.txt (or pyproject.toml) changes.

The import check is our own stdlib AST analyzer shipped as
``docker/import_check.py``; running it inside the dep-installed container is what
lets ``find_spec`` resolve third-party packages (it never executes project code).
"""
from __future__ import annotations

import re
from pathlib import Path

from structured_output import CompileOutput, ErrorOutput
from tools._common import _IGNORE, _vendor_ignore_regex

from .base import (
    _compile_path,
    _compile_result,
    _ensure_project_image,
    _run_docker,
    _run_docker_text,
)

_IMAGE = "crav-python"
# The directory holding import_check.py; mounted read-only at /checker.
_SCRIPT_DIR = Path(__file__).parent / "docker"

# Docker image used to byte-compile the project (syntax check, no execution).
_COMPILE_IMAGE = _IMAGE

# Select files to compile with `find ... -prune` so vendored dirs (.venv,
# node_modules, …) are never *descended into* — `compileall -x` only skips
# compiling matched files, it still os.walks every dir, which stat-walks the
# whole .venv over the slow bind mount and times out. We feed the pruned file
# list to `compileall -i` and run against the read-only /src directly (no copy),
# with PYTHONPYCACHEPREFIX redirecting bytecode to /tmp (set in compile_code).
_COMPILE_PRUNE = " -o ".join(f"-name {name}" for name in sorted(_IGNORE))
_COMPILE_CMD = (
    r"find /src -type d \( " + _COMPILE_PRUNE + r" \) -prune "
    r"-o -type f -name '*.py' -print > /tmp/flist && "
    r"python -m compileall -q -i /tmp/flist"
)

# compileall reports a failing file as a block: a `File "...", line N` header,
# the offending source line, a caret pointing at the column, then `XxxError: msg`.
_PY_FILE_RE = re.compile(r'File "([^"]+)", line (\d+)')
_PY_MSG_RE = re.compile(r"^\s*\w*(?:Error|Warning):")

# Manifests we look for, in priority order.  The first one found in the project
# root is used to build the derived image.
_MANIFESTS = ["requirements.txt", "pyproject.toml", "setup.py", "setup.cfg"]


# ── per-project derived image ─────────────────────────────────────────────

def _pip_install_cmd(manifest_name: str) -> str:
    """Return the pip install command to bake into the derived image's RUN layer.

    For requirements.txt we install directly from the file copied into /tmp.
    For pyproject / setup.* we install from the file path too — pip can resolve
    a pyproject.toml's [project] dependencies from the file alone for the common
    static case. If the project uses dynamic metadata that needs the full source
    tree, the install will partially fail (pip falls back gracefully), but the
    layer cache still works and the common case is fully covered.
    """
    if manifest_name == "requirements.txt":
        return f"pip install --no-cache-dir -q -r /tmp/{manifest_name}"
    # pyproject.toml / setup.py / setup.cfg: install as a project
    return f"pip install --no-cache-dir -q /tmp/{manifest_name}"


def _project_image(path: str) -> "tuple[str, str | None]":
    """Return (image_tag, warning) for the project at `path`.

    Delegates to `_ensure_project_image` with the Python-specific manifest list
    and install command builder.  The returned image is either the pre-baked
    derived image (deps already installed) or the plain _IMAGE base (if no
    manifest was found or the build failed), in which case `warning` is set.
    """
    return _ensure_project_image(
        base_image=_IMAGE,
        project_path=path,
        manifests=_MANIFESTS,
        install_cmd_fn=_pip_install_cmd,
    )


def _prepend_warning(warning: "str | None", result: str) -> str:
    """Prepend an environment warning to a tool result string, if present."""
    return f"{warning}\n{result}" if warning else result


# ── linter / tests / type check ──────────────────────────────────────────

def run_linter(path: str, language: str) -> str:
    """pylint, scoped to skip vendored dirs and the slow duplicate-code checker."""
    # Recurse but skip vendored/ignored dirs (.venv etc.), use all cores, and drop
    # duplicate-code — a bare `pylint .` otherwise lints thousands of vendored
    # files and times out. We prune by directory *basename* via --ignore (pylint
    # matches it against the basename of each dir during its recursive os.walk);
    # every _IGNORE entry is already a plain dir basename.
    image, warn = _project_image(path)
    cmd = (
        f"pylint --recursive=y --ignore={','.join(sorted(_IGNORE))} "
        "--jobs=0 --disable=duplicate-code ."
    )
    result = _run_docker_text(image, cmd, path, name="pylint", timeout=300)
    return _prepend_warning(warn, result)


def run_tests(path: str, language: str, include_coverage: bool = True) -> str:
    """pytest, scoped out of vendored dirs so it doesn't collect .venv's own tests."""
    # -o norecursedirs overrides any project config so collection never descends
    # into .venv/node_modules/build; explicit --ignore for the ones at the root;
    # -q / no cache to trim noise. pytest-cov is baked into the image, so --cov is
    # always available when coverage is requested.
    image, warn = _project_image(path)
    args = ["pytest", "-q", "-p", "no:cacheprovider",
            f"--override-ini=norecursedirs='{' '.join(sorted(_IGNORE))}'"]
    root = Path(path)
    for name in sorted(_IGNORE):
        if (root / name).is_dir():
            args.append(f"--ignore={name}")
    if include_coverage:
        args.append("--cov")
    args.append(".")
    result = _run_docker_text(image, " ".join(args), path, name="pytest", timeout=300)
    return _prepend_warning(warn, result)


def run_type_check(path: str, language: str) -> str:
    """mypy, excluding vendored dirs and silencing missing third-party stubs."""
    image, warn = _project_image(path)
    cmd = (
        f"mypy --exclude='{_vendor_ignore_regex()}' --ignore-missing-imports "
        "--no-error-summary ."
    )
    result = _run_docker_text(image, cmd, path, name="mypy", timeout=300)
    return _prepend_warning(warn, result)


def check_imports(path: str, language: str) -> str:
    """Static import health check (broken / unused / circular) — no code executed.

    Runs the bundled stdlib AST analyzer inside the dep-installed container so
    `find_spec` resolves the project's third-party imports (see import_check.py).
    """
    image, warn = _project_image(path)
    cmd = "python /checker/import_check.py /work"
    result = _run_docker_text(image, cmd, path, name="import check", timeout=300,
                              mounts=[(str(_SCRIPT_DIR), "/checker")])
    return _prepend_warning(warn, result)


# ── compile (byte-compile in Docker) ─────────────────────────────────────

def compile_code(path: str, language: str) -> CompileOutput:
    """Byte-compile every module via `compileall` in a container — a pure syntax
    check (no imports executed). Syntax errors come back as ErrorOutputs.

    Uses the plain base image (not the project image) and copy=False — compileall
    only needs the source files, not the installed deps, and running against /src
    directly avoids the copy entirely.
    """
    dr = _run_docker(_COMPILE_IMAGE, _COMPILE_CMD, path, copy=False,
                     env={"PYTHONPYCACHEPREFIX": "/tmp/pyc"})
    errors = _parse_python_compile(dr.output) if dr.error is None else []
    return _compile_result(dr, language, "compileall", errors, [])


def _parse_python_compile(out: str) -> "list[ErrorOutput]":
    """Pull (file, line, column, message) out of compileall's failure blocks."""
    lines = out.splitlines()
    errors: list[ErrorOutput] = []
    for i, line in enumerate(lines):
        m = _PY_FILE_RE.search(line)
        if not m:
            continue
        file, lineno = _compile_path(m.group(1)), int(m.group(2))
        column, message = 0, "SyntaxError"
        # The caret line (offset column) and the `XxxError: msg` line follow.
        for nxt in lines[i + 1:i + 6]:
            stripped = nxt.strip()
            if stripped and set(stripped) <= {"^", "~"} and column == 0:
                column = len(nxt) - len(nxt.lstrip()) + 1
            if _PY_MSG_RE.match(nxt):
                message = nxt.strip()
                break
        errors.append(ErrorOutput(file=file, line=lineno, column=column,
                                  message=message))
    return errors