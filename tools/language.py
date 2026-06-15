"""Language tooling: linter, test runner, type checker, and architecture map.

Each public `@tool` reads the active language/path off the graph state and routes
to the right command for that language. Python runs go through the project's own
venv interpreter and are scoped to skip vendored dirs (.venv, node_modules, …)
so a whole-root scan doesn't time out or overflow the model's context window.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from structured_output import ReviewState

from ._common import _IGNORE, _truncate, _vendor_ignore_regex


@tool
def run_linter(state: Annotated[ReviewState, InjectedState]) -> str:
    """Run the appropriate linter for the detected language."""
    print(f"Running linter for {state['language']}")
    return _run_linter(state["project_path"], state["language"])


@tool
def run_tests(include_coverage: bool = True, *,
              state: Annotated[ReviewState, InjectedState]) -> str:
    """Execute the test suite and capture coverage."""
    print(f"Running tests with coverage: {include_coverage}")
    return _run_tests(state["project_path"], state["language"], include_coverage)


@tool
def run_type_check(state: Annotated[ReviewState, InjectedState]) -> str:
    """Run type checking / static type analysis."""
    print("Running type check...")
    return _run_type_check(state["project_path"], state["language"])


@tool
def analyze_architecture(depth: int = 3, *,
                         state: Annotated[ReviewState, InjectedState]) -> str:
    """Analyze code structure, imports, and dependencies."""
    print(f"Analyzing architecture with depth: {depth}")
    return _analyze_architecture(state["project_path"], state["language"], depth)


def _run_linter(path: str, language: str) -> str:
    """Route to the right linter for the language"""
    if language == "python":
        # Scope the scan: recurse, but skip vendored/ignored dirs (.venv etc.),
        # run across all cores, and drop the slow duplicate-code checker — a bare
        # `pylint <root>` otherwise tries to lint thousands of .venv files and
        # times out. Give it a longer budget than the default since real projects
        # can still be large.
        args = [
            "--recursive=y",
            f"--ignore-paths={_vendor_ignore_regex()}",
            "--jobs=0",
            "--disable=duplicate-code",
            path,
        ]
        return _run_python_tool(_venv_python(path), "pylint", args, timeout=300)

    linters = {
        "python": ["pylint", path],
        "javascript": ["eslint", path],
        "typescript": ["eslint", path],
        "go": ["golangci-lint", "run", path],
        "rust": ["cargo", "clippy"],
        "java": ["checkstyle", path],
    }

    cmd = linters.get(language)
    if not cmd:
        return f"No linter configured for {language}"

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return _truncate(result.stdout + result.stderr)
    except Exception as e:
        return f"Linter failed: {e}"


def _run_tests(path: str, language: str, include_coverage: bool = True) -> str:
    """Route to test runner based on language"""
    if language == "python":
        # Same scoping concern as the linter/type-check: a bare `pytest <root>`
        # collects EVERY test under the tree — including the thousands shipped
        # inside the project's own .venv/site-packages — which is slow and dumps
        # a giant result that overflows the model's context. Keep pytest out of
        # the vendored/ignored dirs:
        #   * -o norecursedirs=... overrides any project pytest config so
        #     collection never descends into .venv, node_modules, build, etc.
        #   * explicit --ignore=<abs> for the ignored dirs that exist at the root
        #     (belt-and-suspenders if a project config drops the default).
        #   * -q / no cache to trim noise.
        args = ["-q", "-p", "no:cacheprovider",
                f"--override-ini=norecursedirs={' '.join(sorted(_IGNORE))}"]
        root = Path(path)
        for name in sorted(_IGNORE):
            if (root / name).is_dir():
                args.append(f"--ignore={root / name}")
        args.append(path)
        if include_coverage:
            args.append("--cov")
        return _run_python_tool(_venv_python(path), "pytest", args)

    test_commands = {
        "python": (["pytest", path] + (["--cov"] if include_coverage else [])),
        "javascript": (["npm", "test"] + (["--", "--coverage"] if include_coverage else [])),
        "typescript": (["npm", "test"] + (["--", "--coverage"] if include_coverage else [])),
        "go": (["go", "test", "./...", "-v"] + (["-coverprofile=coverage.out"] if include_coverage else [])),
        "rust": (["cargo", "test"] + (["--", "--nocapture"] if include_coverage else [])),
        "java": (["mvn", "test"] + (["jacoco:report"] if include_coverage else [])),
    }

    cmd = test_commands.get(language)
    if not cmd:
        return f"No test runner configured for {language}"

    try:
        result = subprocess.run(cmd, cwd=path, capture_output=True, text=True, timeout=60)
        return _truncate(result.stdout + result.stderr)
    except Exception as e:
        return f"Tests failed: {e}"


def _run_type_check(path: str, language: str) -> str:
    """Route to type checker based on language"""
    if language == "python":
        # Same scoping concern as the linter: exclude vendored/ignored dirs so
        # mypy doesn't try to type-check the whole .venv and time out. Silence
        # missing third-party stubs, which are noise for a review.
        args = [
            f"--exclude={_vendor_ignore_regex()}",
            "--ignore-missing-imports",
            "--no-error-summary",
            path,
        ]
        return _run_python_tool(_venv_python(path), "mypy", args, timeout=300)

    type_checkers = {
        "python": ["mypy", path],
        "typescript": ["tsc", "--noEmit"],
        "java": [],  # Built-in to javac
        "go": ["go", "vet", "./..."],
        "rust": ["cargo", "check"],
    }

    cmd = type_checkers.get(language)
    if not cmd:
        return f"No type checker configured for {language}"

    try:
        result = subprocess.run(cmd, cwd=path, capture_output=True, text=True, timeout=30)
        return _truncate(result.stdout + result.stderr)
    except Exception as e:
        return f"Type check failed: {e}"


def _analyze_architecture(path: str, language: str, depth: int = 3) -> str:
    """Language-agnostic architecture analysis via imports/dependencies"""
    # Use tree command or custom parser
    # Works for any language: look at imports, file structure, module organization
    root = Path(path)
    if not root.exists():
        return f"Directory not found: {path}"

    lines = [str(root)]

    def walk(directory: path, prefix: str, level: int):
        if level > depth:
            return
        try:
            entries = sorted(
                ( e for e in directory.iterdir() if e.name not in _IGNORE),
                key=lambda e: (e.is_file(), e.name.lower()),
            )
        except PermissionError:
            return
        for i, entry in enumerate(entries):
            last = i == len(entries) - 1
            lines.append(f"{prefix}{'└── ' if last else '├── '}{entry.name}")
            if entry.is_dir():
                walk(entry, prefix + ("    " if last else "│   "), level + 1)

    walk(root, "", 1)
    return "\n".join(lines)


def _run_python_tool(python_exe: str, module: str, args: list[str],
                     timeout: int = 120) -> str:
    print(f"Running python: {python_exe} with {module} with args: {args}")

    try:
        result = subprocess.run(
            [python_exe, "-m", module, *args],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return (f"[tool timeout] '{module}' did not finish within {timeout}s. "
                f"The scan is likely too broad — narrow it to the source "
                f"directory or specific files rather than the whole project root.")
    except Exception as e:
        return f"[tool error] could not launch {module}: {e}"

    out = (result.stdout + result.stderr).strip()
    if f"No module named {module}" in out:
        return (f"[tool unavailable] '{module}' is not installed in the agent's "
                f"environment. This is an environment problem, NOT a defect in the "
                f"reviewed project. Do not report it as a code finding.")
    return _truncate(out) or f"{module} ran and produced no output."


def _venv_python(project_path: str) -> str:
    """Find the project's venv interpreter; fall back to the agent's own."""
    root = Path(project_path).resolve()
    # check the project dir and a couple of parents (venv often sits at repo root)
    for base in (root, *list(root.parents)[:2]):
        for name in (".venv", "venv"):
            for rel in ("Scripts/python.exe", "bin/python"):  # Windows, then Unix
                candidate = base / name / rel
                if candidate.exists():
                    return str(candidate)
    return sys.executable
