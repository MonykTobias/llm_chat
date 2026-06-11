import json
import subprocess
import sys
from pathlib import Path

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from typing import Annotated
from structured_output import ReviewState

_IGNORE = {"__pycache__", "node_modules", ".git", ".venv", "venv",
           ".idea", ".mypy_cache", ".pytest_cache", "dist", "build"}

# ── tools (used by ReAct agents) ────────────────────────────────────────
@tool
def run_linter(state: Annotated[ReviewState,InjectedState]) -> str:
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

@tool
def read_file(file_path: str, *, state: Annotated[ReviewState, InjectedState]) -> str:
    """Read the complete text contents of a file inside the project."""
    print(f"Reading file: {file_path}")
    p = Path(file_path)
    if not p.is_absolute():
        p = Path(state["project_path"]) / p
    return _read_file(str(p))

@tool
def list_all_files(*, state: Annotated[ReviewState, InjectedState]) -> str:
    """List all files in the project under review."""
    print(f"Listing all files in {state['project_path']}")
    return _list_all_files(state["project_path"])

# ── tools implementations (used by tool calls) ───────────────────────────

def _read_file(file_path: str) -> str:
    """Read the complete text contents of a file inside the sandbox."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return (
            f"FILE DOES NOT EXIST: '{file_path}'. "
            "STOP reading files. Do NOT attempt to read any variation of this path. "
        )
    except Exception as e:
        return f"Error reading file {file_path}: {e}"


def _run_linter(path: str, language: str) -> str:
    """Route to the right linter for the language"""
    if language == "python":
        return _run_python_tool(_venv_python(path), "pylint", [path])

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
        return result.stdout + result.stderr
    except Exception as e:
        return f"Linter failed: {e}"


def _run_tests(path: str, language: str, include_coverage: bool = True) -> str:
    """Route to test runner based on language"""
    if language == "python":
        return _run_python_tool(_venv_python(path), "pytest", [path] + (["--cov"] if include_coverage else []))

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
        return result.stdout + result.stderr
    except Exception as e:
        return f"Tests failed: {e}"


def _run_type_check(path: str, language: str) -> str:
    """Route to type checker based on language"""
    if language == "python":
        return _run_python_tool(_venv_python(path),"mypy", [path])

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
        return result.stdout + result.stderr
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

def _list_all_files(path: str) -> str:
    """List all files in the sandbox."""
    root = Path(path)
    if not root.exists():
        return f"Directory not found: {path}"

    paths: list[str] = []
    for entry in root.rglob("*"):
        if not entry.is_file():
            continue
        if _IGNORE & set(entry.relative_to(root).parts):
            continue
        paths.append(str(entry))
    return json.dumps(paths, indent=2)

def _run_python_tool(python_exe: str, module: str, args: list[str]) -> str:
    print(f"Running python: {python_exe} with {module} with args: {args}")

    try:
        result = subprocess.run(
            [python_exe, "-m", module, *args],
            capture_output=True, text=True, timeout=120,
        )
    except Exception as e:
        return f"[tool error] could not launch {module}: {e}"

    out = (result.stdout + result.stderr).strip()
    if f"No module named {module}" in out:
        return (f"[tool unavailable] '{module}' is not installed in the agent's "
                f"environment. This is an environment problem, NOT a defect in the "
                f"reviewed project. Do not report it as a code finding.")
    return out or f"{module} ran and produced no output."

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