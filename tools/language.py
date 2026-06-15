"""Language tooling: linter, test runner, type checker, import check, and
architecture map.

Each public `@tool` reads the active language/path off the graph state and
delegates to the per-language implementation in the `tools.languages` package
(one module per language, each documenting the external tools a project needs).
`analyze_architecture` is language-agnostic and stays here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from structured_output import ReviewState

from ._common import _IGNORE
from .languages import (
    check_imports as _lang_check_imports,
    run_linter as _lang_run_linter,
    run_tests as _lang_run_tests,
    run_type_check as _lang_run_type_check,
)


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


@tool
def check_imports(state: Annotated[ReviewState, InjectedState]) -> str:
    """Statically check whether the project's imports are correctly handled.

    Reports three classes of problem: broken/unresolvable imports (a module or
    package that can't be found — typo or missing dependency), unused imports
    (an imported name that is never referenced), and circular imports (modules
    that import each other). For Python this is a pure-stdlib AST analysis that
    never executes the project's code; for other languages it delegates to the
    language's own compiler/type-checker, which surfaces unresolved imports."""
    print(f"Checking imports for {state['language']}")
    return _check_imports(state["project_path"], state["language"])


# ── delegations to the per-language implementations ──────────────────────

def _run_linter(path: str, language: str) -> str:
    return _lang_run_linter(path, language)


def _run_tests(path: str, language: str, include_coverage: bool = True) -> str:
    return _lang_run_tests(path, language, include_coverage)


def _run_type_check(path: str, language: str) -> str:
    return _lang_run_type_check(path, language)


def _check_imports(path: str, language: str) -> str:
    return _lang_check_imports(path, language)


# ── architecture map (language-agnostic) ─────────────────────────────────

def _analyze_architecture(path: str, language: str, depth: int = 3) -> str:
    """Language-agnostic architecture analysis via file structure / module layout."""
    root = Path(path)
    if not root.exists():
        return f"Directory not found: {path}"

    lines = [str(root)]

    def walk(directory: path, prefix: str, level: int):
        if level > depth:
            return
        try:
            entries = sorted(
                (e for e in directory.iterdir() if e.name not in _IGNORE),
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
