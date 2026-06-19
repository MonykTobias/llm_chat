"""Cross-cutting helpers shared by the tool submodules.

Path confinement, output truncation, the vendored-directory ignore set, and the
small file-read helpers live here so the domain modules (filesystem, language,
web, change_tracking) can depend on them without depending on each other.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from langgraph.config import get_stream_writer

_IGNORE = {"__pycache__", "node_modules", ".git", ".venv", "venv",
           ".idea", ".mypy_cache", ".pytest_cache", "dist", "build"}

# Hard cap on how much text any single tool may hand back to the model. A linter
# or test run over a large project can emit tens of thousands of lines; returned
# whole it lands as one giant ToolMessage that blows past the model's num_ctx and
# leaves it stuck. We keep the head and a slice of the tail (the tail usually
# holds the summary / failure count) and drop the middle.
_TOOL_MAX_CHARS = 8_000

from langgraph.config import get_stream_writer

def _w(text: str) -> None:
    try:
        writer = get_stream_writer()
        writer({"kind": "text", "text": text + "\n\n"})
    except Exception:
        pass  # outside streaming context — silently skip

def _truncate(text: str, max_chars: int = _TOOL_MAX_CHARS) -> str:
    """Bound `text` to `max_chars`, keeping head + tail with a dropped-middle note."""
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.7)
    tail = max_chars - head
    omitted = len(text) - head - tail
    return (f"{text[:head]}\n\n[… output truncated, {omitted} chars omitted "
            f"to fit the context window …]\n\n{text[-tail:]}")


def _vendor_ignore_regex() -> str:
    """Regex (matching either path separator) for the _IGNORE directories.

    Used to keep pylint/mypy from descending into .venv, node_modules, etc.,
    which is what makes a whole-project-root scan blow past the timeout.
    """
    names = "|".join(re.escape(n) for n in sorted(_IGNORE))
    sep = r"[\\/]"
    return rf"(^|.*{sep})(?:{names})({sep}.*|$)"


def _read_text_or_none(abs_path: str) -> "str | None":
    """Quietly read a file as UTF-8, or return None if it's missing/unreadable."""
    p = Path(abs_path)
    if not p.is_file():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return None


def _rel(project_path: str, abs_path: str) -> str:
    """Best-effort path relative to the project root, for readable reports."""
    try:
        return str(Path(abs_path).relative_to(Path(project_path).resolve()))
    except ValueError:
        return abs_path


def _safe_path(file_path: str, project_path: str) -> Path:
    """Resolve file_path and confine it to project_path.

    Accepts a relative path (resolved against the project root) or an absolute
    path, but raises ValueError if the resolved location escapes the project
    root via ``..`` or an out-of-tree absolute path.
    """
    root = Path(project_path).resolve()
    target = Path(file_path)
    p = (target if target.is_absolute() else root / target).resolve()
    if p != root and root not in p.parents:
        raise ValueError(f"'{file_path}' is outside the project root '{root}'.")
    return p
