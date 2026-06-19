"""Filesystem tools: read, write, delete, and list files inside the project.

Every write/delete is snapshotted (via change_tracking) before it happens so the
change can be reported or reverted later. All paths are confined to the project
root by `_safe_path`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from langgraph.prebuilt import InjectedState

from structured_output import ReviewState

from ._common import _IGNORE, _read_text_or_none, _safe_path
from .change_tracking import _record_snapshot


@tool
def read_file(file_path: str, *, state: Annotated[dict, InjectedState]) -> str:
    """Read the complete text contents of a file inside the project."""
    print(f"Reading file: {file_path}")
    try:
        p = _safe_path(file_path, state["project_path"])
    except ValueError as e:
        return f"Refused: {e}"
    return _read_file(str(p))


@tool
def list_all_files(*, state: Annotated[dict, InjectedState]) -> str:
    """List all files in the project under review."""
    print(f"Listing all files in {state['project_path']}")
    all_files =_list_all_files(state["project_path"])
    return all_files


@tool
def write_file(file_path: str, content: str, *, state: Annotated[dict, InjectedState]) -> str:
    """Write content to a file inside the project."""
    print(f"Writing file: {file_path}")
    try:
        p = _safe_path(file_path, state["project_path"])
    except ValueError as e:
        return f"Refused: {e}"
    # Stash the pre-edit state before the first write so the change can be
    # reported or reverted later (snapshots live for the server's lifetime).
    _record_snapshot(state["project_path"], str(p))
    return _write_file(str(p), content)


@tool
def delete_file(file_path: str, *, state: Annotated[dict, InjectedState]) -> str:
    """Delete a file from the project."""
    print(f"Deleting file: {file_path}")
    try:
        p = _safe_path(file_path, state["project_path"])
    except ValueError as e:
        return f"Refused: {e}"
    _record_snapshot(state["project_path"], str(p))
    return _delete_file(str(p))


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


def _write_file(file_path: str, content: str) -> str:
    """Write text content to a file inside the sandbox."""
    try:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"File {file_path} written successfully."
    except Exception as e:
        return f"Error writing file {file_path}: {e}"


def _delete_file(file_path: str) -> str:
    """Delete a file inside the sandbox."""
    try:
        Path(file_path).unlink()
        return f"File {file_path} deleted successfully."
    except Exception as e:
        return f"Error deleting file {file_path}: {e}"


# ── Programmatic (non-LLM) helpers for graph nodes ────────────────────────────
# These mirror the `@tool` wrappers above but take an explicit project root and a
# relative path, so a graph node can read/write/delete deterministically from its
# own code instead of depending on the model to emit the right tool call. Writes
# and deletes snapshot first (like write_file/delete_file), so the UI's revert and
# build_change_report keep working. project_path is the folder under review
# (state["project_path"]); paths are confined to it by `_safe_path`.

def safe_read(project_path: str, file_path: str) -> "str | None":
    """Read a file confined to project_path. Returns its text, or None if the path
    is outside the project or the file is missing/unreadable."""
    try:
        p = _safe_path(file_path, project_path)
    except ValueError:
        return None
    text = _read_text_or_none(str(p))
    return text


def safe_write(project_path: str, file_path: str, content: str) -> tuple[bool, str]:
    """Write content to a file confined to project_path. Snapshots the pre-edit
    state first so the change can be reported/reverted. Returns (ok, message)."""
    try:
        p = _safe_path(file_path, project_path)
    except ValueError as e:
        return False, f"Refused: {e}"

    _record_snapshot(project_path, str(p))
    msg = _write_file(str(p), content)
    print(f"Wrote to file: {file_path}")
    return (not msg.startswith("Error")), msg


def safe_delete(project_path: str, file_path: str) -> tuple[bool, str]:
    """Delete a file confined to project_path, snapshotting it first. Returns
    (ok, message)."""
    try:
        p = _safe_path(file_path, project_path)
    except ValueError as e:
        return False, f"Refused: {e}"
    _record_snapshot(project_path, str(p))
    msg = _delete_file(str(p))
    return (not msg.startswith("Error")), msg


def list_workspace_files(project_path: str) -> list[str]:
    """Sorted list of relative file paths currently under project_path.

    The single source of truth for "what files exist right now" — call it after a
    write so context never goes stale. Skips the vendored/build dirs in `_IGNORE`.
    """
    root = Path(project_path)
    if not root.exists():
        return []
    paths: list[str] = []
    for entry in root.rglob("*"):
        if not entry.is_file():
            continue
        if _IGNORE & set(entry.relative_to(root).parts):
            continue
        paths.append(str(entry.relative_to(root)).replace("\\", "/"))
    return sorted(paths)