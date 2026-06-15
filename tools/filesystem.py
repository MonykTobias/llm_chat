"""Filesystem tools: read, write, delete, and list files inside the project.

Every write/delete is snapshotted (via change_tracking) before it happens so the
change can be reported or reverted later. All paths are confined to the project
root by `_safe_path`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from structured_output import ReviewState

from ._common import _IGNORE, _safe_path
from .change_tracking import _record_snapshot


@tool
def read_file(file_path: str, *, state: Annotated[ReviewState, InjectedState]) -> str:
    """Read the complete text contents of a file inside the project."""
    print(f"Reading file: {file_path}")
    try:
        p = _safe_path(file_path, state["project_path"])
    except ValueError as e:
        return f"Refused: {e}"
    return _read_file(str(p))


@tool
def list_all_files(*, state: Annotated[ReviewState, InjectedState]) -> str:
    """List all files in the project under review."""
    print(f"Listing all files in {state['project_path']}")
    return _list_all_files(state["project_path"])


@tool
def write_file(file_path: str, content: str, *, state: Annotated[ReviewState, InjectedState]) -> str:
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
def delete_file(file_path: str, *, state: Annotated[ReviewState, InjectedState]) -> str:
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
