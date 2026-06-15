"""Change tracking: in-memory write snapshots, diff report, and revert.

Before the agent modifies a file we stash its original content here so the
change can be reported (build_change_report) or undone (restore_snapshot).
Snapshots are keyed by resolved project root -> {absolute_file_path:
original_text | None}. A value of None means "the file did not exist before the
first write", so a revert deletes it. This is deliberately NOT persisted to
disk: snapshots live only as long as the server process does.
"""
from __future__ import annotations

import difflib
import threading
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from structured_output import ReviewState

from ._common import _read_text_or_none, _rel, _safe_path

_SNAPSHOTS: dict[str, dict[str, "str | None"]] = {}
_SNAPSHOT_LOCK = threading.RLock()


@tool
def build_change_report(file_paths: "list[str] | None" = None, *,
                        state: Annotated[ReviewState, InjectedState]) -> str:
    """Summarize what changed this session as unified diffs.

    Compares each touched file against the snapshot taken before the agent first
    edited it. With no argument, reports on EVERY file written so far this
    session — a fast way to evaluate the changes without re-reading the tree.
    """
    print(f"Building change report for: {file_paths or 'all written files'}")
    return _build_change_report(state["project_path"], file_paths)


def _snapshot_key(project_path: str) -> str:
    return str(Path(project_path).resolve())


def _record_snapshot(project_path: str, abs_path: str) -> None:
    """Capture a file's pre-edit content once, before its first write this session."""
    key = _snapshot_key(project_path)
    with _SNAPSHOT_LOCK:
        files = _SNAPSHOTS.setdefault(key, {})
        if abs_path in files:           # already snapshotted — keep the original
            return
        files[abs_path] = _read_text_or_none(abs_path)


def _build_change_report(project_path: str, file_paths: "list[str] | None" = None) -> str:
    """Unified-diff report of snapshotted files vs. their current on-disk state."""
    key = _snapshot_key(project_path)
    with _SNAPSHOT_LOCK:
        snapshot = dict(_SNAPSHOTS.get(key, {}))

    if file_paths:
        targets: list[str] = []
        for fp in file_paths:
            try:
                targets.append(str(_safe_path(fp, project_path)))
            except ValueError:
                continue
    else:
        targets = list(snapshot.keys())

    if not targets:
        return "No file changes have been recorded this session."

    parts: list[str] = []
    for abs_path in targets:
        before = snapshot.get(abs_path)
        after = _read_text_or_none(abs_path)
        rel = _rel(project_path, abs_path)

        if after is None:
            parts.append(f"=== {rel} ===\n[DELETED]")
        elif before is None:
            parts.append(f"=== {rel} ===\n[NEW FILE]\n{after}")
        elif before == after:
            parts.append(f"=== {rel} ===\n[UNCHANGED in this session]")
        else:
            diff = "".join(difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"{rel} (before)",
                tofile=f"{rel} (after)",
                n=3,
            ))
            parts.append(
                f"=== {rel} ===\n"
                f"DIFF (what changed in this session):\n{diff}\n"
                f"--- FULL FILE AFTER CHANGES ---\n{after}"
            )
    return "\n\n".join(parts)


def restore_snapshot(project_path: str, file_paths: "list[str] | None" = None) -> tuple[int, int]:
    """Restore snapshotted files to their pre-edit state. Returns (restored, deleted).

    With no file_paths, reverts every file the agent wrote this session. Called
    by the UI's revert button; snapshots are kept so a report still reflects the
    (now reverted) state.
    """
    key = _snapshot_key(project_path)
    with _SNAPSHOT_LOCK:
        snapshot = _SNAPSHOTS.get(key, {})
        if file_paths:
            wanted: set[str] = set()
            for fp in file_paths:
                try:
                    wanted.add(str(_safe_path(fp, project_path)))
                except ValueError:
                    continue
            items = [(p, snapshot[p]) for p in wanted if p in snapshot]
        else:
            items = list(snapshot.items())

        restored = deleted = 0
        for abs_path, content in items:
            p = Path(abs_path)
            try:
                if content is not None:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(content, encoding="utf-8")
                    restored += 1
                elif p.exists():
                    p.unlink()
                    deleted += 1
            except Exception:
                continue
        return restored, deleted
