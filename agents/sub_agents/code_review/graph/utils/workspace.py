"""
Snapshot restore used by the orchestrator's failure-recovery path.

`list_workspace_files` now lives in the central tools package (project_path
based) and is re-exported here so existing imports keep working. The
`restore_snapshot` below uses the orchestrator's own {path: content} snapshot
dict semantics (distinct from the tools package's session-snapshot revert), so
it stays local.
"""
import os

from tools import list_workspace_files  # re-exported for the orchestrator import

__all__ = ["list_workspace_files", "restore_snapshot", "safe_path"]


def safe_path(root: str, rel_path: str) -> str | None:
    """Return absolute path if it stays within `root`, None otherwise."""
    clean_rel_path = rel_path.lstrip("/")
    target = os.path.abspath(os.path.join(root, clean_rel_path))
    if not target.startswith(os.path.abspath(root)):
        return None
    return target


def restore_snapshot(root: str, snapshot: dict) -> tuple[int, int]:
    """Restore a {rel_path: content|None} snapshot. content=None means the file
    did not exist before, so it is removed. Returns (restored, deleted)."""
    restored = deleted = 0
    for rel_path, content in snapshot.items():
        target = safe_path(root, rel_path)
        if not target:
            continue
        if content is not None:
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
            restored += 1
        elif os.path.exists(target):
            os.remove(target)
            deleted += 1
    return restored, deleted
