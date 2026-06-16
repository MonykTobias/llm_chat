"""
Host-side sandbox helpers used programmatically by the graph nodes.

These are the non-LLM filesystem primitives the orchestrator needs directly:
listing what files exist right now, and restoring a pre-task snapshot when a
task fails. The full set of LLM-callable @tool wrappers (read_file,
list_directory, search_files, ...) will land alongside the real node
implementations; only the helpers the orchestrator references are ported here.

Ported from the standalone `orchestrator` project's `tools` package.
"""
import os

_SCAN_SKIP_DIRS = {"__pycache__", "node_modules", ".git", ".venv", "venv", ".idea"}


def safe_path(sandbox_dir: str, rel_path: str) -> str | None:
    """Return absolute path if it stays within the sandbox, None otherwise."""
    clean_rel_path = rel_path.lstrip('/')  # Strip leading slashes
    target = os.path.abspath(os.path.join(sandbox_dir, clean_rel_path))
    if not target.startswith(os.path.abspath(sandbox_dir)):
        return None
    return target


def list_workspace_files(sandbox_dir: str) -> list[str]:
    """Return a sorted list of relative file paths currently in the sandbox.

    The single source of truth for "what files exist right now" — call it
    whenever a fresh view is needed (e.g. after the coder writes files) so
    context never goes stale.
    """
    paths: list[str] = []
    for root, dirs, files in os.walk(sandbox_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in _SCAN_SKIP_DIRS)
        rel_root = os.path.relpath(root, sandbox_dir).replace("\\", "/")
        for fname in sorted(files):
            paths.append(f"{rel_root}/{fname}" if rel_root != "." else fname)
    return paths


def restore_snapshot(sandbox_dir: str, snapshot: dict) -> tuple[int, int]:
    """Restore pre-task file state. Bypasses stats counters. Returns (restored, deleted)."""
    restored = deleted = 0
    for rel_path, content in snapshot.items():
        target = safe_path(sandbox_dir, rel_path)
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
