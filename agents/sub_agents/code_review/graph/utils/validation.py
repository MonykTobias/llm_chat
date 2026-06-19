"""
Shared, deterministic path/tree guardrails used by the graph nodes.

These are HOST-SIDE helpers (plain Python, no LLM, no LangChain @tool, no
Pydantic). They are deliberately NOT in the `tools` package — that namespace is
reserved for LLM-callable tools (list_directory, read_file, search_files). The
functions here run in the orchestrator (and later the architect) to keep agent
decisions grounded in the canonical file tree, and they are the single source of
truth for "what counts as being on the tree" so the two agents can never
disagree.

Ported verbatim from the standalone `orchestrator` project's `validation`
package so the planner logic in `orchestrator.py` behaves identically here.
"""
import os
import re

# File names that the LLM tends to hallucinate or that are build artefacts —
# never legitimate plan targets.
_SYSTEM_FILE_BLOCKLIST = {
    "structured_output.json", "structured_output", "__pycache__",
}
_SYSTEM_FILE_SUFFIXES = (".pyc", ".pyo", ".pyd")

# Source-file extensions, ordered LONGEST-FIRST so the alternation never matches
# a shorter prefix (e.g. 'js' inside 'manifest.json'). Regex alternation is
# first-match-wins, so 'json' must precede 'js', 'jsx'/'tsx' precede 'js'/'ts'.
_PATH_RE = re.compile(
    r'[\w\-./\\]+\.(?:json|jsx|tsx|yaml|yml|html|css|md|txt|py|ts|js)'
)


def normalize(path: str) -> str:
    """Normalize a relative path: forward slashes, no './' prefix, no leading '/'."""
    p = (path or "").replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    return p.lstrip("/")


def is_system_file(path: str) -> bool:
    """True for build artefacts / hallucinated system files that must never be planned."""
    name = os.path.basename(path)
    return (
        name in _SYSTEM_FILE_BLOCKLIST
        or any(name.endswith(s) for s in _SYSTEM_FILE_SUFFIXES)
        or "__pycache__" in path
    )


def extract_paths(text: str) -> list[str]:
    """Extract explicit file paths mentioned in free text, preserving first-seen order."""
    return list(dict.fromkeys(_PATH_RE.findall(text or "")))


def on_tree(path: str, tree: list[str]) -> bool:
    """
    True if `path` belongs to the canonical file tree.

    A path is on the tree if its normalized form is an exact member, OR its
    basename matches the basename of any tree path. This mirrors the membership
    rule used by architect._enforce_file_tree (exact path or basename match) so
    the orchestrator and architect always agree on what is "on the tree".
    """
    if not tree:
        return False
    norm = normalize(path)
    tree_norm = [normalize(p) for p in tree]
    if norm in tree_norm:
        return True
    base = os.path.basename(norm).lower()
    return any(os.path.basename(p).lower() == base for p in tree_norm)
