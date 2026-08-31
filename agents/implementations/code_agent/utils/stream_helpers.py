"""
Tool-activity streaming for the orchestrator's graph nodes.

The graph nodes (coder, architect, validator, inspector) run their tool calls in
an inline `ToolNode.invoke()` loop. Those ToolMessages stay LOCAL to the node —
they are never returned to graph state — so they never reach LangGraph's
`stream_mode="messages"` stream. The UI server can therefore see a tool-call
AIMessage (and emit a `tool_start`) but can NEVER see the matching result, so it
could never emit a `tool_end`. That is what left tool spinners running forever,
and why the coder's programmatic file writes (which aren't tool calls at all)
showed no bubble.

The fix: each node emits its own `tool_start`/`tool_end` events on the graph's
"custom" stream — the channel the UI server already understands (see
`ui/server.py`, kinds `tool_start`/`tool_end`). We emit a matched start→end pair
per call so the frontend's per-name bubble bookkeeping stays unambiguous even
when one step calls the same tool more than once: a pair fully settles before the
next begins, so no bubble is ever left spinning.
"""
from __future__ import annotations

import os

# Arg keys, in priority order, that name a file/dir a tool acted on. Mirrors the
# UI server's native-stream extractor (`_extract_target`) so graph-node bubbles
# read identically to the native-agent ones.
_TARGET_KEYS = ("file_path", "path", "filepath", "filename", "directory", "dir_path")


def tool_target(args: dict | None) -> dict | None:
    """Shape a tool call's args into the bubble target the frontend expects:
    `{path, dir, name}` for file/dir tools, `{url, query}` for web_browse, or
    None for path-less tools (lin/test/etc.), which render as a plain pill."""
    if not isinstance(args, dict):
        return None
    for key in _TARGET_KEYS:
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            val = val.strip()
            return {"path": val,
                    "dir": os.path.dirname(val) or ".",
                    "name": os.path.basename(val) or val}
    url, query = args.get("url"), args.get("query")
    if (isinstance(url, str) and url.strip()) or (isinstance(query, str) and query.strip()):
        return {"url": url.strip() if isinstance(url, str) and url.strip() else None,
                "query": query.strip() if isinstance(query, str) and query.strip() else None}
    return None


def tool_start(writer, name: str, target: dict | None = None) -> None:
    """Open a spinning tool bubble. Pair every call with `tool_end`."""
    if writer is None:
        return
    writer({"kind": "tool_start", "name": name, "target": target})


def tool_end(writer, name: str, target: dict | None = None) -> None:
    """Settle the open bubble for `name` with its final target."""
    if writer is None:
        return
    writer({"kind": "tool_end", "name": name, "target": target})


def stream_tool_calls(writer, tool_calls) -> None:
    """Emit a settled (start→end) bubble for each executed tool call, in order.

    Use this for read-only tool loops, where the calls have already run by the
    time we report them — each pair appears as one finished pill naming the file
    (or site) touched, with no spinner left hanging.
    """
    if writer is None:
        return
    for tc in tool_calls or []:
        name = (tc.get("name") if isinstance(tc, dict) else None) or "tool"
        args = tc.get("args") if isinstance(tc, dict) else None
        target = tool_target(args)
        writer({"kind": "tool_start", "name": name, "target": target})
        writer({"kind": "tool_end", "name": name, "target": target})
