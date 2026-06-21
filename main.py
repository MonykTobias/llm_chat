"""
Minimal entrypoint to run validator_node in isolation.

Usage:
    python main.py --project /path/to/project --files src/foo.py src/bar.py

The root problem: LangGraph >= 1.0.2 requires a runtime object injected by the
graph executor for InjectedState to work. Outside a graph there is no way to
construct this object, so ToolNode always crashes with:
    "Missing required config key 'N/A' for 'tools'."

The fix used here: monkey-patch ToolNode.invoke so that when it is called from
the validator loop it calls the underlying tool functions directly, injecting
the state ourselves — exactly what the graph executor would do.
"""
from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig


# ── stream writer stub ───────────────────────────────────────────────────────

def _make_writer():
    def writer(event: dict):
        kind = event.get("kind")
        if kind == "stage":
            print(f"\n[STAGE] {event.get('label', '')}", flush=True)
        elif kind == "text":
            print(event.get("text", ""), end="", flush=True)
        else:
            print(f"[EVENT] {event}", flush=True)
    return writer


# ── minimal AgentState ───────────────────────────────────────────────────────

def _build_state(project_path: str, files: list[str], language: str) -> dict:
    plan = SimpleNamespace(instruction_for_agent="Validate the changed files.")
    return {
        "plan": plan,
        "project_path": project_path,
        "language": language,
        "context_store": {
            "coder_latest_files": {"current": files},
        },
        "latest_report": "",
        "coder_retries": 0,
        "history": [],
        "messages": [],
    }


# ── ToolNode replacement that injects state directly ─────────────────────────

def _make_tool_node_patch(state: dict):
    """
    Returns a ToolNode subclass whose .invoke() bypasses LangGraph's runtime
    injection machinery and calls tool functions directly with state injected.

    This is needed because LangGraph >= 1.0.2 only populates CONFIG_KEY_RUNTIME
    during real graph execution — there is no way to construct it manually.
    """
    from langgraph.prebuilt import ToolNode
    from langchain_core.tools import BaseTool
    from langgraph.types import Command

    class DirectToolNode(ToolNode):
        def invoke(self, input: dict, config=None, **kwargs):
            messages = input.get("messages", [])
            # Find the last AIMessage with tool_calls
            ai_msg = next(
                (m for m in reversed(messages)
                 if hasattr(m, "tool_calls") and m.tool_calls),
                None,
            )
            if not ai_msg:
                return {"messages": []}

            results = []
            for tc in ai_msg.tool_calls:
                tool_name = tc["name"]
                tool_args = tc.get("args", {})
                tool_id   = tc.get("id", "")

                # Find the matching tool in our list
                matched: BaseTool | None = next(
                    (t for t in self.tools_by_name.values()
                     if t.name == tool_name),
                    None,
                )
                if matched is None:
                    results.append(ToolMessage(
                        content=f"Tool '{tool_name}' not found.",
                        tool_call_id=tool_id,
                        name=tool_name,
                    ))
                    continue

                try:
                    # Inject state and tool_call_id into kwargs if the tool wants them
                    import inspect
                    sig = inspect.signature(matched.func)
                    extra = {}
                    if "state" in sig.parameters:
                        extra["state"] = state
                    if "tool_call_id" in sig.parameters:
                        extra["tool_call_id"] = tool_id

                    result = matched.func(**tool_args, **extra)

                    # Handle Command returns from set_language — extract the
                    # ToolMessage from the Command update and apply language change
                    if isinstance(result, Command):
                        update = result.update or {}
                        if "language" in update:
                            state["language"] = update["language"]
                            print(f"  [language set to: {update['language']}]",
                                  flush=True)
                        msgs = update.get("messages", [])
                        content = msgs[0].content if msgs else "ok"
                        results.append(ToolMessage(
                            content=content,
                            tool_call_id=tool_id,
                            name=tool_name,
                        ))
                    else:
                        results.append(ToolMessage(
                            content=str(result),
                            tool_call_id=tool_id,
                            name=tool_name,
                        ))
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    results.append(ToolMessage(
                        content=f"Error in tool '{tool_name}': {e}",
                        tool_call_id=tool_id,
                        name=tool_name,
                        status="error",
                    ))

            return {"messages": results}

    return DirectToolNode


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run validator_node standalone.")
    parser.add_argument("--project", required=True,
                        help="Absolute path to the project root.")
    parser.add_argument("--files", nargs="+", required=True,
                        help="Project-relative file paths to validate.")
    parser.add_argument("--language", default="python",
                        help="Language hint passed to tools (default: python).")
    args = parser.parse_args()

    state  = _build_state(args.project, args.files, args.language)
    writer = _make_writer()

    from agents.implementations.code_agent.validator.validator import validator_node

    DirectToolNode = _make_tool_node_patch(state)

    with (
        patch("agents.implementations.code_agent.validator.validator.get_stream_writer",
              return_value=writer),
        patch("agents.implementations.code_agent.validator.validator.ToolNode",
              DirectToolNode),
    ):
        config = RunnableConfig(configurable={})
        result = validator_node(state, config)

    print("\n\n── validator_node returned ──────────────────────────────────────")
    print(json.dumps(
        {k: v for k, v in result.items() if k != "context_store"},
        indent=2,
        default=str,
    ))
    if "context_store" in result:
        print("context_store keys:", list(result["context_store"].keys()))
        if "validation_issues" in result["context_store"]:
            print("\nvalidation_issues:")
            print(result["context_store"]["validation_issues"])


if __name__ == "__main__":
    main()