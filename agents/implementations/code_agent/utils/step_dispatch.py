"""
Step-dispatch — pops the next architect step off the queue and loads it for the coder.

Sits between a validator PASS and the coder when architect_step_queue is
non-empty. Resets coder_retries and architect_replans so each step gets its own
full retry budget. Stripped of the snapshot bookkeeping the standalone project
carried (reverts are handled by the tools package's session snapshots).
"""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from agents.implementations.code_agent.structured_output import AgentState

def step_dispatch_node(state: AgentState, config: RunnableConfig) -> dict:
    writer = get_stream_writer() # frontend-provided stream writer
    # helper function to write text on stream for frontend (in Markdown)
    def _w(text: str) -> None:
        writer({"kind": "text", "text": text + "\n\n"})

    queue = list(state.get("architect_step_queue", []))
    step = queue.pop(0)
    _w(f"📦 Step dispatch — running next step ({len(queue)} remaining in queue):\n\n> {step['step_plan'][:]}")

    return {
        "context_store": {
            "architect_plan":            step["step_plan"],
            "architect_files_to_create": step["files_to_create"],
            "architect_files_to_modify": step["files_to_modify"],
            "architect_files_to_delete": step["files_to_delete"],
            "validation_issues":         "",
            "coder_latest_files":        {"current": []},  # clear stale files from previous step
        },
        "architect_step_queue": queue,
        "coder_retries": 0,
        "architect_replans": 0,  # each new step gets its own fresh architect budget
    }
