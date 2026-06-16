"""
Architect node — DUMMY placeholder.

Eventually: turns one orchestrator task into a concrete, ordered implementation
plan (ArchitectOutput) grounded in the canonical file tree, then hands the step
queue to the coder. A `[FAILED]` report routes back to the orchestrator; a
`[COMPLETE]` report means the task was already satisfied (no-op).

For now this is a no-op stub that emits an empty plan and routes to the coder.
Replace with the real implementation ported from the standalone `orchestrator`
project's `agents/architect.py`.
"""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from .structured_output import AgentState


def architect_node(state: AgentState, config: RunnableConfig) -> dict:
    # TODO: produce an ArchitectOutput (plan + steps), enforce the file tree,
    #       and seed architect_step_queue for the coder.
    print("[Architect] (dummy) no-op — emitting empty plan.")
    return {
        "latest_report": "[DUMMY] architect not yet implemented.",
        "architect_step_queue": [],
        "history": ["architect"],
    }
