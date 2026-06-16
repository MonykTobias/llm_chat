"""
Inspector node — DUMMY placeholder.

Eventually: a read-only investigator with two modes.
  * Explore mode — gather information for the orchestrator (returns a report).
  * Verify mode  — the completion gate, triggered when the orchestrator decides
    the objective is `complete`; writes a verification verdict into
    context_store and PASS → END, gaps → orchestrator (plan fixes).

For now this is a no-op stub returning a placeholder report so the graph routes
back to the orchestrator. Replace with the real implementation ported from the
standalone `orchestrator` project's `agents/inspector.py`.
"""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from .structured_output import AgentState


def inspector_node(state: AgentState, config: RunnableConfig) -> dict:
    # TODO: run the read-only ReAct investigation (explore) or the completeness
    #       review (verify), and set context_store["verification_verdict"] in
    #       verify mode so the router can gate on PASS.
    print("[Inspector] (dummy) no-op — returning placeholder report.")
    return {
        "latest_report": "[DUMMY] inspector not yet implemented.",
        "context_store": {"verification_verdict": "pass"},
        "history": ["inspector"],
    }
