"""
Validator node — DUMMY placeholder.

Eventually: checks the coder's changes (syntax, truncation, markdown leakage, and
an LLM review of the change report) and returns a ValidatorOutput verdict. PASS
with a non-empty step queue → step_dispatch (next step); PASS with an empty queue
→ orchestrator. A `[FAILED]` report drives the two-tier coder/architect retry.

For now this is a no-op stub that always passes and routes to the orchestrator.
Replace with the real implementation ported from the standalone `orchestrator`
project's `agents/validator.py`.
"""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from .structured_output import AgentState


def validator_node(state: AgentState, config: RunnableConfig) -> dict:
    # TODO: validate the change report, build a ValidatorOutput, and on PASS
    #       commit module_exports_pending into module_exports.
    print("[Validator] (dummy) no-op — auto-passing.")
    return {
        "latest_report": "[DUMMY] validator not yet implemented (auto-pass).",
        "history": ["validator"],
    }
