"""
Coder node — DUMMY placeholder.

Eventually: takes the current architect step and writes the actual file changes
(CoderOutput — full-content `create` / `delete`), reporting the public exports of
each file so consumers and producers agree on symbol names. Always hands off to
the validator.

For now this is a no-op stub that writes nothing and routes to the validator.
Replace with the real implementation ported from the standalone `orchestrator`
project's `agents/coder.py`.
"""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from .structured_output import AgentState


def coder_node(state: AgentState, config: RunnableConfig) -> dict:
    # TODO: apply FileChange items to the sandbox, refresh workspace_files, and
    #       record module_exports_pending for the validator to commit on PASS.
    print("[Coder] (dummy) no-op — wrote no files.")
    return {
        "latest_report": "[DUMMY] coder not yet implemented.",
        "history": ["coder"],
    }
