"""
Scaffold node — DUMMY placeholder.

Eventually: runs exactly once before any planning, commits the canonical project
file tree (ScaffoldOutput) into context_store["file_tree"] plus a provisional
module-interface map, so every downstream agent plans against one agreed layout.

For now this is a no-op stub that lets the graph compile and flow through to the
orchestrator. Replace with the real implementation ported from the standalone
`orchestrator` project's `agents/scaffold.py`.
"""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from .structured_output import AgentState


def scaffold_node(state: AgentState, config: RunnableConfig) -> dict:
    # TODO: build the canonical file tree (ScaffoldOutput) and seed context_store
    #       with file_tree / module_exports_planned / workspace_files.
    print("[Scaffold] (dummy) no-op — handing straight to orchestrator.")
    return {"history": ["scaffold"]}
