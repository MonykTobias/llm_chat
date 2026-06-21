"""
Inspector node — read-only agent with three modes.

EXPLORE mode (default, dispatched by the orchestrator):
  Gathers information from the project via a ReAct loop over read-only tools.
  Returns raw findings to the orchestrator via context_store["inspector_raw"].
  Never modifies files. See `explorer.py`.

REVIEW mode (triggered by REVIEW_TASK_MARKER prefix in instruction):
  Compares the actual workspace against the canonical file tree and spec. Reads
  expected files, checks imports and content quality, reports gaps or confirms
  completion. Returns [REVIEW_COMPLETE] or [REVIEW_GAPS] to the orchestrator so
  it can make an informed decision about whether to end or plan more work. See
  `review.py`.

VERIFY mode (completion gate, triggered when plan.next_agent == "complete"):
  Inspects the finished project against the objective / file_tree and returns a
  pass/fail verdict plus concrete gaps. On pass the graph ends; on gaps the
  orchestrator plans fix tasks. Uses a plain (non-[FAILED]) report so the
  orchestrator's snapshot-restore cannot roll back the last good task. See
  `verify.py`.

The three modes share their config, helpers and tool loop via `utils.py`. This
package's public surface is unchanged: `from ...inspector import inspector_node`
still works, because the dispatcher below is re-exported here.

Ported from the standalone `orchestrator` project's inspector and adapted to this
project's conventions:
  * the project root comes from `state["project_path"]` (not a config sandbox dir);
  * file ops use this project's `InjectedState` tools — so the ReAct sub-agents are
    built with `state_schema=ReviewState` and the frontend run context
    (project_path / language / enabled_tools) is forwarded into every invocation,
    exactly like every other agent reads it;
  * the real file listing comes from `tools.list_workspace_files`;
  * per-node LLM + prompt config lives in `graph_config.yaml`.
"""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from agents.implementations.code_agent.structured_output import (
    AgentState,
    REVIEW_TASK_MARKER,
)
from agents.implementations.code_agent.inspector.utils import DEFAULT_REVIEW_TOOLS
from agents.implementations.code_agent.inspector.explorer import explore
from agents.implementations.code_agent.inspector.review import review
from agents.implementations.code_agent.inspector.verify import verify

__all__ = ["inspector_node"]


def inspector_node(state: AgentState, config: RunnableConfig) -> dict:
    writer = get_stream_writer()
    writer({"kind": "stage", "stage": "inspector",
            "label": "🔍 Inspector — gathering context"})

    # Frontend-provided run context, read off the graph state the same way every
    # other agent reads it (project_path defaults to '.' just like BaseAgent).
    project_path = state.get("project_path", ".")
    language = state.get("language", "python")

    plan = state.get("plan")
    is_verify = bool(plan and plan.next_agent == "complete")
    is_review = bool(
        plan
        and plan.instruction_for_agent
        and plan.instruction_for_agent.startswith(REVIEW_TASK_MARKER)
    )

    # to the bound set so the forwarded run context is self-consistent.
    base_tool_names = [t.name for t in DEFAULT_REVIEW_TOOLS]
    enabled_tools = state.get("enabled_tools") or base_tool_names

    if is_verify:
        return verify(state, config, project_path, language, enabled_tools)
    if is_review:
        return review(state, config, project_path, language, enabled_tools)
    return explore(state, config, project_path, language, enabled_tools)
