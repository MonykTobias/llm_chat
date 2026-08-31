"""
EXPLORE mode (default, dispatched by the orchestrator).

Gathers information from the project via a ReAct loop over read-only tools and
returns raw findings to the orchestrator via context_store["inspector_raw"].
Never modifies files.
"""
from __future__ import annotations

from string import Template

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables import RunnableConfig

from agents.implementations.code_agent.structured_output import (
    AgentState,
    ExplorerOutput,
)
from agents.implementations.code_agent.inspector.utils import (
    _cfg,
    make_writer,
    run_tool_loop,
    _parse_explore_fallback,
    _explore_is_degenerate,
)


def explore(state: AgentState, config: RunnableConfig, project_path: str,
            language: str, enabled_tools: list[str]) -> dict:

    _w = make_writer()

    instruction = state["plan"].instruction_for_agent
    _w(f"### 🔍 Inspector — Explore\n\n**Task:** {instruction}")

    ####################################################################
    # VALIDATOR'S PATTERN - TOOL LOOP - THEN STRUCTURED OUTPUT         #
    # We create two llm's (one for the tool calls, one for the parser) #
    ####################################################################
    system_prompt = Template(_cfg["prompts"]["inspector"]).safe_substitute(
        instruction=instruction,
        sandbox_dir=project_path,
    )

    messages: list[BaseMessage] = [HumanMessage(content=system_prompt)]
    work: dict = dict(state)
    reason_prompt = (
        "You have finished exploring. Summarise what you found as prose — NOT JSON. "
        "Note the files worth the architect's attention and anything broken, missing, "
        "or suspicious, with specific paths."
    )
    verdict_prompt = (
        "Now convert your findings into the structured ExplorerOutput JSON: a concise "
        "2-3 sentence summary, the specific files of interest, and any issues "
        "discovered. Emit ONLY the JSON object."
    )
    parsed = None

    try:
        parsed = run_tool_loop(
            messages, work, config, ExplorerOutput, verdict_prompt, _w,
            reason_prompt=reason_prompt,
            fallback_parser=_parse_explore_fallback,
            is_degenerate=_explore_is_degenerate,
        )
    except Exception as e:
        msg = f"[FAILED] Inspector crashed: {e}"
        _w(f"❌ Inspector crashed: `{e}`")
        return {"latest_report": msg, "history": ["inspector_failed"]}

    LANGUAGE = work["language"]

    if parsed is None:
        _w("⚠️ Inspector produced no parseable findings.")
        return {
            "latest_report": "Inspector produced no parseable findings.",
            "context_store": {
                "inspector_raw": "",
                "inspector_files": [],
                "inspector_issues": [],
            },
            "history": ["inspector"],
            "language": LANGUAGE,
        }

    _w(f"✅ **Findings:** {parsed.summary}")

    return {
        "latest_report": parsed.summary,
        "context_store": {
            "inspector_raw": parsed.summary,
            "inspector_files": parsed.files_of_interest,
            "inspector_issues": parsed.issues,
        },
        "history": ["inspector"],
        "language": LANGUAGE,
    }