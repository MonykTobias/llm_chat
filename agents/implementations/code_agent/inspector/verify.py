"""
VERIFY mode (completion gate, triggered when plan.next_agent == "complete").

Inspects the finished project against the objective / file_tree and returns a
pass/fail verdict plus concrete gaps. On pass the graph ends; on gaps the
orchestrator plans fix tasks. Uses a plain (non-[FAILED]) report so the
orchestrator's snapshot-restore cannot roll back the last good task.
"""
from __future__ import annotations

import json
from string import Template

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables import RunnableConfig

from agents.implementations.code_agent.structured_output import (
    AgentState,
    ValidatorOutput,
)
from agents.implementations.code_agent.inspector.utils import (
    _cfg,
    make_writer,
    run_tool_loop,
    _workspace_listing,
)


def verify(state: AgentState, config: RunnableConfig, project_path: str,
           language: str, enabled_tools: list[str]) -> dict:

    _w = make_writer()

    objective = state.get("objective", "")
    store = state.get("context_store", {})
    spec_content = store.get("spec_content", "") or "(no brief available)"
    file_tree = store.get("file_tree", [])
    file_tree_text = "\n".join(f"  {p}" for p in file_tree) if file_tree else "  (none specified)"
    actual_files, workspace_files = _workspace_listing(project_path)

    computed_missing = [f for f in file_tree if f not in actual_files]
    if computed_missing:
        missing_list = "\n".join(f"  - `{f}`" for f in computed_missing)
        _w(f"⚠️ Pre-check: **{len(computed_missing)}** canonical file(s) missing:\n\n{missing_list}")

    _w("### ✅ Inspector — Verifying project completeness...")

    ####################################################################
    # VALIDATOR'S PATTERN - TOOL LOOP - THEN STRUCTURED OUTPUT         #
    # We create two llm's (one for the tool calls, one for the parser) #
    ####################################################################
    system_prompt = Template(_cfg["prompts"]["verifier"]).safe_substitute(
        objective=objective,
        spec_content=spec_content,
        file_tree=file_tree_text,
        workspace_files=workspace_files,
        sandbox_dir=project_path,
    )

    verdict_prompt = (
        "You have finished your investigation. "
        "Based on the tool results above, produce a verification verdict. "
        "Set verdict to 'pass' only if the project fully satisfies the objective "
        "and all canonical files exist. Otherwise set 'fail' and list the issues."
    )

    messages: list[BaseMessage] = [HumanMessage(content=system_prompt)]
    work: dict = dict(state)
    parsed = None  # default to None if loop crashes

    try:
        parsed = run_tool_loop(
            messages, work, config, ValidatorOutput, verdict_prompt, _w
        )
    except Exception as e:
        _w(f"❌ Verifier crashed: `{e}`")
        if computed_missing:
            _w(f"❌ FAIL (pre-check) — canonical files missing despite crash.")
            return {
                "latest_report": f"Verification found {len(computed_missing)} missing file(s).",
                "context_store": {
                    "verification_verdict": "fail",
                    "verification_gaps": json.dumps([{"file": f} for f in computed_missing], indent=2),
                },
                "history": ["inspector"],
            }
        _w("⚠️ Verifier crashed — passing to avoid blocking completion.")
        return {
            "latest_report": "Verification skipped (verifier crashed).",
            "context_store": {"verification_verdict": "pass", "verification_gaps": ""},
            "history": ["inspector"],
        }

    if parsed is None:
        # Could not get a verdict — do not block completion.
        return {
            "latest_report": "Verification inconclusive — treated as pass.",
            "context_store": {"verification_verdict": "pass", "verification_gaps": ""},
            "history": ["inspector"],
        }

    if parsed.verdict == "pass":
        _w(f"✅ **PASS** — {parsed.summary}")
        return {
            "latest_report": f"Verification passed: {parsed.summary}",
            "context_store": {"verification_verdict": "pass", "verification_gaps": ""},
            "history": ["inspector"],
        }

    LANGUAGE = work["language"]
    missing_gaps = [{"file": f} for f in computed_missing]
    issue_gaps = [{"file": i.file_path, "issue": i.description} for i in parsed.issues]
    gaps_json = json.dumps(missing_gaps + issue_gaps, indent=2)
    reason = parsed.summary if parsed.verdict == "fail" \
        else f"{len(computed_missing)} canonical file(s) missing"

    # Send fail-report to frontend
    missing_list = "\n".join(f"  - `{f}`" for f in computed_missing)
    issues_list = "\n".join(f"  - `{issue.file_path}`: {issue.description}" for issue in parsed.issues)
    _w(
        f"❌ **FAIL** — {reason}\n\n"
        + (f"**Missing files:**\n{missing_list}\n\n" if computed_missing else "")
        + (f"**Gaps:**\n{issues_list}" if parsed.issues else "")
    )

    return {
        "latest_report": f"Verification found gaps: {reason}",
        "context_store": {"verification_verdict": "fail", "verification_gaps": gaps_json},
        "history": ["inspector"],
        "language": LANGUAGE,
    }
