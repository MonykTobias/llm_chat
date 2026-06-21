"""
REVIEW mode (triggered by REVIEW_TASK_MARKER prefix in instruction).

Compares the actual workspace against the canonical file tree and spec. Reads
expected files, checks imports and content quality, reports gaps or confirms
completion. Returns [REVIEW_COMPLETE] or [REVIEW_GAPS] to the orchestrator so it
can make an informed decision about whether to end or plan more work.
"""
from __future__ import annotations

import json
from string import Template

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables import RunnableConfig

from agents.implementations.code_agent.structured_output import (
    AgentState,
    ReviewIssue,
    ReviewOutput,
)
from agents.implementations.code_agent.inspector.utils import (
    _cfg,
    make_writer,
    run_tool_loop,
    _workspace_listing,
    _classify_file_claims,
    _to_issue,
)


def review(state: AgentState, config: RunnableConfig, project_path: str,
           language: str, enabled_tools: list[str]) -> dict:

    _w = make_writer()

    instruction = state["plan"].instruction_for_agent
    _w(f"### 📋 Inspector — Review\n\n**Task:** {instruction}")

    store = state.get("context_store", {})
    spec_content = store.get("spec_content", "") or "(no spec available)"
    file_tree = store.get("file_tree", [])
    file_tree_text = "\n".join(f"  {p}" for p in file_tree) if file_tree else "  (none specified)"
    actual_files, workspace_files = _workspace_listing(project_path)

    # Compute missing files in Python — reliable, independent of LLM tool-calling.
    computed_missing = [f for f in file_tree if f not in actual_files]
    if computed_missing:
        missing_list = "\n".join(f"  - `{f}`" for f in computed_missing)
        _w(f"⚠️ Pre-check: **{len(computed_missing)}** file(s) missing from file_tree:\n\n{missing_list}")

    ####################################################################
    # VALIDATOR'S PATTERN - TOOL LOOP - THEN STRUCTURED OUTPUT         #
    # We create two llm's (one for the tool calls, one for the parser) #
    ####################################################################
    system_prompt = Template(_cfg["prompts"]["inspector_review"]).safe_substitute(
        instruction=instruction,
        sandbox_dir=project_path,
        spec_content=spec_content,
        file_tree=file_tree_text,
        workspace_files=workspace_files,
    )

    verdict_prompt = (
        "You have finished your review. Based on your investigation, produce a structured "
        "review verdict. List any missing files, new files needed, and issues found. "
        "Set verdict to 'pass' only if the project fully satisfies the spec and file tree."
    )

    messages: list[BaseMessage] = [HumanMessage(content=system_prompt)]
    work: dict = dict(state)
    parsed = None

    try:
        parsed = run_tool_loop(
            messages, work, config, ReviewOutput, verdict_prompt, _w
        )
    except Exception as e:
        msg = f"[FAILED] Inspector review crashed: {e}"
        _w(f"❌ Inspector review crashed: `{e}`")
        return {"latest_report": msg, "history": ["inspector_review_failed"]}

    if parsed is None:
        parsed = ReviewOutput(
            verdict="fail",
            missing_files=[],
            computed_missing_files=[],
            issues=[ReviewIssue(description="Review output could not be parsed.")],
            summary="Review parsing failed — treating as incomplete.",
            new_files=[],
        )

    LANGUAGE = work["language"]

    # ── Validate & re-bucket every file-like claim from the LLM ───────────────
    # The LLM's choice of bucket (missing_files / new_files) is only a hint.
    # Python is authoritative: the pre-check decides contract-missing, existence
    # on disk demotes false positives to issues, and tree membership separates
    # "builder failed a contract file" from "inspector proposes a new file".
    file_tree_set = set(file_tree)
    contract_missing, proposed_new, demoted_issues = _classify_file_claims(
        [*(parsed.missing_files or []), *(parsed.new_files or [])],
        computed_missing,
        file_tree_set,
        actual_files,
    )
    all_issues = [_to_issue(i) for i in (parsed.issues or [])] + demoted_issues

    # Any gap — missing file, proposed new file, or issue — means fail. An
    # explicit LLM "fail" is also respected even if it listed nothing concrete.
    verdict = "fail" if (
        contract_missing or proposed_new or all_issues or parsed.verdict == "fail"
    ) else "pass"

    # Synthesise a summary only when we overrode a too-optimistic LLM "pass".
    if verdict == "fail" and parsed.verdict == "pass":
        bits = []
        if contract_missing:
            bits.append(f"missing {len(contract_missing)} file(s): {', '.join(contract_missing)}")
        if proposed_new:
            bits.append(f"{len(proposed_new)} new file(s) to create: {', '.join(proposed_new)}")
        if all_issues:
            bits.append(f"{len(all_issues)} issue(s)")
        summary = "Overrode pass→fail — " + "; ".join(bits) if bits else parsed.summary
    else:
        summary = parsed.summary

    parsed = ReviewOutput(
        verdict=verdict,
        missing_files=contract_missing,
        computed_missing_files=computed_missing,
        issues=all_issues,
        summary=summary,
        new_files=proposed_new,
    )

    review_report = parsed.model_dump_json()

    # print the final review verdict
    if parsed.verdict == "pass":
        label = "[REVIEW_COMPLETE]"
        _w(f"✅ **REVIEW COMPLETE** — {parsed.summary}")
    else:
        label = "[REVIEW_GAPS]"
        missing_md = "\n".join(f"  - `{f}`" for f in parsed.missing_files)
        new_md = "\n".join(f"  - `{f}`" for f in parsed.new_files)
        issues_md = "\n".join(
            f"  - {'`' + issue.file_path + '`: ' if issue.file_path else ''}{issue.description}"
            for issue in parsed.issues
        )
        _w(
            f"⚠️ **REVIEW GAPS** — {parsed.summary}\n\n"
            + (f"**Missing files:**\n{missing_md}\n\n" if missing_md else "")
            + (f"**New files needed:**\n{new_md}\n\n" if new_md else "")
            + (f"**Issues:**\n{issues_md}" if issues_md else "")
        )

    gaps_json = json.dumps(
        [{"file": f} for f in parsed.missing_files] +
        [{"new_file": f} for f in parsed.new_files] +
        [{"file": i.file_path, "issue": i.description} for i in parsed.issues],
        indent=2,
    ) if parsed.verdict == "fail" else ""

    return {
        "latest_report": f"{label}: {parsed.summary}",
        "context_store": {
            "review_report": review_report,
            "missing_files": parsed.missing_files,
            "new_files": parsed.new_files,
            "verification_verdict": parsed.verdict,
            "verification_gaps": gaps_json,
        },
        "history": ["inspector_review"],
        "language": LANGUAGE,
    }
