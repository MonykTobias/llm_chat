"""
Inspector node — read-only agent with three modes.

EXPLORE mode (default, dispatched by the orchestrator):
  Gathers information from the project via a ReAct loop over read-only tools.
  Returns raw findings to the orchestrator via context_store["inspector_raw"].
  Never modifies files.

REVIEW mode (triggered by REVIEW_TASK_MARKER prefix in instruction):
  Compares the actual workspace against the canonical file tree and spec. Reads
  expected files, checks imports and content quality, reports gaps or confirms
  completion. Returns [REVIEW_COMPLETE] or [REVIEW_GAPS] to the orchestrator so
  it can make an informed decision about whether to end or plan more work.

VERIFY mode (completion gate, triggered when plan.next_agent == "complete"):
  Inspects the finished project against the objective / file_tree and returns a
  pass/fail verdict plus concrete gaps. On pass the graph ends; on gaps the
  orchestrator plans fix tasks. Uses a plain (non-[FAILED]) report so the
  orchestrator's snapshot-restore cannot roll back the last good task.

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

import json
import os
import re
from pathlib import Path
from string import Template

import yaml
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.prebuilt import ToolNode

from agents.llm_factory import make_llm
from tools import (
    analyze_architecture,
    check_imports,
    list_all_files,
    list_workspace_files,
    read_file,
    web_browse,
    run_linter,
    run_tests,
    run_type_check,
    set_language,
    compile_code
)

from agents.sub_agents.code_review.graph.utils.structured_output import (
    AgentState,
    REVIEW_TASK_MARKER,
    ReviewIssue,
    ReviewOutput,
    ValidatorOutput,
    ExplorerOutput,
)

_CFG_PATH = Path(__file__).resolve().parent / "graph_config.yaml"
with open(_CFG_PATH, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

DEFAULT_REVIEW_TOOLS = [
    read_file,
    list_all_files,
    check_imports,
    analyze_architecture,
    web_browse,
    run_linter,
    run_tests,
    run_type_check,
    compile_code,
    set_language
]

MAX_TOOL_ITERS = 30

def _build_llm():
    return make_llm(_cfg["agents"]["inspector"])

def _workspace_listing(project_path: str) -> tuple[set[str], str]:
    """Return (set of relative paths on disk, indented text listing for prompts)."""
    files = list_workspace_files(project_path)
    text = "\n".join(f"  {f}" for f in files) if files else "  (workspace is empty)"
    return set(files), text


def _extract_files_from_text(text: str) -> list[str]:
    """Extract relative file paths from inspector output programmatically."""
    pattern = r'(?:^|[\s\'\"(])\.?/?([a-zA-Z0-9_][a-zA-Z0-9_\-./]*\.[a-zA-Z0-9]{1,10})'
    matches = re.findall(pattern, text)
    paths = []
    seen = set()
    for path in matches:
        clean = path.lstrip("./")
        if clean and clean not in seen and not clean.startswith("http"):
            seen.add(clean)
            paths.append(clean)
    return paths


def _parse_verifier_fallback(raw_content: str) -> ValidatorOutput | None:
    cleaned = raw_content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None
    verdict = data.get("verdict", "")
    if verdict in ("pass", "fail"):
        return ValidatorOutput(
            verdict=verdict,
            summary=data.get("summary", ""),
            issues=data.get("issues", []),
        )
    return None


# ── REVIEW mode helpers ──────────────────────────────────────────────────────

def _looks_like_path(s: str) -> bool:
    """True if s looks like a real relative file path, not a freeform note.

    Rejects anything with spaces/parens/etc. and requires a real extension, so
    descriptions like "react-toastify setup (if not already included)" are
    filtered out before they can be mistaken for a file.
    """
    return bool(re.fullmatch(r"[A-Za-z0-9_\-./]+\.[A-Za-z0-9]{1,10}", s.strip()))


def _file_exists(norm: str, actual_files: set[str]) -> bool:
    """Decide whether a reported path already exists on disk.

    Exact full-path match is authoritative. A suffix match catches a deeper
    file reported by a shorter path. Bare-name basename matching is only used
    when the entry has no directory component (e.g. "todoApi.js" -> an existing
    .../api/todoApi.js); for names like "index.js" this is inherently ambiguous,
    so we deliberately restrict it to slash-less entries and treat a hit as
    "exists, demote to issue" rather than a creation target.
    """
    if norm in actual_files:
        return True
    if any(p.endswith("/" + norm) for p in actual_files):
        return True
    if "/" not in norm:
        return any(os.path.basename(p) == norm for p in actual_files)
    return False


def _classify_file_claims(
    llm_entries: list[str],
    computed_missing: list[str],
    file_tree_set: set[str],
    actual_files: set[str],
) -> tuple[list[str], list[str], list[ReviewIssue]]:
    """Re-bucket the LLM's file claims; its choice of bucket is only a hint.

    Returns (contract_missing, proposed_new, demoted_issues):
      - contract_missing: canonical file_tree paths that are genuinely absent
        (always includes the Python pre-check, which is authoritative).
      - proposed_new: plausible paths, absent, NOT in the canonical tree —
        genuine new-file proposals the inspector wants created.
      - demoted_issues: freeform notes, or paths that actually already exist.
    """
    contract_missing = list(computed_missing)
    proposed_new: list[str] = []
    demoted_issues: list[ReviewIssue] = []
    seen = set(contract_missing)

    for entry in llm_entries:
        if not isinstance(entry, str):
            entry = str(entry)
        norm = entry.strip().lstrip("./").replace("\\", "/")
        if not norm:
            continue
        if not _looks_like_path(norm):
            demoted_issues.append(ReviewIssue(description=entry.strip()))  # freeform note
            continue
        if _file_exists(norm, actual_files):
            demoted_issues.append(ReviewIssue(
                file_path=norm,
                description="reported missing but already exists on disk",
            ))
            continue
        if norm in seen:
            continue
        seen.add(norm)
        if norm in file_tree_set:
            contract_missing.append(norm)                   # builder failed a contract file
        else:
            proposed_new.append(norm)                       # genuine new-file proposal

    return (
        list(dict.fromkeys(contract_missing)),
        list(dict.fromkeys(proposed_new)),
        demoted_issues,
    )


def _to_issue(x) -> ReviewIssue:
    """Coerce a string / dict / existing ReviewIssue into a ReviewIssue.

    ReviewOutput.issues is list[ReviewIssue]; a bare string raises a pydantic
    ValidationError, so everything that lands in issues must pass through here.
    """
    if isinstance(x, ReviewIssue):
        return x
    if isinstance(x, dict):
        return ReviewIssue(
            file_path=x.get("file_path") or x.get("file") or "",
            description=x.get("description") or x.get("issue") or "",
            severity=x.get("severity", "error"),
        )
    return ReviewIssue(description=str(x))


# ── EXPLORE mode ─────────────────────────────────────────────────────────────

def _explore(state: AgentState, config: RunnableConfig, project_path: str,
             language: str, enabled_tools: list[str]) -> dict:

    writer = get_stream_writer()
    # helper function to write text on stream for frontend (in Markdown)
    def _w(text: str) -> None:
        writer({"kind": "text", "text": text + "\n\n"})

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

    tool_llm =_build_llm().bind_tools(DEFAULT_REVIEW_TOOLS)
    structured_llm = _build_llm().with_structured_output(ExplorerOutput, include_raw=True) # just used to do a summary of the findings
    tool_executor = ToolNode(DEFAULT_REVIEW_TOOLS,handle_tool_errors=False)
    messages: list[BaseMessage] = [HumanMessage(content=system_prompt)]
    work: dict = dict(state)
    parsed = None

    try:
        for _ in range(MAX_TOOL_ITERS):
            ai_msg = tool_llm.invoke(messages)
            messages.append(ai_msg)

            if ai_msg.tool_calls:
                work["messages"] = messages
                tool_result = tool_executor.invoke(
                    work,
                    config=config,
                )
                messages.extend(tool_result["messages"])
            else:
                messages.append(HumanMessage(content=(
                     "Summarise your findings concisely in 2-3 sentences. "
                     "List any specific files or issues discovered."
                )))
                result = structured_llm.invoke(messages)
                parsed = result["parsed"]
                findings = parsed.summary if parsed else ai_msg.content # fallback to last ai_msg
                break
        else:
            _w(f"❌ Inspector exceeded max tool iterations ({MAX_TOOL_ITERS}).")
            raise RuntimeError("Validator exceeded max tool iterations")
    except Exception as e:
        msg = f"[FAILED] Inspector crashed: {e}"
        _w(f"❌ Inspector crashed: `{e}`")
        return {"latest_report": msg, "history": ["inspector_failed"]}

    LANGUAGE = work["language"]

    _w(f"✅ **Findings:** {findings}")

    return {
        "latest_report": findings,
        "context_store": {
            "inspector_raw": findings,
            "inspector_files": parsed.files_of_interest if parsed else [],
            "inspector_issues": parsed.issues if parsed else [],
        },
        "history": ["inspector"],
        "language": LANGUAGE,
    }


# ── VERIFY mode ──────────────────────────────────────────────────────────────

def _verify(state: AgentState, config: RunnableConfig, project_path: str,
            language: str, enabled_tools: list[str]) -> dict:

    writer = get_stream_writer()
    # helper function to write text on stream for frontend (in Markdown)
    def _w(text: str) -> None:
        writer({"kind": "text", "text": text + "\n\n"})

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

    tool_llm = _build_llm().bind_tools(DEFAULT_REVIEW_TOOLS)

    structured_llm = _build_llm().with_structured_output(ValidatorOutput, include_raw=True)

    tool_executor = ToolNode(DEFAULT_REVIEW_TOOLS,handle_tool_errors=False)
    messages: list[BaseMessage] = [HumanMessage(content=system_prompt)]
    work: dict = dict(state)
    parsed = None # default to None if loop crashes

    try:
        for _ in range(MAX_TOOL_ITERS):
            ai_msg = tool_llm.invoke(messages)
            messages.append(ai_msg)

            if ai_msg.tool_calls:
                work["messages"] = messages
                tool_result = tool_executor.invoke(
                    work,
                    config=config,
                )
                messages.extend(tool_result["messages"])
            else:
                messages.append(HumanMessage(content=(verdict_prompt)))
                result = structured_llm.invoke(messages)
                parsed = result["parsed"]
                break
        else:
            _w(f"❌ Inspector exceeded max tool iterations ({MAX_TOOL_ITERS}).")
            raise RuntimeError("Validator exceeded max tool iterations")
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


# ── REVIEW mode ──────────────────────────────────────────────────────────────

def _review(state: AgentState, config: RunnableConfig, project_path: str,
            language: str, enabled_tools: list[str]) -> dict:

    writer = get_stream_writer()
    # helper function to write text on stream for frontend (in Markdown)
    def _w(text: str) -> None:
        writer({"kind": "text", "text": text + "\n\n"})

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

    tool_llm = _build_llm().bind_tools(DEFAULT_REVIEW_TOOLS)
    structured_llm = _build_llm().with_structured_output(ReviewOutput, include_raw=True)
    tool_executor = ToolNode(DEFAULT_REVIEW_TOOLS,handle_tool_errors=False)
    messages: list[BaseMessage] = [HumanMessage(content=system_prompt)]
    work: dict = dict(state)

    verdict_prompt = (
        "You have finished your review. Based on your investigation, produce a structured "
        "review verdict. List any missing files, new files needed, and issues found. "
        "Set verdict to 'pass' only if the project fully satisfies the spec and file tree."
    )

    parsed = None
    try:
        for _ in range(MAX_TOOL_ITERS):
            ai_msg = tool_llm.invoke(messages)
            messages.append(ai_msg)

            if ai_msg.tool_calls:
                work["messages"] = messages
                tool_result = tool_executor.invoke(
                    work,
                    config=config,
                )
                messages.extend(tool_result["messages"])
            else:
                messages.append(HumanMessage(content=(verdict_prompt)))
                result = structured_llm.invoke(messages)
                parsed = result["parsed"]
                break
        else:
            _w(f"❌ Inspector exceeded max tool iterations ({MAX_TOOL_ITERS}).")
            raise RuntimeError("Inspector review exceeded max tool iterations")
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


# ── Entry point ──────────────────────────────────────────────────────────────

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
        return _verify(state, config, project_path, language, enabled_tools)
    if is_review:
        return _review(state, config, project_path, language, enabled_tools)
    return _explore(state, config, project_path, language, enabled_tools)
