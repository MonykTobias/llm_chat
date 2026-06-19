"""
Validator — checks the coder's changes and decides pass / fail.

Stripped-down port. The deterministic work is done by the project's own tooling,
called HERE (not by the model): the linter, type checker and import check run
programmatically, and the change report (diffs) is built directly. Their output
plus the diff is handed to the model, which returns a structured ValidatorOutput
verdict. The standalone project's ast/markdown/truncation phase-1 checks and the
module-export commit logic are dropped — the real linter/type-checker subsume the
former, and exports are no longer tracked.

On fail, the verdict + issues go to context_store["validation_issues"] and
coder_retries is incremented so the graph routes back to the coder / architect.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path
from string import Template

import yaml
from langchain_core.callbacks import CallbackManager
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables import RunnableConfig, patch_config
from langgraph.config import get_stream_writer
from langgraph.prebuilt import ToolNode

from agents.llm_factory import make_llm
from tools import list_workspace_files, safe_read, set_language, compile_code
from tools.change_tracking import _build_change_report
from tools import check_imports, run_type_check, run_linter

from agents.sub_agents.code_review.graph.utils.structured_output import AgentState, ValidatorOutput
from agents.sub_agents.code_review.graph.utils.stats import stats

_CFG_PATH = Path(__file__).resolve().parent / "graph_config.yaml"
with open(_CFG_PATH, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

_max_retries = _cfg.get("coder_max_retries", 5)

DEFAULT_REVIEW_TOOLS = [
    set_language,
    run_linter,
    run_type_check,
    check_imports,
    compile_code
]


def _parse_validator_fallback(raw_content: str) -> ValidatorOutput | None:
    cleaned = (raw_content or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None
    if data.get("verdict") in ("pass", "fail"):
        return ValidatorOutput(verdict=data["verdict"], issues=data.get("issues", []),
                               summary=data.get("summary", ""))
    return None


def _read_files_under_review(project_path: str, files: list[str]) -> str:
    """Full current content of every in-scope file the coder wrote this step.

    The diff alone is blind on a no-op write (identical content reads as
    UNCHANGED), so we always hand the model the actual file content to validate —
    a `create` rewrite makes the whole file the coder's output."""
    parts = []
    for rel in dict.fromkeys(files):
        content = safe_read(project_path, rel)
        if content is None:
            parts.append(f"=== {rel} ===\n[MISSING — file not found on disk]")
        else:
            parts.append(f"=== {rel} ===\n{content}")
    return "\n\n".join(parts) if parts else "(no files under review)"

def validator_node(state: AgentState, config: RunnableConfig) -> dict:
    writer = get_stream_writer() # frontend-provided stream writer
    # helper function to write text on stream for frontend (in Markdown)
    def _w(text: str) -> None:
        writer({"kind": "text", "text": text + "\n\n"})

    def _w_scrollable(text: str) -> None:
        html = (
            f"<div style='max-height:150px;overflow-y:auto;padding:8px;"
            f"border:1px solid #ccc;border-radius:6px;font-size:0.9em;"
            f"white-space:pre-wrap;'>{text}</div>"
        )
        writer({"kind": "text", "text": html + "\n\n"})

    writer({"kind": "stage", "stage": "validator",
            "label": "✅ Validator — Validating Coders output"})

    instruction = state["plan"].instruction_for_agent
    project_path = state.get("project_path", ".")
    files_written = state.get("context_store", {}).get("coder_latest_files", {}).get("current", [])

    # Fast-fail: propagate a coder failure and bump retries so the router reacts.
    coder_report = state.get("latest_report", "")
    if coder_report.startswith("[FAILED]"):
        retries = state.get("coder_retries", 0) + 1
        _w(f"⚠️ Fast-fail: coder reported failure (retry **{retries}/{_max_retries}**)")
        return {
            "latest_report": coder_report,
            "context_store": {"validation_issues": coder_report},
            "history": ["validator_failed"],
            "coder_retries": retries,
        }

    if not files_written:
        return {"latest_report": "Validator skipped: no files to check.", "history": ["validator"]}

    _w(f"### ✅ Validator\n\n**Checking {len(files_written)} file(s):**")
    _w_scrollable(instruction)

    # Deterministic evidence: the full current content of every in-scope file
    # (always present, even on a no-op write), the diff for context, and real
    # tool output.

    files_content = _read_files_under_review(project_path, files_written)

    change_report = _build_change_report(project_path, files_written)

    file_contents = (
        '<files_under_review description="FULL current content of each file the coder '
        'wrote this step — the coder owns all of this content; validate it in full">\n'
        f"{files_content}\n"
        "</files_under_review>\n\n"
        '<diff description="unified diff of what changed this session — context only">\n'
        f"{change_report}\n"
        "</diff>\n\n"
    )

    orchestrate_prompt = Template(_cfg["prompts"]["validator_orchestrate"]).safe_substitute(
        instruction=instruction,
        file_contents=file_contents,
    )

    verdict_prompt = Template(_cfg["prompts"]["validator_verdict"]).safe_substitute(
        files_written=", ".join(files_written),
    )

    ####################################################################
    # VALIDATOR'S PATTERN - TOOL LOOP - THEN STRUCTURED OUTPUT         #
    # We create two llm's (one for the tool calls, one for the parser) #
    ####################################################################
    tool_llm = (make_llm(_cfg["agents"]["validator"])
                .bind_tools(DEFAULT_REVIEW_TOOLS))

    structured_llm = (make_llm(_cfg["agents"]["validator"])
                      .with_structured_output(ValidatorOutput, include_raw=True))

    # Guard the tool loop
    MAX_TOOL_ITERS = 20
    tool_executor = ToolNode(DEFAULT_REVIEW_TOOLS, handle_tool_errors=False)

    messages: list[BaseMessage] = [HumanMessage(content=orchestrate_prompt)]
    work: dict = dict(state)

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
                messages.append(HumanMessage(content=verdict_prompt))
                result = structured_llm.invoke(messages)
                break
        else:
            _w(f"❌ Validator exceeded max tool iterations ({MAX_TOOL_ITERS}).")
            raise RuntimeError("Validator exceeded max tool iterations")
    except Exception as e:
        msg = f"[FAILED] Validator crashed: {e}"
        _w(f"❌ Validator crashed: `{e}`")
        return {"latest_report": msg, "history": ["validator_failed"],
                "coder_retries": state.get("coder_retries", 0) + 1}

    LANGUAGE = work["language"] #get language to pass it to the next node if it switched

    if result.get("raw"):
        stats.record_tokens(result["raw"])
    parsed = result["parsed"]
    if parsed is None and result.get("raw"):
        raw_text = result["raw"].content if hasattr(result["raw"], "content") else str(result["raw"])
        parsed = _parse_validator_fallback(raw_text)

    if parsed is None:
        msg = "[FAILED] Validator produced no usable output."
        _w("❌ Validator produced no usable output.")
        return {"latest_report": msg, "history": ["validator_failed"],
                "coder_retries": state.get("coder_retries", 0) + 1}

    if parsed.verdict == "pass":
        _w(f"✅ **PASS** — {parsed.summary}")
        return {
            "latest_report": f"Validator passed: {parsed.summary}",
            "context_store": {"workspace_files": list_workspace_files(project_path)},
            "history": ["validator"],
            "language": LANGUAGE
        }

    retries = state.get("coder_retries", 0) + 1
    issues_json = json.dumps(
        [{"file": i.file_path, "severity": i.severity, "issue": i.description} for i in parsed.issues],
        indent=2,
    )

    # write issues to frontend stream
    issues_md = "\n".join(
        f"  - `{i.file_path}` [{i.severity}]: {i.description}"
        for i in parsed.issues
    )
    _w(f"❌ **FAIL** (retry **{retries}/{_max_retries}**) — {parsed.summary}\n\n{issues_md}")

    return {
        "latest_report": f"[FAILED] Validator: {parsed.summary}",
        "context_store": {"validation_issues": issues_json},
        "history": ["validator_failed"],
        "coder_retries": retries,
        "language": LANGUAGE
    }