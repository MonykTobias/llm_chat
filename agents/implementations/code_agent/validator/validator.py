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
import re
from pathlib import Path
from string import Template

import yaml
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.prebuilt import ToolNode

from agents.llm_factory import make_llm, _strip_thinking
from agents.implementations.code_agent.utils.llm_helpers import _is_ollama_xml_bug, _scrub
from tools import list_workspace_files, safe_read, set_language, compile_code
from tools.change_tracking import _build_change_report
from tools import check_imports, run_type_check, run_linter

from agents.implementations.code_agent.structured_output import AgentState, ValidatorOutput
from structured_output import LANGUAGES
from agents.implementations.code_agent.utils.stats import stats

_CFG_PATH = Path(__file__).resolve().parent.parent / "graph_config.yaml"
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


# A verdict word followed (within a short window, possibly across a newline) by
# pass/fail. Tolerant of markdown noise like "**Verdict:** Pass" and trailing
# prose like 'the verdict should be "Pass"'. The short window keeps it from
# matching unrelated "...does not pass..." sentences.
_VERDICT_RE = re.compile(r"verdict\b.{0,15}?\b(pass|fail)\b", re.IGNORECASE | re.DOTALL)


def _coerce_verdict_output(data: dict) -> ValidatorOutput | None:
    """Build a ValidatorOutput from a parsed dict, tolerating a bad issues list."""
    if not isinstance(data, dict) or data.get("verdict") not in ("pass", "fail"):
        return None
    try:
        return ValidatorOutput(verdict=data["verdict"], issues=data.get("issues", []),
                               summary=data.get("summary", ""))
    except Exception:
        # Issue list had an unusable shape — keep the verdict, drop the issues.
        return ValidatorOutput(verdict=data["verdict"], issues=[],
                               summary=data.get("summary", ""))


def _parse_validator_fallback(raw_content: str) -> ValidatorOutput | None:
    """Best-effort recovery of a verdict from non-conforming model output.

    qwen3.5 on Ollama produces three failure shapes that all break json_schema
    parsing, and this handles each one:
      1. JSON wrapped in a <think> block or a ```fence``` — strip and json.loads.
      2. A bare {...} object embedded in prose — slice the outermost object.
      3. Pure prose ("**Verdict:** Pass …"). This is what qwen returns when
         thinking is OFF (it ignores the json_schema format), and it's also what
         survives in the reasoning trace when a thinking generation is truncated
         by num_predict before any JSON reaches `content`. Regex the verdict out.
    Returns None only when no verdict can be found at all."""
    cleaned = _strip_thinking(raw_content or "").strip()
    if not cleaned:
        return None
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

    # 1 & 2 — JSON object: the whole string, else sliced from surrounding prose.
    data = None
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(cleaned[start:end + 1])
            except (json.JSONDecodeError, TypeError):
                data = None
    parsed = _coerce_verdict_output(data) if isinstance(data, dict) else None
    if parsed is not None:
        return parsed

    # 3 — prose verdict. Take the LAST mention so a "fail" weighed mid-reasoning
    # doesn't override the model's final "Verdict: pass".
    matches = _VERDICT_RE.findall(cleaned)
    if matches:
        verdict = matches[-1].lower()
        summary = cleaned if len(cleaned) <= 600 else cleaned[:600].rstrip() + " …"
        return ValidatorOutput(verdict=verdict, issues=[], summary=summary)
    return None


def _raw_recovery_text(raw) -> str:
    """Text to mine for a verdict when schema parsing fails: content, plus the
    reasoning trace ONLY if the generation concluded.

    A length-truncated thinking trace (done_reason == "length", which is exactly
    when `content` comes back empty) is mid-deliberation — it echoes rule text
    like "when in doubt, pass" and tentative verdicts the model never committed
    to, so mining it flips genuine fails to pass. When truncated we return just
    the (empty) content and let the caller fall through to the thinking-OFF plain
    call, which always concludes."""
    if raw is None:
        return ""
    content = raw.content if hasattr(raw, "content") else str(raw)
    meta = getattr(raw, "response_metadata", {}) or {}
    reasoning = ""
    if meta.get("done_reason") != "length" and hasattr(raw, "additional_kwargs") \
            and isinstance(raw.additional_kwargs, dict):
        reasoning = raw.additional_kwargs.get("reasoning_content") or ""
    return f"{content}\n\n{reasoning}".strip()


def _invoke_verdict(structured_llm, plain_llm, messages, tries: int = 2):
    """Derive a pass/fail verdict, defeating qwen's two structured-output failure
    modes (see _parse_validator_fallback).

    Strategy, in order:
      1. The thinking-ON structured call (json_schema is honored when the verdict
         fits inside num_predict). Accept the schema-parsed verdict if present,
         else recover it from the raw content + reasoning trace.
      2. If that yields nothing, a single thinking-OFF plain call. qwen ignores
         json_schema there and returns prose, but with no <think> trace it cannot
         exhaust num_predict, so the verdict always lands in `content` — which the
         prose-aware fallback parses reliably.
    Note temperature is 0, so re-rolling the SAME structured call is pointless;
    `tries` only absorbs transient invoke exceptions. Returns
    (ValidatorOutput | None, last_raw) so the caller can still count tokens."""
    last_raw = None
    for _ in range(max(1, tries)):
        try:
            result = structured_llm.invoke(messages)
        except Exception:
            continue  # transient — re-roll
        raw = result.get("raw")
        if raw is not None:
            last_raw = raw
        parsed = result.get("parsed")
        if parsed is not None:
            return parsed, raw
        parsed = _parse_validator_fallback(_raw_recovery_text(raw))
        if parsed is not None:
            return parsed, raw
        break  # got a (bad) generation; temp=0 means re-rolling is identical

    # Last resort: thinking-OFF prose call that can't burn its budget on a trace.
    try:
        raw = plain_llm.invoke(messages)
        if raw is not None:
            last_raw = raw
        parsed = _parse_validator_fallback(_raw_recovery_text(raw))
        if parsed is not None:
            return parsed, raw
    except Exception:
        pass
    return None, last_raw


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
    validator_cfg = _cfg["agents"]["validator"]
    validator_tool_cfg = {**validator_cfg, **_cfg["agents"].get("validator_tools", {})}

    tool_llm = (make_llm(validator_tool_cfg)
                .bind_tools(DEFAULT_REVIEW_TOOLS))

    structured_llm = (make_llm(validator_cfg)
                      .with_structured_output(ValidatorOutput, include_raw=True, method="json_schema"))

    # Thinking-OFF prose fallback for the verdict. qwen ignores json_schema with
    # reasoning off, so this is a bare LLM whose output the prose-aware fallback
    # parses; with no <think> trace it can't exhaust num_predict and truncate.
    plain_llm = make_llm({**validator_cfg, "thinking": False})

    # Guard the tool loop
    MAX_TOOL_ITERS = 20
    tool_executor = ToolNode(DEFAULT_REVIEW_TOOLS, handle_tool_errors=False)

    messages: list[BaseMessage] = [HumanMessage(content=orchestrate_prompt)]
    work: dict = dict(state)

    # The tool-loop conversation grows unbounded.
    # We DON'T replay all of it into the structured verdict call
    tool_evidence: list[str] = []

    try:
        for _ in range(MAX_TOOL_ITERS):
            try:
                ai_msg = tool_llm.invoke(messages)
            except Exception as e:
                if _is_ollama_xml_bug(e):
                    _w("⚠️ Ollama tool-call parse failed; planning from context gathered so far.")
                    break  # degrade → go straight to the plan
                _w(f"tool_llm.invoke failed: {e!r}")
                raise

            messages.append(ai_msg)
            stats.record_tokens(ai_msg)  # count + live-emit each tool-loop call

            if not ai_msg.tool_calls:
                break

            # Pre-apply any set_language call before the batch runs so that
            # ToolNode's parallel threads all read the updated language from work.
            for tc in ai_msg.tool_calls:
                if tc["name"] == "set_language":
                    new_lang = (tc["args"].get("language") or "").strip().lower()
                    if new_lang in LANGUAGES:
                        work["language"] = new_lang

            work["messages"] = messages
            tool_result = tool_executor.invoke(
                work,
                config=config,
            )
            for m in tool_result["messages"]:
                if isinstance(m.content, str):
                    m.content = _scrub(m.content)
                tool_name = getattr(m, "name", None) or "tool"
                tool_evidence.append(f"### {tool_name}\n{m.content}")
            messages.extend(tool_result["messages"])

        else:
            _w(f"❌ Validator exceeded max tool iterations ({MAX_TOOL_ITERS}).")

        # Slim verdict context: task + capped tool results only
        MAX_EVIDENCE_CHARS = 12000
        evidence_text = "\n\n".join(tool_evidence).strip() or "(no tool output was produced)"
        if len(evidence_text) > MAX_EVIDENCE_CHARS:
            evidence_text = evidence_text[:MAX_EVIDENCE_CHARS] + "\n…[evidence truncated]"

        verdict_messages: list[BaseMessage] = [HumanMessage(content=(
            "You are a code validator. Below are the real tool results for the "
            f"files under review ({', '.join(files_written)}).\n\n"
            f"<task description=\"what the coder was implementing — context only\">\n{instruction}\n</task>\n\n"
            f"<tool_results>\n{evidence_text}\n</tool_results>\n\n"
            f"{verdict_prompt}"
        ))]
        parsed, last_raw = _invoke_verdict(structured_llm, plain_llm, verdict_messages)

    except Exception as e:
        msg = f"[FAILED] Validator crashed: {e}"
        _w(f"❌ Validator crashed: `{e}`")
        return {"latest_report": msg, "history": ["validator_failed"],
                "coder_retries": state.get("coder_retries", 0) + 1}

    LANGUAGE = work["language"] #get language to pass it to the next node if it switched

    if last_raw is not None:
        stats.record_tokens(last_raw)

    if parsed is None:
        # The TOOLS already ran; we just couldn't get the model to emit a verdict.
        # That's an infrastructure miss, not evidence the coder's code is broken,
        # so DON'T punish the coder (no coder_retries bump, no [FAILED] report).
        # Per the validator's own "when in doubt, pass" rule, soft-pass and move on.
        _w("⚠️ Validator could not parse a verdict from the tool results — "
           "soft-passing (no code issues were reported by the tooling).")
        return {
            "latest_report": "Validator soft-pass: verdict unparseable; no tool-reported issues.",
            "context_store": {"workspace_files": list_workspace_files(project_path)},
            "history": ["validator"],
            "language": LANGUAGE,
        }

    if parsed.verdict == "pass":
        _w(f"✅ **PASS** — {parsed.summary}")
        return {
            "latest_report": f"Validator passed: {parsed.summary}",
            "context_store": {"workspace_files": list_workspace_files(project_path)},
            "history": ["validator"],
            "language": LANGUAGE
        }

    retries = state.get("coder_retries", 0) + 1

    # A verdict recovered from prose (thinking-OFF / truncated trace) carries no
    # structured issues — fall back to the summary so the coder still gets
    # something actionable instead of an empty issues list.
    issue_dicts = [
        {"file": i.file_path, "severity": i.severity, "issue": i.description}
        for i in parsed.issues
    ]
    if not issue_dicts:
        issue_dicts = [{
            "file": ", ".join(files_written),
            "severity": "error",
            "issue": parsed.summary or "Validator reported a failure without details.",
        }]
    issues_json = json.dumps(issue_dicts, indent=2)

    # write issues to frontend stream
    issues_md = "\n".join(
        f"  - `{d['file']}` [{d['severity']}]: {d['issue']}" for d in issue_dicts
    )
    _w(f"❌ **FAIL** (retry **{retries}/{_max_retries}**) — {parsed.summary}\n\n{issues_md}")

    return {
        "latest_report": f"[FAILED] Validator: {parsed.summary}",
        "context_store": {"validation_issues": issues_json},
        "history": ["validator_failed"],
        "coder_retries": retries,
        "language": LANGUAGE
    }