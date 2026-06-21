"""
Shared building blocks for the inspector's three modes (explore / review / verify).

This module holds everything the mode implementations have in common:
  * config + LLM/tool wiring loaded once from `graph_config.yaml`;
  * the deterministic, host-side helpers that keep agent claims grounded in the
    canonical file tree (path classification, existence checks, issue coercion);
  * `make_writer`, the small Markdown stream-writer used to talk to the frontend;
  * `run_tool_loop`, the Validator's-Pattern tool loop (ReAct over read-only
    tools, then a single structured-output call) that all three modes drive.

Keeping these here means `explorer.py`, `review.py` and `verify.py` only contain
the bits that actually differ between modes (prompts, output schema, result
shaping), with no duplicated loop logic.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.prebuilt import ToolNode

from agents.llm_factory import make_llm
from agents.implementations.code_agent.utils.llm_helpers import _is_ollama_xml_bug, _scrub
from agents.implementations.code_agent.utils.stats import stats
from structured_output import LANGUAGES
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
    compile_code,
)

from agents.implementations.code_agent.structured_output import ReviewIssue

_CFG_PATH = Path(__file__).resolve().parent.parent / "graph_config.yaml"
with open(_CFG_PATH, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

inspector_config = _cfg["agents"]["inspector"]
inspector_tool_config = {**inspector_config, **_cfg["agents"].get("inspector_tools", {})}

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
    set_language,
]

MAX_TOOL_ITERS = 30


def make_writer():
    """Return a `_w(text)` helper that streams Markdown text to the frontend."""
    writer = get_stream_writer()

    def _w(text: str) -> None:
        writer({"kind": "text", "text": text + "\n\n"})

    return _w


def run_tool_loop(
    messages: list[BaseMessage],
    work: dict,
    config: RunnableConfig,
    output_schema,
    verdict_prompt: str,
    write,
):
    """Drive the Validator's Pattern: a ReAct tool loop, then one structured call.

    Two LLMs are used — one bound to the read-only tools for the loop, one with
    structured output for the final verdict. Mutates `messages` (and `work`,
    whose "language" is updated when the model calls `set_language`) in place and
    returns the parsed structured-output object (may be None if parsing failed).

    Raises on a genuine tool-call failure (other than the tolerated Ollama XML
    bug, which degrades to "plan from context gathered so far"); callers wrap
    this in their own try/except to build a mode-specific failure report.
    """
    tool_llm = make_llm(inspector_tool_config).bind_tools(DEFAULT_REVIEW_TOOLS)
    structured_llm = make_llm(inspector_config).with_structured_output(
        output_schema, include_raw=True, method="json_schema"
    )
    tool_executor = ToolNode(DEFAULT_REVIEW_TOOLS, handle_tool_errors=False)

    for _ in range(MAX_TOOL_ITERS):
        try:
            ai_msg = tool_llm.invoke(messages)
        except Exception as e:
            if _is_ollama_xml_bug(e):
                write("⚠️ Ollama tool-call parse failed; planning from context gathered so far.")
                break  # degrade → go straight to the plan
            write(f"tool_llm.invoke failed: {e!r}")
            raise

        messages.append(ai_msg)
        stats.record_tokens(ai_msg)  # count + live-emit each tool-loop call
        if not ai_msg.tool_calls:
            break

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
        messages.extend(tool_result["messages"])

    else:
        write(f"❌ Inspector exceeded max tool iterations ({MAX_TOOL_ITERS}).")

    messages.append(HumanMessage(content=verdict_prompt))
    result = structured_llm.invoke(messages)
    if result.get("raw"):
        stats.record_tokens(result["raw"])
    return result["parsed"]


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


def _parse_verifier_fallback(raw_content: str):
    from agents.implementations.code_agent.structured_output import ValidatorOutput

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
