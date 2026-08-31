"""
Coder — implements one architect step in the project.

Stripped-down port: the model returns a structured CoderOutput (full-content
`create` / `delete`), and THIS node applies each change deterministically via the
programmatic safe_write / safe_delete helpers — so the model never has to pick a
file-writing tool. The module-export registry, import-path hints and snapshot
bookkeeping from the standalone project are dropped; reverts are handled by the
tools package's session snapshots (safe_write records them).

On a validator retry (coder_retries > 0) the validator's issues are folded into
the prompt. Files written this step are published in coder_latest_files for the
validator to check.
"""
from __future__ import annotations

import json
from pathlib import Path
from string import Template

import yaml
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.prebuilt import ToolNode

from agents.llm_factory import make_llm
from agents.implementations.code_agent.utils.llm_helpers import _is_ollama_xml_bug, _scrub
from tools import safe_write, safe_delete, list_workspace_files, read_file, safe_read
from tools.languages.symbols import extract_definitions, language_for_path

from agents.implementations.code_agent.structured_output import AgentState, CoderOutput, FileChange
from agents.implementations.code_agent.utils.stats import stats
from agents.implementations.code_agent.utils.stream_helpers import (
    tool_start, tool_end, stream_tool_calls, tool_target,
)

_CFG_PATH = Path(__file__).resolve().parent.parent / "graph_config.yaml"
with open(_CFG_PATH, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

_max_retries = _cfg.get("coder_max_retries", 5)

CODER_TOOLS = [
    read_file,
]

MAX_TOOL_ITERS = 15

def _parse_coder_fallback(raw_content: str) -> CoderOutput | None:
    """Best-effort parse if the structured call returns nothing usable."""
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
    if isinstance(data.get("changes"), list):
        # Normalize a deprecated "modify" action into "create".
        changes = [{**c, "action": "create"} if isinstance(c, dict) and c.get("action") == "modify" else c
                   for c in data["changes"]]
        try:
            return CoderOutput.model_validate({"changes": changes})
        except Exception:
            return None
    return None


def _apply_change(change: FileChange, project_path: str) -> tuple[bool, str]:
    if change.action == "create":
        return safe_write(project_path, change.file_path, change.content)
    if change.action == "delete":
        return safe_delete(project_path, change.file_path)
    return False, f"Unknown action '{change.action}' — only 'create' and 'delete' are supported."


def coder_node(state: AgentState, config: RunnableConfig) -> dict:
    writer = get_stream_writer() # frontend-provided stream writer
    # helper function to write text on stream for frontend (in Markdown)
    def _w(text: str) -> None:
        writer({"kind": "text", "text": text + "\n\n"})

    writer({"kind": "stage", "stage": "coder",
            "label": "⌨️ Coder — implementing architect plan"})

    project_path = state.get("project_path", ".")
    store = state.get("context_store", {})
    retries = state.get("coder_retries", 0)

    files_to_create = store.get("architect_files_to_create", [])
    files_to_modify = store.get("architect_files_to_modify", [])
    files_to_delete = store.get("architect_files_to_delete", [])
    architect_plan = store.get("architect_plan", "No architect plan available.")
    allowed_files = list(dict.fromkeys(files_to_create + files_to_modify + files_to_delete))

    _w(f"### ⌨️ Coder{'  — retry **' + str(retries) + '/' + str(_max_retries) + '**' if retries else ''}\n\n**Plan:** {architect_plan[:]}")

    # Feed the CURRENT on-disk content of every modify target straight into the
    # prompt — the same "don't rely on the model to call a tool" approach the
    # architect uses for the file tree. A blind, non-thinking coder otherwise
    # rewrites the file from the step_plan alone and silently drops everything
    # earlier steps added (the overwrite bug). With the real content inlined it
    # extends instead of overwriting. We also snapshot each file's public symbols
    # so we can catch a drop deterministically after the model responds. Symbol
    # extraction is language-aware (tree-sitter) so the guard covers every language
    # the pipeline supports, not just Python.
    existing_blocks: list[str] = []
    existing_symbols: dict[str, set[str]] = {}
    for rel in files_to_modify:
        content = safe_read(project_path, rel)
        if content is None:
            existing_blocks.append(f"=== {rel} ===\n[does not exist yet — create it from scratch per the plan]")
        else:
            existing_blocks.append(f"=== {rel} ===\n{content}")
            syms = extract_definitions(content, language_for_path(rel))
            if syms:
                existing_symbols[rel] = syms
    existing_file_contents = (
        "\n\n".join(existing_blocks)
        if existing_blocks else "  (none — this step only creates or deletes files)"
    )

    feedback = ""
    if retries:
        feedback += f'<retry attempt="{retries}">Fix the issues below; rewrite each file in full.</retry>\n'
    if store.get("validation_issues"):
        feedback += f"<validation_issues>\n{store['validation_issues']}\n</validation_issues>\n"

    ####################################################################
    # VALIDATOR'S PATTERN - TOOL LOOP - THEN STRUCTURED OUTPUT         #
    # We create two llm's (one for the tool calls, one for the parser) #
    ####################################################################
    prompt = Template(_cfg["prompts"]["coder"]).safe_substitute(
        architect_plan=architect_plan,
        architect_files_to_create=files_to_create,
        architect_files_to_modify=files_to_modify,
        architect_files_to_delete=files_to_delete,
        existing_file_contents=existing_file_contents,
        allowed_files="\n".join(f"  - {f}" for f in allowed_files),
        feedback=feedback,
        sandbox_dir=project_path,
        required_exports="",  # stripped: no module-export registry
    )

    coder_cfg = _cfg["agents"]["coder"]
    coder_tool_cfg = {**coder_cfg, **_cfg["agents"].get("coder_tools", {})}

    tool_llm = make_llm(coder_tool_cfg).bind_tools(CODER_TOOLS)
    structured_llm = ((make_llm(coder_cfg))
                      .with_structured_output(CoderOutput, include_raw=True, method="json_schema"))
    tool_executor = ToolNode(CODER_TOOLS, handle_tool_errors=False)
    messages: list[BaseMessage] = [HumanMessage(content=prompt)]

    code_prompt=(
        "You have read all the files you need. "
        "Now produce your changes as a JSON CoderOutput. "
        "For files_to_modify: output the COMPLETE new version incorporating your changes — "
        "every line, no truncation, no placeholders. "
        "For files_to_create: write the complete file from scratch per the architect plan. "
        "Only write files listed in your scope."
    )
    parsed = None

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

            tool_result = tool_executor.invoke(
                {**state, "messages": messages},
                config=config,
            )
            for m in tool_result["messages"]:
                if isinstance(m.content, str):
                    m.content = _scrub(m.content)
            messages.extend(tool_result["messages"])
            # Surface each read as a settled bubble. The ToolNode runs inline, so
            # its results never hit the "messages" stream the UI server watches —
            # the node must announce its own tool activity (see stream_helpers).
            stream_tool_calls(writer, ai_msg.tool_calls)

        else:
            _w(f"❌ Coder exceeded max tool iterations ({MAX_TOOL_ITERS}).")
        messages.append(HumanMessage(content=code_prompt))
        result = structured_llm.invoke(messages)
        if result.get("raw"):
            stats.record_tokens(result["raw"])
        parsed = result["parsed"]

        if parsed is None and result.get("raw"):
            raw_text = result["raw"].content if hasattr(result["raw"], "content") else str(result["raw"])
            parsed = _parse_coder_fallback(raw_text)

    except Exception as e:
        msg = f"[FAILED] Coder could not generate output: {e}"
        _w(f"❌ Coder could not generate output: `{e}`")
        return {"latest_report": msg, "history": ["coder_failed"]}

    if parsed is None:
        msg = "[FAILED] Coder produced no usable output."
        _w("❌ Coder produced no usable output.")
        return {"latest_report": msg, "history": ["coder_failed"]}


    # Keep only the last change per path (most complete) and enforce the step scope.
    seen: dict[str, FileChange] = {c.file_path: c for c in parsed.changes}
    changes = list(seen.values())
    if allowed_files:
        scoped = [c for c in changes if c.file_path in allowed_files]
        rejected = [c.file_path for c in changes if c.file_path not in allowed_files]
        if rejected:
            _w(f"⚠️ Scope violation — dropping out-of-plan file(s): `{rejected}`")
        changes = scoped
        if not changes:
            msg = f"[FAILED] Coder wrote none of the planned files. Expected: {sorted(allowed_files)}."
            _w(f"❌ Coder wrote none of the planned files. Expected: `{sorted(allowed_files)}`")
            return {
                "latest_report": msg,
                "history": ["coder_failed"],
                "context_store": {"validation_issues": msg},
            }

    # Regression guard: a modify step must EXTEND the file, never silently drop
    # public defs an earlier step added. The deterministic validator tools
    # (lint/type/import/compile) can't see this — the truncated file still
    # compiles — so catch it HERE and bounce it back to the coder with feedback
    # instead of writing the broken file and letting the validator pass it. We
    # skip the check when the plan signals an intentional restructure, to avoid
    # false positives on a legitimate rename/removal.
    plan_lower = architect_plan.lower()
    intentional_removal = any(
        w in plan_lower for w in ("remove", "delete", "replace", "rename", "refactor")
    )
    if not intentional_removal:
        regressions = []
        for change in changes:
            if change.action != "create":
                continue
            old_syms = existing_symbols.get(change.file_path)
            if not old_syms:
                continue
            new_syms = extract_definitions(change.content or "", language_for_path(change.file_path))
            dropped = sorted(old_syms - new_syms)
            if dropped:
                regressions.append((change.file_path, dropped))
        if regressions:
            detail = "; ".join(f"`{f}` dropped {d}" for f, d in regressions)
            msg = (
                "[FAILED] Coder regression: the modify output removed definitions that "
                f"already existed ({detail}). A modify must output the COMPLETE file — "
                "keep every existing definition verbatim and ADD the new ones."
            )
            _w(f"❌ Regression — modify dropped existing definitions: {detail}")
            return {
                "latest_report": msg,
                "history": ["coder_failed"],
                "context_store": {"validation_issues": msg},
            }

    succeeded, failed = [], []
    for change in changes:
        # Writes/deletes are applied programmatically (not via an LLM tool call),
        # so emit the tool bubble ourselves: a live spinner over the write, then a
        # settled amber 📝 (or red 🗑️) pill naming the file — with a revert button.
        tool_name = "delete_file" if change.action == "delete" else "write_file"
        target = tool_target({"file_path": change.file_path})
        tool_start(writer, tool_name, target)
        ok, msg = _apply_change(change, project_path)
        tool_end(writer, tool_name, target)
        (succeeded if ok else failed).append(msg)
        if not ok:
            _w(f"  ❌ `{change.file_path}` — {msg}")

    written = [c.file_path for c in changes if c.action != "delete"]
    if failed and not succeeded:
        report = f"[FAILED] Coder could not apply any changes. Errors: {failed}"
        history = ["coder_failed"]
    elif failed:
        report = f"Coder applied {len(succeeded)} change(s) with {len(failed)} failure(s): {failed}"
        history = ["coder"]
    else:
        report = f"Coder applied {len(succeeded)} change(s): {[c.file_path for c in changes]}"
        history = ["coder"]

    # write final report to stream
    if failed and not succeeded:
        _w(f"❌ Could not apply any changes.\n\n**Errors:** {failed}")
    elif failed:
        _w(f"⚠️ Applied **{len(succeeded)}** change(s) with **{len(failed)}** failure(s):\n\n{failed}")
    else:
        _w(f"✅ Applied **{len(succeeded)}** change(s).")

    return {
        "latest_report": report,
        "context_store": {
            "coder_latest_files": {"current": written},
            "workspace_files": list_workspace_files(project_path),
        },
        "history": history,
    }