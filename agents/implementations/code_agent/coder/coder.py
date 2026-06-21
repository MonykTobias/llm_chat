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
from tools import safe_write, safe_delete, list_workspace_files, read_file

from agents.implementations.code_agent.structured_output import AgentState, CoderOutput, FileChange
from agents.implementations.code_agent.utils.stats import stats

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

    succeeded, failed = [], []
    for change in changes:
        ok, msg = _apply_change(change, project_path)
        (succeeded if ok else failed).append(msg)
        _w(f"  {'✅' if ok else '❌'} `{change.file_path}` — {msg}")

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
        files_md = "\n".join(f"  - `{c.file_path}`" for c in changes)
        _w(f"✅ Applied **{len(succeeded)}** change(s):\n\n{files_md}")

    return {
        "latest_report": report,
        "context_store": {
            "coder_latest_files": {"current": written},
            "workspace_files": list_workspace_files(project_path),
        },
        "history": history,
    }