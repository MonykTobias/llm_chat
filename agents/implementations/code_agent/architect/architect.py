"""
Architect — drafts a sequential step plan for one orchestrator task.

Stripped-down port of the standalone project's architect: the heavy machinery
(ReAct investigation with a file-tree enforcement, path
canonicalization, required-file correction passes, snapshot restores) is gone.
Instead the node feeds the model the real file listing programmatically
(list_workspace_files) and asks for a structured ArchitectOutput in one shot —
no reliance on the model calling the right tool.

Step 1 is dispatched to the coder immediately; steps 2..N are queued in
architect_step_queue for step_dispatch. A re-plan (coder_retries > 0) folds the
validator's feedback into the prompt so the model varies its approach.
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
from tools import list_workspace_files, read_file, list_all_files, analyze_architecture

from agents.implementations.code_agent.structured_output import AgentState, ArchitectOutput, ArchitectStep
from agents.implementations.code_agent.utils.stats import stats

ARCHITECT_TOOLS = [
    read_file,
    analyze_architecture,
    list_all_files,
]

MAX_TOOL_ITERS = 15

_CFG_PATH = Path(__file__).resolve().parent.parent / "graph_config.yaml"
with open(_CFG_PATH, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

def _as_file_list(value) -> list[str]:
    """Coerce a files_to_* field into a clean list of path strings (models often
    emit a bare string or null instead of the required array)."""
    if value is None:
        return []
    if isinstance(value, str):
        v = value.strip()
        return [v] if v else []
    if isinstance(value, list):
        return [s.strip() for s in value if isinstance(s, str) and s.strip()]
    return []


def _parse_architect_fallback(raw_content: str) -> ArchitectOutput | None:
    """Best-effort parse of a plan the structured call failed to return cleanly."""
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

    plan_text = data.get("plan", "")
    if not isinstance(plan_text, str):
        return None
    if data.get("task_complete"):
        return ArchitectOutput(plan=plan_text, steps=[], task_complete=True)

    steps = []
    for s in data.get("steps", []):
        if isinstance(s, dict) and s.get("step_plan"):
            sp = s["step_plan"]
            steps.append(ArchitectStep(
                step_plan=sp if isinstance(sp, str) else json.dumps(sp),
                files_to_create=_as_file_list(s.get("files_to_create")),
                files_to_modify=_as_file_list(s.get("files_to_modify")),
                files_to_delete=_as_file_list(s.get("files_to_delete")),
            ))
    return ArchitectOutput(plan=plan_text, steps=steps) if steps else None


def architect_node(state: AgentState, config: RunnableConfig) -> dict:
    writer = get_stream_writer() # frontend-provided stream writer
    # helper function to write text on stream for frontend (in Markdown)
    def _w(text: str) -> None:
        writer({"kind": "text", "text": text + "\n\n"})

    writer({"kind": "stage", "stage": "architect",
            "label": "📐 Architect — planning for coder"})

    instruction = state["plan"].instruction_for_agent
    project_path = state.get("project_path", ".")
    store = state.get("context_store", {})
    retries = state.get("coder_retries", 0)
    is_replan = retries > 0

    prompt_key = "architect_recovery_coder_fails" if is_replan else "architect"
    _w(f"### 📐 Architect{'  — re-plan after **' + str(retries) + '** failed attempt(s)' if is_replan else ''}\n\n**Task:** {instruction[:]}")

    arch_cfg = _cfg["agents"]["architect"]
    arch_tool_cfg = {**arch_cfg, **_cfg["agents"].get("architect_tools", {})}
    # Feed the model the real, current file listing instead of relying on it to call tools
    file_tree = "\n".join(f"  {p}" for p in list_workspace_files(project_path)) or "  (empty project)"
    inspector_findings = store.get("inspector_raw") or "No prior exploration found."
    # if really long cut it down from the end (most relevant part)
    if len(inspector_findings) > 1500:
        inspector_findings = "... (truncated)\n" + inspector_findings[-1500:]
    inspector_files = store.get("inspector_files") or []
    inspector_issues = store.get("inspector_issues") or []

    ####################################################################
    # VALIDATOR'S PATTERN - TOOL LOOP - THEN STRUCTURED OUTPUT         #
    # We create two llm's (one for the tool calls, one for the parser) #
    ####################################################################
    system_prompt = Template(_cfg["prompts"][prompt_key]).safe_substitute(
        instruction=instruction,
        inspector_findings=inspector_findings,
        inspector_files="\n".join(f"  {i}" for i in inspector_files) or "  (none)",
        inspector_issues="\n".join(f"  - {i}" for i in inspector_issues) or "  (none)",
        spec=store.get("spec_content", ""),
        file_tree=file_tree,
    )

    # Add re-plan feedback to the prompt if this is a re-plan.
    if is_replan:
        system_prompt += (
            "\n\n<replan>Your prior plan for this task failed validation "
            f"({retries} attempt(s)). Validator feedback:\n"
            f"{store.get('validation_issues', '')}\n"
            "Produce a genuinely different plan — do not repeat the failed approach.</replan>"
        )

    tool_llm = make_llm(arch_tool_cfg).bind_tools(ARCHITECT_TOOLS)
    structured_llm = (make_llm(arch_cfg).with_structured_output(ArchitectOutput, include_raw=True, method="json_schema"))
    tool_executor = ToolNode(ARCHITECT_TOOLS,handle_tool_errors=False)
    messages: list[BaseMessage] = [HumanMessage(content=system_prompt)]

    plan_prompt = (
        "You have gathered enough context. Now produce your structured implementation plan."
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
            for m in tool_result["messages"]:  # scrub file contents before re-feeding
                if isinstance(m.content, str):
                    m.content = _scrub(m.content)
            messages.extend(tool_result["messages"])

        else:
            _w(f"⚠️ Architect hit max tool iterations ({MAX_TOOL_ITERS}); planning with current context.")


        messages.append(HumanMessage(content=plan_prompt))
        result = structured_llm.invoke(messages)
        if result.get("raw"):
            stats.record_tokens(result["raw"])
        parsed = result["parsed"]
        if parsed is None and result.get("raw"):
            raw_text = result["raw"].content if hasattr(result["raw"], "content") else str(result["raw"])
            parsed = _parse_architect_fallback(raw_text)

    except Exception as e:
        msg = f"[FAILED] Architect could not generate a plan: {e}"
        _w(f"❌ Architect could not generate a plan: `{e}`")
        return {"latest_report": msg, "history": ["architect_failed"]}

    if parsed is None:
        msg = "[FAILED] Architect produced no usable output."
        _w("❌ Architect produced no usable output.")
        return {"latest_report": msg, "history": ["architect_failed"]}

    # Legitimate no-op: task already satisfied. '[COMPLETE]' routes back to the
    # orchestrator (and is NOT '[FAILED]', so it records as done).
    if parsed.task_complete:
        reason = (parsed.plan or "Task already satisfied; no changes required.").strip()
        _w(f"✅ No changes needed: {reason[:]}")
        return {
            "latest_report": f"[COMPLETE] {reason[:]}",
            "history": ["architect_noop"],
            "coder_retries": 0,
            "architect_replans": 0,
            "architect_step_queue": [],
        }

    if not parsed.steps:
        msg = "[FAILED] Architect produced no steps."
        _w("❌ Architect produced no steps.")
        return {"latest_report": msg, "history": ["architect_failed"]}

    # Print plan with step details
    steps_md = "\n".join(
        f"  {i + 1}. create={s.files_to_create} modify={s.files_to_modify} delete={s.files_to_delete}"
        for i, s in enumerate(parsed.steps)
    )
    _w(f"✅ Plan ready: **{len(parsed.steps)}** step(s)\n\n{steps_md}")

    # Dispatch step 1; queue steps 2..N for step_dispatch.
    step1 = parsed.steps[0]
    remaining_queue = [
        {
            "step_plan":        s.step_plan,
            "files_to_create":  s.files_to_create,
            "files_to_modify":  s.files_to_modify,
            "files_to_delete":  s.files_to_delete,
        }
        for s in parsed.steps[1:]
    ]
    ctx_store = {
        "architect_plan":            step1.step_plan,
        "architect_files_to_create": step1.files_to_create,
        "architect_files_to_modify": step1.files_to_modify,
        "architect_files_to_delete": step1.files_to_delete,
        "validation_issues":         "",
        "coder_latest_files":        {"current": []},
    }
    current_replans = state.get("architect_replans", 0)
    return {
        "latest_report": f"Architect plan ready ({len(parsed.steps)} step(s)): {step1.step_plan[:150]}",
        "context_store": ctx_store,
        "history": ["architect"],
        "coder_retries": 0,
        "architect_replans": current_replans + 1 if is_replan else 0,
        "architect_step_queue": remaining_queue,
    }
