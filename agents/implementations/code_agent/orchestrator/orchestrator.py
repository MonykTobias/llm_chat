"""
Orchestrator — the central planner that drives the graph workflow.

Reads the objective, reviews completed work, and decides what to do next:
  - Send to Inspector to gather information (read-only)
  - Send to Architect to plan + trigger an implementation cycle
    (architect → coder → validator)
  - Mark the objective as complete

Has a fast-path queue: if the LLM produced multiple subtasks, it dispatches
them one-by-one without re-invoking the LLM until the queue is drained or
a failure occurs.

Two surfaces live in this module:

  * `orchestrator_node(state, config)` — the LangGraph node itself, ported
    (nearly verbatim) from the standalone `orchestrator` project. The graph in
    this package's `__init__.py` wires it together with the (currently dummy)
    inspector / architect / coder / validator nodes.

  * `Orchestrator` — a thin object that mimics the slice of `agents.base.BaseAgent`
    the UI wiring relies on (`name`, `requires_project`, `kickoff_message`,
    `tool_names`, `pool`, `default`, `get`, `stream`, `invoke`). It compiles the
    graph (via `get_app`) and exposes it the same way `BaseAgent` exposes a
    `create_agent` graph — WITHOUT subclassing it, because the per-node LLMs are
    configured from `graph_config.yaml` rather than from the prompt+tools contract
    `BaseAgent` enforces. This is the object intended to eventually replace
    `agents/sub_agents/code_review/orchestrator.py`'s `CodeReviewOrchestrator` in
    the registry. It is deliberately NOT wired into the frontend yet.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from string import Template
from typing import Any

import yaml
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from agents.llm_factory import make_llm

from agents.implementations.code_agent.structured_output import (
    AgentState,
    TaskPlan,
    PlannerOutput,
    SubTask,
    REVIEW_TASK_MARKER,
)
from agents.implementations.code_agent.utils.workspace import restore_snapshot, list_workspace_files
from agents.implementations.code_agent.utils.validation import extract_paths, on_tree
from agents.implementations.code_agent.utils.stats import stats

# Token-limit recovery is keyed on openai's LengthFinishReasonError. The graph
# talks to Ollama, so openai may not be installed; degrade gracefully to an empty
# tuple, which `isinstance(e, ())` always treats as "not that error".
try:
    from openai import LengthFinishReasonError
except ImportError:  # pragma: no cover - openai is optional here
    LengthFinishReasonError = ()

# Config lives next to this module (graph_config.yaml), not at the CWD root, so
# the orchestrator graph keeps its own model + prompt set independent of the
# project's top-level config.yaml.
_CFG_PATH = Path(__file__).resolve().parent.parent / "graph_config.yaml"
with open(_CFG_PATH, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

MAX_ITERATIONS = _cfg.get("max_iterations", 100)
MAX_RAW_CHARS = _cfg.get("max_raw_chars", 2000)
MAX_WORKSPACE_FILES = _cfg.get("max_workspace_files", 120)
MAX_TASK_ATTEMPTS = _cfg.get("max_task_attempts", 3)
TASK_RECONSIDERATION_BUDGET = _cfg.get("task_reconsideration_budget", 1)
# When True, a full re-plan (after a task is abandoned + re-planned) asks the
# scaffold node to re-run in diff mode so the canonical tree covers any new files
# the fresh plan introduced. Default False = scaffold runs exactly once, right
# after the first plan. The scaffold node clears the request flag when it runs.
SCAFFOLD_ON_REPLAN = _cfg.get("scaffold_on_replan", False)


def _parse_plan_fallback(raw_content: str) -> PlannerOutput | None:
    """Handle common schema deviations from the LLM."""
    try:
        data = json.loads(raw_content)
    except (json.JSONDecodeError, TypeError):
        return None

    valid_tasks = []
    for item in data.get("todo_list", []):
        if not isinstance(item, dict):
            continue
        agent = item.get("agent", "")
        if agent == "coder":
            agent = "architect"
        if agent in ("inspector", "architect"):
            valid_tasks.append(SubTask(agent=agent, instruction=item.get("instruction", "")))

    return PlannerOutput(
        thinking_process=data.get("thinking_process", ""),
        todo_list=valid_tasks,
    )


def _sanitize_todo(todo_list: list[SubTask], file_tree: list[str]) -> tuple[list[SubTask], list[tuple[SubTask, list[str]]]]:
    """
    Drop planner tasks that name ONLY files absent from the canonical tree.

    The orchestrator may name files as scope hints, but a task whose every named
    file is off-tree is a hallucination (e.g. 'verify manifest.json for PWA' on a
    project that has no manifest). Letting it through forces the architect into a
    correction loop chasing a file that cannot exist. Tasks that name no files, or
    name at least one real tree file, pass through unchanged.

    Returns (kept, dropped) where dropped is a list of (task, off_tree_files).
    """
    if not file_tree:
        return list(todo_list), []
    kept: list[SubTask] = []
    dropped: list[tuple[SubTask, list[str]]] = []
    for t in todo_list:
        named = extract_paths(t.instruction)
        off = [f for f in named if not on_tree(f, file_tree)]
        if named and len(off) == len(named):
            dropped.append((t, off))
        else:
            kept.append(t)
    return kept, dropped


def _run_reconsideration(
    state: AgentState,
    current_plan: TaskPlan,
    latest_report: str,
    attempt: int,
) -> PlannerOutput | None:
    """
    Focused LLM call that produces a 1–3 task replacement sub-plan for the
    single failing task. The preserved remaining queue is passed as read-only
    context so the LLM does not re-emit it. Returns None on any failure so the
    caller falls through to the mechanical retry path gracefully.
    """
    orch = _cfg["agents"]["orchestrator"]
    prompt_template = _cfg["prompts"].get("orchestrator_reconsideration")
    if not prompt_template:
        print("[Orchestrator] orchestrator_reconsideration prompt missing — skipping reconsideration.")
        return None

    llm = make_llm(orch)
    structured_llm = llm.with_structured_output(PlannerOutput, include_raw=True)
    store = state.get("context_store", {})

    remaining = "\n".join(
        f"  {i+1}. [{t.agent}] {t.instruction}"
        for i, t in enumerate(current_plan.todo_list)
    ) or "  (none)"

    full_prompt = Template(prompt_template).safe_substitute(
        objective=state["objective"],
        failing_task_agent=current_plan.next_agent,
        failing_task_instruction=current_plan.instruction_for_agent,
        failure_report=latest_report,
        attempt_number=attempt,
        remaining_queue=remaining,
        spec_content=store.get("spec_content", ""),
        file_tree="\n".join(f"  {p}" for p in (store.get("file_tree", []) or [])),
    )

    try:
        result = structured_llm.invoke([HumanMessage(content=full_prompt)])
    except Exception as e:
        print(f"[Orchestrator] Reconsideration LLM call failed: {e}")
        return None

    if result.get("raw"):
        stats.record_tokens(result["raw"])

    planner_out = result["parsed"]
    if planner_out is None and result.get("raw"):
        raw_text = result["raw"].content if hasattr(result["raw"], "content") else str(result["raw"])
        planner_out = _parse_plan_fallback(raw_text)

    if planner_out is None or not planner_out.todo_list:
        print("[Orchestrator] Reconsideration produced no tasks — falling through to mechanical retry.")
        return None

    print(f"[Orchestrator] Reconsideration produced {len(planner_out.todo_list)} replacement task(s).")
    return planner_out


def orchestrator_node(state: AgentState, config: RunnableConfig):
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

    writer({"kind": "stage", "stage": "orchestrator",
            "label": "🧭 Orchestrator — planning next step"})

    iteration = state.get("iteration_count", 0)

    # ── Safety: iteration cap ────────────────────────────────────────────
    if iteration >= MAX_ITERATIONS:
        _w(f"⛔ Iteration limit ({MAX_ITERATIONS}) reached — forcing completion.")
        completed = state["plan"].completed_tasks if state.get("plan") else []
        return {
            "plan": TaskPlan(
                thinking_process="Iteration limit reached, forcing completion.",
                completed_tasks=completed,
                next_agent="complete",
                instruction_for_agent="",
            ),
            "context_store": {"skip_verification": True},
            "iteration_count": 1,
            "coder_retries": 0,
            "architect_replans": 0,
        }

    current_plan = state.get("plan")
    latest_report = state.get("latest_report", "")
    has_failed = latest_report.startswith("[FAILED]")

    # ── Per-task failure tracking: abandon a task that keeps failing ──────
    # active_task_original is the instruction text of the task as first dispatched.
    # It acts as a stable failure-count key that survives across reconsiderations —
    # even if reconsideration rewrites the instruction, the count is keyed on the
    # original so we never inadvertently reset it by rephrasing. The count reaches
    # MAX_TASK_ATTEMPTS → task is abandoned and Step 3 re-plans from scratch.
    _store = state.get("context_store", {})
    failure_counts = dict(_store.get("task_failure_counts", {}))
    abandoned_tasks = list(_store.get("abandoned_tasks", []))
    active_original = _store.get("active_task_original", "")
    reconsideration_count = _store.get("task_reconsideration_count", 0)
    just_abandoned = None
    if has_failed and current_plan and current_plan.instruction_for_agent:
        stable_key = active_original or current_plan.instruction_for_agent
        failure_counts[stable_key] = failure_counts.get(stable_key, 0) + 1
        attempts = failure_counts[stable_key]
        _w(f"⚠️ Task failure **{attempts}/{MAX_TASK_ATTEMPTS}:** `{stable_key[:]}`")
        if attempts >= MAX_TASK_ATTEMPTS and stable_key not in abandoned_tasks:
            abandoned_tasks.append(stable_key)
            failure_counts.pop(stable_key, None)
            just_abandoned = stable_key
            _w(f"🚫 Abandoning task after **{attempts}** failed attempts: `{stable_key[:]}`")

    # ── Silent full-task restore on failure ──────────────────────────────
    # Restore only when task_snapshot is non-empty, meaning the coder actually
    # wrote files during this task that may now be in a broken state.
    # Fresh tasks have task_snapshot cleared by the orchestrator on dispatch,
    # so an architect failure on a brand-new task never triggers a restore.
    restored_ws = None  # set when a failure restore reverts files and workspace_files must be re-synced
    restored_exports = None  # set when a failure restore must prune the confirmed-exports registry
    if has_failed:
        store = state.get("context_store", {})
        task_snap = store.get("task_snapshot", {})
        if task_snap:
            sandbox_dir = state.get("project_path", ".")
            r, d = restore_snapshot(sandbox_dir, task_snap)
            if r or d:
                _w(f"♻️ Snapshot restored: **{r}** file(s) reverted, **{d}** removed.")
                # Steps that passed before the failure had the validator refresh
                restored_ws = list_workspace_files(sandbox_dir)
                # Those passed steps also committed confirmed exports for files now
                # reverted. Prune every path the task touched from the registry so a
                # rolled-back file never leaves stale exports behind (worst case: a
                # missing entry, never a wrong one).
                reverted_paths = {p.replace("\\", "/") for p in task_snap}
                restored_exports = {
                    k: v for k, v in store.get("module_exports", {}).items()
                    if k.replace("\\", "/") not in reverted_paths
                }
                state = {**state, "context_store": {
                    **store,
                    "workspace_files": restored_ws,
                    "module_exports": restored_exports,
                }}

    # ── Step 1: Record outcome of the previous task ──────────────────────
    if current_plan and current_plan.instruction_for_agent:
        if has_failed:
            _w(f"❌ Task failed — surfacing to LLM:\n\n> {latest_report[:]}")
            if just_abandoned:
                current_plan.completed_tasks = list(current_plan.completed_tasks) + [
                    f"[ABANDONED after {MAX_TASK_ATTEMPTS} attempts] "
                    f"[{current_plan.next_agent}] {just_abandoned}"
                ]
        else:
            # Task succeeded (or was a no-op '[COMPLETE]') — clear its failure count.
            success_key = active_original or current_plan.instruction_for_agent
            failure_counts.pop(success_key, None)
            current_plan.completed_tasks = list(current_plan.completed_tasks) + [
                f"[{current_plan.next_agent}] {current_plan.instruction_for_agent}"
            ]

    # ── Step 2: Dispatch next queued task (skip LLM call) ────────────────
    if current_plan and current_plan.todo_list and not has_failed:
        todo = list(current_plan.todo_list)
        next_task = todo.pop(0)
        _w(f"▶️ Queue → **{next_task.agent}:** {next_task.instruction}")
        return {
            "plan": TaskPlan(
                thinking_process="Dispatching next queued task.",
                todo_list=todo,
                completed_tasks=list(current_plan.completed_tasks),
                next_agent=next_task.agent,
                instruction_for_agent=next_task.instruction,
            ),
            # Clear stale snapshot so an architect failure on this new task cannot
            # accidentally roll back files written by the previous task. Also reset
            # the reconsideration state so the new task starts with a clean slate.
            "context_store": {
                "task_snapshot": {},
                "task_failure_counts": failure_counts,
                "abandoned_tasks": abandoned_tasks,
                "task_retry_feedback": "",
                "active_task_original": next_task.instruction,
                "task_reconsideration_count": 0,
            },
            "iteration_count": 1,
            "coder_retries": 0,
            "architect_replans": 0,
        }

    # ── Step 2b: Reconsider or mechanically retry a failed task ─────────
    # The queue is preserved in both cases — a single task failure must never
    # silently wipe unrelated queued work. Two sub-paths:
    #
    #   a) Within reconsideration budget: run a focused LLM call that produces a
    #      short replacement sub-plan (1–3 tasks). Those tasks are prepended to the
    #      preserved queue, giving the architect a genuinely new angle rather than
    #      re-attempting the exact same instruction.
    #
    #   b) Budget exhausted (or reconsideration LLM failed): mechanical re-dispatch
    #      of the same instruction with the failure report as extra context. The
    #      architect's task_retry_feedback channel surfaces it.
    #
    # Only when just_abandoned (failure_counts hit MAX_TASK_ATTEMPTS) do we fall
    # through to Step 3, which discards the queue and re-evaluates from scratch.
    if has_failed and current_plan and current_plan.instruction_for_agent and not just_abandoned:
        stable_key = active_original or current_plan.instruction_for_agent
        attempt = failure_counts.get(stable_key, 0)

        if reconsideration_count < TASK_RECONSIDERATION_BUDGET:
            _w(
                f"🔄 Reconsideration **{reconsideration_count + 1}/{TASK_RECONSIDERATION_BUDGET}** "
                f"for task (attempt **{attempt}/{MAX_TASK_ATTEMPTS}**), "
                f"queue preserved ({len(current_plan.todo_list)} task(s))."
            )
            recon_out = _run_reconsideration(state, current_plan, latest_report, attempt)
            if recon_out is not None:
                # Prepend replacement tasks; preserved queue follows at the back.
                new_todo = list(recon_out.todo_list) + list(current_plan.todo_list)
                first = new_todo.pop(0)
                recon_ctx = {
                    "task_snapshot": {},
                    "task_failure_counts": failure_counts,
                    "abandoned_tasks": abandoned_tasks,
                    "task_retry_feedback": "",
                    "active_task_original": stable_key,
                    "task_reconsideration_count": reconsideration_count + 1,
                }
                if restored_ws is not None:
                    recon_ctx["workspace_files"] = restored_ws
                if restored_exports is not None:
                    recon_ctx["module_exports"] = restored_exports
                return {
                    "plan": TaskPlan(
                        thinking_process=(
                            f"Reconsideration {reconsideration_count + 1}/{TASK_RECONSIDERATION_BUDGET}: "
                            f"replacement sub-plan prepended, queue preserved."
                        ),
                        todo_list=new_todo,
                        completed_tasks=list(current_plan.completed_tasks),
                        next_agent=first.agent,
                        instruction_for_agent=first.instruction,
                    ),
                    "context_store": recon_ctx,
                    "iteration_count": 1,
                    "coder_retries": 0,
                    "architect_replans": 0,
                }
            # Reconsideration LLM failed — fall through to mechanical retry below.

        # Budget exhausted or reconsideration LLM unavailable: mechanical re-dispatch.
        _w(
            f"🔁 Mechanical retry (attempt **{attempt + 1}/{MAX_TASK_ATTEMPTS}**), "
            f"queue preserved ({len(current_plan.todo_list)} task(s) still queued)."
        )
        retry_ctx = {
            "task_snapshot": {},
            "task_failure_counts": failure_counts,
            "abandoned_tasks": abandoned_tasks,
            "task_retry_feedback": (
                f"This task has already failed {attempt} time(s). Most recent failure report:\n"
                f"{latest_report}\n"
                f"Produce a genuinely different plan this attempt — do not repeat the approach that failed."
            ),
            "active_task_original": stable_key,
            "task_reconsideration_count": reconsideration_count,
        }
        if restored_ws is not None:
            retry_ctx["workspace_files"] = restored_ws
        if restored_exports is not None:
            retry_ctx["module_exports"] = restored_exports
        return {
            "plan": TaskPlan(
                thinking_process=f"Mechanical retry (attempt {attempt + 1}/{MAX_TASK_ATTEMPTS}), queue preserved.",
                todo_list=list(current_plan.todo_list),
                completed_tasks=list(current_plan.completed_tasks),
                next_agent=current_plan.next_agent,
                instruction_for_agent=current_plan.instruction_for_agent,
            ),
            "context_store": retry_ctx,
            "iteration_count": 1,
            "coder_retries": 0,
            "architect_replans": 0,
        }

    # ── Review-first: evidence-based first plan on a resumed/existing project ──
    # First round (current_plan is None) AND the sandbox already has files →
    # dispatch a review BEFORE planning so the first plan reacts to real gaps
    # instead of being committed blind. Greenfield (empty sandbox) skips this and
    # plans immediately. The review returns to the orchestrator via the normal
    # inspector routing; the next round's Step 3 plans with review_report in context.
    # reviewed_at_start guards re-entry (current_plan is None is itself once-only,
    # since no path resets plan to None — the flag is belt-and-suspenders).
    _rf_store = state.get("context_store", {})
    if (
        current_plan is None
        and _rf_store.get("workspace_files")
        and not _rf_store.get("reviewed_at_start", False)
    ):
        _w("🔍 Review-first: existing workspace detected — reviewing before planning.")
        review_instruction = (
            f"{REVIEW_TASK_MARKER} <goal>Evaluate project completeness: compare actual workspace "
            "against canonical file tree, check all file contents and imports against the "
            "specification, report gaps or confirm completion."
        )
        return {
            "plan": TaskPlan(
                thinking_process="Review-first: inspecting existing workspace before planning.",
                todo_list=[],
                completed_tasks=[],
                next_agent="inspector",
                instruction_for_agent=review_instruction,
            ),
            "context_store": {"reviewed_at_start": True},
            "iteration_count": 1,
            "coder_retries": 0,
            "architect_replans": 0,
        }

    # ── Step 3: LLM planning (queue empty, first run, or abandonment) ────
    _w("🧠 Running LLM planner...")

    # Get config for Orchestrator
    orch = _cfg["agents"]["orchestrator"]

    # If has_failed = true, inject the recovery prompt instead of the default one
    if has_failed:
        _w("💉 Task failed — injecting recovery prompt.")
        prompt = _cfg["prompts"]["orchestrator_recovery"]
    else:
        prompt = _cfg["prompts"]["orchestrator"]

    llm = make_llm(orch)

    structured_llm = llm.with_structured_output(PlannerOutput, include_raw=True)
    existing_completed = current_plan.completed_tasks if current_plan else []

    # Dynamic context for the prompt's $variables. Lists are joined into lines so
    # they don't render as Python list reprs; everything else is passed as-is.
    store = state.get("context_store", {})
    context_vars = {
        "spec_content": store.get("spec_content", ""),
        "inspector_raw": store.get("inspector_raw", ""),
        "inspector_files": "\n".join(f"  {f}" for f in (store.get("inspector_files", []) or [])),
        "inspector_issues": "\n".join(f"  - {i}" for i in (store.get("inspector_issues", []) or [])),
        "verification_gaps": store.get("verification_gaps", ""),
        "validation_issues": store.get("validation_issues", ""),
        "workspace_files": "\n".join(store.get("workspace_files", []) or []),
        "file_tree": "\n".join(f"  {p}" for p in (store.get("file_tree", []) or [])),
        "review_report": store.get("review_report", ""),
        "missing_files": "\n".join(store.get("missing_files", []) or []),
        "new_files": "\n".join(store.get("new_files", []) or []),
    }

    failure_context = ""
    if has_failed and current_plan and current_plan.instruction_for_agent:
        remaining = "\n".join(
            f"  {i+1}. [{t.agent}] {t.instruction}"
            for i, t in enumerate(current_plan.todo_list)
        ) or "  (none)"
        failure_context = (
            f"\nFAILING TASK: [{current_plan.next_agent}] {current_plan.instruction_for_agent}"
            f"\n\nREMAINING PLANNED TASKS (queued before the failure — keep, modify, or discard):\n{remaining}\n"
        )
        if just_abandoned:
            failure_context += (
                f"\nABANDONED: the task above has now failed {MAX_TASK_ATTEMPTS} times and is "
                f"PERMANENTLY ABANDONED. Do NOT re-issue this instruction or a paraphrase of it. "
                f"Either find a genuinely different approach to the underlying objective or move on "
                f"to other work / complete the objective without it.\n"
            )
        elif abandoned_tasks:
            abandoned_list = "\n".join(f"  - {t}" for t in abandoned_tasks)
            failure_context += (
                f"\nPREVIOUSLY ABANDONED TASKS (do NOT re-issue these or paraphrases of them):\n{abandoned_list}\n"
            )

    # only load max. last 10 completed Task to reduce context
    COMPLETED_PROMPT_CAP = 10
    if len(existing_completed) > COMPLETED_PROMPT_CAP:
        completed_display = [
                                f"({len(existing_completed) - COMPLETED_PROMPT_CAP} earlier tasks omitted)"
                            ] + existing_completed[-COMPLETED_PROMPT_CAP:]
    else:
        completed_display = existing_completed

    # The selected prompt (orchestrator / orchestrator_recovery) is itself a
    # template: it carries the $variable annotations for the dynamic context, so
    # we substitute them in and send the whole thing as one message (scaffold-style).
    full_prompt = Template(prompt).safe_substitute(
        objective=state["objective"],
        completed_tasks=completed_display,
        failure_context=failure_context,
        latest_report=latest_report,
        **context_vars,
    )
    messages = [HumanMessage(content=full_prompt)]

    try:
        result = structured_llm.invoke(messages)
    except Exception as e:
        # Try one more time with stripped-down prompt
        if isinstance(e, LengthFinishReasonError):
            _w("⚠️ Token limit hit — retrying with minimal context...")
            minimal_prompt = Template(prompt).safe_substitute(
                objective=state["objective"],
                completed_tasks=f"{len(existing_completed)} tasks done. Last 3: {existing_completed[-3:]}",
                failure_context=failure_context,
                latest_report=latest_report[:400],
                **{k: "" for k in context_vars},
            )
            minimal_messages = [HumanMessage(content=minimal_prompt)]
            result = structured_llm.invoke(minimal_messages)  # still raises if this fails too
        else:
            _w(f"❌ LLM call failed: `{e}`")
            raise

    if result.get("raw"):
        # record_tokens both accumulates the run total and pushes this call's usage
        # onto the custom stream so the UI's live stats meters update mid-run.
        stats.record_tokens(result["raw"])

    planner_out = result["parsed"]

    if planner_out is None and result["raw"]:
        raw_text = result["raw"].content if hasattr(result["raw"], "content") else str(result["raw"])
        _w("⚠️ Schema parse failed — trying fallback parser...")
        planner_out = _parse_plan_fallback(raw_text)

    if planner_out is None:
        _w("⚠️ LLM returned empty response — forcing completion.")
        return {
            "plan": TaskPlan(
                thinking_process="LLM returned no plan.",
                completed_tasks=existing_completed,
                next_agent="complete",
                instruction_for_agent="",
            ),
            "context_store": {
                "skip_verification": True,
                "task_failure_counts": failure_counts,
                "abandoned_tasks": abandoned_tasks,
                **({"module_exports": restored_exports} if restored_exports is not None else {}),
            },
            "iteration_count": 1,
            "coder_retries": 0,
            "architect_replans": 0,
        }

    # ── Ground the plan in the canonical tree ────────────────────────────
    # Drop tasks that name only off-tree files (hallucinations). If that empties
    # the plan, give the planner ONE chance to re-plan against the tree before
    # giving up — mirrors the architect's single-shot correction pass.
    file_tree = state.get("context_store", {}).get("file_tree", [])
    if file_tree:
        kept, dropped = _sanitize_todo(planner_out.todo_list, file_tree)
        for t, off in dropped:
            _w(f"🌲 Dropped off-tree task (`{off}`): {t.instruction[:]}")
        if dropped and not kept:
            _w("🌲 All tasks were off-tree — requesting one grounded re-plan...")
            tree_text = "\n".join(f"  {p}" for p in file_tree)
            off_all = sorted({f for _, off in dropped for f in off})
            correction_prompt = Template(prompt).safe_substitute(
                objective=state["objective"],
                completed_tasks=completed_display,
                failure_context=failure_context,
                latest_report=latest_report,
                **context_vars,
            )
            correction = [
                HumanMessage(content=correction_prompt),
                HumanMessage(content=(
                    f"OBJECTIVE: {state['objective']}\n\n"
                    f"Your previous plan referenced files that do NOT exist in the canonical "
                    f"file tree and were rejected: {off_all}\n\n"
                    f"CANONICAL FILE TREE — every file you name MUST be one of these exact paths:\n"
                    f"{tree_text}\n\n"
                    f"Re-plan using ONLY files from the tree above. Do not invent files, "
                    f"features, or concepts the tree does not contain."
                )),
            ]
            try:
                fix = structured_llm.invoke(correction)
                if fix.get("raw"):
                    stats.record_tokens(fix["raw"])
                fixed_out = fix["parsed"]
                if fixed_out is None and fix.get("raw"):
                    raw_text = fix["raw"].content if hasattr(fix["raw"], "content") else str(fix["raw"])
                    fixed_out = _parse_plan_fallback(raw_text)
                if fixed_out is not None:
                    kept, _ = _sanitize_todo(fixed_out.todo_list, file_tree)
                    planner_out = fixed_out
            except Exception as e:
                _w(f"❌ Grounded re-plan failed: `{e}`")
        planner_out = PlannerOutput(thinking_process=planner_out.thinking_process, todo_list=kept)

    # Drop any (re)planned task whose instruction exactly matches one we already
    # abandoned, so the planner cannot resurrect the task we just gave up on.
    if abandoned_tasks:
        before = len(planner_out.todo_list)
        kept_tasks = [t for t in planner_out.todo_list if t.instruction not in abandoned_tasks]
        if len(kept_tasks) != before:
            _w(f"🚫 Dropped **{before - len(kept_tasks)}** re-planned task(s) matching abandoned instructions.")
        planner_out = PlannerOutput(thinking_process=planner_out.thinking_process, todo_list=kept_tasks)

    # ── Force-append a review inspector task at the end of every non-empty plan ─
    # This ensures the orchestrator always gets a rich completeness report
    # (missing files, import gaps, unimplemented spec features) before it decides
    # to end the workflow — rather than declaring complete based on its own judgment
    # alone. The inspector detects the REVIEW_TASK_MARKER prefix and switches into
    # review mode instead of the normal explore mode.
    if planner_out.todo_list:
        last = planner_out.todo_list[-1]
        if not last.instruction.startswith(REVIEW_TASK_MARKER):
            review_task = SubTask(
                agent="inspector",
                instruction=(
                    f"{REVIEW_TASK_MARKER} Evaluate project completeness: compare actual workspace "
                    "against canonical file tree, check all file contents and imports against the "
                    "specification, report gaps or confirm completion."
                ),
            )
            planner_out = PlannerOutput(
                thinking_process=planner_out.thinking_process,
                todo_list=list(planner_out.todo_list) + [review_task],
            )

    # Derive TaskPlan from PlannerOutput — code owns next_agent, instruction, completed_tasks
    todo = list(planner_out.todo_list)
    if todo:
        first = todo.pop(0)
        next_agent = first.agent
        instruction_for_agent = first.instruction
    else:
        next_agent = "complete"
        instruction_for_agent = ""

    new_plan = TaskPlan(
        thinking_process=planner_out.thinking_process,
        todo_list=todo,
        completed_tasks=list(existing_completed),
        next_agent=next_agent,
        instruction_for_agent=instruction_for_agent,
    )

    # print a final plan summary (next plan + queue)
    remaining_md = "\n".join(
        f"  {i + 1}. `{task.agent}` — {task.instruction[:100]}"
        for i, task in enumerate(new_plan.todo_list)
    )
    _w(f"### 📋 Plan\n\n**Next:** `{new_plan.next_agent}` —")
    _w_scrollable(new_plan.instruction_for_agent)
    _w(
        (f"**Queue:**\n{remaining_md}" if remaining_md else "_No remaining tasks._")
    )

    return_context = {
        "task_snapshot": {},
        "task_failure_counts": failure_counts,
        "abandoned_tasks": abandoned_tasks,
        "task_retry_feedback": "",
        "active_task_original": instruction_for_agent,
        "task_reconsideration_count": 0,
    }
    if restored_ws is not None:
        return_context["workspace_files"] = restored_ws
    if restored_exports is not None:
        return_context["module_exports"] = restored_exports
    # Opt-in: once the tree already exists, a fresh full plan may need new files.
    # Ask the scaffold node to re-run in diff mode. Guarded on `scaffolded` so the
    # very first plan (scaffold hasn't run yet) doesn't set this — the router still
    # diverts to scaffold then via its not-yet-scaffolded branch.
    if SCAFFOLD_ON_REPLAN and state.get("context_store", {}).get("scaffolded"):
        return_context["rescaffold_requested"] = True
    return {
        "plan": new_plan,
        "iteration_count": 1,
        "coder_retries": 0,
        "context_store": return_context,
        "architect_replans": 0,
    }


# ── BaseAgent-compatible wrapper ─────────────────────────────────────────────
class Orchestrator:
    """Thin object that exposes the compiled orchestrator graph the same way
    `agents.base.BaseAgent` exposes a `create_agent` graph, WITHOUT subclassing it.

    BaseAgent's contract (`tools` + `render_prompt`, one react-agent per model) does
    not fit a multi-node planner whose per-node LLMs are configured from
    `graph_config.yaml`. So this mirrors only the public surface the UI wiring reads
    (`name`, `requires_project`, `kickoff_message`, `tool_names`, `pool`, `default`,
    `get`, `stream`, `invoke`) — exactly the slice `CodeReviewOrchestrator` already
    mimics today. It is intended to eventually replace that class in the registry; it
    is deliberately left unwired for now.
    """

    name = "code-review-graph"
    requires_project = True
    kickoff_message = (
        "Drive the orchestrator graph (scaffold -> orchestrator -> "
        "inspector/architect -> coder -> validator) to completion on this task."
    )

    def __init__(
        self,
        model_configs: dict[str, dict],
        *,
        checkpointer: Any = None,
        recursion_limit: int = 1000,
        default_model: str | None = None,
    ) -> None:
        if not model_configs:
            raise ValueError("Orchestrator needs at least one model config")

        # Lazy import avoids a circular import: this module is imported BY the
        # package __init__ (where get_app lives), so we only reach in for get_app
        # at instantiation time, after the package has finished importing.
        from agents.implementations.code_agent import get_app

        self._model_configs = model_configs
        self._recursion_limit = recursion_limit

        # The graph's per-node LLMs come from graph_config.yaml
        app = get_app(checkpointer=checkpointer)
        self._pool: dict[str, Any] = {name: app for name in model_configs}
        self._default_name = default_model or next(iter(self._pool))
        if self._default_name not in self._pool:
            raise KeyError(
                f"default_model {self._default_name!r} is not one of {list(self._pool)}"
            )

    # ── BaseAgent-compatible surface used by the wiring/server ───────────────
    @property
    def tool_names(self) -> list[str]:
        # The graph exposes no single tool set: each node owns its own tools.
        return []

    @property
    def pool(self) -> dict[str, Any]:
        return self._pool

    @property
    def default(self) -> Any:
        return self._pool[self._default_name]

    def get(self, model_name: str | None, default: Any = None) -> Any:
        if model_name is None:
            return self.default
        return self._pool.get(model_name, default if default is not None else self.default)

    def _ensure_config(self, config: dict | None) -> dict:
        cfg = dict(config or {})
        cfg.setdefault("configurable",{})
        cfg["configurable"].setdefault("thread_id", str(uuid.uuid4()))
        return cfg

    def stream(self, payload: dict, *, model: str | None = None,
               config: dict | None = None, stream_mode: Any = "messages"):
        yield from self.get(model).stream(payload, config=self._ensure_config(config), stream_mode=stream_mode)

    def invoke(self, payload, *, model=None, config=None):
        return self.get(model).invoke(payload, config=self._ensure_config(config))