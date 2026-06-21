"""
LangGraph workflow definition for the orchestrator graph.

Flow:
                          INTAKE
                            |
                      ORCHESTRATOR
                     /     |      \\  \\
              INSPECTOR ARCHITECT  SCAFFOLD (complete)
                  |        |          |        |
              ORCHESTRATOR CODER  (next_agent) INSPECTOR (verify)
                              |    architect/    /     \\
                           VALIDATOR inspector  END  ORCHESTRATOR
                           /     \\
               STEP_DISPATCH   ORCHESTRATOR
                    |
                  CODER  (next step)

Intake (entry point):
  - Turns the incoming chat turn into the planner `objective` and seeds
    `language`. This used to live inside the scaffold node; it was split out so
    the orchestrator (which reads `state["objective"]`) can run *before* scaffold.

Scaffold (now AFTER the orchestrator's first real plan, not before it):
  - The orchestrator diverts to scaffold exactly once — the first time it emits a
    plan whose next_agent is a real work agent — so the canonical file tree is
    built with knowledge of the plan instead of blind. Scaffold then routes on to
    the agent the plan asked for (architect / inspector).
  - It runs once by default. If `scaffold_on_replan` is set in graph_config.yaml,
    a full orchestrator re-plan (task abandoned + re-planned) sets
    context_store["rescaffold_requested"], diverting through scaffold again in
    diff mode (it never relocates existing files; it only adds what the new plan
    needs).

Inspector has two modes:
  - Explore mode: read-only investigation, returns to orchestrator.
  - Verify mode: completion gate (triggered when next_agent == 'complete').
    PASS → END, gaps → orchestrator (plan fixes).

Two-tier retry on Validator failure:
  - Retries 1-(threshold-1): route back to Coder
  - Retries threshold+:      route back to Architect (re-plan current step only)
  - Retry max+:              escalate to Orchestrator (full task cleanup + replan)

Validator PASS routing:
  - architect_step_queue non-empty → step_dispatch → coder (next step)
  - queue empty                   → orchestrator

Thresholds are configurable via graph_config.yaml:
  coder_max_retries and architect_retry_threshold.

Only the orchestrator node is a real port right now; inspector / architect /
coder / validator / scaffold / step_dispatch are dummy stubs (see their modules)
to be filled in later. This package is NOT wired into the frontend yet — build a
runnable with `get_app()` or via the `Orchestrator` wrapper in `orchestrator.py`.
"""
from pathlib import Path
from typing import Any, Literal

import yaml
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from agents.implementations.code_agent.structured_output import AgentState
from .orchestrator import orchestrator_node, Orchestrator
from .inspector import inspector_node
from .architect import architect_node
from .coder import coder_node
from .validator import validator_node
from .scaffold import scaffold_node, intake_node
from agents.implementations.code_agent.utils.step_dispatch import step_dispatch_node

__all__ = ["get_app", "Orchestrator"]

# Config lives next to this package, independent of the project-level config.yaml.
_CFG_PATH = Path(__file__).resolve().parent / "graph_config.yaml"
with open(_CFG_PATH, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

MAX_CODER_RETRIES = _cfg.get("coder_max_retries", 3)
ARCHITECT_RETRY_THRESHOLD = _cfg.get("architect_retry_threshold", 2)


def _needs_scaffold(state: AgentState) -> bool:
    """True when control should be diverted through the scaffold node.

    Two cases:
      * First run — scaffolding has never happened (the scaffold node stamps
        context_store["scaffolded"] = True on every one of its return paths,
        including the skip/empty no-ops, so this flips exactly once).
      * Re-plan — the orchestrator requested a rescaffold after a full re-plan
        (only happens when `scaffold_on_replan` is enabled; see orchestrator).
    """
    store = state.get("context_store", {}) or {}
    if not store.get("scaffolded"):
        return True
    return bool(store.get("rescaffold_requested"))


def _route_from_orchestrator(
    state: AgentState,
) -> Literal["scaffold", "inspector", "architect", "__end__"]:
    plan = state.get("plan")
    if not plan:
        return "__end__"
    if plan.next_agent == "complete":
        # Forced completions (iteration cap / empty LLM) bypass the verifier.
        if state.get("context_store", {}).get("skip_verification"):
            print("\n[Router] Objective complete (verification skipped).")
            return "__end__"
        return "inspector"
    # A real work agent is up next. The first time we reach this point (or when a
    # re-plan asks for it) we build/refresh the canonical tree first, THEN let the
    # scaffold node forward control to the planned agent.
    if _needs_scaffold(state):
        return "scaffold"
    return plan.next_agent


def _route_from_scaffold(state: AgentState) -> Literal["architect", "inspector", "__end__"]:
    """Hand control to whatever the orchestrator's plan asked for, now that the
    canonical tree exists. Mirrors the non-scaffold branches of the orchestrator
    router so scaffold is a transparent pass-through in the flow."""
    plan = state.get("plan")
    if not plan:
        return "__end__"
    if plan.next_agent == "complete":
        if state.get("context_store", {}).get("skip_verification"):
            return "__end__"
        return "inspector"
    return plan.next_agent


def _route_from_inspector(state: AgentState) -> Literal["orchestrator", "__end__"]:
    plan = state.get("plan")
    # Verify mode: the inspector was reached via a 'complete' decision.
    if plan and plan.next_agent == "complete":
        verdict = state.get("context_store", {}).get("verification_verdict", "pass")
        if verdict == "pass":
            return "__end__"
        return "orchestrator"
    # Explore mode: back to the orchestrator as before.
    return "orchestrator"


def _route_from_architect(state: AgentState) -> Literal["coder", "orchestrator"]:
    report = state.get("latest_report", "")
    if report.startswith("[FAILED]"):
        return "orchestrator"
    if report.startswith("[COMPLETE]"):
        # No-op: task already satisfied. Skip the coder and let the orchestrator
        # record it as done and plan the next task.
        return "orchestrator"
    return "coder"


def _retry_destination(state: AgentState, source: str) -> Literal["coder", "architect", "orchestrator"]:
    """Nested retry: coder gets coder_max_retries attempts per architect plan;
    architect gets architect_retry_threshold re-plan attempts before escalating."""
    coder_retries = state.get("coder_retries", 0)
    architect_replans = state.get("architect_replans", 0)

    if coder_retries >= MAX_CODER_RETRIES:
        if architect_replans >= ARCHITECT_RETRY_THRESHOLD:
            return "orchestrator"
        return "architect"
    return "coder"


def _route_from_validator(state: AgentState) -> Literal["orchestrator", "coder", "architect", "step_dispatch"]:
    if not state.get("latest_report", "").startswith("[FAILED]"):
        if state.get("architect_step_queue"):
            return "step_dispatch"
        return "orchestrator"
    return _retry_destination(state, "Validator")


def get_app(dashboard=None, checkpointer: Any = None):
    graph = StateGraph(AgentState)

    if checkpointer is None:
        checkpointer = MemorySaver()

    def node(fn, name):
        return dashboard.wrap(fn, name) if dashboard else fn

    graph.add_node("intake",        node(intake_node,        "intake"))
    graph.add_node("scaffold",      node(scaffold_node,      "scaffold"))
    graph.add_node("orchestrator",  node(orchestrator_node,  "orchestrator"))
    graph.add_node("inspector",     node(inspector_node,     "inspector"))
    graph.add_node("architect",     node(architect_node,     "architect"))
    graph.add_node("coder",         node(coder_node,         "coder"))
    graph.add_node("validator",     node(validator_node,     "validator"))
    graph.add_node("step_dispatch", node(step_dispatch_node, "step_dispatch"))

    # Intake turns the chat turn into an objective, THEN the orchestrator plans.
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "orchestrator")

    # Orchestrator diverts to scaffold the first time it dispatches real work
    # (and on re-plan when scaffold_on_replan is enabled); otherwise straight on.
    graph.add_conditional_edges("orchestrator", _route_from_orchestrator, {
        "scaffold":  "scaffold",
        "inspector": "inspector",
        "architect": "architect",
        "__end__":   END,
    })

    # Scaffold runs, commits the canonical tree, then forwards to the planned agent.
    graph.add_conditional_edges("scaffold", _route_from_scaffold, {
        "architect": "architect",
        "inspector": "inspector",
        "__end__":   END,
    })

    graph.add_conditional_edges("inspector", _route_from_inspector, {
        "orchestrator": "orchestrator",
        "__end__":      END,
    })

    graph.add_conditional_edges("architect", _route_from_architect, {
        "coder":        "coder",
        "orchestrator": "orchestrator",
    })

    graph.add_edge("coder", "validator")

    # step_dispatch always routes to coder (next step in queue)
    graph.add_edge("step_dispatch", "coder")

    # Validator: pass → step_dispatch (queue non-empty) | orchestrator | fail → coder/architect/orchestrator
    graph.add_conditional_edges("validator", _route_from_validator, {
        "orchestrator":  "orchestrator",
        "coder":         "coder",
        "architect":     "architect",
        "step_dispatch": "step_dispatch",
    })

    return graph.compile(checkpointer=checkpointer)