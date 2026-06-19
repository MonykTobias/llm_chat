"""
LangGraph workflow definition for the orchestrator graph.

Flow:
                      ORCHESTRATOR
                     /     |       \\
              INSPECTOR  ARCHITECT  (complete)
                  |         |          |
              ORCHESTRATOR CODER    INSPECTOR (verify)
                              |      /     \\
                           VALIDATOR  END  ORCHESTRATOR
                           /     \\
               STEP_DISPATCH   ORCHESTRATOR
                    |
                  CODER  (next step)

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

from agents.sub_agents.code_review.graph.utils.structured_output import AgentState
from .orchestrator import orchestrator_node, Orchestrator
from .inspector import inspector_node
from .architect import architect_node
from .coder import coder_node
from .validator import validator_node
from .scaffold import scaffold_node
from agents.sub_agents.code_review.graph.utils.step_dispatch import step_dispatch_node

__all__ = ["get_app", "Orchestrator"]

# Config lives next to this package, independent of the project-level config.yaml.
_CFG_PATH = Path(__file__).resolve().parent / "graph_config.yaml"
with open(_CFG_PATH, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

MAX_CODER_RETRIES = _cfg.get("coder_max_retries", 3)
ARCHITECT_RETRY_THRESHOLD = _cfg.get("architect_retry_threshold", 2)


def _route_from_orchestrator(state: AgentState) -> Literal["inspector", "architect", "__end__"]:
    plan = state.get("plan")
    if not plan:
        return "__end__"
    if plan.next_agent == "complete":
        # Forced completions (iteration cap / empty LLM) bypass the verifier.
        if state.get("context_store", {}).get("skip_verification"):
            print("\n[Router] Objective complete (verification skipped).")
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

    graph.add_node("scaffold",      node(scaffold_node,      "scaffold"))
    graph.add_node("orchestrator",  node(orchestrator_node,  "orchestrator"))
    graph.add_node("inspector",     node(inspector_node,     "inspector"))
    graph.add_node("architect",     node(architect_node,     "architect"))
    graph.add_node("coder",         node(coder_node,         "coder"))
    graph.add_node("validator",     node(validator_node,     "validator"))
    graph.add_node("step_dispatch", node(step_dispatch_node, "step_dispatch"))

    # Scaffold runs exactly once (nothing routes back to it), then hands off.
    graph.add_edge(START, "scaffold")
    graph.add_edge("scaffold", "orchestrator")

    graph.add_conditional_edges("orchestrator", _route_from_orchestrator, {
        "inspector": "inspector",
        "architect": "architect",
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
