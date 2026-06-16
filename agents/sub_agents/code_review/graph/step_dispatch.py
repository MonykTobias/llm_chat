"""
Step-dispatch node — DUMMY placeholder.

Eventually: pops the next ArchitectStep off architect_step_queue and stages its
context (files to create/modify/delete, step plan) into context_store so the
coder works one step at a time. Always routes to the coder.

For now this is a no-op stub. It is only reached when the validator passes with a
non-empty step queue, which the dummy validator never produces — so in the
skeleton graph this node never actually runs. Replace with the real
implementation ported from the standalone `orchestrator` project's
`agents/step_dispatch.py`.
"""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from .structured_output import AgentState


def step_dispatch_node(state: AgentState, config: RunnableConfig) -> dict:
    # TODO: pop architect_step_queue[0] and stage the step's file lists + plan
    #       into context_store for the coder.
    print("[StepDispatch] (dummy) no-op.")
    return {"history": ["step_dispatch"]}
