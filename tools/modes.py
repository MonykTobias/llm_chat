"""State-switch tools: active language + review-pipeline stage handoffs.

These tools don't do filesystem or network work — they write a flag into the
graph state and let the UI server act on it after the turn finishes. `set_language`
switches the language used by the linter/test/type-check tools and the prompt;
the `change_mode_*` tools request a handoff to the next stage of the four-stage
review pipeline (explore → plan → act → verify).
"""
from __future__ import annotations

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from structured_output import LANGUAGES, ReviewState


@tool
def set_language(language: str, *,
                 tool_call_id: Annotated[str, InjectedToolCallId],
                 state: Annotated[ReviewState, InjectedState]) -> Command:
    """Switch the active project language used by the linter, test runner, type
    checker and the system prompt. Use when you need to inspect or test part of
    the project written in a different language than the current one.
    Supported: english, python, javascript, typescript, go, rust, java."""
    new = (language or "").strip().lower()
    old = state.get("language", "unknown")
    print(f"Changing language from {old} to {new}")
    if new not in LANGUAGES:
        return Command(update={"messages": [ToolMessage(
            f"Unknown language '{language}'. Supported: {', '.join(LANGUAGES)}.",
            tool_call_id=tool_call_id)]})
    return Command(update={
        "language": new,
        "messages": [ToolMessage(f"Language changed from {old} to {new}.",
                                 tool_call_id=tool_call_id)],
    })


# Pipeline stages the change_mode_* tools can switch between. These are the
# config.yaml `prompt:` keys / AGENTS roles for the four-stage review pipeline.
REVIEW_MODES = {"cr_explore", "cr_plan", "cr_act", "cr_verify"}


def _switch_mode(target: str, tool_call_id: str, state: ReviewState) -> Command:
    """Flag `target` as the stage the server should run next, via graph state.

    Mirrors `set_language`: the tool writes into the graph state and the UI
    server reads `review_mode` back after the turn, advances the session's role,
    and auto-runs the next stage's kickoff. The switch takes effect only once the
    current turn has fully finished.
    """
    old = state.get("review_mode") or state.get("role") or "current stage"
    print(f"Requesting mode switch from {old} to {target}")
    return Command(update={
        "review_mode": target,
        "messages": [ToolMessage(
            f"Handoff requested: the {target} stage will run next, after this "
            f"turn completes.", tool_call_id=tool_call_id)],
    })


@tool(return_direct=True)
def change_mode_explore(tool_call_id: Annotated[str, InjectedToolCallId],
                        state: Annotated[ReviewState, InjectedState]) -> Command:
    """Hand off to the EXPLORE stage. Call this when the task needs to go back to
    investigating the codebase. The switch takes effect after your current turn
    ends; the explore stage then starts automatically."""
    return _switch_mode("cr_explore", tool_call_id, state)


@tool(return_direct=True)
def change_mode_plan(tool_call_id: Annotated[str, InjectedToolCallId],
                     state: Annotated[ReviewState, InjectedState]) -> Command:
    """Hand off to the PLAN stage. Call this once you have finished the current
    stage's job and are ready for planning to begin. The switch takes effect
    after your current turn ends; the plan stage then starts automatically."""
    return _switch_mode("cr_plan", tool_call_id, state)


@tool(return_direct=True)
def change_mode_act(tool_call_id: Annotated[str, InjectedToolCallId],
                    state: Annotated[ReviewState, InjectedState]) -> Command:
    """Hand off to the ACT stage. Call this once the plan is ready to be executed,
    or to send failed work back for fixing. The switch takes effect after your
    current turn ends; the act stage then starts automatically."""
    return _switch_mode("cr_act", tool_call_id, state)


@tool(return_direct=True)
def change_mode_verify(tool_call_id: Annotated[str, InjectedToolCallId],
                       state: Annotated[ReviewState, InjectedState]) -> Command:
    """Hand off to the VERIFY stage. Call this once the implementation is complete
    and ready to be checked. The switch takes effect after your current turn ends;
    the verify stage then starts automatically."""
    return _switch_mode("cr_verify", tool_call_id, state)
