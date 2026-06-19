from __future__ import annotations

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from structured_output import LANGUAGES


@tool
def set_language(language: str, *,
                 tool_call_id: Annotated[str, InjectedToolCallId],
                 state: Annotated[dict, InjectedState]) -> Command | str:
    """Switch the active project language used by the linter, test runner, type
    checker and the system prompt. Use when you need to inspect or test part of
    the project written in a different language than the current one.
    Supported: english, python, javascript, typescript, go, rust, java."""
    print(f"Setting language to {language}")
    new = (language or "").strip().lower()
    old = state.get("language", "unknown")

    if new not in LANGUAGES:
        msg = f"Unknown language '{language}'. Supported: {', '.join(LANGUAGES)}."
        # In a manual loop, state is a plain mutable dict — just return the string.
        # In the graph executor, return a Command so the state update propagates.
        if isinstance(state, dict):
            return msg
        return Command(update={"messages": [ToolMessage(
            content=msg, tool_call_id=tool_call_id,
        )]})

    # Mutate the dict in-place for the manual loop (validator/inspector/architect/coder)
    # so subsequent tools in the same loop see the updated language immediately.
    if isinstance(state, dict):
        state["language"] = new
        return f"Language changed from {old} to {new}."

    # Inside the real graph executor — use Command to update graph state properly.
    return Command(update={
        "language": new,
        "messages": [ToolMessage(
            content=f"Language changed from {old} to {new}.",
            tool_call_id=tool_call_id,
        )],
    })