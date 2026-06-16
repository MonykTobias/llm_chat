"""State-switch tools: active language
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