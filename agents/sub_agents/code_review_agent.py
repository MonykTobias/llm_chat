"""
The code-review agent: a senior-engineer assistant that reviews, implements,
compares, and explains code, grounded in tool output.

This is the reference implementation of `BaseAgent`. Its prompt is a
`string.Template` (filled per run with the project language and path) and its
tool set is the full filesystem/lint/test/type-check suite. Adding a different
agent type is just another `BaseAgent` subclass with its own prompt and tools.
"""
from __future__ import annotations

from string import Template
from typing import Any

from agents.base import BaseAgent
from agents.llm_factory import make_system_prompt
from tools.tools import (
    analyze_architecture,
    list_all_files,
    read_file,
    run_linter,
    run_tests,
    run_type_check,
    write_file,
)

# Default tools for a code review: read the tree, read files, inspect
# architecture, run the linter / type-checker / test suite, and write files
# when the user asks the agent to implement or fix something. Every write is
# snapshotted so it shows up in the UI with a one-click revert.
DEFAULT_REVIEW_TOOLS = [
    run_linter,
    run_tests,
    analyze_architecture,
    run_type_check,
    read_file,
    list_all_files,
    write_file,
]


class CodeReviewAgent(BaseAgent):
    """Reviews / implements / compares / explains code across a model pool."""

    kickoff_message = "Review this project."

    def __init__(
        self,
        model_configs: dict[str, dict],
        prompt_template: str,
        *,
        name: str | None = None,
        tools: list | None = None,
        checkpointer: Any = None,
        recursion_limit: int = 1000,
        default_model: str | None = None,
    ) -> None:
        # Stored before super().__init__ — the base reads `self.tools` while
        # building the pool.
        self._prompt_template = prompt_template
        self._tools = tools if tools is not None else list(DEFAULT_REVIEW_TOOLS)
        super().__init__(
            model_configs,
            name=name,
            checkpointer=checkpointer,
            recursion_limit=recursion_limit,
            default_model=default_model,
        )

    @property
    def tools(self) -> list:
        return self._tools

    def render_prompt(self, request, cfg: dict = {}) -> str:
        """System prompt rebuilt per run, with project language and path filled in."""
        language = request.state.get("language", "unknown")
        project_path = request.state.get("project_path", ".")
        base = Template(self._prompt_template).safe_substitute(
            language=language,
            project_path=project_path,
        )
        return make_system_prompt(base,cfg)