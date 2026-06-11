from __future__ import annotations

from string import Template
from typing import Any, List

from agents.base import BaseAgent
from tools.tools import (
    list_all_files, read_file, analyze_architecture,
    run_type_check, run_tests, run_linter, build_change_report, web_browse,
)

DEFAULT_REVIEW_TOOLS = [
    read_file,
    list_all_files,
    analyze_architecture,
    run_linter,
    run_tests,
    run_type_check,
    build_change_report,
    web_browse,
]

class CodeReviewVerify(BaseAgent):
    """A conversational text assistant with no tools — pure chat mode."""

    kickoff_message = "Use your tools on changed files to verify the codebase is compiling and passing tests."

    def __init__(
            self,
            model_configs: dict[str,dict],
            prompt_template: str,
            *,
            name: str | None = None,
            tools: list | None = None,
            checkpointer: Any = None,
            recursion_limit: int = 1000,
            default_model: str | None = None,
    ) -> None:
        # building the pool
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
    def tools(self) -> List:  # type: ignore[return-type]
        return self._tools

    def render_prompt(self, request) -> str:
        """System prompt rebuilt per run, with project language and path filled in."""
        language = request.state.get("language", "unknown")
        project_path = request.state.get("project_path", ".")
        return Template(self._prompt_template).safe_substitute(
            language=language,
            project_path=project_path,
        )