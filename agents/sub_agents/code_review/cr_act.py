from __future__ import annotations

from string import Template
from typing import Any, List

from agents.base import BaseAgent
from agents.llm_factory import make_system_prompt
from tools.tools import (
    list_all_files, read_file, delete_file, analyze_architecture, write_file,
    set_language, web_browse
)

DEFAULT_REVIEW_TOOLS = [
    read_file,
    list_all_files,
    analyze_architecture,
    write_file,
    delete_file,
    web_browse,
    set_language,
]

class CodeReviewAct(BaseAgent):
    """A conversational text assistant with no tools — pure chat mode."""

    kickoff_message = "Explore the codebase to get a better understanding of the project."

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

    def render_prompt(self, request, cfg: dict = {}) -> str:
        """System prompt rebuilt per run, with project language and path filled in."""
        language = request.state.get("language", "unknown")
        project_path = request.state.get("project_path", ".")

        base = Template(self._prompt_template).safe_substitute(
            language=language,
            project_path=project_path,
        )
        return make_system_prompt(base,cfg)