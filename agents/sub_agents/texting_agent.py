from __future__ import annotations

from string import Template
from typing import Any, List

from agents.base import BaseAgent

DEFAULT_REVIEW_TOOLS = []

class TextingAgent(BaseAgent):
    """A conversational text assistant with no tools — pure chat mode."""

    name = "texting_agent"
    kickoff_message = "Hi! I'm your friendly chat companion. Ask me anything or just say hello!"

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