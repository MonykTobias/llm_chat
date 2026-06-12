from __future__ import annotations

from string import Template
from typing import Any, List

from agents.base import BaseAgent
from agents.llm_factory import make_system_prompt
from tools.tools import web_browse

# Chat mode is folder-less, so it gets web search only — no file tools, which
# would need a project_path this role never has.
DEFAULT_REVIEW_TOOLS = [web_browse]

class TextingAgent(BaseAgent):
    """A conversational chat assistant — web search only, no project folder."""

    # Folder-less: the UI offers this role without a directory and keeps it
    # permanently separate from the code-oriented (project) roles.
    requires_project = False

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

    def render_prompt(self, request, cfg: dict = {}) -> str:
        """System prompt rebuilt per run, with project language and path filled in."""
        language = request.state.get("language", "unknown")
        project_path = request.state.get("project_path", ".")
        base = Template(self._prompt_template).safe_substitute(
            language=language,
            project_path=project_path,
        )
        return make_system_prompt(base,cfg)