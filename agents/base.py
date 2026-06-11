"""
Abstract base for every agent in this project.

An "agent" here is one *kind* of assistant (a code reviewer, a security
auditor, a doc writer, ...) defined by two things:

  * its **prompt**  — implemented by `render_prompt`
  * its **tools**   — exposed via the `tools` property

Everything else is shared and lives here: building one LangGraph agent per
model in `config.yaml`, wiring the dynamic system prompt, sharing the SQLite
checkpointer, and the invoke / stream / run_session helpers.

Each agent instance owns a *pool* of compiled graphs — one per model config —
so the caller can switch models on the go (`agent.get(model_name)`), exactly
like the original single-purpose implementation did, but now per agent type.

To add a new agent type, subclass `BaseAgent` and implement `tools` and
`render_prompt`. See `agents.code_review_agent.CodeReviewAgent` for an example.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, wrap_model_call

from agents.llm_factory import make_llm
from structured_output import ReviewState


class BaseAgent(ABC):
    """A pool of model-variants for one kind of agent.

    Subclasses MUST implement `tools` and `render_prompt`. They MAY override
    `state_schema`, `kickoff_message`, and `name`.
    """

    #: Short identifier for the agent type (used for default thread ids, etc.).
    name: str = "agent"

    #: LangGraph state schema shared by every run of this agent.
    state_schema: type = ReviewState

    #: First user message sent by `run_session` to kick a session off.
    kickoff_message: str = "Hello."

    def __init__(
        self,
        model_configs: dict[str, dict],
        *,
        name: str | None = None,
        checkpointer: Any = None,
        recursion_limit: int = 1000,
        default_model: str | None = None,
    ) -> None:
        if not model_configs:
            raise ValueError(f"{type(self).__name__} needs at least one model config")

        # Instance name overrides the class default — lets several instances of
        # the same class act as distinct roles (each gets its own thread space).
        if name is not None:
            self.name = name

        self._recursion_limit = recursion_limit
        self._checkpointer = checkpointer

        # Middleware shared across the pool:
        #  * dynamic prompt — rebuilt per run from `render_prompt`.
        #  * tool gate — advertises only the per-turn `enabled_tools` subset to
        #    the model, so a deselected tool can't be called (all tools stay
        #    registered, so the tool node can still execute the allowed ones).
        prompt_middleware = self._build_prompt_middleware()
        tool_gate_middleware = self._build_tool_gate_middleware()

        self._pool: dict[str, Any] = {
            model_name: create_agent(
                model=make_llm(model_cfg),
                tools=self.tools,
                middleware=[prompt_middleware, tool_gate_middleware],
                state_schema=self.state_schema,
                checkpointer=checkpointer,
            )
            for model_name, model_cfg in model_configs.items()
        }

        self._default_name = default_model or next(iter(self._pool))
        if self._default_name not in self._pool:
            raise KeyError(
                f"default_model {self._default_name!r} is not one of "
                f"{list(self._pool)}"
            )

    # ── subclass contract ───────────────────────────────────────────────────
    @property
    @abstractmethod
    def tools(self) -> list:
        """Tools this agent may call. Must be ready before `__init__` runs."""

    @abstractmethod
    def render_prompt(self, request) -> str:
        """Build the system prompt for one run from `request.state`."""

    @property
    def tool_names(self) -> list[str]:
        """Names of every tool this agent can run — the toggle set for the UI."""
        return [t.name for t in self.tools]

    # ── pool access ─────────────────────────────────────────────────────────
    @property
    def pool(self) -> dict[str, Any]:
        """The {model_name: compiled_graph} mapping backing this agent."""
        return self._pool

    @property
    def default(self) -> Any:
        """The compiled graph for the default model."""
        return self._pool[self._default_name]

    def get(self, model_name: str | None, default: Any = None) -> Any:
        """Return the graph for `model_name`, falling back to the default agent."""
        if model_name is None:
            return self.default
        return self._pool.get(model_name, default if default is not None else self.default)

    # ── execution helpers ───────────────────────────────────────────────────
    def invoke(self, payload: dict, *, model: str | None = None, config: dict | None = None):
        return self.get(model).invoke(payload, config=config)

    def stream(
        self,
        payload: dict,
        *,
        model: str | None = None,
        config: dict | None = None,
        stream_mode: str = "messages",
    ):
        yield from self.get(model).stream(payload, config=config, stream_mode=stream_mode)

    def run_session(
        self,
        project_path: str,
        language: str,
        *,
        model: str | None = None,
        thread_id: str | None = None,
    ) -> None:
        """Interactive CLI session: kickoff message, then a follow-up loop.

        Same thread_id across turns -> the checkpointer keeps full history.
        """
        graph = self.get(model)
        config = {
            "configurable": {"thread_id": thread_id or f"{self.name}-1"},
            "recursion_limit": self._recursion_limit,
            "callbacks": [StreamingStdOutCallbackHandler()],
        }

        # First turn: full state, triggers the agent's primary task.
        result = graph.invoke(
            {
                "messages": [{"role": "user", "content": self.kickoff_message}],
                "project_path": project_path,
                "language": language,
            },
            config=config,
        )
        print(result["messages"][-1].content)

        # Follow-up loop: send ONLY the new question; thread_id continues history.
        while True:
            q = input("\nFollow-up (blank to quit): ").strip()
            if not q:
                break
            result = graph.invoke(
                {"messages": [{"role": "user", "content": q}]},
                config=config,
            )
            print(result["messages"][-1].content)

    # ── internals ───────────────────────────────────────────────────────────
    def _build_prompt_middleware(self):
        """Wrap `render_prompt` in a `@dynamic_prompt` middleware bound to self."""

        @dynamic_prompt
        def _prompt(request) -> str:
            return self.render_prompt(request)

        return _prompt

    def _build_tool_gate_middleware(self):
        """Per-turn tool gate: keep only the tools named in `state.enabled_tools`.

        `None`/absent means "no gate" (all tools). An explicit list — including
        the empty list — restricts the model to exactly those tools.
        """

        @wrap_model_call
        def _gate(request, handler):
            enabled = request.state.get("enabled_tools")
            if enabled is not None:
                allowed = set(enabled)
                request = request.override(
                    tools=[t for t in request.tools if t.name in allowed]
                )
            return handler(request)

        return _gate
