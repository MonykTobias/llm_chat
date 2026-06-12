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


def _msg_field(message: Any, field: str) -> Any:
    """Read `field` from a LangChain message object or a plain-dict message."""
    if isinstance(message, dict):
        return message.get(field)
    return getattr(message, field, None)


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

    #: Whether this agent needs a project directory to do its job. True for every
    #: code-oriented role (they read/write files under `state["project_path"]`);
    #: a chat-only role sets this False so the UI can offer it without a folder
    #: and keep it permanently separate from the folder-requiring roles.
    requires_project: bool = True

    #: Per-turn cap on how many times a given tool may run before it is dropped
    #: from the set advertised to the model. This is the hard loop-breaker: small
    #: local models often re-call a tool forever (e.g. web_browse on the same
    #: query); once the cap is hit the tool disappears from their options, so they
    #: must answer instead of looping. Tools absent from this map are uncapped.
    tool_call_budgets: dict[str, int] = {"web_browse": 10}

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
        tool_budget_middleware = self._build_tool_budget_middleware()

        self._model_configs = model_configs

        self._pool: dict[str, Any] = {
            model_name: create_agent(
                model=make_llm(model_cfg),
                tools=self.tools,
                middleware=[prompt_middleware, tool_gate_middleware,
                            tool_budget_middleware],
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
    def render_prompt(self, request, cfg: dict = {}) -> str:
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
            # resolve active model for this request
            model_name = request.state.get("model_name") or self._default_name
            cfg = self._model_configs.get(model_name, {})
            return self.render_prompt(request, cfg)
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

    def _build_tool_budget_middleware(self):
        """Hard loop-breaker: drop a tool once it has run too many times this turn.

        Counts how often each tool has already returned *since the last user
        message* (so the budget resets every turn) and removes any tool that has
        hit its `tool_call_budgets` cap from the tools advertised to the model.
        A model stuck re-calling `web_browse` then simply loses the option and is
        forced to answer with what it has, instead of looping to the recursion
        limit. Tools not listed in `tool_call_budgets` are never capped.
        """

        @wrap_model_call
        def _budget(request, handler):
            budgets = self.tool_call_budgets or {}
            if budgets:
                messages = request.state.get("messages", []) or []
                # Scope to the current turn: only count tool results that came
                # after the most recent human message.
                start = 0
                for i, m in enumerate(messages):
                    if _msg_field(m, "type") == "human":
                        start = i
                counts: dict[str, int] = {}
                for m in messages[start:]:
                    if _msg_field(m, "type") == "tool":
                        name = _msg_field(m, "name")
                        if name:
                            counts[name] = counts.get(name, 0) + 1
                exhausted = {n for n, cap in budgets.items()
                             if counts.get(n, 0) >= cap}
                if exhausted:
                    request = request.override(
                        tools=[t for t in request.tools if t.name not in exhausted]
                    )
            return handler(request)

        return _budget
