"""
The code-assistant orchestrator: one role that drives the whole coding pipeline.

Selecting the `code-assistant` role and sending a task runs the four stage agents
end to end in a single turn:

    explore  ->  plan  ->  act  ->  verify  ->  (PASS? END : plan)

Design
------
* **One shared context.** All four stages append to the *same* message list
  (the graph's `messages` channel). Each stage sees everything the earlier
  stages did and reported. There is no per-stage isolated context.
* **Each stage keeps its own identity.** A stage's system prompt comes straight
  from its `config.yaml` entry (via the already-built `cr_explore` / `cr_plan`
  / `cr_act` / `cr_verify` agents passed in as `stage_agents`), and a stage may
  only call *its own* tools.
* **Structured output, not native tool-calling.** Every model turn is a single
  `AgentTurnSchema` JSON object (grammar-constrained, so it is always valid
  JSON). The orchestrator reads `tool_to_call` and dispatches the tool itself;
  `tool_to_call is None` means the stage is done and `stage_report` carries the
  hand-off. This keeps structured output and tool execution cleanly separated.
* **Thinking stays on.** The per-model `reasoning` flag from `config.yaml` is
  honoured (see `make_llm`). The model reasons privately in its thinking channel
  and only the compact JSON reaches us — so thinking never corrupts the JSON.
  Per-stage step budgets and a repeat-call guard stop think/tool loops.
* **Live stage banners.** Entering a stage and every action emit on LangGraph's
  `custom` stream channel, which `ui/server.py` turns into a dedicated stage
  bubble, progress text, and tool pills (see `STAGE_NODE_NAMES`, which tells the
  server to suppress the stages' raw grammar-constrained JSON tokens).

This object mimics the slice of `BaseAgent` the wiring relies on (`get`,
`default`, `pool`, `tool_names`, `requires_project`, `kickoff_message`); each
`get(model)` returns a fully compiled LangGraph the server streams directly.
"""
from __future__ import annotations

import inspect
import json
import os
import re
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agents.llm_factory import make_llm
from structured_output import AgentTurnSchema, ReviewState

# ── pipeline shape ───────────────────────────────────────────────────────────
# stage key -> config.yaml `prompt:` role that supplies that stage's prompt+tools.
STAGE_ROLES: dict[str, str] = {
    "explore": "cr_explore",
    "plan": "cr_plan",
    "act": "cr_act",
    "verify": "cr_verify",
}
STAGE_ORDER = ["explore", "plan", "act", "verify"]

# The graph node names. ui/server.py drops every "messages"-channel chunk whose
# `langgraph_node` is in this set, so the stages' raw structured-output JSON never
# reaches the chat raw — we surface clean text/tool events on the custom channel
# instead. The node names below MUST equal these keys.
STAGE_NODE_NAMES = set(STAGE_ORDER)

STAGE_LABELS = {"explore": "Explore", "plan": "Plan", "act": "Act", "verify": "Verify"}
STAGE_EMOJI = {"explore": "🔍", "plan": "🧭", "act": "🛠️", "verify": "✅"}

# ── loop / size guards ───────────────────────────────────────────────────────
MAX_STEPS_PER_STAGE = 24   # hard cap on tool steps in one stage (loop-breaker)
MAX_REPLANS = 2            # verify->plan retries before we stop and END anyway
MAX_REPEAT = 3            # identical (tool,args) calls before we refuse to repeat
MAX_OBS_CHARS = 6000       # cap a single tool observation kept in shared context

_SCHEMA_REMINDER = (
    "<turn_contract>\n"
    "Reply with exactly ONE JSON object per turn: "
    '{"note", "tool_to_call", "tool_arguments", "stage_report"}.\n'
    "- To act: set tool_to_call to a tool name and tool_arguments to its JSON "
    "args; leave stage_report null.\n"
    "- When this stage is fully finished: set tool_to_call to null, tool_arguments "
    "to {}, and put your complete hand-off report in stage_report.\n"
    '- "note" is ONE short line about the current step only. Do all detailed '
    "reasoning silently in your own thinking; never repeat yourself or loop.\n"
    "</turn_contract>"
)


class OrchestratorState(ReviewState):
    """Shared pipeline state: ReviewState (messages, project_path, language,
    model, enabled_tools) plus the orchestrator's own loop bookkeeping."""
    verify_cycles: int   # how many times verify has bounced back to plan
    verdict: str         # last verify verdict: "PASS" | "FAIL"


# ── small helpers ────────────────────────────────────────────────────────────
class _PromptRequest:
    """Minimal stand-in for the `request` object a BaseAgent.render_prompt reads
    (`request.state.get(...)`), so we can reuse the stage agents' own prompt
    rendering verbatim against the shared graph state."""

    def __init__(self, state: dict) -> None:
        self.state = state


def _short_json(obj: Any, limit: int = 200) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        s = str(obj)
    return s if len(s) <= limit else s[:limit] + "…"


def _truncate(text: str, limit: int = MAX_OBS_CHARS) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[… truncated {len(text) - limit} chars …]"


def _extract_json(content: str) -> "dict | None":
    """Best-effort: pull the first JSON object out of raw model text (used only
    as a fallback if grammar-constrained parsing ever returns nothing)."""
    if not content:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = content.find("{")
        end = content.rfind("}")
        candidate = content[start:end + 1] if start != -1 and end > start else None
    if candidate is None:
        return None
    try:
        data = json.loads(candidate)
        return data if isinstance(data, dict) else None
    except (ValueError, TypeError):
        return None


def _target_from_args(args: Any) -> "dict | None":
    """Mirror ui/server.py `_extract_target` so custom tool pills label the
    file/dir/url a tool touched, exactly like native tool events do."""
    if not isinstance(args, dict):
        return None
    for key in ("file_path", "path", "filepath", "filename", "directory", "dir_path"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            val = val.strip()
            return {"path": val, "dir": os.path.dirname(val) or ".",
                    "name": os.path.basename(val) or val}
    url, query = args.get("url"), args.get("query")
    if (isinstance(url, str) and url.strip()) or (isinstance(query, str) and query.strip()):
        return {"url": url.strip() if isinstance(url, str) and url.strip() else None,
                "query": query.strip() if isinstance(query, str) and query.strip() else None}
    return None


def _is_pass(report: str) -> bool:
    """A verify stage passes only on an explicit `VERDICT: PASS`. Anything else
    (FAIL, or no parseable verdict) is treated as a failure — the bounded
    verify->plan loop then either fixes it or gives up after MAX_REPLANS."""
    m = re.search(r"VERDICT\s*:?\s*(PASS|FAIL)", report or "", re.I)
    return bool(m) and m.group(1).upper() == "PASS"


class CodeReviewOrchestrator:
    """Drives explore->plan->act->verify as one compiled graph per model."""

    name = "code-assistant"
    requires_project = True
    kickoff_message = (
        "Run the full explore -> plan -> act -> verify pipeline on this task."
    )

    def __init__(
        self,
        model_configs: dict[str, dict],
        *,
        stage_agents: dict[str, Any],
        checkpointer: Any = None,
        recursion_limit: int = 1000,
        default_model: str | None = None,
    ) -> None:
        if not model_configs:
            raise ValueError("CodeReviewOrchestrator needs at least one model config")
        missing = [s for s in STAGE_ORDER if s not in stage_agents]
        if missing:
            raise KeyError(f"stage_agents is missing stages: {missing}")

        self._model_configs = model_configs
        self._recursion_limit = recursion_limit
        self._stage_agents = stage_agents
        # {stage: {tool_name: tool}} — a stage may only ever call its own tools.
        self._tools_by_stage = {
            stage: {t.name: t for t in agent.tools}
            for stage, agent in stage_agents.items()
        }
        # Union of every stage's tools (config order) — the toggle set the UI
        # shows for this role; per-stage gating still applies at dispatch time.
        names: list[str] = []
        for stage in STAGE_ORDER:
            for n in self._tools_by_stage[stage]:
                if n not in names:
                    names.append(n)
        self._tool_names = names

        self._pool: dict[str, Any] = {
            model_name: self._build_graph(model_cfg, checkpointer)
            for model_name, model_cfg in model_configs.items()
        }
        self._default_name = default_model or next(iter(self._pool))
        if self._default_name not in self._pool:
            raise KeyError(
                f"default_model {self._default_name!r} is not one of {list(self._pool)}"
            )

    # ── BaseAgent-compatible surface used by the wiring/server ───────────────
    @property
    def tool_names(self) -> list[str]:
        return list(self._tool_names)

    @property
    def pool(self) -> dict[str, Any]:
        return self._pool

    @property
    def default(self) -> Any:
        return self._pool[self._default_name]

    def get(self, model_name: str | None, default: Any = None) -> Any:
        if model_name is None:
            return self.default
        return self._pool.get(model_name, default if default is not None else self.default)

    def stream(self, payload: dict, *, model: str | None = None,
               config: dict | None = None, stream_mode: Any = "messages"):
        yield from self.get(model).stream(payload, config=config, stream_mode=stream_mode)

    def invoke(self, payload: dict, *, model: str | None = None, config: dict | None = None):
        return self.get(model).invoke(payload, config=config)

    # ── graph construction ───────────────────────────────────────────────────
    def _build_graph(self, model_cfg: dict, checkpointer: Any):
        llm = make_llm(model_cfg)
        # Grammar-constrained JSON (always valid) while the model's own thinking
        # channel stays on — include_raw lets us fall back to manual JSON
        # extraction in the rare case parsing returns nothing.
        structured = llm.with_structured_output(AgentTurnSchema, include_raw=True)

        graph = StateGraph(OrchestratorState)
        for stage in STAGE_ORDER:
            graph.add_node(stage, self._make_node(stage, structured, model_cfg))
        graph.add_edge(START, "explore")
        graph.add_edge("explore", "plan")
        graph.add_edge("plan", "act")
        graph.add_edge("act", "verify")
        graph.add_conditional_edges(
            "verify", self._route_after_verify, {"plan": "plan", END: END}
        )
        return graph.compile(checkpointer=checkpointer)

    def _make_node(self, stage: str, structured: Any, model_cfg: dict):
        def node(state: OrchestratorState) -> dict:
            return self._run_stage(stage, state, structured, model_cfg)
        node.__name__ = stage  # cosmetic; add_node(stage, ...) sets the real name
        return node

    @staticmethod
    def _stage_kickoff(stage: str, cycles: int) -> str:
        """The human turn that opens each stage. Keeps the transcript alternating
        System/Human/AI/Human… so the model always generates after a human turn
        (see the invariant noted in `_run_stage`)."""
        base = {
            "explore": "Begin the EXPLORE stage for the task above. Read every file "
                       "you need to understand it, then hand off what you found.",
            "plan": "Begin the PLAN stage. Turn the exploration above into a concrete, "
                    "file-by-file execution plan.",
            "act": "Begin the ACT stage. Apply the approved plan above by writing the "
                   "required file changes with write_file.",
            "verify": "Begin the VERIFY stage. Run the checks needed to confirm the "
                      "changes work, then give a VERDICT (PASS or FAIL).",
        }[stage]
        if stage == "plan" and cycles:
            base = ("Begin the PLAN stage AGAIN — verification FAILED above. Revise the "
                    "plan to fix the specific failures reported; do not repeat the plan "
                    "that just failed.")
        return (base + " Respond with exactly ONE JSON object per the turn contract: "
                "emit a tool call to keep working, or set tool_to_call to null and fill "
                "stage_report when this stage is done.")

    @staticmethod
    def _route_after_verify(state: OrchestratorState) -> str:
        if state.get("verdict") == "PASS":
            return END
        if state.get("verify_cycles", 0) >= MAX_REPLANS:
            return END   # give up after enough re-plans rather than loop forever
        return "plan"

    # ── one stage = a bounded structured-output ReAct loop ───────────────────
    def _run_stage(self, stage: str, state: OrchestratorState,
                   structured: Any, model_cfg: dict) -> dict:
        writer = get_stream_writer()
        cycles = state.get("verify_cycles", 0)
        label = STAGE_LABELS[stage]
        banner = f"{label} (revision {cycles})" if stage == "plan" and cycles else label
        # The dedicated stage-switch bubble in the chat.
        writer({"kind": "stage", "stage": stage,
                "label": f"{STAGE_EMOJI[stage]} {banner} stage"})

        agent = self._stage_agents[stage]
        tools = self._tools_by_stage[stage]
        enabled = state.get("enabled_tools")
        usable = ({n: t for n, t in tools.items() if n in set(enabled)}
                  if enabled is not None else dict(tools))

        sys_prompt = "\n\n".join((
            agent.render_prompt(_PromptRequest(state), model_cfg),
            self._tool_spec(usable),
            _SCHEMA_REMINDER,
        ))

        history = list(state.get("messages") or [])
        # Every stage opens with a HUMAN kickoff turn. This is the load-bearing
        # invariant of the whole pipeline: a local thinking model only emits clean
        # schema JSON when it is generating *in response to a human turn*. Earlier
        # stages' reports are assistant (AI) messages, so without this kickoff the
        # next stage would be asked to "continue after its own message" — which
        # makes the model spill prose or stop with empty output instead of JSON.
        new_msgs: list = [HumanMessage(content=self._stage_kickoff(stage, cycles))]
        work_state = dict(state)      # mutated locally so set_language etc. apply now
        extra: dict[str, Any] = {}    # state updates to persist (e.g. language)
        repeats: dict[str, int] = {}
        report: str | None = None

        for _ in range(MAX_STEPS_PER_STAGE):
            turn, usage = self._next_turn(
                structured,
                [SystemMessage(content=sys_prompt)] + history + new_msgs)
            # Surface this model call's token usage on the custom channel. The
            # server can't see the orchestrator's internal model calls (their
            # native "messages" chunks are suppressed), so without this the UI's
            # tok/s and context-window meters stay empty for the code-assistant role.
            if usage.get("output_tokens") or usage.get("input_tokens"):
                writer({"kind": "usage", "usage": usage})
            note = (turn.note or "").strip()
            tool_name = (turn.tool_to_call or "").strip() or None

            if tool_name is None:                      # stage finished
                report = (turn.stage_report or note or "").strip() or "(no report produced)"
                break

            args = turn.tool_arguments if isinstance(turn.tool_arguments, dict) else {}
            if note:
                writer({"kind": "text", "text": note + "\n"})

            if tool_name not in usable:
                obs = (f"Tool '{tool_name}' is not available in the {stage} stage. "
                       f"Available tools: {', '.join(usable) or 'none'}.")
                new_msgs.append(AIMessage(content=f"{note}\nAction: {tool_name}({_short_json(args)})"))
                new_msgs.append(HumanMessage(content=f"[tool error] {obs}"))
                continue

            target = _target_from_args(args)
            sig = f"{tool_name}::{_short_json(args)}"
            repeats[sig] = repeats.get(sig, 0) + 1
            writer({"kind": "tool_start", "name": tool_name, "target": target})
            if repeats[sig] > MAX_REPEAT:
                result = (f"[stop] You already ran {tool_name} with these exact "
                          f"arguments {repeats[sig] - 1} times — the result will not "
                          "change. Use what you already have; do not repeat it.")
            else:
                result, updates = self._dispatch(usable[tool_name], args, work_state)
                if updates:
                    work_state.update(updates)
                    extra.update(updates)
            writer({"kind": "tool_end", "name": tool_name, "target": target})

            new_msgs.append(AIMessage(content=f"{note}\nAction: {tool_name}({_short_json(args)})"))
            new_msgs.append(HumanMessage(
                content=f"[tool result: {tool_name}]\n{_truncate(result)}"))
        else:
            report = ("(stage stopped: step budget exhausted before it reported "
                      "completion)")

        # Final stage report -> its own clean bubble + carried into shared context.
        writer({"kind": "text", "text": "\n" + report + "\n"})
        new_msgs.append(AIMessage(content=f"[{label} stage report]\n{report}"))

        update: dict[str, Any] = {"messages": new_msgs, **extra}
        if stage == "verify":
            verdict = "PASS" if _is_pass(report) else "FAIL"
            update["verdict"] = verdict
            if verdict != "PASS":
                update["verify_cycles"] = cycles + 1
        return update

    # ── model turn (structured output, with a robust fallback) ───────────────
    @staticmethod
    def _usage_from(resp: Any) -> dict:
        """Pull token usage from a structured-output response's raw AIMessage.

        The orchestrator's model calls never reach the server's native token
        stream, so this is the only place that knows how many tokens each turn
        cost — the UI's tok/s and context-window meters are fed from it."""
        raw = resp.get("raw") if isinstance(resp, dict) else resp
        um = getattr(raw, "usage_metadata", None) or {}
        try:
            return {"input_tokens": int(um.get("input_tokens", 0) or 0),
                    "output_tokens": int(um.get("output_tokens", 0) or 0),
                    "total_tokens": int(um.get("total_tokens", 0) or 0)}
        except (TypeError, ValueError):
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    @staticmethod
    def _next_turn(structured: Any, messages: list) -> "tuple[AgentTurnSchema, dict]":
        try:
            resp = structured.invoke(messages)
        except Exception as e:  # noqa: BLE001 — never crash the pipeline on a bad turn
            return (AgentTurnSchema(note="", tool_to_call=None,
                    stage_report=f"(model call failed: {type(e).__name__}: {e})"), {})
        usage = CodeReviewOrchestrator._usage_from(resp)
        parsed = resp.get("parsed") if isinstance(resp, dict) else resp
        if parsed is not None:
            return parsed, usage
        # Grammar parse returned nothing — recover the JSON from the raw text.
        raw = resp.get("raw") if isinstance(resp, dict) else None
        content = getattr(raw, "content", "") if raw is not None else ""
        if isinstance(content, list):
            content = "".join(p.get("text", "") if isinstance(p, dict) else str(p)
                              for p in content)
        data = _extract_json(content)
        if data is not None:
            try:
                return AgentTurnSchema.model_validate(data), usage
            except Exception:  # noqa: BLE001
                pass
        # Give up cleanly: end the stage using whatever text we got as the report.
        return (AgentTurnSchema(note="", tool_to_call=None,
                stage_report=(content or "").strip() or "(no output)"), usage)

    # ── tool dispatch (manual, with injected state/tool_call_id) ─────────────
    @staticmethod
    def _dispatch(tool: Any, args: dict, work_state: dict) -> "tuple[str, dict]":
        """Run one tool. Tools take their `state` (project_path/language) and any
        `tool_call_id` as *injected* args, hidden from the model — so we fill them
        from the shared state here. A tool that returns a Command (e.g.
        set_language) has its state update folded back into the pipeline."""
        fn = getattr(tool, "func", None) or getattr(tool, "coroutine", None)
        call_args = dict(args or {})
        try:
            params = inspect.signature(fn).parameters if fn else {}
        except (TypeError, ValueError):
            params = {}
        if "state" in params:
            call_args["state"] = work_state
        if "tool_call_id" in params:
            call_args["tool_call_id"] = "orch-" + uuid.uuid4().hex[:8]
        try:
            result = fn(**call_args) if fn else tool.invoke(call_args)
        except Exception as e:  # noqa: BLE001
            return f"[tool error] {type(e).__name__}: {e}", {}

        if isinstance(result, Command):
            upd = result.update or {}
            msgs = upd.get("messages") or []
            text = "; ".join(str(getattr(m, "content", m)) for m in msgs) or "(state updated)"
            return text, {k: v for k, v in upd.items() if k != "messages"}
        return str(result), {}

    @staticmethod
    def _tool_spec(usable: dict) -> str:
        """A compact, accurate tool list injected into the stage prompt. Tools
        aren't natively bound (we use structured output), so the model needs the
        exact names + visible arg names to fill tool_to_call/tool_arguments."""
        lines = ["<available_tools>",
                 "Call a tool via tool_to_call + tool_arguments. ONLY these tools "
                 "exist in this stage:"]
        for name, tool in usable.items():
            try:
                arg_names = ", ".join((tool.args or {}).keys())
            except Exception:  # noqa: BLE001
                arg_names = ""
            desc = ((tool.description or "").strip().splitlines() or [""])[0]
            lines.append(f"- {name}({arg_names}): {desc}")
        if len(lines) == 2:
            lines.append("- (no tools available — finish the stage now)")
        lines.append("</available_tools>")
        return "\n".join(lines)
