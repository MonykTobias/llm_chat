"""
Agent wiring.

This package exposes:
  * `BaseAgent`         — abstract agent (prompt + tools + a model pool).
  * `CodeReviewAgent`   — the concrete code-review implementation.
  * `AGENTS`            — registry of ready-to-use agents, one per *role*.
  * `ROLES`             — the role names (keys of AGENTS), for the UI to list.
  * `graph_for(role, model)` — pick the compiled graph for a role + model.
  * `review_session`    — interactive CLI entry point (used by main.py).

Roles
-----
A *role* is one prompt "personality" (code-review, explore, ...). Every role is
a `CodeReviewAgent` built from its own prompt template in `config.yaml` under
`prompt:`, and each owns a full pool of model variants. Because the checkpointer
keys only on `thread_id`, the caller can switch roles or models mid-session and
keep the full conversation history — only the system prompt (and tools) change.

The module-level `_cfg`, `_checkpointer`, `_agent_pool`, and `_agent` names are
kept for backward compatibility with `ui/server.py`, which reaches into them
directly. New code should prefer `graph_for(...)` / the `AGENTS` registry.
"""
import sqlite3
from pathlib import Path

import yaml
from langgraph.checkpoint.sqlite import SqliteSaver

from agents.base import BaseAgent
from agents.sub_agents.code_review.cr_act import CodeReviewAct
from agents.sub_agents.code_review.cr_explore import CodeReviewExplore
from agents.sub_agents.code_review.cr_plan import CodeReviewPlan
from agents.sub_agents.code_review.cr_verify import CodeReviewVerify
from agents.sub_agents.code_review_agent import CodeReviewAgent
from agents.sub_agents.texting_agent import TextingAgent

__all__ = [
    "BaseAgent",
    "CodeReviewAgent",
    "AGENTS",
    "ROLES",
    "DEFAULT_ROLE",
    "TOOLS_BY_ROLE",
    "graph_for",
    "review_agent",
    "review_session",
]

# ── config ──────────────────────────────────────────────────────────────────
with open("config.yaml", "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

# The role offered first / used as the fallback. Falls back to whatever prompt
# is defined first if "code-review" is ever renamed away.
DEFAULT_ROLE = "code-review" if "code-review" in _cfg["prompt"] else next(iter(_cfg["prompt"]))

# ── shared checkpointer ─────────────────────────────────────────────────────
# Persistent SQLite checkpointer — survives server restarts so sessions can be
# resumed. Every role/model in every pool shares one instance; thread_id is the
# key, which is what lets role/model switches keep the same conversation.
_DB_PATH = Path(__file__).resolve().parent.parent / "ui" / "checkpoints.db"
_conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
_checkpointer = SqliteSaver(_conn)

# ── the agent registry (each entry owns its own pool of model variants) ─────
# Every value here becomes a selectable role in the UI. Shared build options so
# each agent gets the same model pool, checkpointer, and limits.
_BUILD_OPTS = dict(
    checkpointer=_checkpointer,
    recursion_limit=_cfg.get("recursion_limit", 1000),
    default_model="main_agent",
)

# Which class implements each role. Any prompt in config.yaml not listed here is
# built with CodeReviewAgent (the default). A custom class that loads its
# template from config only needs ONE line here — WIRE A NEW AGENT HERE.
AGENT_CLASSES: dict[str, type[BaseAgent]] = {
    "texting_agent": TextingAgent,
    "cr_explore" : CodeReviewExplore,
    "cr_plan": CodeReviewPlan,
    "cr_act" : CodeReviewAct,
    "cr_verify" : CodeReviewVerify,
    # "code-review": CodeReviewAgent,   # default, no entry needed
    # "explore":     CodeReviewAgent,   # default, no entry needed
}

# One agent per prompt in config.yaml, each built from its own template and its
# mapped class. The role name is the single identity: it is the config `prompt:`
# key, the AGENTS registry key (what the UI shows / routes by), and the agent's
# `name` — all the same string.
AGENTS: dict[str, BaseAgent] = {
    role: AGENT_CLASSES.get(role, CodeReviewAgent)(
        _cfg["agents"], template, name=role, **_BUILD_OPTS
    )
    for role, template in _cfg["prompt"].items()
}

ROLES = list(AGENTS)

# Tools each role can run — the UI renders a checkbox per name and sends back
# the enabled subset per turn (see BaseAgent's tool gate).
TOOLS_BY_ROLE: dict[str, list[str]] = {role: agent.tool_names for role, agent in AGENTS.items()}

# The default role's agent — convenient handle and backward-compat fallback.
review_agent = AGENTS[DEFAULT_ROLE]


def graph_for(role: str | None, model: str | None):
    """Compiled graph for a (role, model) pair, falling back to sane defaults.

    Unknown role -> default role's agent. Unknown model -> that agent's default
    model. This is the single place the UI resolves a session to a runnable.
    """
    agent = AGENTS.get(role, review_agent)
    return agent.get(model)


# ── backward-compat aliases for ui/server.py (reaches into these directly) ──
_agent_pool = review_agent.pool
_agent = review_agent.default


def review_session(project_path: str, language: str, thread_id: str = "review-1"):
    """Interactive code-review CLI session (used by main.py)."""
    return review_agent.run_session(project_path, language, thread_id=thread_id)
