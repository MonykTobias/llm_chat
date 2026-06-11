import sqlite3
from string import Template
from typing import Any

from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.messages import SystemMessage
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langchain_core.runnables import RunnableConfig
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from agents.llm_factory import make_llm
from structured_output import ReviewState
import yaml

from tools.tools import run_linter, run_tests, analyze_architecture, run_type_check, read_file, list_all_files

# Load config
with open("config.yaml" , "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

with open("tools.json", "r", encoding="utf-8") as f:
    _tools = yaml.safe_load(f)

@dynamic_prompt
def _language_prompt(request) -> str:
    """System prompt rebuild per run, with project language and path filled in."""
    language = request.state.get("language", "unknown")
    project_path = request.state.get("project_path", ".")
    return Template(_cfg["prompt"]["agent"]).safe_substitute(
        language=language,
        project_path=project_path,
    )

_tools_list = [run_linter, run_tests, analyze_architecture, run_type_check, read_file, list_all_files]

# Persistent SQLite checkpointer — survives server restarts so sessions can be
# resumed. All agents in the pool share one instance; thread_id is the key.
_DB_PATH = Path(__file__).resolve().parent.parent / "ui" / "checkpoints.db"
_conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
_checkpointer = SqliteSaver(_conn)

_agent_pool: dict[str, Any] = {
    name: create_agent(
        model=make_llm(cfg),
        tools=_tools_list,
        middleware=[_language_prompt],
        state_schema=ReviewState,
        checkpointer=_checkpointer,
    )
    for name, cfg in _cfg["agents"].items()
}

_agent = _agent_pool["main_agent"]   # kept for backward compat with main.py

def review_session(project_path: str, language: str, thread_id: str = "review-1"):
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": _cfg.get("recursion_limit", 1000),
        "callbacks":  [StreamingStdOutCallbackHandler()],
    }

    # First turn: full state, triggers the review.
    result = _agent.invoke(
        {
            "messages": [{"role": "user", "content": "Review this project."}],
            "project_path": project_path,
            "language": language,
        },
        config=config,
    )
    print(result["messages"][-1].content)

    # Follow-up loop: send ONLY the new question.
    while True:
        q = input("\nFollow-up (blank to quit): ").strip()
        if not q:
            break
        result = _agent.invoke(
            {"messages": [{"role": "user", "content": q}]},
            config=config,  # same thread_id -> continues, keeps full history
        )
        print(result["messages"][-1].content)