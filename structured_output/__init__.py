from langchain.agents import AgentState

# Canonical list of languages a session can use. Shared by the UI server
# (selectors + validation) and the `set_language` tool, so both validate against
# one source. Kept here (a low-level module already imported by tools/tools.py)
# to avoid an import cycle with ui/server.py.
LANGUAGES = ["english", "python", "javascript", "typescript", "go", "rust", "java"]


class ReviewState(AgentState):
# inherited from AgentState:
#   messages: Annotated[list, add_messages]  -> conversation + tool results
#   remaining_steps: int
    project_path: str
    language: str
    model: str
    # Names of the tools the model is allowed to call this turn. When absent
    # (None), every registered tool is available; an explicit list (incl. the
    # empty list) gates the model down to exactly those tools. Passed per turn,
    # so toggles take effect immediately. See BaseAgent's tool-gate middleware.
    enabled_tools: list[str]
    # Set by the change_mode_* tools to request the pipeline stage (role) the
    # server should switch to AFTER the current turn finishes. The server reads
    # it back post-turn, advances session["role"], and auto-runs the next stage's
    # kickoff. Empty/absent means "no switch". Reset to "" at the start of every
    # turn so a stale value never re-triggers.
    review_mode: str