from langchain.agents import AgentState


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