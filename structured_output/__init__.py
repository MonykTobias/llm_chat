from langchain.agents import AgentState


class ReviewState(AgentState):
# inherited from AgentState:
#   messages: Annotated[list, add_messages]  -> conversation + tool results
#   remaining_steps: int
    project_path: str
    language: str
    model: str