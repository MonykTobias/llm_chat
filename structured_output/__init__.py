import json
from typing import Optional, Any, Dict

from langchain.agents import AgentState
from pydantic import BaseModel, Field, field_validator

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

class AgentTurnSchema(BaseModel):
    """One turn of a pipeline-stage agent (explore / plan / act / verify).

    Designed to be driven with the model's *native thinking* enabled: the model
    reasons privately in its thinking channel, so this schema deliberately does
    NOT carry a verbose `internal_reasoning` field (which used to make the model
    reason twice — once in thinking, once in JSON — bloating output and inviting
    think-loops). What survives is a single short, user-facing progress line
    (`note`), the tool to run this step, its arguments, and — only on the final
    turn of a stage — the hand-off report for the next stage.

    The orchestrator (agents/sub_agents/code_review/orchestrator.py) interprets
    `tool_to_call is None` as "this stage is finished" and advances the pipeline.
    """
    note: str = Field(
        default="",
        description="ONE short line (max ~15 words) naming the single action you are taking THIS step, e.g. 'Reading config.yaml to inspect model settings.' No analysis, no multi-step plans, no lists."
    )
    tool_to_call: Optional[str] = Field(
        default=None,
        description="Exact name of the tool to run this step (e.g. 'list_all_files', 'read_file'). Set to null ONLY when this stage is complete and you are emitting stage_report."
    )
    tool_arguments: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Arguments for tool_to_call as a JSON object. Empty dict {} when the tool needs no arguments or when tool_to_call is null."
    )
    stage_report: Optional[str] = Field(
        default=None,
        description="Fill ONLY when tool_to_call is null: the final hand-off report for this stage, in the stage's required layout (TASK:/PLAN:/VERDICT: ...). Leave null while you are still calling tools."
    )

    @field_validator("note", "stage_report", mode="before")
    @classmethod
    def _coerce_to_text(cls, v: Any) -> Any:
        """The stage prompts ask the model to fill `stage_report` with a STRUCTURED
        report (TASK:/PLAN:/VERDICT: ... ), so models frequently emit a nested
        JSON object/array here instead of a string and fail strict validation
        ("Input should be a valid string"). Flatten any non-string value to text
        so the turn still parses instead of crashing the pipeline."""
        if v is None or isinstance(v, str):
            return v
        if isinstance(v, (dict, list)):
            return json.dumps(v, indent=2, ensure_ascii=False)
        return str(v)

    @field_validator("tool_arguments", mode="before")
    @classmethod
    def _coerce_args(cls, v: Any) -> Any:
        """Accept a JSON-string tool_arguments (some models stringify it) and a
        null value, normalising both to a dict so tool dispatch never sees None."""
        if v is None:
            return {}
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                return parsed if isinstance(parsed, dict) else {}
            except (ValueError, TypeError):
                return {}
        return v