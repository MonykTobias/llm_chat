import operator
from pydantic import BaseModel, Field, computed_field
from typing import Annotated, Literal, TypedDict

# Instruction prefix that marks a force-appended review task. The inspector
# detects this prefix to switch into review mode instead of explore mode.
REVIEW_TASK_MARKER = "[REVIEW]"

# Keys that must always be replaced (not accumulated) when context_store is updated.
_REPLACE_KEYS = {
    "architect_plan",
    "architect_files_to_create",
    "architect_files_to_modify",
    "architect_files_to_delete",
    "validation_issues",
    "step_snapshot",
    "coder_latest_files",
    "workspace_files",
    "verification_verdict",
    "verification_gaps",
    "skip_verification",
    "file_tree",
    "workspace_dirs",
    "task_failure_counts",
    "abandoned_tasks",
    "task_retry_feedback",
    "active_task_original",       # stable failure-count key (survives instruction rewording)
    "task_reconsideration_count", # how many reconsiderations have fired for the current task
    "review_report",              # JSON ReviewOutput written by the review inspector
    "reviewed_at_start",          # True once the first-round review-first pass has fired
    "module_exports",          # confirmed exports, committed only on validator PASS
    "module_exports_planned",  # scaffold's provisional interface map (forward refs)
    "module_exports_pending",  # coder self-report for current step, awaiting commit
    "inspector_raw",  # full findings text from explore mode
    "inspector_files",  # files of interest from ExploreOutput
    "inspector_issues",  # issues found from ExploreOutput
    "explore_summary",
    "explore_files",
    "explore_issues"
}


def merge_dicts(current: dict, update: dict) -> dict:
    merged = {**current}
    for key, value in update.items():
        if key in _REPLACE_KEYS:
            merged[key] = value
        elif key in merged and isinstance(merged[key], list) and isinstance(value, list):
            merged[key] = merged[key] + value
        else:
            merged[key] = value
    return merged


# ── Orchestrator Models ──────────────────────────────────────────────────────

class SubTask(BaseModel):
    agent: Literal["inspector", "architect"] = Field(
        description="Which agent handles this step: 'inspector' (read-only search) or 'architect' (plan + implement)."
    )
    instruction: str = Field(
        description="Specific, detailed instruction for the agent."
    )


class PlannerOutput(BaseModel):
    """What the LLM actually produces. Kept small to reduce schema deviations."""
    thinking_process: str = Field(
        default="",
        description="Brief reasoning about what needs to happen next."
    )
    todo_list: list[SubTask] = Field(
        default=[],
        description="Ordered list of remaining tasks. Empty when objective is complete."
    )


class TaskPlan(BaseModel):
    """Internal runtime plan — derived from PlannerOutput by the orchestrator code."""
    thinking_process: str = Field(default="")
    todo_list: list[SubTask] = Field(default=[])
    completed_tasks: list[str] = Field(default=[])
    next_agent: Literal["inspector", "architect", "complete"] = Field(default="complete")
    instruction_for_agent: str = Field(default="")


# ── Scaffold Models ──────────────────────────────────────────────────────────

class ScaffoldOutput(BaseModel):
    """The canonical project file tree, committed once before any planning."""
    rationale: str = Field(
        default="",
        description="One or two sentences on how the layout is organized by layer."
    )
    files: list[str] = Field(
        default=[],
        description="Complete canonical project file tree as full relative paths "
                    "(e.g. 'backend/models/session.py'). Empty if there is not "
                    "enough information to commit a layout."
    )
    interfaces: dict[str, list[str]] = Field(
        default={},
        description="Provisional public interface per source file: maps a file path "
                    "from 'files' to the list of public symbol names other files will "
                    "import from it (e.g. 'backend/routes/todos.py': ['todos_router']). "
                    "This is the shared naming contract so producers and consumers "
                    "agree on symbol names regardless of build order. Only include "
                    "files that expose something; omit entrypoints/configs."
    )


# ── Architect Models ─────────────────────────────────────────────────────────

class ArchitectStep(BaseModel):
    step_plan: str = Field(description="Detailed plan for this step with pseudo-code, function signatures, and key logic.")
    files_to_create: list[str] = Field(default=[], description="New files to create in this step (relative paths).")
    files_to_modify: list[str] = Field(default=[], description="Existing files to modify in this step (relative paths).")
    files_to_delete: list[str] = Field(default=[], description="Files to delete in this step (relative paths).")


class ArchitectOutput(BaseModel):
    plan: str = Field(description="Brief overall summary of the full task.")
    steps: list[ArchitectStep] = Field(
        default=[],
        description="Ordered implementation steps. Each step targets at most 2 files, grouped by dependency.",
    )
    task_complete: bool = Field(
        default=False,
        description="Set true ONLY when the task is already fully satisfied by the "
                    "existing code and no file changes are needed. When true, 'steps' "
                    "must be empty and 'plan' must explain why no work is required.",
    )


# ── Coder Models ─────────────────────────────────────────────────────────────
# The coder supports two actions:
#   "create"  — write or overwrite a file with full content (new files AND updates)
#   "delete"  — remove a file
# "modify" (search-and-replace) has been removed — full rewrites are more reliable.

class FileChange(BaseModel):
    file_path: str = Field(
        description="Relative path within the sandbox (e.g., 'app.py', 'src/utils.py')."
    )
    action: Literal["create", "delete"] = Field(
        description="'create' to write or overwrite a file with full content, 'delete' to remove it."
    )
    content: str = Field(
        default="",
        description="Complete file content. Required for 'create', ignored for 'delete'."
    )
    exports: list[str] = Field(
        default=[],
        description="The public symbols this file exposes for OTHER files to import "
                    "(e.g. function/class/router/constant names like 'todos_router', "
                    "'Todo'). List the exact names other modules would import. Leave "
                    "empty for files nothing imports from (entrypoints, configs). "
                    "Ignored for 'delete'."
    )


class CoderOutput(BaseModel):
    changes: list[FileChange] = Field(
        description="All file changes to apply."
    )

# ── Explorer Models ──────────────────────────────────────────────────────────

class ExplorerOutput(BaseModel):
    summary: str = Field(
        description="2-3 sentence overview of what was found."
    )
    files_of_interest: list[str] = Field(
        default=[],
        description="Relative file paths worth the architect's attention."
    )
    issues: list[str] = Field(
        default=[],
        description="Anything broken, missing, or suspicious found during investigation."
    )

# ── Validator Models ─────────────────────────────────────────────────────────

class ValidationIssue(BaseModel):
    file_path: str = Field(default="")
    description: str = Field(default="")
    severity: Literal["error", "warning"] = Field(default="error")


class ValidatorOutput(BaseModel):
    verdict: Literal["pass", "fail"] = Field(default="pass")
    issues: list[ValidationIssue] = Field(default=[])
    summary: str = Field(default="")


# ── Review Models ────────────────────────────────────────────────────────────

class ReviewIssue(BaseModel):
    file_path: str = Field(default="")
    description: str = Field(default="")
    severity: Literal["error", "warning"] = Field(default="error")


class ReviewOutput(BaseModel):
    verdict: Literal["pass", "fail"] = Field(default="pass")
    issues: list[ReviewIssue] = Field(default=[])
    summary: str = Field(default="")
    computed_missing_files: list[str] = Field(default=[])
    missing_files: list[str] = Field(default=[])
    new_files: list[str] = Field(default=[])

# ── Graph State ──────────────────────────────────────────────────────────────
# This is the shared state that flows between all agents in the LangGraph.
#
# Reducers:
#   context_store — merge_dicts: merges new keys into existing dict
#   history       — operator.add: appends new entries to the list
#   iteration_count — operator.add: increments by the returned value
#
# Plain fields (overwrite on update):
#   objective, plan, latest_report, coder_retries

class AgentState(TypedDict):
    objective: str
    plan: TaskPlan | None
    context_store: Annotated[dict, merge_dicts]
    latest_report: str
    history: Annotated[list[str], operator.add]
    iteration_count: Annotated[int, operator.add]
    coder_retries: int
    architect_replans: int      # how many times architect has re-planned the current step
    architect_step_queue: list  # remaining ArchitectStep dicts to dispatch
    project_path: str
    language: str
    enabled_tools: list[str]
    messages: list