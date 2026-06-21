"""
Scaffold node — canonical-file-tree builder, now running AFTER the orchestrator's
first plan (it used to be the graph entry point, before any planning).

The chat-turn → objective INTAKE that used to live here as "job 1" has been split
into `intake_node` (also in this module), which runs at START so the orchestrator
— which reads `state["objective"]` — has it before it plans. `scaffold_node` is
now diverted to by the orchestrator the first time it dispatches real work, so the
tree is built with the plan in hand rather than blind. See this package's
`__init__.py` flow docstring for the wiring.

This node's single job:

  1. SCAFFOLD: commit ONE canonical project file tree into
     context_store["file_tree"] before any planning starts. It reads the spec
     (if any), the real files on disk under `state["project_path"]`, and a
     spec-seeded hint tree, then writes the final canonical list every downstream
     agent is locked to. This is what stops the architect from inventing a fresh,
     flattened structure on every task. It also seeds the planned-interface map
     (module_exports_planned) and, for files already on disk, the CONFIRMED export
     registry (module_exports) by reading their real code.

If there is not enough information to commit a layout (no spec, no seed hint, and
an empty workspace) it returns no tree and the system runs exactly as before —
downstream enforcement simply becomes a no-op.

Ported from the standalone `orchestrator` project's scaffold and adapted to this
project's conventions: the project root comes from `state["project_path"]` (not a
config sandbox dir), the real file listing comes from `tools.list_workspace_files`,
per-node LLM/prompt config lives in `graph_config.yaml`, and the objective intake
above is preserved.

Idempotency: the node stamps context_store["scaffolded"] = True on every return
path (including the skip/empty no-ops), so the orchestrator router diverts here
exactly once. If `scaffold_on_replan` is enabled, a full orchestrator re-plan sets
context_store["rescaffold_requested"] and control is diverted here again; the
reconciliation below is inherently a diff (it always re-includes the existing
workspace files and the prior tree as the seed), so a re-run only ADDS what the
new plan needs and never relocates existing files. After running, scaffold forwards
to the agent the plan named (architect / inspector).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from string import Template

import yaml
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from agents.llm_factory import make_llm
from tools import list_workspace_files, safe_read

from agents.implementations.code_agent.structured_output import AgentState, ScaffoldOutput
from agents.implementations.code_agent.utils.stats import stats
from agents.implementations.code_agent.utils.validation import extract_paths

# Config lives next to this module (graph_config.yaml), like every other node.
_CFG_PATH = Path(__file__).resolve().parent.parent / "graph_config.yaml"
with open(_CFG_PATH, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

# Scaffold knobs — read from config with the standalone project's defaults so the
# node works whether or not graph_config.yaml declares them.
_SCAFFOLD_ENABLED = _cfg.get("scaffold_enabled", True)
# Existing-file export extraction (option 2): on resume / existing codebases the
# in-memory registry is empty, so real exports of files already on disk must be
# re-derived at startup and seeded as CONFIRMED (ground truth from real code).
_EXTRACT_EXISTING_EXPORTS = _cfg.get("scaffold_extract_existing_exports", True)
_EXPORT_SCAN_MAX_FILES = _cfg.get("scaffold_export_scan_max_files", 200)
_EXPORT_SCAN_PER_FILE_CHARS = _cfg.get("scaffold_export_scan_per_file_chars", 8000)
_EXPORT_SCAN_BATCH_CHARS = _cfg.get("scaffold_export_scan_batch_chars", 16000)
_SOURCE_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue", ".svelte",
    ".go", ".rs", ".java", ".kt", ".swift", ".rb", ".php", ".cs", ".scala",
    ".c", ".cc", ".cpp", ".h", ".hpp",
}

# Stamped into context_store on EVERY scaffold return path so the orchestrator
# router (`_needs_scaffold`) diverts here exactly once. `rescaffold_requested` is
# cleared here so a re-plan-triggered re-run consumes the request and doesn't loop.
_SCAFFOLD_DONE_FLAGS = {"scaffolded": True, "rescaffold_requested": False}


def _derive_objective(state: AgentState) -> str:
    """Latest user message text, used as the planner objective."""
    for m in reversed(state.get("messages") or []):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "type", None)
        if role in ("user", "human"):
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def intake_node(state: AgentState, config: RunnableConfig) -> dict:
    """Entry node: turn the incoming chat turn into the planner objective.

    Split out of the old scaffold node so it can run at START, *before* the
    orchestrator (which reads `state["objective"]`). It is intentionally tiny:
    no LLM, no disk access, no file tree. If the caller already supplied an
    explicit `objective` we keep it; otherwise we derive it from the latest user
    message — exactly as scaffold used to. `language` is passed through with a
    sane default so downstream nodes that read it still see a value.
    """
    writer = get_stream_writer()
    objective = state.get("objective") or _derive_objective(state)
    writer({"kind": "stage", "stage": "intake",
            "label": "📥 Intake — reading the request"})
    return {
        "objective": objective,
        "language": state.get("language", "python"),
        "history": ["intake"],
    }


def _format_planned_tasks(plan) -> str:
    """Render the orchestrator's current plan as a short bullet list for the
    scaffold prompt, so the tree is shaped by the work that's actually planned.

    Returns "" when there is no usable plan, so callers can fall back to the
    objective-only prompt unchanged.
    """
    if plan is None:
        return ""
    lines: list[str] = []
    head = getattr(plan, "instruction_for_agent", "") or ""
    head_agent = getattr(plan, "next_agent", "") or "agent"
    if head.strip():
        lines.append(f"  - [{head_agent}] {head.strip()}")
    for t in (getattr(plan, "todo_list", None) or []):
        instr = getattr(t, "instruction", "") or ""
        agent = getattr(t, "agent", "") or "agent"
        if instr.strip():
            lines.append(f"  - [{agent}] {instr.strip()}")
    return "\n".join(lines)


def _plan_referenced_paths(plan) -> list[str]:
    """Every file path the orchestrator's plan names, normalized and de-duped.

    These are force-included in the canonical tree so a planned-but-not-yet-created
    file (e.g. 'backend/models/session.py' from a 'create ...' task) is guaranteed
    to be on the tree. That keeps the tree a superset of what the plan references,
    so later grounding (`_sanitize_todo`) can never drop a legitimate create-task,
    and the inspector reports such files as missing-to-implement rather than ignoring
    them. Uses the same `extract_paths` the orchestrator trusts for sanitizing.
    """
    if plan is None:
        return []
    texts: list[str] = []
    head = getattr(plan, "instruction_for_agent", "") or ""
    if head.strip():
        texts.append(head)
    for t in (getattr(plan, "todo_list", None) or []):
        instr = getattr(t, "instruction", "") or ""
        if instr.strip():
            texts.append(instr)
    out: list[str] = []
    for text in texts:
        for p in extract_paths(text):
            np = _norm(p)
            if np and not _is_junk(np) and np not in out:
                out.append(np)
    return out


def _parse_scaffold_fallback(raw_content: str) -> ScaffoldOutput | None:
    cleaned = (raw_content or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    files = data.get("files", [])
    if not isinstance(files, list):
        return None
    files = [f for f in files if isinstance(f, str)]
    raw_ifaces = data.get("interfaces", {})
    interfaces: dict[str, list[str]] = {}
    if isinstance(raw_ifaces, dict):
        for path, syms in raw_ifaces.items():
            if isinstance(path, str) and isinstance(syms, list):
                names = [s for s in syms if isinstance(s, str) and s.strip()]
                if names:
                    interfaces[path] = names
    return ScaffoldOutput(
        rationale=str(data.get("rationale", "")),
        files=files,
        interfaces=interfaces,
    )


def _norm(path: str) -> str:
    """Normalize a relative path: forward slashes, no './' prefix, no leading '/'."""
    p = path.replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    return p.lstrip("/")


def _is_junk(path: str) -> bool:
    base = os.path.basename(path)
    return (
        not path
        or "__pycache__" in path
        or base.endswith((".pyc", ".pyo", ".pyd"))
        or path.endswith("/")
    )


def _collapse_double_root(files: list[str], anchors: set[str]) -> list[str]:
    """
    Collapse an accidental double root, e.g. both 'backend/main.py' and
    'workout-logger/backend/main.py' appearing because the spec-seed convention
    and the scaffold-LLM convention disagreed. Only collapses when BOTH 'X' and
    'wrapper/X' literally exist; prefers the form matching an authoritative
    workspace/seed anchor so real on-disk files are never relocated, otherwise
    keeps the un-prefixed form.
    """
    paths = set(files)
    roots = {p.split("/", 1)[0] for p in files if "/" in p}
    drop: set[str] = set()
    for p in files:
        head, _, tail = p.partition("/")
        if tail and head in roots and tail in paths:
            # 'head/tail' and 'tail' are the same file under two roots.
            if p in anchors and tail not in anchors:
                drop.add(tail)   # workspace/seed says the wrapped form is real
            else:
                drop.add(p)      # default: keep the un-prefixed form
    return [f for f in files if f not in drop]


def _norm_dirname(name: str) -> str:
    """Collapse case and separator differences so 'workout-logger' == 'workout_logger'."""
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def _strip_sandbox_wrapper(files: list[str], project_root: str) -> list[str]:
    """Drop a single top-level folder that is just the project root renamed.

    The project directory IS the root, so a spec tree wrapped in a project-name
    folder (e.g. 'workout-logger/backend/main.py' under root 'workout_logger')
    would land doubly-nested on disk. Only strips when EVERY path lives under one
    top-level dir whose normalized name equals the root basename's — a legitimate
    'backend/'-only or 'src/'-only tree is never touched.
    """
    roots = {p.split("/", 1)[0] for p in files if "/" in p}
    if len(roots) != 1:
        return files
    root = next(iter(roots))
    if not all(p.startswith(root + "/") for p in files):
        return files  # a loose root-level file means this isn't a pure wrapper
    root_base = os.path.basename(project_root.rstrip("/\\"))
    if _norm_dirname(root) != _norm_dirname(root_base):
        return files
    stripped = [p[len(root) + 1:] for p in files]
    print(f"[Scaffold] Stripped project-name wrapper '{root}/' from {len(files)} path(s).")
    return stripped


def _resolve_package_module_collisions(files: list[str]) -> list[str]:
    """
    Drop a flat module when a same-named package directory also exists, e.g. keep
    the 'backend/models/' package and drop the stale 'backend/models.py'.

    These collisions come from earlier botched runs leaving BOTH forms on disk;
    the "never relocate existing workspace files" rule then faithfully re-adds the
    stale flat module into the canonical tree. The result is a tree where 'models'
    is simultaneously a file and a folder, so the architect can't pick a
    consistent path and emits off-tree ones (the file-vs-folder confusion).

    The multi-file package form is authoritative: a flat module 'X.<ext>' is
    dropped only when the directory 'X/' contains at least two files in the tree
    (a deliberate package, not a single model-invented file). Singular/plural
    leftovers like 'repository.py' next to 'repositories/' have no same-stem
    folder, so they are not a file/folder collision and are left untouched.
    """
    # Count files living directly or transitively under each directory prefix.
    dir_counts: dict[str, int] = {}
    for p in files:
        parts = p.split("/")
        for i in range(1, len(parts)):
            prefix = "/".join(parts[:i])
            dir_counts[prefix] = dir_counts.get(prefix, 0) + 1

    kept: list[str] = []
    for p in files:
        stem, ext = os.path.splitext(p)
        if ext and dir_counts.get(stem, 0) >= 2:
            print(f"[Scaffold] Dropped module '{p}' — collides with same-named package '{stem}/'.")
            continue
        kept.append(p)
    return kept


def _derive_dirs(files: list[str]) -> list[str]:
    """Every directory prefix implied by the file list (forward-slash, sorted)."""
    dirs: set[str] = set()
    for f in files:
        parts = f.split("/")
        for i in range(1, len(parts)):
            dirs.add("/".join(parts[:i]))
    return sorted(dirs)


def _clip(content: str, limit: int) -> str:
    """Keep a file within a char budget for extraction, preserving head AND tail
    (exports can sit at either end — e.g. JS `export default` / `module.exports`)."""
    if len(content) <= limit:
        return content
    head = content[: (limit * 2) // 3]
    tail = content[-(limit // 3):]
    return f"{head}\n... (truncated) ...\n{tail}"


def _parse_json_object(raw: str) -> dict | None:
    """Parse a JSON object from an LLM reply, tolerating ```fences and preamble."""
    cleaned = (raw or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        # Last resort: grab the outermost {...} span.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(cleaned[start:end + 1])
            except (json.JSONDecodeError, TypeError):
                return None
        else:
            return None
    return data if isinstance(data, dict) else None


def _extract_existing_exports(project_path: str, paths: list[str], llm) -> dict[str, list[str]]:
    """Read existing source files and have the LLM report their real public symbols.

    Language-agnostic (the model reads the code), seeded as CONFIRMED. Batched and
    capped so a large/resumed codebase doesn't blow the token budget or stall start-up.
    Returns {normalized_path: [symbol, ...]} for files that expose something.
    """
    candidates: list[tuple[str, str]] = []
    for p in paths:
        if os.path.splitext(p)[1].lower() not in _SOURCE_EXTS:
            continue
        content = safe_read(project_path, p)
        if not content or not content.strip():
            continue
        candidates.append((p, _clip(content, _EXPORT_SCAN_PER_FILE_CHARS)))
        if len(candidates) >= _EXPORT_SCAN_MAX_FILES:
            print(f"[Scaffold] Export scan hit the {_EXPORT_SCAN_MAX_FILES}-file cap; remaining files skipped.")
            break
    if not candidates:
        return {}

    # Group files into batches under the per-call char budget.
    batches: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    cur_chars = 0
    for p, c in candidates:
        block = len(c) + len(p) + 16
        if cur and cur_chars + block > _EXPORT_SCAN_BATCH_CHARS:
            batches.append(cur)
            cur, cur_chars = [], 0
        cur.append((p, c))
        cur_chars += block
    if cur:
        batches.append(cur)

    tmpl = _cfg["prompts"].get("export_extractor")
    if not tmpl:
        return {}

    result: dict[str, list[str]] = {}
    for batch in batches:
        files_block = "\n\n".join(f"=== {p} ===\n{c}" for p, c in batch)
        prompt = Template(tmpl).safe_substitute(files=files_block)
        try:
            r = llm.invoke([HumanMessage(content=prompt)])
            stats.record_tokens(r)
            data = _parse_json_object(r.content)
        except Exception as e:
            print(f"[Scaffold] Export extraction batch failed: {e}")
            continue
        if not data:
            continue
        for path, syms in data.items():
            if not isinstance(path, str) or not isinstance(syms, list):
                continue
            names = [s for s in syms if isinstance(s, str) and s.strip()]
            if names:
                result[_norm(path)] = names
    return result



def scaffold_node(state: AgentState, config: RunnableConfig) -> dict:
    writer = get_stream_writer() # frontend-provided stream writer
    # helper function to write text on stream for frontend (in Markdown)
    def _w(text: str) -> None:
        writer({"kind": "text", "text": text + "\n\n"})

    # write stage bubble to stream
    writer({"kind": "stage", "stage": "scaffold",
            "label": "📁 Scaffold — Gathering project information"})

    objective = state.get("objective") or _derive_objective(state)

    _w(f"## Objective\n\n{objective}") # write to stream

    # ── Job 2: SCAFFOLD ──────────────────────────────────────────────────────
    if not _SCAFFOLD_ENABLED:
        return {
            "objective": objective,
            "language": state.get("language", "python"),
            "context_store": dict(_SCAFFOLD_DONE_FLAGS),
            "history": ["scaffold_skipped"]
        }

    store = state.get("context_store", {})
    spec_content = store.get("spec_content", "")
    seed = list(store.get("file_tree", []) or [])  # spec-derived hint / fallback

    # This project's source of truth for "what exists right now" is the real disk
    # listing under project_path (the orchestrator/architect read it the same way),
    # not a pre-seeded context key — so scan it here rather than trusting the store.
    project_path = state.get("project_path", ".")
    workspace_files = list_workspace_files(project_path)
    project_name = os.path.basename(os.path.abspath(project_path).rstrip("/\\"))

    # Nothing to build a layout from — stay a no-op and let the system run as before.
    if not spec_content and not seed and not workspace_files:
        _w("⚠️ No spec, seed, or workspace — scaffold skipped.") # write to stream
        return {
            "objective": objective,
            "language": state.get("language", "python"),
            "context_store": dict(_SCAFFOLD_DONE_FLAGS),
            "history": ["scaffold_skipped"]
        }

    _w("### 🗂 Committing canonical project file tree...")

    sc = _cfg["agents"]["scaffold"]
    llm = make_llm(sc)

    workspace_dirs = _derive_dirs(workspace_files)

    # Scaffold now runs after the orchestrator's first plan, so fold the planned
    # tasks into the objective the tree-builder sees. This is what lets the tree be
    # shaped by the actual work instead of guessed blind. We enrich only the prompt
    # copy; state["objective"] itself is left untouched.
    planned_tasks = _format_planned_tasks(state.get("plan"))
    objective_for_tree = objective
    if planned_tasks:
        objective_for_tree = (
            f"{objective}\n\n"
            f"PLANNED TASKS (the implementation plan this file tree must support):\n"
            f"{planned_tasks}"
        )

    prompt = Template(_cfg["prompts"]["scaffold"]).safe_substitute(
        objective=objective_for_tree,
        sandbox_name=project_name or "(project root)",
        spec=spec_content or "(no spec provided)",
        workspace_files="\n".join(f"  {p}" for p in workspace_files) or "(empty workspace)",
        existing_dirs="\n".join(f"  {p}" for p in workspace_dirs) or "(none)",
        required_paths="\n".join(f"  {p}" for p in seed) or "(none extracted)",
    )

    parsed: ScaffoldOutput | None = None
    try:
        result = llm.invoke([HumanMessage(content=prompt)])
        stats.record_tokens(result)
        parsed = _parse_scaffold_fallback(result.content)
    except Exception as e:
        _w(f"❌ LLM call failed: {e}") # write to stream

    if parsed is None:
        # Structured-output fallback (same pattern the architect uses).
        try:
            structured = llm.with_structured_output(ScaffoldOutput, include_raw=True)
            r = structured.invoke([HumanMessage(content=prompt)])
            if r.get("raw"):
                stats.record_tokens(r["raw"])
            parsed = r.get("parsed")
            if parsed is None and r.get("raw") is not None:
                raw_text = r["raw"].content if hasattr(r["raw"], "content") else str(r["raw"])
                parsed = _parse_scaffold_fallback(raw_text)
        except Exception as e:
            _w(f"❌ LLM structured output fallback failed: {e}")

    model_files = [_norm(f) for f in parsed.files] if parsed else []

    # Paths the plan explicitly names — force-included so no planned file can be
    # left off the tree (and thus grounded away later). Weaker than workspace files
    # (real on disk) but treated as anchors so they survive double-root collapse.
    plan_paths = _plan_referenced_paths(state.get("plan"))

    # Reconcile: the tree MUST contain every existing workspace file (never relocate),
    # every spec-required path, and every path the plan references, on top of whatever
    # the model proposed.
    files: list[str] = []
    for f in model_files + [_norm(p) for p in workspace_files] + [_norm(p) for p in seed] + plan_paths:
        if f and not _is_junk(f) and f not in files:
            files.append(f)

    forced = [p for p in plan_paths if p not in model_files
              and p not in {_norm(x) for x in workspace_files}]
    if forced:
        _w(f"📌 Force-included **{len(forced)}** plan-referenced path(s) not proposed by the tree LLM.")

    # Collapse any accidental double root (e.g. 'backend/x' vs 'wrapper/backend/x').
    # Existing workspace + spec-required + plan-referenced paths are authoritative forms.
    anchors = ({_norm(p) for p in workspace_files}
               | {_norm(p) for p in seed}
               | set(plan_paths))
    before = len(files)
    files = _collapse_double_root(files, anchors)
    if len(files) < before:
        _w(f"🔧 Collapsed double-root duplicates: {before} → {len(files)} path(s).")

    # The project dir IS the root — never keep a top-level folder named after it.
    files = _strip_sandbox_wrapper(files, project_path)

    # Resolve file/folder collisions ('backend/models.py' vs 'backend/models/'),
    # so the architect is never offered the same logical layer as both a file and
    # a package. The package form wins; the stale flat module is dropped.
    before = len(files)
    files = _resolve_package_module_collisions(files)
    if len(files) < before:
        _w(f"🔧 Resolved package/module collisions: {before} → {len(files)} path(s).") # write to stream

    if not files:
        print("[Scaffold] Empty tree committed — downstream enforcement disabled for this run.")
        return {
            "objective": objective,
            "language": state.get("language", "python"),
            "context_store": {"workspace_files": workspace_files, "workspace_dirs": workspace_dirs,
                              **_SCAFFOLD_DONE_FLAGS},
            "history": ["scaffold_empty"],
        }

    # Write to stream for frontend
    file_list = "\n".join(f"  {f}" for f in files)
    writer({"kind": "text", "text": (
        f"### 📁 Canonical Tree — {len(files)} file(s)\n\n"
        f"```\n{file_list}\n```\n\n"
    )})

    # Provisional interface map (planned tier): the shared naming contract so a
    # producer and a consumer of a symbol agree even if written in either order.
    # Keyed only to paths that survived into the canonical tree; normalized to
    # match how the tree and downstream agents reference paths.
    tree_set = set(files)
    planned_interfaces: dict[str, list[str]] = {}
    if parsed and parsed.interfaces:
        for path, syms in parsed.interfaces.items():
            norm = _norm(path)
            if norm in tree_set and syms:
                planned_interfaces[norm] = syms

    # Option 2: seed CONFIRMED exports for files already on disk (resume / existing
    # codebase) by reading their real code. Confirmed overrides planned, so drop any
    # planned guess that is now backed by ground truth.
    confirmed_exports: dict[str, list[str]] = {}
    if _EXTRACT_EXISTING_EXPORTS and workspace_files:
        on_disk = {_norm(p) for p in workspace_files}
        scan_targets = [f for f in files if f in on_disk]
        if scan_targets:
            _w(f"🔍 Scanning {len(scan_targets)} existing file(s) for real exports...\n\n") # write to stream
            confirmed_exports = _extract_existing_exports(project_path, scan_targets, llm)
            if confirmed_exports:
                _w(f"✅ Extracted real exports from **{len(confirmed_exports)}** file(s).\n\n") # write to stream
    for k in confirmed_exports:
        planned_interfaces.pop(k, None)

    if planned_interfaces:
        _w(f"📐 Planned interfaces for **{len(planned_interfaces)}** file(s).") # write to stream

    ctx: dict = {
        "file_tree": files,
        "workspace_files": workspace_files,
        "workspace_dirs": workspace_dirs,
        "module_exports_planned": planned_interfaces,
        **_SCAFFOLD_DONE_FLAGS,
    }
    if confirmed_exports:
        ctx["module_exports"] = confirmed_exports

    return {
        "objective": objective,
        "language": state.get("language", "python"),
        "context_store": ctx,
        "history": ["scaffold"],
    }