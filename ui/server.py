"""
Web UI server for the code_review_agent.

A dependency-free (standard-library only) HTTP server that wraps the existing
LangChain/LangGraph review agent and exposes it to a browser chat front-end.

Why stdlib http.server instead of FastAPI/Flask?
  * The agent (`agents._agent`) is fully synchronous and streams via
    `_agent.stream(..., stream_mode="messages")`. A threaded blocking server
    maps onto that 1:1 with no async glue.
  * No pip install required -> works offline, nothing new to manage.

Run it (from anywhere):
    .venv\\Scripts\\python.exe ui\\server.py
then open http://localhost:8765 in a browser.

Endpoints
  GET  /                     -> index.html
  GET  /static/*             -> css / js
  POST /api/browse           -> native Windows folder picker, returns {path}
  GET  /api/sessions         -> all sessions (metadata + transcripts) for re-reading
  POST /api/session          -> create a new session {path, language} -> {id, ...}
  POST /api/chat             -> Server-Sent-Events stream of one agent turn
"""
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

# ── make the agent importable & give it the cwd it expects ──────────────────
# agents/__init__.py and tools open "config.yaml" / "tools.json" with relative
# paths, so the process MUST run with the project root as its working dir.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import agents  # noqa: E402  (after chdir/sys.path tweak, by design)
from langchain_core.messages import AIMessageChunk, ToolMessage, AIMessage  # noqa: E402
from tools.tools import restore_snapshot  # noqa: E402  (revert button backend)

STATIC_DIR = Path(__file__).resolve().parent / "static"
SESSIONS_FILE = Path(__file__).resolve().parent / "sessions.json"
HOST, PORT = "127.0.0.1", 8765

from structured_output import LANGUAGES  # noqa: E402  (single source of truth)
with open("config.yaml", "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)
MODELS = []
for agent in _cfg.get("agents", []):
    if isinstance(agent, dict):
        MODELS.append(agent["model"])
    else:
        MODELS.append(agent)
print(MODELS)

# Roles = the prompt "personalities" the user can switch between on the go
# (code-review, explore, ...). Keys of config.yaml's `prompt:` map.
ROLES = agents.ROLES
DEFAULT_ROLE = agents.DEFAULT_ROLE
print(ROLES)

# Tools each role can run; the UI shows a checkbox per name and the user can
# disable any of them per session (applied to the next message).
TOOLS_BY_ROLE = agents.TOOLS_BY_ROLE

# Whether each role needs a project folder. Drives the Chat/Project split: a chat
# session may only select/switch to roles where this is False, a project session
# only to roles where it is True.
ROLE_REQUIRES_PROJECT = agents.ROLE_REQUIRES_PROJECT

# Roles offered as the default for each mode (first matching role in config order).
_DEFAULT_CHAT_ROLE = next((r for r in ROLES if not ROLE_REQUIRES_PROJECT.get(r, True)), None)

# The orchestrator-graph role ("Coder" in the UI). It is a multi-node planner
# driven entirely by graph_config.yaml (its own per-node models + prompts), so the
# UI exposes it as its own mode with NO role/model pickers. An optional spec file
# rides along into the graph's context_store as spec_content. See agents/__init__.py
# (AGENTS["code-assistant"] = Orchestrator graph) for the wiring.
CODER_ROLE = "code-assistant"
# Model is config-driven for the coder graph; the value stored on the session only
# feeds the stats meters (context-window bar). Prefer the build default model.
_CODER_MODEL = "main_agent" if "main_agent" in MODELS else (MODELS[0] if MODELS else "")


def _tools_for_role(role: str) -> list[str]:
    return list(TOOLS_BY_ROLE.get(role, TOOLS_BY_ROLE.get(DEFAULT_ROLE, [])))


def _mode_for_role(role: str) -> str:
    """'coder' for the orchestrator-graph role, 'project' if the role needs a
    folder, else 'chat'."""
    if role == CODER_ROLE:
        return "coder"
    return "project" if ROLE_REQUIRES_PROJECT.get(role, True) else "chat"

# ── session store (in-memory, mirrored to disk for re-reading) ──────────────
# The agent's own InMemorySaver keeps the LangGraph state per thread_id while
# the process lives. We keep a parallel transcript here so old chats can be
# re-read in the UI (and survive a restart, read-only).
_sessions: dict[str, dict] = {}
_store_lock = threading.Lock()

# ── stop/cancel registry ────────────────────────────────────────────────────
# One threading.Event per session that currently has a turn streaming. The Stop
# button POSTs to /api/chat/stop, which sets the event; the stream loop in
# _run_turn checks it each chunk and breaks. agent.stream() is pull-based, so
# once we stop pulling the graph stops advancing — no further LLM/tool steps run.
_cancels: dict[str, threading.Event] = {}
_cancels_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_sessions() -> None:
    if SESSIONS_FILE.exists():
        try:
            data = json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
            for s in data.get("sessions", []):
                s["restored"] = True  # cosmetic badge only — no longer read-only
                # Back-fill fields added after this session was first saved.
                s.setdefault("role", DEFAULT_ROLE)
                s.setdefault("enabled_tools", _tools_for_role(s["role"]))
                # Sessions saved before chat mode all had folders -> project.
                s.setdefault("mode", _mode_for_role(s.get("role", DEFAULT_ROLE)))
                s.setdefault("spec_content", "")  # added with coder mode
                # If the SQLite checkpointer has state for this thread the session
                # is fully resumable; otherwise the next message re-initialises it.
                cfg = {"configurable": {"thread_id": s["id"]}}
                has_state = agents._checkpointer.get(cfg) is not None
                s["started"] = has_state
                _sessions[s["id"]] = s
        except Exception as e:  # noqa: BLE001
            print(f"[ui] could not load sessions.json: {e}")


def _save_sessions() -> None:
    with _store_lock:
        ordered = sorted(_sessions.values(), key=lambda s: s["created"])
        tmp = SESSIONS_FILE.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"sessions": ordered}, indent=2), encoding="utf-8"
        )
        tmp.replace(SESSIONS_FILE)


def _new_session(path: str, language: str, model: str, role: str,
                 mode: str = "project", spec_content: str = "") -> dict:
    sid = uuid.uuid4().hex[:12]
    # Chat sessions have no folder, so fall back to a friendly default title.
    title = Path(path).name or path or ("Chat" if mode == "chat" else "session")
    session = {
        "id": sid,
        "title": title,
        "path": path,
        "language": language,
        "model": model,
        "role": role,
        "mode": mode,            # 'chat' (folder-less), 'project', or 'coder'
        # Optional spec/context text for the coder graph; fed into the graph's
        # context_store as spec_content on the first turn (see _run_turn).
        "spec_content": spec_content,
        "enabled_tools": _tools_for_role(role),  # all on by default
        "created": _now_iso(),
        "messages": [],          # [{role, content, ts, usage?, elapsed?}]
        "started": False,        # has the first (stateful) turn run yet?
        "restored": False,
        "totals": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                   "turns": 0},
        "last": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                 "elapsed": 0.0},
    }
    _sessions[sid] = session
    _save_sessions()
    return session


# ── folder picker (native Windows dialog, run out-of-process) ───────────────
def _pick_folder() -> str | None:
    """Open a native folder-select dialog and return the chosen path.

    Run in a short-lived subprocess: tkinter is not thread-safe and the HTTP
    server handles each request on its own thread, so spawning a fresh process
    is the reliable cross-thread way to show the dialog.
    """
    code = (
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
        "p = filedialog.askdirectory(title='Select the project to review')\n"
        "print(p)\n"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=300,
        )
        path = out.stdout.strip()
        return path or None
    except Exception as e:  # noqa: BLE001
        print(f"[ui] folder picker failed: {e}")
        return None


# ── tool-call target extraction ─────────────────────────────────────────────
# Arg keys, in priority order, that name a file or directory a tool acted on.
_TARGET_KEYS = ("file_path", "path", "filepath", "filename", "directory", "dir_path")


def _extract_target(args_str: str) -> dict | None:
    """Parse accumulated tool-call args JSON and pull out the file/dir touched.

    Returns {"path", "dir", "name"} or None when no path-like arg is present
    (e.g. linter/test tools that operate on the injected project path).
    """
    if not args_str:
        return None
    try:
        data = json.loads(args_str)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    for key in _TARGET_KEYS:
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            val = val.strip()
            return {
                "path": val,
                "dir": os.path.dirname(val) or ".",
                "name": os.path.basename(val) or val,
            }
    # web_browse has no path-like arg — surface the site + query instead so the
    # chat bubble can show what was searched (mirrors read/write file labels).
    url = data.get("url")
    query = data.get("query")
    if (isinstance(url, str) and url.strip()) or (isinstance(query, str) and query.strip()):
        return {
            "url": url.strip() if isinstance(url, str) and url.strip() else None,
            "query": query.strip() if isinstance(query, str) and query.strip() else None,
        }
    return None


# ── attachment handling (drag-and-drop files from the UI) ───────────────────
# The browser sends each attachment as {name, type, size, data}, where `data`
# is a base64 data URL. We turn them into LangChain content blocks: images
# become multimodal image blocks (understood by vision-capable Ollama models),
# while text and PDF files are extracted to text and embedded inline so ANY
# model can use them. Unreadable binaries are noted but not sent.
_TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csv", ".tsv", ".html",
    ".css", ".scss", ".xml", ".sh", ".bat", ".ps1", ".java", ".c", ".h", ".cpp",
    ".hpp", ".cc", ".cs", ".go", ".rs", ".rb", ".php", ".sql", ".log", ".env",
    ".gitignore", ".dockerfile", ".kt", ".swift", ".r", ".lua", ".pl",
}
_TEXT_MIMES = {
    "application/json", "application/xml", "application/x-yaml",
    "application/javascript", "application/typescript", "application/x-sh",
    "application/x-python", "application/sql",
}
_MAX_ATTACH_TEXT_CHARS = 20_000  # per-file cap so a huge paste can't blow the context


def _data_url_bytes(data_url: str) -> "bytes | None":
    """Decode a `data:<mime>;base64,<payload>` URL (or bare base64) into bytes."""
    if not isinstance(data_url, str) or not data_url:
        return None
    try:
        payload = data_url.split(",", 1)[1] if "," in data_url else data_url
        return base64.b64decode(payload)
    except Exception:  # noqa: BLE001
        return None


def _looks_textual(name: str, mime: str) -> bool:
    return (
        mime.startswith("text/")
        or mime in _TEXT_MIMES
        or Path(name).suffix.lower() in _TEXT_EXTS
    )


def _extract_pdf_text(raw: bytes) -> "str | None":
    """Best-effort PDF text extraction; None if no parser is installed/usable."""
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:  # noqa: BLE001
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except Exception:  # noqa: BLE001
            return None
    try:
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except Exception:  # noqa: BLE001
        return None


def _attachment_blocks(att: dict) -> list[dict]:
    """Convert one UI attachment into one or more LangChain content blocks."""
    name = str(att.get("name") or "file")
    mime = str(att.get("type") or "")
    data_url = att.get("data") or ""

    # Images -> multimodal block (used by vision-capable Ollama models; a
    # text-only model simply ignores it).
    if mime.startswith("image/"):
        return [{"type": "image_url", "image_url": {"url": data_url}}]

    raw = _data_url_bytes(data_url)
    if raw is None:
        return [{"type": "text", "text": f"\n[attachment `{name}` could not be decoded]\n"}]

    # PDFs -> extracted text.
    if mime == "application/pdf" or name.lower().endswith(".pdf"):
        text = _extract_pdf_text(raw)
        if text:
            return [{"type": "text",
                     "text": f"\n[attached PDF: `{name}`]\n```\n"
                             f"{text[:_MAX_ATTACH_TEXT_CHARS]}\n```\n"}]
        return [{"type": "text",
                 "text": f"\n[attached PDF `{name}` — text could not be extracted "
                         f"(install `pypdf` to enable PDF reading)]\n"}]

    # Text / code files -> inline fenced block tagged with the file's extension.
    if _looks_textual(name, mime):
        text = raw.decode("utf-8", errors="replace")[:_MAX_ATTACH_TEXT_CHARS]
        lang = Path(name).suffix.lstrip(".").lower()
        return [{"type": "text",
                 "text": f"\n[attached file: `{name}`]\n```{lang}\n{text}\n```\n"}]

    # Anything else: raw binary can't be fed to a text model.
    return [{"type": "text",
             "text": f"\n[attachment `{name}` ({mime or 'unknown type'}) is binary "
                     f"and was not included]\n"}]


def _build_user_content(user_text: str, attachments: "list | None"):
    """Message content for one user turn.

    Returns a plain string when there are no attachments (the common case), or a
    list of content blocks (text + images + extracted file text) when there are.
    """
    if not attachments:
        return user_text
    blocks: list[dict] = []
    if user_text:
        blocks.append({"type": "text", "text": user_text})
    for att in attachments:
        if isinstance(att, dict):
            blocks.extend(_attachment_blocks(att))
    return blocks or user_text


def _attachment_meta(attachments: "list | None") -> list[dict]:
    """Lightweight {name,type,size} records for the saved transcript.

    The raw bytes are deliberately NOT persisted to sessions.json (they live in
    the SQLite checkpoint for the model's benefit); the transcript only needs
    enough to show a chip when an old session is re-opened.
    """
    meta: list[dict] = []
    for att in attachments or []:
        if isinstance(att, dict):
            meta.append({
                "name": str(att.get("name") or "file"),
                "type": str(att.get("type") or ""),
                "size": int(att.get("size") or 0),
            })
    return meta


# ── the streaming agent turn ────────────────────────────────────────────────
def _run_turn(session: dict, user_text: str, emit, attachments: "list | None" = None,
              cancel: "threading.Event | None" = None):
    """Run one agent turn, pushing SSE events through `emit(dict)`.

    Emits events: token, tool, status, done, error. If `cancel` is set partway
    through, the stream is stopped and whatever was produced so far is saved and
    returned as a (stopped) `done` event.
    """
    cancel = cancel or threading.Event()

    is_coder = session.get("role") == CODER_ROLE

    # The Coder role is a multi-node planner that runs autonomously to completion;
    # each user turn is a NEW objective, not a continuation of one chat thread.
    # Reusing a single thread_id leaks run-scoped state across objectives: the old
    # `objective` short-circuits scaffold's derivation, `iteration_count`
    # (operator.add) keeps climbing and can't be reset from input, and sticky
    # context_store flags (`skip_verification`, `reviewed_at_start`,
    # `abandoned_tasks`, `task_failure_counts`) poison the next run. So mint a fresh
    # thread_id per Coder turn — scaffold rebuilds file_tree/exports from disk, so
    # nothing material is lost. A UUID (not a per-session counter) guarantees the id
    # never collides with a leftover SQLite checkpoint after a restart. Other roles
    # keep the stable per-session thread so their conversation memory survives.
    thread_id = f"{session['id']}#{uuid.uuid4().hex}" if is_coder else session["id"]
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": agents._cfg.get("recursion_limit", 1000),
    }

    # Fold any drag-and-dropped files into the user message: images become
    # multimodal blocks, text/PDF files are extracted and embedded inline.
    content = _build_user_content(user_text, attachments)

    # A fresh thread starts with empty state, so the Coder graph needs the full
    # bootstrap payload (project_path, spec) on EVERY turn — not just the first.
    if is_coder or not session.get("started"):
        payload = {
            "messages": [{"role": "user", "content": content}],
            "project_path": session["path"],
            "model": session["model"],
        }
        # Coder sessions may carry an optional spec file: seed the orchestrator
        # graph's shared store so the scaffold and downstream agents can read it
        # under context_store["spec_content"]. Re-sent on every Coder turn because
        # the fresh thread's store is empty; for other roles it's sent only on the
        # first turn and persists in the checkpoint thereafter (merge_dicts reducer).
        spec = session.get("spec_content")
        if spec:
            payload["context_store"] = {"spec_content": spec}
        session["started"] = True
    else:
        payload = {"messages": [{"role": "user", "content": content}]}

    # Sent every turn so switching language mid-session (via the header switcher
    # or the set_language tool) takes effect immediately. session["language"] is
    # the canonical value; the read-back below folds any tool change back in.
    payload["language"] = session["language"]

    # Sent every turn so toggling tools mid-session takes effect immediately.
    # The tool-gate middleware restricts the model to exactly these names.
    payload["enabled_tools"] = session.get(
        "enabled_tools", _tools_for_role(session.get("role", DEFAULT_ROLE))
    )

    # Reset the mode-switch flag at the start of every turn so a stale value from
    # a previous stage can never re-trigger. Only a change_mode_* tool call during
    # THIS turn re-sets it; we read it back below to advance the pipeline.
    payload["review_mode"] = ""

    started = time.monotonic()
    answer_parts: list[str] = []
    # Ordered record of the turn so the UI can interleave text and tool bubbles
    # in stream order (and re-render that layout when the session is reopened).
    parts: list[dict] = []      # [{type:"text",content} | {type:"tool",name,target}]
    cur_text: list[str] = []    # buffer for the current (open) text segment

    def flush_text() -> None:
        if cur_text:
            parts.append({"type": "text", "content": "".join(cur_text)})
            cur_text.clear()

    last_usage: dict | None = None
    # The code-assistant orchestrator hides its internal model calls from the native
    # "messages" token stream, so it reports each call's usage on the custom
    # channel instead. Aggregate it here: sum output tokens (real generation
    # throughput across all four stages) and keep the peak input (how full the
    # context got) so tok/s and the context-window bar work for that role too.
    orch_out_tokens = 0
    orch_peak_input = 0
    orch_usage_seen = False
    active_tools: set[str] = set()
    # Accumulate streamed tool-call args (they arrive as JSON fragments) so we
    # can report which file/directory a tool touched once it completes.
    tool_args: dict[int, str] = {}     # tool_call index -> partial args JSON
    tool_index_by_id: dict[str, int] = {}  # tool_call id  -> index
    tool_index_by_name: dict[str, int] = {}  # tool name    -> latest index

    # Shared emit helpers so native streaming (below) and the code-assistant
    # orchestrator's "custom" channel produce identical token/tool SSE events and
    # transcript `parts`.
    def emit_text(text: str) -> None:
        if not text:
            return
        answer_parts.append(text)
        cur_text.append(text)
        emit({"type": "token", "text": text})

    def emit_tool_start(name: str, target: dict | None = None) -> None:
        if name in active_tools:
            # A previous call to this tool never reported completion (its result
            # was dropped, or a stage errored before the end event). Close it out
            # now so its pill stops spinning AND this fresh call isn't suppressed
            # — otherwise the stuck name permanently blocks every later call.
            emit_tool_end(name)
        active_tools.add(name)
        flush_text()  # text before the tool call ends its own segment here
        emit({"type": "tool", "name": name, "phase": "start"})

    def emit_tool_end(name: str, target: dict | None = None) -> None:
        active_tools.discard(name)
        flush_text()
        parts.append({"type": "tool", "name": name, "target": target})
        emit({"type": "tool", "name": name, "phase": "end", "target": target})

    stopped = False
    try:
        agent = agents.graph_for(session.get("role"), session.get("model"))
        # Keep a handle on the generator so we can close() it promptly on cancel,
        # which sends GeneratorExit into the graph and stops it advancing.
        # "messages" carries native LLM tokens/tool calls; "custom" carries the
        # code-assistant orchestrator's per-stage progress (its own grammar-constrained
        # JSON messages are suppressed below so they never reach the UI raw).
        stream = agent.stream(payload, config=config,
                              stream_mode=["messages", "custom"])
        for mode, data in stream:
            # User hit Stop: quit pulling, close the generator, keep partial work.
            if cancel.is_set():
                stopped = True
                stream.close()
                break

            # Orchestrator stages render through the custom channel: clean reasoning
            # text + tool pills, already shaped like the native events above.
            if mode == "custom":
                if isinstance(data, dict):
                    kind = data.get("kind")
                    if kind == "text":
                        emit_text(data.get("text", ""))
                    elif kind == "tool_start":
                        emit_tool_start(data.get("name", "tool"), data.get("target"))
                    elif kind == "tool_end":
                        emit_tool_end(data.get("name", "tool"), data.get("target"))
                    elif kind == "stage":
                        # The code-assistant orchestrator switched to a new pipeline
                        # stage. End the current text segment, record the handoff
                        # so it re-renders, and push a dedicated stage bubble.
                        flush_text()
                        label = data.get("label") or data.get("stage") or "next stage"
                        parts.append({"type": "stage", "label": label})
                        emit({"type": "stage", "label": label,
                              "stage": data.get("stage")})
                    elif kind == "usage":
                        # One orchestrator model call's token usage. Sum the output
                        # (throughput) and track the peak input (context fill) so the
                        # done event below can drive the meters like a native turn.
                        u = data.get("usage") or {}
                        orch_usage_seen = True
                        orch_out_tokens += int(u.get("output_tokens", 0) or 0)
                        orch_peak_input = max(
                            orch_peak_input, int(u.get("input_tokens", 0) or 0))
                        # Push a running snapshot so the stats meters update live as
                        # the graph runs, instead of only on the final `done` event.
                        # Mirrors the `done` aggregation (peak input + summed output)
                        # and previews the session totals as saved + running.
                        running = {
                            "input_tokens": orch_peak_input,
                            "output_tokens": orch_out_tokens,
                            "total_tokens": orch_peak_input + orch_out_tokens,
                        }
                        t = session["totals"]
                        live_totals = {
                            "input_tokens": t["input_tokens"] + running["input_tokens"],
                            "output_tokens": t["output_tokens"] + running["output_tokens"],
                            "total_tokens": t["total_tokens"] + running["total_tokens"],
                            "turns": t["turns"] + 1,
                        }
                        emit({
                            "type": "usage",
                            "usage": running,
                            "totals": live_totals,
                            "context_window": agents._cfg["agents"].get(
                                session["model"], {}).get("context_window"),
                        })
                continue

            chunk, _meta = data

            # Orchestrator pipeline stages talk to the UI via the "custom" channel; their
            # native LLM token stream is grammar-constrained JSON — drop it here.
            is_graph_node = (_meta or {}).get("langgraph_node") in {
                "scaffold", "orchestrator", "inspector", "architect", "coder",
                "validator", "step_dispatch",
            }

            if is_graph_node:
                # Only let tool-related chunks through — drop raw text content
                is_tool_chunk = isinstance(chunk, AIMessageChunk) and bool(chunk.tool_call_chunks)
                is_tool_result = isinstance(chunk, ToolMessage)
                if not is_tool_chunk and not is_tool_result:
                    continue
            # Full AIMessage (non-streaming) with tool calls — graph nodes deliver these
            # as complete messages rather than chunks, so register IDs here too.
            if isinstance(chunk, AIMessage) and not isinstance(chunk, AIMessageChunk):
                for tc in (chunk.tool_calls or []):
                    name = tc.get("name", "")
                    idx = len(tool_args)
                    tool_index_by_id[tc.get("id", "")] = idx
                    tool_index_by_name[name] = idx
                    tool_args[idx] = json.dumps(tc.get("args", {}))
                    emit_tool_start(name)
                continue

            # Tool result coming back
            if isinstance(chunk, ToolMessage):
                name = chunk.name or "tool"
                tool_call_id = getattr(chunk, "tool_call_id", None)
                idx = tool_index_by_id.get(tool_call_id)
                if idx is None:
                    idx = tool_index_by_name.get(name)
                target = _extract_target(tool_args.get(idx, "")) if idx is not None else None
                # Force-add to active_tools if it never got a start event
                # (happens when the AIMessage arrived as a complete message, not chunks)
                if name not in active_tools:
                    emit_tool_start(name, target)
                emit_tool_end(name, target)
                continue

            if isinstance(chunk, AIMessageChunk):
                # Tool call(s) being requested
                for tc in (chunk.tool_call_chunks or []):
                    idx = tc.get("index") or 0
                    if tc.get("id"):
                        tool_index_by_id[tc["id"]] = idx
                        # A new tool-call id reuses index 0 on each ReAct step;
                        # reset the accumulator so this call's args aren't
                        # concatenated onto the previous call's (which produced
                        # invalid JSON -> no target -> a generic "project" label
                        # for every read/write after the first).
                        tool_args[idx] = ""
                    frag = tc.get("args")
                    if frag:
                        tool_args[idx] = tool_args.get(idx, "") + frag
                    name = tc.get("name")
                    if name:
                        tool_index_by_name[name] = idx
                        emit_tool_start(name)

                # Streamed answer text
                text = chunk.content
                if isinstance(text, list):  # some providers chunk content as parts
                    text = "".join(
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in text
                    )
                emit_text(text)

                # Token accounting (Ollama fills this on the final chunk)
                if getattr(chunk, "usage_metadata", None):
                    last_usage = chunk.usage_metadata

        flush_text()  # trailing text segment after the last tool (or whole answer)
        # Any tool still marked active never produced an end event — close them
        # out so no pill is left spinning once the turn (or a Stop) finishes.
        for stuck in list(active_tools):
            emit_tool_end(stuck)
        elapsed = round(time.monotonic() - started, 2)
        answer = "".join(answer_parts).strip()

        # The set_language tool writes language into the graph state. Fold any
        # such change back into the session so it persists and the header
        # switcher (via the `done` event below) reflects reality.
        try:
            final_lang = agent.get_state(config).values.get("language")
            if final_lang and final_lang != session.get("language"):
                session["language"] = final_lang
        except Exception:  # noqa: BLE001
            pass

        # A change_mode_* tool may have requested the next pipeline stage by
        # writing `review_mode` into graph state. Read it back and, if it names a
        # different role valid for THIS session's mode, flag the switch so the
        # caller (_handle_chat) can advance the role and auto-run the next stage.
        # The mode check keeps a folder-less chat session from ever jumping into a
        # project-only stage.
        pending_mode = None
        try:
            requested = (agent.get_state(config).values.get("review_mode") or "").strip()
            cur = session.get("role", DEFAULT_ROLE)
            if (requested in ROLES and requested != cur
                    and _mode_for_role(requested) == session.get("mode", "project")):
                pending_mode = requested
        except Exception:  # noqa: BLE001
            pass

        if orch_usage_seen:
            # code-assistant pipeline: peak context fill + summed generation.
            usage = {
                "input_tokens": orch_peak_input,
                "output_tokens": orch_out_tokens,
                "total_tokens": orch_peak_input + orch_out_tokens,
            }
        else:
            usage = {
                "input_tokens": (last_usage or {}).get("input_tokens", 0),
                "output_tokens": (last_usage or {}).get("output_tokens", 0),
                "total_tokens": (last_usage or {}).get("total_tokens", 0),
            }

        # Persist transcript + stats
        ts = _now_iso()
        session["messages"].append(
            {"role": "user", "content": user_text, "ts": session.get(
                "_pending_user_ts", ts),
             "attachments": _attachment_meta(attachments)}
        )
        session["messages"].append(
            {"role": "assistant", "content": answer, "ts": ts,
             "usage": usage, "elapsed": elapsed, "parts": parts}
        )
        session["last"] = {**usage, "elapsed": elapsed}
        t = session["totals"]
        t["input_tokens"] += usage["input_tokens"]
        t["output_tokens"] += usage["output_tokens"]
        t["total_tokens"] += usage["total_tokens"]
        t["turns"] += 1
        _save_sessions()

        emit({
            "type": "done",
            "answer": answer,
            "parts": parts,
            "stopped": stopped,
            "usage": usage,
            "elapsed": elapsed,
            "totals": session["totals"],
            "last": session["last"],
            "language": session["language"],
            "role": session.get("role"),     # the stage that produced this turn
            "next_mode": pending_mode,        # stage about to auto-run, or None
            "context_window": agents._cfg["agents"].get(
                session["model"], {}).get("context_window"),
            "ts": ts,
        })
        return pending_mode
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
        return None


# ── HTTP handler ────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    server_version = "CodeReviewUI/1.0"

    def log_message(self, fmt, *args):  # quieter console
        pass

    # -- helpers --
    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str):
        if not path.exists():
            self._send_json({"error": "not found"}, 404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # Never cache the UI assets: they change during development and a stale
        # mix of old CSS / new JS produces confusing, inconsistent behaviour.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    # Serve any file under static/ (app modules, css, …), guarding against path
    # traversal and picking a content type from the extension.
    _STATIC_TYPES = {
        ".js":   "application/javascript; charset=utf-8",
        ".mjs":  "application/javascript; charset=utf-8",
        ".css":  "text/css; charset=utf-8",
        ".html": "text/html; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".svg":  "image/svg+xml",
    }

    def _serve_static(self, rel: str):
        target = (STATIC_DIR / rel).resolve()
        if not target.is_relative_to(STATIC_DIR.resolve()):
            self._send_json({"error": "not found"}, 404)  # path traversal attempt
            return
        ctype = self._STATIC_TYPES.get(target.suffix.lower(), "application/octet-stream")
        self._send_file(target, ctype)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:  # noqa: BLE001
            return {}

    # -- routing --
    def do_GET(self):
        route = self.path.split("?", 1)[0]
        if route == "/":
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif route.startswith("/static/"):
            self._serve_static(route[len("/static/"):])
        elif route == "/api/sessions":
            self._send_json({
                "sessions": sorted(_sessions.values(),
                                   key=lambda s: s["created"]),
                "languages": LANGUAGES,
                "models": MODELS,
                "roles": ROLES,
                "tools_by_role": TOOLS_BY_ROLE,
                "role_modes": {r: _mode_for_role(r) for r in ROLES},
            })
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        route = self.path.split("?", 1)[0]
        if route == "/api/browse":
            path = _pick_folder()
            self._send_json({"path": path})
        elif route == "/api/session":
            data = self._read_body()
            mode = (data.get("mode") or "project").strip().lower()
            if mode not in ("project", "chat", "coder"):
                self._send_json({"error": f"unknown mode '{mode}'"}, 400)
                return
            path = (data.get("path") or "").strip()
            language = (data.get("language") or "").strip().lower()
            model = (data.get("model") or "").strip().lower()
            spec = data.get("spec")
            spec = spec if isinstance(spec, str) else ""
            role = (data.get("role")
                    or (_DEFAULT_CHAT_ROLE if mode == "chat" else DEFAULT_ROLE)
                    or DEFAULT_ROLE).strip()
            # Coder mode is driven entirely by graph_config.yaml: the role + model
            # are fixed here (the UI offers no pickers for them), and an optional
            # spec file rides along into the graph's context_store.
            if mode == "coder":
                role = CODER_ROLE
                model = _CODER_MODEL
                language = language or "english"
            # Chat sessions are folder-less and default to English; project and
            # coder sessions still require a real path + language.
            if mode == "chat":
                path = ""
                language = language or "english"
            elif not path:
                self._send_json({"error": "path is required"}, 400)
                return
            if language not in LANGUAGES:
                self._send_json({"error": f"unknown language '{language}'"}, 400)
                return
            if model not in MODELS:
                self._send_json({"error": f"unknown model '{model}'"}, 400)
                return
            if role not in ROLES:
                self._send_json({"error": f"unknown role '{role}'"}, 400)
                return
            if _mode_for_role(role) != mode:
                self._send_json(
                    {"error": f"role '{role}' is not available in {mode} mode"}, 400)
                return
            # The spec only feeds the coder graph; ignore it for other modes.
            session = _new_session(path, language, model, role, mode,
                                   spec_content=spec if mode == "coder" else "")
            self._send_json({"session": session})
        elif route == "/api/session/model":
            data = self._read_body()
            sid = data.get("id")
            new_model = (data.get("model") or "").strip()
            session = _sessions.get(sid)
            if not session:
                self._send_json({"error": "unknown session"}, 404)
                return
            if new_model not in MODELS:
                self._send_json({"error": f"unknown model '{new_model}'"}, 400)
                return
            session["model"] = new_model
            _save_sessions()
            self._send_json({"ok": True, "model": new_model})
        elif route == "/api/session/language":
            data = self._read_body()
            sid = data.get("id")
            new_language = (data.get("language") or "").strip().lower()
            session = _sessions.get(sid)
            if not session:
                self._send_json({"error": "unknown session"}, 404)
                return
            if new_language not in LANGUAGES:
                self._send_json({"error": f"unknown language '{new_language}'"}, 400)
                return
            # Picked up on the next turn, which re-injects session["language"]
            # into the graph state (see _run_turn).
            session["language"] = new_language
            _save_sessions()
            self._send_json({"ok": True, "language": new_language})
        elif route == "/api/session/role":
            data = self._read_body()
            sid = data.get("id")
            new_role = (data.get("role") or "").strip()
            session = _sessions.get(sid)
            if not session:
                self._send_json({"error": "unknown session"}, 404)
                return
            if new_role not in ROLES:
                self._send_json({"error": f"unknown role '{new_role}'"}, 400)
                return
            # Session type is fixed at creation: a chat session can only switch
            # among chat roles, a project session only among project roles. This
            # is what keeps a folder-less chat from ever reaching a file tool.
            session_mode = session.get("mode", _mode_for_role(session.get("role", DEFAULT_ROLE)))
            if _mode_for_role(new_role) != session_mode:
                self._send_json(
                    {"error": f"role '{new_role}' is not available in "
                              f"{session_mode} mode"}, 400)
                return
            # Same thread_id -> the next turn keeps the full history but is
            # handled by the new role's prompt (and tools). Roles can expose
            # different tools, so reset the toggle set to the new role's full
            # list (all enabled).
            session["role"] = new_role
            session["enabled_tools"] = _tools_for_role(new_role)
            _save_sessions()
            self._send_json({
                "ok": True,
                "role": new_role,
                "tools": _tools_for_role(new_role),       # selectable set
                "enabled_tools": session["enabled_tools"],  # currently enabled
            })
        elif route == "/api/session/tools":
            data = self._read_body()
            sid = data.get("id")
            requested = data.get("enabled_tools")
            session = _sessions.get(sid)
            if not session:
                self._send_json({"error": "unknown session"}, 404)
                return
            if not isinstance(requested, list):
                self._send_json({"error": "enabled_tools must be a list"}, 400)
                return
            # Keep only names this session's role actually offers, preserving the
            # role's canonical tool order.
            allowed = _tools_for_role(session.get("role", DEFAULT_ROLE))
            session["enabled_tools"] = [t for t in allowed if t in requested]
            _save_sessions()
            self._send_json({"ok": True, "enabled_tools": session["enabled_tools"]})
        elif route == "/api/session/revert":
            data = self._read_body()
            sid = data.get("id")
            # A single path (from a write bubble) or None to revert everything.
            path = data.get("path")
            session = _sessions.get(sid)
            if not session:
                self._send_json({"error": "unknown session"}, 404)
                return
            paths = [path] if isinstance(path, str) and path.strip() else None
            restored, deleted = restore_snapshot(session["path"], paths)
            if not restored and not deleted:
                self._send_json(
                    {"error": "Nothing to revert — no snapshot for this file "
                              "(it may predate this server run)."}, 404)
                return
            self._send_json({"ok": True, "restored": restored, "deleted": deleted})
        elif route == "/api/session/delete":
            data = self._read_body()
            sid = data.get("id")
            removed = _sessions.pop(sid, None) is not None
            if removed:
                _save_sessions()
                self._send_json({"ok": True})
            else:
                self._send_json({"error": "unknown session"}, 404)
        elif route == "/api/chat":
            self._handle_chat()
        elif route == "/api/chat/stop":
            data = self._read_body()
            sid = data.get("session_id")
            with _cancels_lock:
                ev = _cancels.get(sid)
            if ev is not None:
                ev.set()
                self._send_json({"ok": True})
            else:
                self._send_json({"ok": False, "error": "no active turn"}, 404)
        else:
            self._send_json({"error": "not found"}, 404)

    # -- SSE chat --
    def _handle_chat(self):
        data = self._read_body()
        sid = data.get("session_id")
        message = (data.get("message") or "").strip()
        attachments = data.get("attachments")
        if not isinstance(attachments, list):
            attachments = []
        session = _sessions.get(sid)
        if not session:
            self._send_json({"error": "unknown session"}, 404)
            return
        if not message and not attachments:
            self._send_json({"error": "empty message"}, 400)
            return

        session["_pending_user_ts"] = _now_iso()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        lock = threading.Lock()

        def emit(obj: dict):
            with lock:
                try:
                    self.wfile.write(
                        f"data: {json.dumps(obj)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        # Heartbeat so the browser timer/UI knows the stream is alive even
        # while a long tool call produces no tokens.
        stop = threading.Event()

        def heartbeat():
            while not stop.wait(2.0):
                emit({"type": "status", "alive": True})

        hb = threading.Thread(target=heartbeat, daemon=True)
        hb.start()

        # Register a cancel flag the Stop button can trip for this session.
        cancel = threading.Event()
        with _cancels_lock:
            _cancels[sid] = cancel
        try:
            # Pipeline auto-advance: a stage may call a change_mode_* tool to hand
            # off to the next stage. _run_turn returns the requested next role (or
            # None); when set, we switch the session's role and re-run with the new
            # stage's kickoff message. Bounded so a plan<->act ping-pong on repeated
            # failures can't loop forever.
            MAX_STAGES = 8
            text, atts = message, attachments
            for _ in range(MAX_STAGES):
                pending = _run_turn(session, text, emit, atts, cancel)
                if not pending or cancel.is_set():
                    break
                session["role"] = pending
                session["enabled_tools"] = _tools_for_role(pending)
                _save_sessions()
                text = agents.AGENTS[pending].kickoff_message
                atts = None     # auto-stage kickoffs carry no attachments
        finally:
            stop.set()
            with _cancels_lock:
                if _cancels.get(sid) is cancel:
                    del _cancels[sid]


class _Server(ThreadingHTTPServer):
    # Refuse to share the port: without this, a second `python server.py` can
    # silently bind the same port and requests get split between two processes
    # with divergent in-memory session state.
    allow_reuse_address = False


def main():
    _load_sessions()
    STATIC_DIR.mkdir(exist_ok=True)
    try:
        httpd = _Server((HOST, PORT), Handler)
    except OSError:
        print(f"[ui] ERROR: port {PORT} is already in use — a server is "
              f"probably already running. Close it first, then retry.")
        sys.exit(1)
    url = f"http://{HOST}:{PORT}"
    print(f"[ui] code_review_agent UI running at {url}")
    print(f"[ui] project root: {PROJECT_ROOT}")
    print("[ui] press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[ui] shutting down.")
        httpd.shutdown()


if __name__ == "__main__":
    main()
