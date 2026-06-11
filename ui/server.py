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
from langchain_core.messages import AIMessageChunk, ToolMessage  # noqa: E402
from tools.tools import restore_snapshot  # noqa: E402  (revert button backend)

STATIC_DIR = Path(__file__).resolve().parent / "static"
SESSIONS_FILE = Path(__file__).resolve().parent / "sessions.json"
HOST, PORT = "127.0.0.1", 8765

LANGUAGES = ["english","python", "javascript", "typescript", "go", "rust", "java"]
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
ROLES = list(_cfg.get("prompt", {}))
DEFAULT_ROLE = agents.DEFAULT_ROLE
print(ROLES)

# Tools each role can run; the UI shows a checkbox per name and the user can
# disable any of them per session (applied to the next message).
TOOLS_BY_ROLE = agents.TOOLS_BY_ROLE


def _tools_for_role(role: str) -> list[str]:
    return list(TOOLS_BY_ROLE.get(role, TOOLS_BY_ROLE.get(DEFAULT_ROLE, [])))

# ── session store (in-memory, mirrored to disk for re-reading) ──────────────
# The agent's own InMemorySaver keeps the LangGraph state per thread_id while
# the process lives. We keep a parallel transcript here so old chats can be
# re-read in the UI (and survive a restart, read-only).
_sessions: dict[str, dict] = {}
_store_lock = threading.Lock()


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


def _new_session(path: str, language: str, model: str, role: str) -> dict:
    sid = uuid.uuid4().hex[:12]
    title = Path(path).name or path or "session"
    session = {
        "id": sid,
        "title": title,
        "path": path,
        "language": language,
        "model": model,
        "role": role,
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
    return None


# ── the streaming agent turn ────────────────────────────────────────────────
def _run_turn(session: dict, user_text: str, emit):
    """Run one agent turn, pushing SSE events through `emit(dict)`.

    Emits events: token, tool, status, done, error.
    """
    config = {
        "configurable": {"thread_id": session["id"]},
        "recursion_limit": agents._cfg.get("recursion_limit", 1000),
    }

    if not session.get("started"):
        payload = {
            "messages": [{"role": "user", "content": user_text}],
            "project_path": session["path"],
            "language": session["language"],
            "model": session["model"],
        }
        session["started"] = True
    else:
        payload = {"messages": [{"role": "user", "content": user_text}]}

    # Sent every turn so toggling tools mid-session takes effect immediately.
    # The tool-gate middleware restricts the model to exactly these names.
    payload["enabled_tools"] = session.get(
        "enabled_tools", _tools_for_role(session.get("role", DEFAULT_ROLE))
    )

    started = time.monotonic()
    answer_parts: list[str] = []
    last_usage: dict | None = None
    active_tools: set[str] = set()
    # Accumulate streamed tool-call args (they arrive as JSON fragments) so we
    # can report which file/directory a tool touched once it completes.
    tool_args: dict[int, str] = {}     # tool_call index -> partial args JSON
    tool_index_by_id: dict[str, int] = {}  # tool_call id  -> index
    tool_index_by_name: dict[str, int] = {}  # tool name    -> latest index

    try:
        agent = agents.graph_for(session.get("role"), session.get("model"))
        for chunk, _meta in agent.stream(
            payload, config=config, stream_mode="messages"
        ):
            # Tool result coming back
            if isinstance(chunk, ToolMessage):
                name = chunk.name or "tool"
                active_tools.discard(name)
                idx = tool_index_by_id.get(getattr(chunk, "tool_call_id", None))
                if idx is None:
                    idx = tool_index_by_name.get(name)
                target = _extract_target(tool_args.get(idx, "")) if idx is not None else None
                emit({"type": "tool", "name": name, "phase": "end", "target": target})
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
                        if name not in active_tools:
                            active_tools.add(name)
                            emit({"type": "tool", "name": name, "phase": "start"})

                # Streamed answer text
                text = chunk.content
                if isinstance(text, list):  # some providers chunk content as parts
                    text = "".join(
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in text
                    )
                if text:
                    answer_parts.append(text)
                    emit({"type": "token", "text": text})

                # Token accounting (Ollama fills this on the final chunk)
                if getattr(chunk, "usage_metadata", None):
                    last_usage = chunk.usage_metadata

        elapsed = round(time.monotonic() - started, 2)
        answer = "".join(answer_parts).strip()

        usage = {
            "input_tokens": (last_usage or {}).get("input_tokens", 0),
            "output_tokens": (last_usage or {}).get("output_tokens", 0),
            "total_tokens": (last_usage or {}).get("total_tokens", 0),
        }

        # Persist transcript + stats
        ts = _now_iso()
        session["messages"].append(
            {"role": "user", "content": user_text, "ts": session.get(
                "_pending_user_ts", ts)}
        )
        session["messages"].append(
            {"role": "assistant", "content": answer, "ts": ts,
             "usage": usage, "elapsed": elapsed}
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
            "usage": usage,
            "elapsed": elapsed,
            "totals": session["totals"],
            "last": session["last"],
            "context_window": agents._cfg["agents"].get(
                session["model"], {}).get("context_window"),
            "ts": ts,
        })
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        emit({"type": "error", "message": f"{type(e).__name__}: {e}"})


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
        elif route == "/static/app.js":
            self._send_file(STATIC_DIR / "app.js",
                            "application/javascript; charset=utf-8")
        elif route == "/static/style.css":
            self._send_file(STATIC_DIR / "style.css", "text/css; charset=utf-8")
        elif route == "/api/sessions":
            self._send_json({
                "sessions": sorted(_sessions.values(),
                                   key=lambda s: s["created"]),
                "languages": LANGUAGES,
                "models": MODELS,
                "roles": ROLES,
                "tools_by_role": TOOLS_BY_ROLE,
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
            path = (data.get("path") or "").strip()
            language = (data.get("language") or "").strip().lower()
            model = (data.get("model") or "").strip().lower()
            role = (data.get("role") or DEFAULT_ROLE).strip()
            if not path:
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
            session = _new_session(path, language, model, role)
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
        else:
            self._send_json({"error": "not found"}, 404)

    # -- SSE chat --
    def _handle_chat(self):
        data = self._read_body()
        sid = data.get("session_id")
        message = (data.get("message") or "").strip()
        session = _sessions.get(sid)
        if not session:
            self._send_json({"error": "unknown session"}, 404)
            return
        if not message:
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
        try:
            _run_turn(session, message, emit)
        finally:
            stop.set()


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
