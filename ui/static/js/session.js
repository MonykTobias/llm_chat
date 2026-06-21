"use strict";

import { state, activeSession } from "./state.js";
import { $, el, fmtTime, fillSelect, fillSelectOnce, scrollMessages } from "./dom.js";
import { getJSON, postJSON } from "./api.js";
import { renderTools } from "./tools.js";
import { renderStatsFor, resetStats, setStatus } from "./stats.js";
import { renderAssistantTurn, addMessageBubble } from "./messages.js";
import { clearSpec } from "./attachments.js";

// ── Loading sessions + populating the new-session form ─────────────────
export async function refreshSessions() {
  try {
    const data = await getJSON("/api/sessions");
    state.sessions = data.sessions || [];
    state.languages = data.languages || [];
    state.models = data.models || [];
    state.roles = data.roles || [];
    state.toolsByRole = data.tools_by_role || {};
    state.roleModes = data.role_modes || {};
    fillSelectOnce($("language-select"), state.languages);
    fillSelectOnce($("model-select"), state.models);
    setCreateMode(state.createMode);  // also (re)populates the role dropdown
    renderSessionList();
    if (!state.activeId && state.sessions.length) {
      selectSession(state.sessions[state.sessions.length - 1].id);
    } else if (state.activeId) {
      renderActive();
    }
  } catch (e) {
    console.error("Failed to fetch sessions:", e);
  }
}

// Roles offered for a given mode (chat roles need no folder, project roles do).
function rolesForMode(mode) {
  return state.roles.filter((r) => (state.roleModes[r] || "project") === mode);
}

// Switch the new-session form between "project", "chat" and "coder": highlight the
// chosen button, show/hide the folder + language fields, the role/model pickers and
// the coder spec drop zone, refilter roles, relabel Start.
export function setCreateMode(mode) {
  state.createMode = mode;
  for (const btn of document.querySelectorAll("#mode-toggle .mode-btn")) {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  }
  // Chat needs no folder; coder needs one. Both project + coder show the folder.
  $("project-fields").classList.toggle("hidden", mode === "chat");
  // Coder is driven entirely by graph_config.yaml — no role/model choice. It gets
  // an optional spec file instead.
  $("role-model-fields").classList.toggle("hidden", mode === "coder");
  $("spec-fields").classList.toggle("hidden", mode !== "coder");
  fillSelect($("role-select"), rolesForMode(mode));
  $("start-btn").textContent = startLabel(mode);
}

const startLabel = (mode) =>
  mode === "chat" ? "Start chat" : mode === "coder" ? "Start coder" : "Start review";

// ── Folder picker (native dialog via backend) ──────────────────────────
export async function browseFolder() {
  const btn = $("browse-btn");
  btn.disabled = true;
  btn.textContent = "…";
  try {
    const data = await postJSON("/api/browse");
    if (data.path) $("path-input").value = data.path;
  } catch (e) {
    console.error(e);
  } finally {
    btn.disabled = false;
    btn.textContent = "📁";
  }
}

// ── Create a new session ───────────────────────────────────────────────
export async function startSession() {
  const mode = state.createMode;
  const isChat = mode === "chat";
  const isCoder = mode === "coder";
  const path = $("path-input").value.trim();
  const errBox = $("new-session-error");
  errBox.textContent = "";
  if (!isChat && !path) { errBox.textContent = "Please choose a project path."; return; }

  const btn = $("start-btn");
  btn.disabled = true;
  btn.textContent = "Starting…";
  try {
    const data = await postJSON("/api/session", {
      mode,
      path: isChat ? "" : path,
      language: $("language-select").value,
      // Coder mode ignores these server-side (role + model come from
      // graph_config.yaml); send them anyway for the other modes.
      model: $("model-select").value,
      role: $("role-select").value,
      // Optional spec/context file text, only meaningful for coder mode.
      spec: isCoder ? state.specContent : "",
    });
    if (data.error) { errBox.textContent = data.error; return; }
    state.sessions.push(data.session);
    renderSessionList();
    selectSession(data.session.id);
    $("path-input").value = "";
    if (isCoder) clearSpec();  // hand the spec off to this session; reset the form
  } catch (e) {
    errBox.textContent = "Could not start session.";
    console.error(e);
  } finally {
    btn.disabled = false;
    btn.textContent = startLabel(mode);
  }
}

// ── Session list / switching ───────────────────────────────────────────
function renderSessionList() {
  const ul = $("session-list");
  ul.innerHTML = "";
  for (const s of state.sessions) {
    const active = s.id === state.activeId;
    const li = el("li", "session-item" + (active ? " active" : ""));

    const info = el("div", "flex-1 min-w-0");
    info.appendChild(el("div", "session-title truncate", s.title));
    const sMode = s.mode || "project";
    const label = sMode === "chat" ? "chat" : sMode === "coder" ? "coder" : s.language;
    const meta = `${label} · ${fmtTime(s.created)}`;
    info.appendChild(el("div", "session-meta truncate",
      meta + (s.restored ? " · restored" : "")));
    info.addEventListener("click", () => selectSession(s.id));

    const del = el("button", "session-del", "✕");
    del.title = "Delete this session";
    del.addEventListener("click", (e) => { e.stopPropagation(); deleteSession(s.id); });

    li.appendChild(info);
    li.appendChild(del);
    ul.appendChild(li);
  }
}

function selectSession(id) {
  state.activeId = id;
  renderSessionList();
  renderActive();
}

async function deleteSession(id) {
  if (state.streaming && state.streamingId === id) {
    alert("This session is still generating a response — wait for it to finish before deleting.");
    return;
  }
  if (!confirm("Delete this session? This cannot be undone.")) return;
  try {
    const data = await postJSON("/api/session/delete", { id });
    if (data.error) { alert(data.error); return; }
    state.sessions = state.sessions.filter((s) => s.id !== id);
    if (state.activeId === id) {
      state.activeId = null;
      if (state.sessions.length) {
        selectSession(state.sessions[state.sessions.length - 1].id);
      } else {
        renderActive();
      }
    }
    renderSessionList();
  } catch (e) {
    console.error(e);
    alert("Could not delete the session.");
  }
}

// The three header switchers share one flow: POST the change, mirror the server's
// echoed value onto the session, and refresh the subtitle. `apply` handles the
// per-field bookkeeping (role also resets its tool set).
async function switchAttr(endpoint, key, value, apply) {
  const s = activeSession();
  if (!s || s.restored) return;
  try {
    const data = await postJSON(endpoint, { id: s.id, [key]: value });
    if (data.error) { alert(data.error); return; }
    apply(s, data);
    $("chat-subtitle").textContent = subtitleFor(s);
  } catch (e) {
    console.error(e);
    alert(`Could not switch ${key}.`);
  }
}

export const switchModel = (v) =>
  switchAttr("/api/session/model", "model", v, (s, d) => { s.model = d.model; });

export const switchLanguage = (v) =>
  switchAttr("/api/session/language", "language", v, (s, d) => { s.language = d.language; });

export const switchRole = (v) =>
  switchAttr("/api/session/role", "role", v, (s, d) => {
    s.role = d.role;
    if (d.enabled_tools) s.enabled_tools = d.enabled_tools;  // reset for new role
    renderTools(s);
  });

// One source of truth for the header subtitle. Chat sessions have no folder, so
// they lead with "Chat" instead of a path/language.
export function subtitleFor(s) {
  const mode = s.mode || "project";
  // Coder is config-driven (no user-chosen role/model): lead with the folder only.
  if (mode === "coder") return `Coder  ·  ${s.path}`;
  const lead = mode === "chat" ? "Chat" : `${s.path}  ·  ${s.language}`;
  return `${lead}  ·  ${s.model}` + (s.role ? `  ·  ${s.role}` : "");
}

// ── Rendering the active session ───────────────────────────────────────
export function renderActive() {
  const s = activeSession();
  $("messages").innerHTML = "";

  if (!s) {
    $("chat-title").textContent = "No session selected";
    $("chat-subtitle").textContent = "Create a session on the left to begin.";
    $("restored-badge").classList.add("hidden");
    setComposerEnabled(false);
    resetStats();
    renderTools(null);
    return;
  }

  $("chat-title").textContent = s.title;
  $("chat-subtitle").textContent = subtitleFor(s);
  $("restored-badge").classList.toggle("hidden", !s.restored);

  // Populate and sync the in-header model + role + language switchers.
  // Only roles of this session's own mode — a chat session can never switch to a
  // folder-requiring role, and vice versa (type is fixed at creation).
  const sessionMode = s.mode || (state.roleModes[s.role] || "project");
  const sw = $("model-switch");
  const rsw = $("role-switch");
  const lsw = $("language-switch");
  fillSelect(sw, state.models, s.model);
  fillSelect(rsw, rolesForMode(sessionMode), s.role);
  fillSelect(lsw, state.languages, s.language);

  // Hide all switchers while streaming, show them once idle. Coder sessions are
  // fully config-driven (graph_config.yaml), so they never expose the switchers.
  const showSwitchers = !state.streaming && sessionMode !== "coder";
  for (const node of [sw, rsw, lsw]) node.classList.toggle("hidden", !showSwitchers);

  for (const m of s.messages) {
    if (m.role === "assistant" && Array.isArray(m.parts) && m.parts.length) {
      renderAssistantTurn(m);
    } else {
      addMessageBubble(m.role, m.content, m.ts, m);
    }
  }

  // ALLOW INPUT: only lock the composer when a *different* session is streaming
  // in the background. While the active session streams, keep it open so the
  // user can queue follow-ups.
  setComposerEnabled(!(state.streaming && state.activeId !== state.streamingId));
  updateComposerMode();

  renderStatsFor(s);
  renderTools(s);
  scrollMessages();
}

export function setComposerEnabled(enabled) {
  $("message-input").disabled = !enabled;
  $("send-btn").disabled = !enabled;
  $("attach-btn").disabled = !enabled;
}

// Reflect whether the next submit will send immediately or queue behind the
// in-progress turn for the active session.
export function updateComposerMode() {
  const queueing = state.streaming && state.activeId === state.streamingId;
  $("message-input").placeholder = queueing
    ? "Queue a follow-up… (sent when the agent finishes)"
    : "Ask a follow-up question…";
  $("send-btn").textContent = queueing ? "Queue" : "Send";

  // Stop button: shown only while the *active* session is mid-turn, so it sits
  // beside the Send/Queue button and lets the user abort the runaway agent.
  const stopBtn = $("stop-btn");
  if (queueing) {
    stopBtn.classList.remove("hidden");
    stopBtn.disabled = false;
  } else {
    stopBtn.classList.add("hidden");
  }
}

// Ask the server to stop the in-flight turn for the active session. The stream
// loop trips its cancel flag, ends the turn, and emits a (stopped) done event —
// so the normal reader path tears everything down; we just disable the button.
export async function onStop() {
  const s = activeSession();
  if (!s || !(state.streaming && state.streamingId === s.id)) return;
  $("stop-btn").disabled = true;
  setStatus("working", "Stopping…");
  try {
    await postJSON("/api/chat/stop", { session_id: s.id });
  } catch (e) {
    console.error(e);
  }
}
