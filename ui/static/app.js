"use strict";

// ── State ──────────────────────────────────────────────────────────────
const state = {
  sessions: [],          // [{id,title,path,language,messages,...}]
  activeId: null,
  languages: [],
  models: [],
  roles: [],
  toolsByRole: {},
  streaming: false,      // is an agent turn currently running?
  streamingId: null,     // tracks which session is streaming
  queue: [],             // follow-up messages waiting to be sent: {text, attachments}
  attachments: [],       // pending files for the next message: {name,type,size,data}
  timer: null,           // live response-time interval handle
  turnStart: 0,
};

// Per-file size guard so a giant drop can't lock up the browser / blow context.
const MAX_ATTACH_BYTES = 15 * 1024 * 1024; // 15 MB

// ── DOM helpers ────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};
function fmtTime(iso) {
  try { return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); }
  catch { return ""; }
}
function fmtNum(n) { return (n ?? 0).toLocaleString(); }

// ── Bootstrap ──────────────────────────────────────────────────────────
async function init() {
  bindUI();
  applyTheme();
  applyPanelState();
  await refreshSessions();
}

// ── Theme (light / dark, persisted in localStorage) ────────────────────
function applyTheme() {
  const light = localStorage.getItem("theme") === "light";
  document.documentElement.classList.toggle("dark", !light);
  const btn = $("theme-toggle");
  if (btn) btn.textContent = light ? "☀️" : "🌙";
}
function toggleTheme() {
  const isDark = document.documentElement.classList.contains("dark");
  localStorage.setItem("theme", isDark ? "light" : "dark");
  applyTheme();
}

// ── Collapsible side panels (state persisted in localStorage) ──────────
function applyPanelState() {
  setPanel("stats", localStorage.getItem("statsCollapsed") === "1");
}
function setPanel(which, collapsed) {
  const cls = which + "-collapsed";
  $("app").classList.toggle(cls, collapsed);
  localStorage.setItem(which + "Collapsed", collapsed ? "1" : "0");
  $(which === "sidebar" ? "toggle-sidebar" : "toggle-stats")
    .classList.toggle("active", !collapsed);
}
function togglePanel(which) {
  setPanel(which, !$("app").classList.contains(which + "-collapsed"));
}

async function refreshSessions() {
  try {
    const res = await fetch("/api/sessions");
    const data = await res.json();
    state.sessions = data.sessions || [];
    state.languages = data.languages || [];
    state.models = data.models || [];
    state.roles = data.roles || [];
    state.toolsByRole = data.tools_by_role || {};
    populateLanguages();
    populateModels();
    populateRoles();
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

function populateLanguages() {
  const sel = $("language-select");
  if (sel.options.length) return; // only once
  for (const lang of state.languages) {
    const o = el("option", null, lang);
    o.value = lang;
    sel.appendChild(o);
  }
}

function populateModels(){
  const sel = $("model-select");
  if(sel.options.length) return; // only once
  for (const lang of state.models){
    const o = el("option", null, lang);
    o.value = lang;
    sel.appendChild(o);
  }
}

function populateRoles(){
  const sel = $("role-select");
  if(sel.options.length) return; // only once
  for (const role of state.roles){
    const o = el("option", null, role);
    o.value = role;
    sel.appendChild(o);
  }
}

// ── UI bindings ────────────────────────────────────────────────────────
function bindUI() {
  $("browse-btn").addEventListener("click", browseFolder);
  $("start-btn").addEventListener("click", startSession);
  $("toggle-sidebar").addEventListener("click", () => togglePanel("sidebar"));
  $("toggle-stats").addEventListener("click", () => togglePanel("stats"));
  $("theme-toggle").addEventListener("click", toggleTheme);
  $("send-btn").addEventListener("click", onSend);
  $("model-switch").addEventListener("change", (e) => switchModel(e.target.value));
  $("role-switch").addEventListener("change", (e) => switchRole(e.target.value));

  const input = $("message-input");
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  });
  input.addEventListener("input", () => autoGrow(input));
  // Paste images / files straight into the composer.
  input.addEventListener("paste", (e) => {
    const items = (e.clipboardData && e.clipboardData.items) || [];
    const files = [];
    for (const it of items) {
      if (it.kind === "file") { const f = it.getAsFile(); if (f) files.push(f); }
    }
    if (files.length) { e.preventDefault(); addFiles(files); }
  });

  // Attachments: file picker button + hidden input.
  $("attach-btn").addEventListener("click", () => $("file-input").click());
  $("file-input").addEventListener("change", (e) => {
    addFiles(e.target.files);
    e.target.value = "";  // allow re-selecting the same file
  });

  bindDragAndDrop();
}

// ── Drag & drop files onto the chat pane ───────────────────────────────
function bindDragAndDrop() {
  const pane = $("chat-pane");
  const overlay = $("drop-overlay");
  const hasFiles = (e) =>
    e.dataTransfer && Array.from(e.dataTransfer.types || []).includes("Files");

  let depth = 0;  // track nested dragenter/leave so the overlay doesn't flicker
  pane.addEventListener("dragenter", (e) => {
    if (!hasFiles(e) || !activeSession()) return;
    e.preventDefault();
    depth++;
    overlay.classList.remove("hidden");
  });
  pane.addEventListener("dragover", (e) => {
    if (!hasFiles(e) || !activeSession()) return;
    e.preventDefault();  // required to allow a drop
  });
  pane.addEventListener("dragleave", (e) => {
    if (!hasFiles(e)) return;
    depth = Math.max(0, depth - 1);
    if (depth === 0) overlay.classList.add("hidden");
  });
  pane.addEventListener("drop", (e) => {
    depth = 0;
    overlay.classList.add("hidden");
    if (!hasFiles(e)) return;
    e.preventDefault();
    if (!activeSession()) return;
    if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
  });
}

// ── Attachments (pending files for the next message) ───────────────────
function addFiles(fileList) {
  for (const f of Array.from(fileList || [])) {
    if (f.size > MAX_ATTACH_BYTES) {
      alert(`"${f.name}" is too large (max 15 MB).`);
      continue;
    }
    const reader = new FileReader();
    reader.onload = () => {
      state.attachments.push({
        name: f.name, type: f.type || "", size: f.size, data: reader.result,
      });
      renderAttachments();
    };
    reader.onerror = () => alert(`Could not read "${f.name}".`);
    reader.readAsDataURL(f);  // -> "data:<mime>;base64,<payload>"
  }
}

function removeAttachment(i) {
  state.attachments.splice(i, 1);
  renderAttachments();
}

function fileIcon(type, name) {
  type = type || "";
  if (type.startsWith("image/")) return "🖼️";
  if (type === "application/pdf" || /\.pdf$/i.test(name || "")) return "📕";
  return "📄";
}

function renderAttachments() {
  const box = $("attachment-preview");
  box.innerHTML = "";
  if (!state.attachments.length) { box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  state.attachments.forEach((a, i) => {
    const chip = el("div",
      "flex items-center gap-1.5 text-xs pl-1.5 pr-1 py-1 rounded-lg bg-gray-100 dark:bg-slate-800 " +
      "border border-gray-200 dark:border-slate-700 text-gray-700 dark:text-slate-200");
    if ((a.type || "").startsWith("image/") && a.data) {
      const img = el("img", "w-7 h-7 object-cover rounded");
      img.src = a.data;
      chip.appendChild(img);
    } else {
      chip.appendChild(el("span", "text-base", fileIcon(a.type, a.name)));
    }
    chip.appendChild(el("span", "font-mono truncate max-w-[140px]", a.name));
    const x = el("button",
      "flex-none w-5 h-5 flex items-center justify-center rounded text-gray-400 hover:text-red-500 hover:bg-red-500/10 transition-colors",
      "✕");
    x.title = "Remove";
    x.addEventListener("click", () => removeAttachment(i));
    chip.appendChild(x);
    box.appendChild(chip);
  });
}

function autoGrow(t) {
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight, 160) + "px";
}

// ── Folder picker (native dialog via backend) ──────────────────────────
async function browseFolder() {
  const btn = $("browse-btn");
  btn.disabled = true;
  btn.textContent = "…";
  try {
    const res = await fetch("/api/browse", { method: "POST" });
    const data = await res.json();
    if (data.path) $("path-input").value = data.path;
  } catch (e) {
    console.error(e);
  } finally {
    btn.disabled = false;
    btn.textContent = "📁";
  }
}

// ── Create a new session ───────────────────────────────────────────────
async function startSession() {
  const path = $("path-input").value.trim();
  const language = $("language-select").value;
  const errBox = $("new-session-error");
  errBox.textContent = "";
  if (!path) { errBox.textContent = "Please choose a project path."; return; }

  const btn = $("start-btn");
  btn.disabled = true;
  btn.textContent = "Starting…";
  try {
    const res = await fetch("/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, language, model: $("model-select").value, role: $("role-select").value }),
    });
    const data = await res.json();
    if (data.error) { errBox.textContent = data.error; return; }
    state.sessions.push(data.session);
    renderSessionList();
    selectSession(data.session.id);
    $("path-input").value = "";
    sendMessage("Review this project.");
  } catch (e) {
    errBox.textContent = "Could not start session.";
    console.error(e);
  } finally {
    btn.disabled = false;
    btn.textContent = "Start review";
  }
}

// ── Session list / switching ───────────────────────────────────────────
function renderSessionList() {
  const ul = $("session-list");
  ul.innerHTML = "";
  for (const s of state.sessions) {
    const active = s.id === state.activeId;
    const li = el("li",
      "group flex items-center gap-1.5 px-2.5 py-2 rounded-lg border transition-colors cursor-pointer " +
      (active
        ? "bg-blue-50 dark:bg-blue-950/40 border-blue-400 dark:border-blue-600"
        : "bg-gray-50 dark:bg-slate-800 border-transparent hover:border-gray-300 dark:hover:border-slate-600"));

    const info = el("div", "flex-1 min-w-0");
    info.appendChild(el("div", "text-sm font-semibold truncate", s.title));
    const meta = `${s.language} · ${fmtTime(s.created)}`;
    info.appendChild(el("div", "text-[11px] text-gray-400 dark:text-slate-500 mt-0.5 truncate",
      meta + (s.restored ? " · restored" : "")));
    info.addEventListener("click", () => selectSession(s.id));

    const del = el("button",
      "flex-none text-xs px-1.5 py-1 rounded-md text-gray-400 dark:text-slate-500 opacity-0 group-hover:opacity-100 hover:text-red-500 hover:bg-red-500/10 transition-all",
      "✕");
    del.title = "Delete this session";
    del.addEventListener("click", (e) => { e.stopPropagation(); deleteSession(s.id); });

    li.appendChild(info);
    li.appendChild(del);
    ul.appendChild(li);
  }
}

function activeSession() {
  return state.sessions.find((s) => s.id === state.activeId);
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
    const res = await fetch("/api/session/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    });
    const data = await res.json();
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

async function switchModel(newModel) {
  const s = activeSession();
  if (!s || s.restored) return;
  try {
    const res = await fetch("/api/session/model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: s.id, model: newModel }),
    });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    s.model = data.model;
    $("chat-subtitle").textContent = subtitleFor(s);
  } catch (e) {
    console.error(e);
    alert("Could not switch model.");
  }
}

async function switchRole(newRole) {
  const s = activeSession();
  if (!s || s.restored) return;
  try {
    const res = await fetch("/api/session/role", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: s.id, role: newRole }),
    });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    s.role = data.role;
    if (data.enabled_tools) s.enabled_tools = data.enabled_tools;  // reset for new role
    $("chat-subtitle").textContent = subtitleFor(s);
    renderTools(s);
  } catch (e) {
    console.error(e);
    alert("Could not switch role.");
  }
}

// One source of truth for the header subtitle: path · language · model · role.
function subtitleFor(s) {
  return `${s.path}  ·  ${s.language}  ·  ${s.model}` + (s.role ? `  ·  ${s.role}` : "");
}

// ── Tool toggles (per session, applied to the next message) ─────────────
function renderTools(s) {
  const ul = $("tool-toggles");
  ul.innerHTML = "";
  const tools = (s && state.toolsByRole[s.role]) || [];
  if (!s || !tools.length) {
    ul.innerHTML = '<li class="muted text-gray-400 dark:text-slate-500">no session</li>';
    return;
  }
  const enabled = new Set(s.enabled_tools || tools);
  for (const name of tools) {
    const li = el("li", "flex items-center gap-2");
    const cb = el("input");
    cb.type = "checkbox";
    cb.id = "tool-cb-" + name;
    cb.checked = enabled.has(name);
    cb.className = "accent-blue-600 cursor-pointer";
    cb.addEventListener("change", () => toggleTool(name, cb.checked));
    const label = el("label", "cursor-pointer select-none font-mono text-[11px] truncate", name);
    label.htmlFor = cb.id;
    li.appendChild(cb);
    li.appendChild(label);
    ul.appendChild(li);
  }
}

async function toggleTool(name, on) {
  const s = activeSession();
  if (!s) return;
  const tools = state.toolsByRole[s.role] || [];
  const set = new Set(s.enabled_tools || tools);
  if (on) set.add(name); else set.delete(name);
  const enabled = tools.filter((t) => set.has(t));  // keep canonical order
  s.enabled_tools = enabled;  // optimistic
  try {
    const res = await fetch("/api/session/tools", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: s.id, enabled_tools: enabled }),
    });
    const data = await res.json();
    if (data.error) { alert(data.error); renderTools(s); return; }
    s.enabled_tools = data.enabled_tools;
  } catch (e) {
    console.error(e);
    alert("Could not update tools.");
    renderTools(s);
  }
}

function renderActive() {
  const s = activeSession();
  const msgs = $("messages");
  msgs.innerHTML = "";

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

  // Populate and sync the in-header model + role switchers
  const sw = $("model-switch");
  sw.innerHTML = "";
  for (const m of state.models) {
    const o = el("option", null, m);
    o.value = m;
    if (m === s.model) o.selected = true;
    sw.appendChild(o);
  }

  const rsw = $("role-switch");
  rsw.innerHTML = "";
  for (const r of state.roles) {
    const o = el("option", null, r);
    o.value = r;
    if (r === s.role) o.selected = true;
    rsw.appendChild(o);
  }

  // Hide both switchers while streaming, show them once idle.
  sw.classList.add("hidden");
  rsw.classList.add("hidden");
  if (!state.streaming) {
    sw.classList.remove("hidden");
    rsw.classList.remove("hidden");
  }

  for (const m of s.messages) {
    addMessageBubble(m.role, m.content, m.ts, m);
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

function setComposerEnabled(enabled) {
  $("message-input").disabled = !enabled;
  $("send-btn").disabled = !enabled;
  $("attach-btn").disabled = !enabled;
}

// Reflect whether the next submit will send immediately or queue behind the
// in-progress turn for the active session.
function updateComposerMode() {
  const queueing = state.streaming && state.activeId === state.streamingId;
  $("message-input").placeholder = queueing
    ? "Queue a follow-up… (sent when the agent finishes)"
    : "Ask a follow-up question…";
  $("send-btn").textContent = queueing ? "Queue" : "Send";
}

// ── Message rendering ──────────────────────────────────────────────────
function addMessageBubble(role, content, ts, meta) {
  const isUser = role === "user";
  const wrap = el("div",
    "flex flex-col gap-1 max-w-[80%] " + (isUser ? "self-end items-end" : "self-start items-start"));

  // Attachments (user messages only): thumbnails for images, pills for files.
  const atts = (meta && meta.attachments) || [];
  if (isUser && atts.length) {
    const row = el("div", "flex flex-wrap gap-1.5 justify-end");
    for (const a of atts) row.appendChild(attachmentThumb(a));
    wrap.appendChild(row);
  }

  const bubble = el("div",
    isUser
      ? "px-3.5 py-2.5 rounded-2xl rounded-br-sm bg-blue-600 text-white text-sm leading-relaxed whitespace-pre-wrap break-words"
      : "ai-bubble px-3.5 py-2.5 rounded-2xl rounded-bl-sm bg-gray-100 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 text-gray-900 dark:text-slate-100 text-sm leading-relaxed break-words");

  if (role === "assistant" && content) {
    bubble.innerHTML = marked.parse(content);
  } else {
    bubble.textContent = content;
  }

  // Skip an empty bubble when a user message carries only attachments.
  if (content || !isUser || !atts.length) wrap.appendChild(bubble);
  let metaText = fmtTime(ts);
  if (role === "assistant" && meta && meta.elapsed != null) {
    metaText += `  ·  ${meta.elapsed}s`;
    if (meta.usage && meta.usage.total_tokens) metaText += `  ·  ${fmtNum(meta.usage.total_tokens)} tok`;
    if (meta.usage && meta.usage.output_tokens && meta.elapsed) {
      metaText += `  ·  ${(meta.usage.output_tokens / meta.elapsed).toFixed(1)} tok/s`;
    }
  }
  wrap.appendChild(el("div", "text-[11px] text-gray-400 dark:text-slate-500", metaText));
  $("messages").appendChild(wrap);
  return bubble;
}

// One attachment in a sent message: an image thumbnail when we still have the
// data (live turn), otherwise a labelled pill. Re-opened sessions only keep
// {name,type,size} metadata, so they always show the pill.
function attachmentThumb(a) {
  if ((a.type || "").startsWith("image/") && a.data) {
    const img = el("img",
      "max-w-[180px] max-h-[180px] rounded-lg border border-blue-300 dark:border-blue-700");
    img.src = a.data;
    img.title = a.name;
    return img;
  }
  const pill = el("div",
    "flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-lg bg-blue-50 dark:bg-blue-950/40 " +
    "border border-blue-200 dark:border-blue-800/60 text-blue-700 dark:text-blue-300");
  pill.appendChild(el("span", "text-base", fileIcon(a.type, a.name)));
  pill.appendChild(el("span", "font-mono truncate max-w-[180px]", a.name));
  pill.title = a.name;
  return pill;
}

function scrollMessages() {
  const m = $("messages");
  // Clean cross-browser anchor scrolling
  m.scrollTo({
    top: m.scrollHeight,
    behavior: "auto"
  });
}

// ── Sending / queueing ─────────────────────────────────────────────────
function onSend() {
  const input = $("message-input");
  const text = input.value.trim();
  const attachments = state.attachments;
  if ((!text && !attachments.length) || input.disabled) return;
  input.value = "";
  autoGrow(input);
  state.attachments = [];  // hand the pending files off to this message
  renderAttachments();

  if (state.streaming && state.activeId === state.streamingId) {
    state.queue.push({ text, attachments });
    renderQueue();
  } else {
    sendMessage(text, attachments);
  }
}

function renderQueue() {
  const q = $("queue-indicator");
  if (!state.queue.length) { q.classList.add("hidden"); return; }
  q.classList.remove("hidden");
  q.textContent = `⏳ ${state.queue.length} queued: ` +
    state.queue.map((m) => {
      const label = m.text || `📎 ${m.attachments.length} file(s)`;
      return `"${label.length > 30 ? label.slice(0, 30) + "…" : label}"`;
    }).join(", ");
}

function drainQueue() {
  if (state.queue.length && !state.streaming) {
    const next = state.queue.shift();
    renderQueue();
    sendMessage(next.text, next.attachments);
  }
}

// ── The streaming turn ─────────────────────────────────────────────────
async function sendMessage(text, attachments = []) {
  const s = activeSession();
  if (!s) return;

  state.streaming = true;
  state.streamingId = s.id;
  // Composer stays enabled so the user can type and queue follow-ups while the
  // agent works (onSend routes input to the queue during streaming).
  setComposerEnabled(true);
  updateComposerMode();

  const userTs = new Date().toISOString();
  addMessageBubble("user", text, userTs, { attachments });
  scrollMessages();

  const aiBubble = addMessageBubble("assistant", "", new Date().toISOString());
  aiBubble.classList.add("streaming");
  scrollMessages();

  startTimer();
  setStatus("working", "Working…");
  clearTools();
  state.openToolBubbles = {};

  let answer = "";
  let reader = null;
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: s.id, message: text, attachments }),
    });

    reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let finished = false;

    while (!finished) {
      const { value, done } = await reader.read();
      if (done) break;  // connection closed (fallback)
      buf += decoder.decode(value, { stream: true });

      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const raw = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 2);
        if (!raw.startsWith("data:")) continue;
        const evt = JSON.parse(raw.slice(5).trim());
        answer = handleEvent(evt, aiBubble, answer);
        // The turn is over on a terminal event — stop reading instead of
        // blocking on the kept-alive socket that the server won't close.
        if (evt.type === "done" || evt.type === "error") finished = true;
      }
    }
  } catch (e) {
    console.error(e);
    setStatus("error", "Connection error");
    aiBubble.textContent = answer || "⚠️ The stream was interrupted.";
  } finally {
    if (reader) { try { await reader.cancel(); } catch (_) { /* already closed */ } }
    aiBubble.classList.remove("streaming");
    stopTimer();
    state.streaming = false;
    state.streamingId = null;

    // Safely reinstate the UI interaction state if looking at current context
    if (state.activeId === s.id) {
      setComposerEnabled(true);
    }
    updateComposerMode();
    drainQueue();
  }
}

function handleEvent(evt, aiBubble, answer) {
  switch (evt.type) {
    case "token":
      answer += evt.text;
      if (state.tokenChunks === 0) state.firstTokenAt = performance.now();
      state.tokenChunks++;
      aiBubble.innerHTML = marked.parse(answer);
      scrollMessages();
      break;
    case "tool":
      updateTool(evt.name, evt.phase);
      renderToolBubble(evt, aiBubble);
      break;
    case "status":
      break;
    case "done":
      if (evt.answer) { answer = evt.answer; aiBubble.innerHTML = marked.parse(answer); }
      applyDoneStats(evt);
      setStatus("done", `Done in ${evt.elapsed}s`);
      mergeSessionStats(evt, answer);
      break;
    case "error":
      setStatus("error", "Agent error");
      aiBubble.innerHTML = marked.parse((answer ? answer + "\n\n" : "") + "⚠️ " + evt.message);
      break;
  }
  return answer;
}

function mergeSessionStats(evt, answer) {
  const s = activeSession();
  if (!s) return;
  s.totals = evt.totals;
  s.last = evt.last;

  // Backfill the message to the state session history array so it re-renders
  // as markdown (not raw text) when the session is reselected.
  if (s.messages) {
    s.messages.push({
      role: "assistant",
      content: evt.answer || answer || "",
      ts: evt.ts || new Date().toISOString()
    });
  }
}

// ── Live response timer ────────────────────────────────────────────────
function startTimer() {
  state.turnStart = performance.now();
  state.tokenChunks = 0;
  $("stat-timer").textContent = "0.0s";
  setSpeed(null, true);
  if (state.timer) clearInterval(state.timer);
  state.timer = setInterval(() => {
    const secs = (performance.now() - state.turnStart) / 1000;
    $("stat-timer").textContent = secs.toFixed(1) + "s";
    if (state.tokenChunks > 0) {
      const gen = (performance.now() - state.firstTokenAt) / 1000;
      if (gen > 0) setSpeed(state.tokenChunks / gen, true);
    }
  }, 100);
}
function stopTimer() {
  if (state.timer) { clearInterval(state.timer); state.timer = null; }
}

// ── Stats rendering ────────────────────────────────────────────────────
function setStatus(cls, text) {
  const node = $("stat-status");
  node.className = "stat-status font-semibold text-sm " + cls;
  node.textContent = text;
}

function setSpeed(tps, live) {
  const node = $("stat-speed");
  const sub = $("stat-speed-sub");
  if (!tps || !isFinite(tps)) {
    node.innerHTML = '— <span class="unit">tok/s</span>';
    if (sub) sub.textContent = "output tokens / sec";
    return;
  }
  node.innerHTML = `${live ? "~" : ""}${tps.toFixed(1)} <span class="unit">tok/s</span>`;
  if (sub) sub.textContent = live ? "live estimate" : "output tokens / sec";
}

function applyDoneStats(evt) {
  const u = evt.usage || {};
  $("tok-in").textContent = fmtNum(u.input_tokens);
  $("tok-out").textContent = fmtNum(u.output_tokens);
  $("tok-total").textContent = fmtNum(u.total_tokens);
  $("stat-timer").textContent = evt.elapsed + "s";
  $("stat-timer-sub").textContent = "last turn: " + evt.elapsed + "s";
  setSpeed((u.output_tokens && evt.elapsed) ? u.output_tokens / evt.elapsed : 0, false);

  const t = evt.totals || {};
  $("tot-turns").textContent = fmtNum(t.turns);
  $("tot-in").textContent = fmtNum(t.input_tokens);
  $("tot-out").textContent = fmtNum(t.output_tokens);
  $("tot-total").textContent = fmtNum(t.total_tokens);

  const ctx = evt.context_window;
  updateCtxBar(u.input_tokens, ctx);
}

function updateCtxBar(used, window) {
  const bar = $("ctx-bar");
  const txt = $("ctx-text");
  if (!window) { bar.style.width = "0%"; txt.textContent = `${fmtNum(used)} / —`; return; }
  const pct = Math.min(100, Math.round((used / window) * 100));
  bar.style.width = pct + "%";
  txt.textContent = `${fmtNum(used)} / ${fmtNum(window)} (${pct}%)`;
}

function renderStatsFor(s) {
  const last = s.last || {};
  const totals = s.totals || {};
  $("tok-in").textContent = fmtNum(last.input_tokens);
  $("tok-out").textContent = fmtNum(last.output_tokens);
  $("tok-total").textContent = fmtNum(last.total_tokens);
  $("stat-timer").textContent = (last.elapsed || 0) + "s";
  $("stat-timer-sub").textContent = last.elapsed ? "last turn: " + last.elapsed + "s" : "last turn: —";
  setSpeed((last.output_tokens && last.elapsed) ? last.output_tokens / last.elapsed : 0, false);
  $("tot-turns").textContent = fmtNum(totals.turns);
  $("tot-in").textContent = fmtNum(totals.input_tokens);
  $("tot-out").textContent = fmtNum(totals.output_tokens);
  $("tot-total").textContent = fmtNum(totals.total_tokens);
  updateCtxBar(last.input_tokens || 0, null);
  if (!state.streaming) setStatus("idle", "Idle");
  clearTools();
}

function resetStats() {
  ["tok-in","tok-out","tok-total","tot-turns","tot-in","tot-out","tot-total"]
    .forEach((id) => $(id).textContent = "0");
  $("stat-timer").textContent = "0.0s";
  $("stat-timer-sub").textContent = "last turn: —";
  setSpeed(null, false);
  updateCtxBar(0, null);
  setStatus("idle", "Idle");
  clearTools();
}

// ── Tool activity ──────────────────────────────────────────────────────
const toolNodes = {};
function clearTools() {
  for (const k in toolNodes) delete toolNodes[k];
  $("tool-activity").innerHTML = '<li class="muted text-gray-400 dark:text-slate-500">none yet</li>';
}
function updateTool(name, phase) {
  const list = $("tool-activity");
  if (list.querySelector(".muted")) list.innerHTML = "";
  let li = toolNodes[name];
  if (!li) { li = el("li", "flex items-center gap-1.5"); toolNodes[name] = li; list.appendChild(li); }
  if (phase === "start") {
    li.innerHTML = `<span class="text-blue-500 animate-spin inline-block">⟳</span> ${name}`;
  } else {
    li.innerHTML = `<span class="text-emerald-500">✓</span> ${name}`;
  }
}

// ── In-chat tool bubbles (persistent, distinct colour) ─────────────────
// A small violet pill inserted above the streaming answer for each tool the
// agent runs, showing the affected file + directory once the call completes.
function renderToolBubble(evt, aiBubble) {
  state.openToolBubbles = state.openToolBubbles || {};

  if (evt.phase === "start") {
    const wrap = buildToolBubble(evt.name, null, false);
    // Insert just above the assistant's (streaming) bubble so tools read
    // in-order before the answer they inform.
    const aiWrap = aiBubble ? aiBubble.parentElement : null;
    if (aiWrap && aiWrap.parentElement === $("messages")) {
      $("messages").insertBefore(wrap, aiWrap);
    } else {
      $("messages").appendChild(wrap);
    }
    state.openToolBubbles[evt.name] = wrap;
    scrollMessages();
    return;
  }

  // phase === "end": finalise the open bubble for this tool, or create a
  // already-completed one if the start was de-duplicated server-side.
  const open = state.openToolBubbles[evt.name];
  if (open) {
    open.replaceWith(buildToolBubble(evt.name, evt.target, true));
    delete state.openToolBubbles[evt.name];
  } else {
    const wrap = buildToolBubble(evt.name, evt.target, true);
    const aiWrap = aiBubble ? aiBubble.parentElement : null;
    if (aiWrap && aiWrap.parentElement === $("messages")) {
      $("messages").insertBefore(wrap, aiWrap);
    } else {
      $("messages").appendChild(wrap);
    }
  }
  scrollMessages();
}

function buildToolBubble(name, target, done) {
  const isWrite = name === "write_file";
  const isWeb = name === "web_browse";
  // Writes get a distinct amber tint, web browsing a sky tint, so each reads
  // clearly apart from the violet reads.
  const wrap = el("div",
    "self-start flex items-center gap-2 max-w-[80%] text-xs px-3 py-1.5 rounded-lg " + (isWrite
      ? "bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800/60 " +
        "text-amber-700 dark:text-amber-300"
      : isWeb
      ? "bg-sky-50 dark:bg-sky-950/40 border border-sky-200 dark:border-sky-800/60 " +
        "text-sky-700 dark:text-sky-300"
      : "bg-violet-50 dark:bg-violet-950/40 border border-violet-200 dark:border-violet-800/60 " +
        "text-violet-700 dark:text-violet-300"));

  const accent = isWrite
    ? "text-amber-500 dark:text-amber-400"
    : isWeb
    ? "text-sky-500 dark:text-sky-400"
    : "text-violet-500 dark:text-violet-400";
  const icon = el("span", done ? accent : accent + " animate-spin inline-block",
    done ? (isWrite ? "📝" : isWeb ? "🌐" : "🔧") : "⟳");
  wrap.appendChild(icon);
  wrap.appendChild(el("span", "font-semibold", name));

  // web_browse: show the site searched and the query, mirroring the file label.
  if (isWeb && target && (target.url || target.query)) {
    const sep = el("span", "text-sky-300 dark:text-sky-700", "·");
    wrap.appendChild(sep);
    const detail = el("span",
      "font-mono text-[11px] truncate text-sky-600 dark:text-sky-400/90");
    const bits = [];
    if (target.url) bits.push(`🔗 ${target.url}`);
    if (target.query) bits.push(`🔎 ${target.query}`);
    detail.textContent = bits.join("  ");
    detail.title = bits.join("  ");
    wrap.appendChild(detail);
    return wrap;
  }

  if (target) {
    const sep = el("span", isWrite
      ? "text-amber-300 dark:text-amber-700" : "text-violet-300 dark:text-violet-700", "·");
    wrap.appendChild(sep);
    const detail = el("span", "font-mono text-[11px] truncate " + (isWrite
      ? "text-amber-600 dark:text-amber-400/90" : "text-violet-500 dark:text-violet-400/90"));
    detail.title = target.path;
    detail.textContent = `📄 ${target.name}` + (target.dir && target.dir !== "." ? `  📁 ${target.dir}` : "");
    wrap.appendChild(detail);

    // Writes are snapshotted server-side; offer a one-click undo to the
    // file's state from before the agent first edited it (this session only).
    if (isWrite && done) {
      const btn = el("button",
        "flex-none text-[11px] px-1.5 py-0.5 rounded border " +
        "border-amber-300 dark:border-amber-700/70 hover:bg-amber-500/10 transition-colors",
        "↩ revert");
      btn.title = "Restore this file to its state before the agent edited it";
      btn.addEventListener("click", () => revertFile(target.path, btn));
      wrap.appendChild(btn);
    }
  } else if (done) {
    wrap.appendChild(el("span",
      (isWrite ? "text-amber-400 dark:text-amber-500" : "text-violet-400 dark:text-violet-500") + " italic",
      "project"));
  }
  return wrap;
}

// Restore a written file to its pre-edit snapshot via the backend.
async function revertFile(path, btn) {
  const s = activeSession();
  if (!s) return;
  if (!confirm(`Revert "${path}" to its state before the agent edited it?`)) return;
  const original = btn ? btn.textContent : "";
  if (btn) { btn.disabled = true; btn.textContent = "…"; }
  try {
    const res = await fetch("/api/session/revert", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: s.id, path }),
    });
    const data = await res.json();
    if (data.error) {
      alert(data.error);
      if (btn) { btn.disabled = false; btn.textContent = original; }
      return;
    }
    if (btn) { btn.textContent = "✓ reverted"; btn.classList.add("opacity-60"); }
  } catch (e) {
    console.error(e);
    alert("Could not revert the file.");
    if (btn) { btn.disabled = false; btn.textContent = original; }
  }
}

init();