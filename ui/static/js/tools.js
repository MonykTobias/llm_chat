"use strict";

import { state, activeSession } from "./state.js";
import { $, el, scrollMessages } from "./dom.js";
import { postJSON } from "./api.js";

// ── Tool toggles (per session, applied to the next message) ─────────────
export function renderTools(s) {
  const ul = $("tool-toggles");
  ul.innerHTML = "";
  const tools = (s && state.toolsByRole[s.role]) || [];
  if (!s || !tools.length) {
    ul.innerHTML = '<li class="muted">no session</li>';
    return;
  }
  const enabled = new Set(s.enabled_tools || tools);
  for (const name of tools) {
    const li = el("li", "flex items-center gap-2");
    const cb = el("input");
    cb.type = "checkbox";
    cb.id = "tool-cb-" + name;
    cb.checked = enabled.has(name);
    cb.className = "cursor-pointer";
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
    const data = await postJSON("/api/session/tools", { id: s.id, enabled_tools: enabled });
    if (data.error) { alert(data.error); renderTools(s); return; }
    s.enabled_tools = data.enabled_tools;
  } catch (e) {
    console.error(e);
    alert("Could not update tools.");
    renderTools(s);
  }
}

// ── Tool activity (the live "what ran" list in the stats pane) ─────────
const toolNodes = {};

export function clearTools() {
  for (const k in toolNodes) delete toolNodes[k];
  $("tool-activity").innerHTML = '<li class="muted">none yet</li>';
}

export function updateTool(name, phase) {
  const list = $("tool-activity");
  if (list.querySelector(".muted")) list.innerHTML = "";
  let li = toolNodes[name];
  if (!li) { li = el("li", "flex items-center gap-1.5"); toolNodes[name] = li; list.appendChild(li); }
  if (phase === "start") {
    li.innerHTML = `<span class="animate-spin inline-block" style="color:var(--accent)">⟳</span> ${name}`;
  } else {
    li.innerHTML = `<span style="color:var(--ok)">✓</span> ${name}`;
  }
}

// ── In-chat tool bubbles (persistent, distinct colour) ─────────────────
// A small pill inserted above the streaming answer for each tool the agent
// runs, showing the affected file + directory once the call completes.
function ensureToolRow() {
  // Recreate if missing, or if a mid-stream re-render (renderActive clears
  // #messages) left our cached row detached — appending into a detached row
  // would make new tool pills silently vanish.
  if (!state.openToolRow || !state.openToolRow.isConnected) {
    state.openToolRow = el("div", "self-start flex flex-row flex-wrap gap-2");
    $("messages").appendChild(state.openToolRow);
  }
  return state.openToolRow;
}

export function renderToolBubble(evt) {
  state.openToolBubbles = state.openToolBubbles || {};

  if (evt.phase === "start") {
    const wrap = buildToolBubble(evt.name, null, false);
    ensureToolRow().appendChild(wrap);
    state.openToolBubbles[evt.name] = wrap;
    scrollMessages();
    return;
  }

  // phase === "end": finalise the open bubble for this tool, or create an
  // already-completed one if the start was de-duplicated server-side.
  const open = state.openToolBubbles[evt.name];
  if (open) {
    open.replaceWith(buildToolBubble(evt.name, evt.target, true));
    delete state.openToolBubbles[evt.name];
  } else {
    ensureToolRow().appendChild(buildToolBubble(evt.name, evt.target, true));
  }
  scrollMessages();
}

// Per-tool-kind styling. Writes get a distinct amber tint, web browsing a sky
// tint, deletes a red tint (destructive — stands apart), so each reads clearly
// apart from the violet reads.
const TOOL_KINDS = {
  write: {
    bubble: "bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800/60 text-amber-700 dark:text-amber-300",
    accent: "text-amber-500 dark:text-amber-400",
    sep:    "text-amber-300 dark:text-amber-700",
    detail: "text-amber-600 dark:text-amber-400/90",
    empty:  "text-amber-400 dark:text-amber-500",
    icon:   "📝",
  },
  web: {
    bubble: "bg-sky-50 dark:bg-sky-950/40 border border-sky-200 dark:border-sky-800/60 text-sky-700 dark:text-sky-300",
    accent: "text-sky-500 dark:text-sky-400",
    sep:    "text-sky-300 dark:text-sky-700",
    detail: "text-sky-600 dark:text-sky-400/90",
    empty:  "text-sky-400 dark:text-sky-500",
    icon:   "🌐",
  },
  delete: {
    bubble: "bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-800/60 text-red-700 dark:text-red-300",
    accent: "text-red-500 dark:text-red-400",
    sep:    "text-red-300 dark:text-red-700",
    detail: "text-red-600 dark:text-red-400/90",
    empty:  "text-red-400 dark:text-red-500",
    icon:   "🗑️",
  },
  read: {
    bubble: "bg-violet-50 dark:bg-violet-950/40 border border-violet-200 dark:border-violet-800/60 text-violet-700 dark:text-violet-300",
    accent: "text-violet-500 dark:text-violet-400",
    sep:    "text-violet-300 dark:text-violet-700",
    detail: "text-violet-500 dark:text-violet-400/90",
    empty:  "text-violet-400 dark:text-violet-500",
    icon:   "🔧",
  },
};

function toolKind(name) {
  if (name === "write_file") return "write";
  if (name === "web_browse") return "web";
  if (name === "delete_file") return "delete";
  return "read";
}

function buildToolBubble(name, target, done) {
  const kind = toolKind(name);
  const k = TOOL_KINDS[kind];

  const wrap = el("div",
    "self-start flex items-center gap-2 max-w-[80%] text-xs px-3 py-1.5 rounded-lg " + k.bubble);
  wrap.appendChild(el("span",
    done ? k.accent : k.accent + " animate-spin inline-block",
    done ? k.icon : "⟳"));
  wrap.appendChild(el("span", "font-semibold", name));

  // web_browse: show the site searched and the query, as a clickable link.
  if (kind === "web" && target && (target.url || target.query)) {
    wrap.appendChild(el("span", k.sep, "·"));
    const detail = el("a",
      `font-mono text-[11px] truncate ${k.detail} hover:underline cursor-pointer`);
    detail.target = "_blank";
    detail.rel = "noopener noreferrer";
    let bit;
    if (target.url) {
      bit = `🔗 ${target.url}`;
      detail.href = target.url;
    } else {
      bit = `🔎 ${target.query}`;
      detail.href = `https://www.google.com/search?q=${encodeURIComponent(target.query)}`;
    }
    detail.textContent = bit;
    detail.title = bit;
    wrap.appendChild(detail);
    return wrap;
  }

  if (target) {
    wrap.appendChild(el("span", k.sep, "·"));
    const detail = el("span", "font-mono text-[11px] truncate " + k.detail);
    detail.title = target.path;
    detail.textContent = `📄 ${target.name}` + (target.dir && target.dir !== "." ? `  📁 ${target.dir}` : "");
    wrap.appendChild(detail);

    // Writes are snapshotted server-side; offer a one-click undo to the
    // file's state from before the agent first edited it (this session only).
    if (kind === "write" && done) {
      const btn = el("button",
        "flex-none text-[11px] px-1.5 py-0.5 rounded border " +
        "border-amber-300 dark:border-amber-700/70 hover:bg-amber-500/10 transition-colors",
        "↩ revert");
      btn.title = "Restore this file to its state before the agent edited it";
      btn.addEventListener("click", () => revertFile(target.path, btn));
      wrap.appendChild(btn);
    }
  } else if (done) {
    wrap.appendChild(el("span", k.empty + " italic", "project"));
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
    const data = await postJSON("/api/session/revert", { id: s.id, path });
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

// Re-export for renderAssistantTurn (messages.js) which rebuilds finished pills.
export { buildToolBubble };
