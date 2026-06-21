"use strict";

// ── DOM + formatting helpers ────────────────────────────────────────────
export const $ = (id) => document.getElementById(id);

export const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

export function fmtTime(iso) {
  try { return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); }
  catch { return ""; }
}

export function fmtNum(n) { return (n ?? 0).toLocaleString(); }

// Grow a textarea with its content, capped so it never eats the whole viewport.
export function autoGrow(t) {
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight, 160) + "px";
}

// Scroll the message list to the bottom — but only steal scroll position from
// the user when they were already near the bottom (or we force it).
export function scrollMessages(force = false) {
  const m = $("messages");
  const isNearBottom = (m.scrollHeight - m.scrollTop - m.clientHeight) < 60;
  if (force || isNearBottom) {
    m.scrollTo({ top: m.scrollHeight, behavior: "auto" });
  }
}

// Fill a <select> with options, optionally marking one selected. Replaces the
// current options each call (used for the header switchers + role dropdown).
export function fillSelect(sel, items, selected) {
  sel.innerHTML = "";
  for (const v of items) {
    const o = el("option", null, v);
    o.value = v;
    if (v === selected) o.selected = true;
    sel.appendChild(o);
  }
}

// Like fillSelect but a no-op once the <select> already has options — for the
// new-session pickers that are populated a single time on first load.
export function fillSelectOnce(sel, items) {
  if (sel.options.length) return;
  fillSelect(sel, items);
}
