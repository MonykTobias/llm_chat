"use strict";

import { $ } from "./dom.js";

// ── Theme (light / dark, persisted in localStorage) ────────────────────
export function applyTheme() {
  const light = localStorage.getItem("theme") === "light";
  document.documentElement.classList.toggle("dark", !light);
  const btn = $("theme-toggle");
  if (btn) btn.textContent = light ? "☀️" : "🌙";
}

export function toggleTheme() {
  const isDark = document.documentElement.classList.contains("dark");
  localStorage.setItem("theme", isDark ? "light" : "dark");
  applyTheme();
}

// ── Collapsible side panels (state persisted in localStorage) ──────────
export function applyPanelState() {
  setPanel("stats", localStorage.getItem("statsCollapsed") === "1");
}

function setPanel(which, collapsed) {
  const cls = which + "-collapsed";
  $("app").classList.toggle(cls, collapsed);
  localStorage.setItem(which + "Collapsed", collapsed ? "1" : "0");
  $(which === "sidebar" ? "toggle-sidebar" : "toggle-stats")
    .classList.toggle("active", !collapsed);
}

export function togglePanel(which) {
  setPanel(which, !$("app").classList.contains(which + "-collapsed"));
}
