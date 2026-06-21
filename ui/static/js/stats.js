"use strict";

import { state } from "./state.js";
import { $, fmtNum } from "./dom.js";
import { clearTools } from "./tools.js";

// ── Live response timer ────────────────────────────────────────────────
export function startTimer() {
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

export function stopTimer() {
  if (state.timer) { clearInterval(state.timer); state.timer = null; }
}

// ── Stats rendering ────────────────────────────────────────────────────
export function setStatus(cls, text) {
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

export function applyDoneStats(evt) {
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

  updateCtxBar(u.input_tokens, evt.context_window);
}

// Live, mid-run snapshot pushed by the orchestrator graph (server `usage`
// events). Same meters as applyDoneStats minus the per-turn timer fields, which
// only settle once the turn finishes. Lets the context-window bar, last-turn
// tokens and session totals climb while the graph is still running.
export function applyLiveStats(evt) {
  const u = evt.usage || {};
  $("tok-in").textContent = fmtNum(u.input_tokens);
  $("tok-out").textContent = fmtNum(u.output_tokens);
  $("tok-total").textContent = fmtNum(u.total_tokens);

  const t = evt.totals || {};
  $("tot-turns").textContent = fmtNum(t.turns);
  $("tot-in").textContent = fmtNum(t.input_tokens);
  $("tot-out").textContent = fmtNum(t.output_tokens);
  $("tot-total").textContent = fmtNum(t.total_tokens);

  updateCtxBar(u.input_tokens, evt.context_window);
}

function updateCtxBar(used, window) {
  const bar = $("ctx-bar");
  const txt = $("ctx-text");
  if (!window) { bar.style.width = "0%"; txt.textContent = `${fmtNum(used)} / —`; return; }
  const pct = Math.min(100, Math.round((used / window) * 100));
  bar.style.width = pct + "%";
  txt.textContent = `${fmtNum(used)} / ${fmtNum(window)} (${pct}%)`;
}

export function renderStatsFor(s) {
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

export function resetStats() {
  ["tok-in", "tok-out", "tok-total", "tot-turns", "tot-in", "tot-out", "tot-total"]
    .forEach((id) => $(id).textContent = "0");
  $("stat-timer").textContent = "0.0s";
  $("stat-timer-sub").textContent = "last turn: —";
  setSpeed(null, false);
  updateCtxBar(0, null);
  setStatus("idle", "Idle");
  clearTools();
}
