"use strict";

import { state, AI_BUBBLE_CLS, activeSession } from "./state.js";
import { $, el, autoGrow, scrollMessages } from "./dom.js";
import { addMessageBubble, makeAssistantBubble, addStageBanner, addStageDivider } from "./messages.js";
import { updateTool, clearTools, renderToolBubble, renderTools } from "./tools.js";
import { setStatus, startTimer, stopTimer, applyDoneStats, applyLiveStats } from "./stats.js";
import { renderAttachments } from "./attachments.js";
import { setComposerEnabled, updateComposerMode, subtitleFor } from "./session.js";

// ── Sending / queueing ─────────────────────────────────────────────────
export function onSend() {
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
  q.innerHTML = "";
  if (!state.queue.length) { q.classList.add("hidden"); return; }
  q.classList.remove("hidden");

  q.appendChild(el("div", "font-semibold mb-1.5", `⏳ ${state.queue.length} queued`));

  const list = el("div", "flex flex-wrap gap-1.5");
  state.queue.forEach((m, i) => {
    const label = m.text || `📎 ${m.attachments.length} file(s)`;
    const short = label.length > 40 ? label.slice(0, 40) + "…" : label;
    const chip = el("div", "queue-chip text-xs");
    const txt = el("span", "truncate max-w-[220px]", `"${short}"`);
    txt.title = label;
    chip.appendChild(txt);
    const x = el("button", "chip-x", "✕");
    x.title = "Remove from queue";
    x.addEventListener("click", () => removeQueued(i));
    chip.appendChild(x);
    list.appendChild(chip);
  });
  q.appendChild(list);
}

// Drop a still-pending follow-up before the agent gets to it.
function removeQueued(i) {
  state.queue.splice(i, 1);
  renderQueue();
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
  // Keep the in-memory transcript in sync with what the server persists, so
  // switching away and back (which re-renders from s.messages) still shows the
  // user's own messages — not just the assistant's.
  if (s.messages) {
    s.messages.push({
      role: "user",
      content: text,
      ts: userTs,
      attachments: attachments.map((a) => ({ name: a.name, type: a.type, size: a.size })),
    });
  }
  scrollMessages(true);

  // Assistant text bubbles are created lazily, one per segment, so tool pills can
  // be interleaved between them in stream order (see handleEvent).
  startTimer();
  setStatus("working", "Working…");
  clearTools();
  state.openToolBubbles = {};
  state.openToolRow = null;
  showTyping();  // blinking dots until the first token / through tool calls

  const ctx = { bubble: null, answer: "", full: "" };
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
        handleEvent(evt, ctx);
        // The turn is over on a terminal event — stop reading instead of
        // blocking on the kept-alive socket that the server won't close. A `done`
        // carrying `next_mode` is NOT terminal: the pipeline is auto-advancing to
        // the next stage, which streams more events on this same connection.
        if ((evt.type === "done" && !evt.next_mode) || evt.type === "error") {
          finished = true;
        }
      }
    }
  } catch (e) {
    console.error(e);
    setStatus("error", "Connection error");
    ensureStreamBubble(ctx);
    ctx.bubble.textContent = ctx.full || "⚠️ The stream was interrupted.";
  } finally {
    if (reader) { try { await reader.cancel(); } catch (_) { /* already closed */ } }
    closeStreamBubble(ctx);
    hideTyping();
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

// ── Streaming-bubble lifecycle ─────────────────────────────────────────
// A blinking-dots bubble pinned at the bottom of the chat while the turn is
// active but no text is currently streaming (before the first token and across
// tool calls). Reused across turns; just detached/re-appended.
function showTyping() {
  let wrap = state.typingEl;
  if (!wrap) {
    wrap = el("div", "flex flex-col gap-1 max-w-[80%] self-start items-start");
    const bubble = el("div", AI_BUBBLE_CLS + " flex items-center py-3");
    for (let i = 0; i < 3; i++) bubble.appendChild(el("span", "typing-dot"));
    wrap.appendChild(bubble);
    state.typingEl = wrap;
  }
  $("messages").appendChild(wrap);  // (re)pin it to the very bottom
  scrollMessages();
}

function hideTyping() {
  if (state.typingEl) state.typingEl.remove();
}

// Open the current streaming text bubble (creating one if the previous segment
// was closed by a tool call). `ctx.answer` holds only the current segment's text;
// `ctx.full` holds the whole turn for persistence/fallback.
function ensureStreamBubble(ctx) {
  if (!ctx.bubble) {
    hideTyping();  // the bubble's own ▋ cursor takes over the loading signal
    ctx.bubble = makeAssistantBubble().bubble;
    ctx.answer = "";
    ctx.bubble.classList.add("streaming");
  }
  return ctx.bubble;
}

// Finalise the current segment so the next token starts a fresh bubble below
// whatever tool pill comes next.
function closeStreamBubble(ctx) {
  if (ctx.bubble) {
    ctx.bubble.classList.remove("streaming");
    ctx.bubble = null;
    state.openToolRow = null;  // text ended; next tools start a fresh row below it
  }
}

// ── SSE event handling ─────────────────────────────────────────────────
function handleEvent(evt, ctx) {
  switch (evt.type) {
    case "token":
      ensureStreamBubble(ctx);
      ctx.answer += evt.text;
      ctx.full += evt.text;
      if (state.tokenChunks === 0) state.firstTokenAt = performance.now();
      state.tokenChunks++;
      ctx.bubble.innerHTML = marked.parse(ctx.answer);
      scrollMessages();
      break;
    case "tool":
      updateTool(evt.name, evt.phase);
      // A tool starting ends the current text segment so the pill reads after the
      // text it follows, and any post-tool text opens a new bubble below it.
      if (evt.phase === "start") closeStreamBubble(ctx);
      renderToolBubble(evt);
      showTyping();  // keep the loading signal pinned below the tool pill
      break;
    case "status":
      break;
    case "usage":
      // Live mid-run stats snapshot from the orchestrator graph: update the
      // meters as the pipeline progresses rather than only on `done`.
      applyLiveStats(evt);
      break;
    case "stage":
      // code-assistant orchestrator entered a new pipeline stage: close the current
      // text segment, drop a dedicated stage bubble, and keep the loader pinned.
      closeStreamBubble(ctx);
      addStageBanner(evt.label);
      state.openToolRow = null;
      setStatus("working", `${evt.label}…`);
      showTyping();
      break;
    case "done":
      // Only fall back to a single bubble when nothing streamed (e.g. the model
      // returned its whole answer at once); otherwise keep the segmented view.
      if (!ctx.full && evt.answer) {
        ensureStreamBubble(ctx);
        ctx.answer = ctx.full = evt.answer;
        ctx.bubble.innerHTML = marked.parse(evt.answer);
      }
      closeStreamBubble(ctx);
      applyDoneStats(evt);
      mergeSessionStats(evt, ctx.full);
      if (evt.next_mode) {
        // Pipeline is auto-advancing: this stage finished but the next one is
        // about to stream on the same connection. Mark the handoff, reset the
        // stream context so the next stage gets a fresh bubble, and keep the
        // "working" status instead of declaring the whole turn done.
        addStageDivider(evt.next_mode);
        ctx.full = "";
        ctx.answer = "";
        state.openToolRow = null;
        setStatus("working", `Switched to ${evt.next_mode}…`);
        showTyping();
      } else {
        setStatus(evt.stopped ? "idle" : "done",
                  evt.stopped ? `Stopped after ${evt.elapsed}s` : `Done in ${evt.elapsed}s`);
      }
      break;
    case "error":
      setStatus("error", "Agent error");
      ensureStreamBubble(ctx);
      ctx.bubble.innerHTML = marked.parse((ctx.full ? ctx.full + "\n\n" : "") + "⚠️ " + evt.message);
      break;
  }
}

function mergeSessionStats(evt, answer) {
  const s = activeSession();
  if (!s) return;
  s.totals = evt.totals;
  s.last = evt.last;

  // The agent may have changed the language mid-turn via the set_language tool;
  // reflect that in the session, the header switcher and the subtitle.
  if (evt.language && evt.language !== s.language) {
    s.language = evt.language;
    const lsw = $("language-switch");
    if (lsw) lsw.value = evt.language;
    $("chat-subtitle").textContent = subtitleFor(s);
  }

  // The stage may have called a change_mode_* tool to hand off to the next
  // pipeline stage; the server already advanced the role and will auto-run it.
  // Mirror that here: update the session, the header role switcher, the tool
  // checkboxes and the subtitle, just like a manual role switch.
  if (evt.next_mode && evt.next_mode !== s.role) {
    s.role = evt.next_mode;
    s.enabled_tools = (state.toolsByRole[s.role] || []).slice();
    const rsw = $("role-switch");
    if (rsw) rsw.value = s.role;
    renderTools(s);
    $("chat-subtitle").textContent = subtitleFor(s);
  }

  // Backfill the message to the state session history array so it re-renders
  // as markdown (not raw text) when the session is reselected.
  if (s.messages) {
    s.messages.push({
      role: "assistant",
      content: evt.answer || answer || "",
      ts: evt.ts || new Date().toISOString(),
      parts: evt.parts || null,
    });
  }
}
