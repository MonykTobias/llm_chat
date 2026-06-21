"use strict";

import { AI_BUBBLE_CLS } from "./state.js";
import { $, el, fmtTime, fmtNum, scrollMessages } from "./dom.js";
import { attachmentThumb } from "./attachments.js";
import { buildToolBubble } from "./tools.js";

// ── Message rendering ──────────────────────────────────────────────────

// The "time · 4.2s · 1,234 tok · 18.3 tok/s" line shown under an assistant turn.
function bubbleMetaText(role, ts, meta) {
  let metaText = fmtTime(ts);
  if (role === "assistant" && meta && meta.elapsed != null) {
    metaText += `  ·  ${meta.elapsed}s`;
    if (meta.usage && meta.usage.total_tokens) metaText += `  ·  ${fmtNum(meta.usage.total_tokens)} tok`;
    if (meta.usage && meta.usage.output_tokens && meta.elapsed) {
      metaText += `  ·  ${(meta.usage.output_tokens / meta.elapsed).toFixed(1)} tok/s`;
    }
  }
  return metaText;
}

// A standalone assistant text bubble appended to the chat, in its own wrap so it
// reads as one segment among interleaved tool pills. Returns {wrap, bubble}.
export function makeAssistantBubble() {
  const wrap = el("div", "flex flex-col gap-1 max-w-[80%] self-start items-start");
  const bubble = el("div", AI_BUBBLE_CLS);
  wrap.appendChild(bubble);
  $("messages").appendChild(wrap);
  return { wrap, bubble };
}

// A centered "→ switched to <stage>" marker shown when the review pipeline
// auto-advances from one stage to the next within a single send.
export function addStageDivider(nextMode) {
  const wrap = el("div", "self-center my-2 text-xs opacity-60");
  wrap.appendChild(el("span", null, `→ switched to ${nextMode}`));
  $("messages").appendChild(wrap);
  scrollMessages();
}

// A dedicated, centered "bubble" announcing that the code-assistant orchestrator
// has moved to a new pipeline stage (explore / plan / act / verify). Distinct from
// text bubbles and tool pills so the stage hand-off is unmistakable in the chat.
export function addStageBanner(label) {
  const wrap = el("div", "self-center my-3 flex items-center gap-2 w-full max-w-[80%]");
  const line = () => { const d = el("div", "flex-1"); d.style.height = "1px"; d.style.background = "var(--border)"; return d; };
  wrap.appendChild(line());
  const tag = el("div", "px-3 py-1 rounded-full text-xs font-semibold whitespace-nowrap", label);
  tag.style.color = "var(--accent)";
  tag.style.background = "var(--accent-soft)";
  tag.style.border = "1px solid color-mix(in srgb, var(--accent) 35%, transparent)";
  wrap.appendChild(tag);
  wrap.appendChild(line());
  $("messages").appendChild(wrap);
  scrollMessages();
}

export function addMessageBubble(role, content, ts, meta) {
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

  const bubble = el("div", isUser ? "bubble-user" : AI_BUBBLE_CLS);

  if (role === "assistant" && content) {
    bubble.innerHTML = marked.parse(content);
  } else {
    bubble.textContent = content;
  }

  // Skip an empty bubble when a user message carries only attachments.
  if (content || !isUser || !atts.length) wrap.appendChild(bubble);
  wrap.appendChild(el("div", "bubble-meta", bubbleMetaText(role, ts, meta)));
  $("messages").appendChild(wrap);
  return bubble;
}

// Re-render a finished assistant turn from its persisted `parts`, interleaving
// text bubbles and tool pills in the exact order they occurred. The stats footer
// hangs under the last text bubble (or stands alone if the turn ended on a tool).
export function renderAssistantTurn(m) {
  let lastWrap = null;
  let toolRow = null;
  for (const part of m.parts) {
    if (part.type === "text") {
      if (!part.content) continue;
      const { wrap, bubble } = makeAssistantBubble();
      bubble.innerHTML = marked.parse(part.content);
      lastWrap = wrap;
      toolRow = null;
    } else if (part.type === "tool") {
      if (!toolRow) {
        toolRow = el("div", "self-start flex flex-row flex-wrap gap-2");
        $("messages").appendChild(toolRow);
      }
      toolRow.appendChild(buildToolBubble(part.name, part.target, true));
      lastWrap = null;
    } else if (part.type === "stage") {
      addStageBanner(part.label);
      lastWrap = null;
      toolRow = null;
    }
  }
  const footer = el("div", "bubble-meta", bubbleMetaText("assistant", m.ts, m));
  if (lastWrap) {
    lastWrap.appendChild(footer);
  } else {
    const fwrap = el("div", "flex flex-col self-start");
    fwrap.appendChild(footer);
    $("messages").appendChild(fwrap);
  }
}
