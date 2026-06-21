"use strict";

// Entry point: wire up the DOM, restore persisted UI state, load sessions.
import { $, autoGrow } from "./dom.js";
import { applyTheme, toggleTheme, applyPanelState, togglePanel } from "./layout.js";
import { addFiles, bindDragAndDrop, bindSpecDrop } from "./attachments.js";
import {
  refreshSessions, setCreateMode, startSession, browseFolder,
  switchModel, switchRole, switchLanguage, onStop,
} from "./session.js";
import { onSend } from "./stream.js";

function bindUI() {
  $("browse-btn").addEventListener("click", browseFolder);
  for (const btn of document.querySelectorAll("#mode-toggle .mode-btn")) {
    btn.addEventListener("click", () => setCreateMode(btn.dataset.mode));
  }
  $("start-btn").addEventListener("click", startSession);
  $("toggle-sidebar").addEventListener("click", () => togglePanel("sidebar"));
  $("toggle-stats").addEventListener("click", () => togglePanel("stats"));
  $("theme-toggle").addEventListener("click", toggleTheme);
  $("send-btn").addEventListener("click", onSend);
  $("stop-btn").addEventListener("click", onStop);
  $("model-switch").addEventListener("change", (e) => switchModel(e.target.value));
  $("role-switch").addEventListener("change", (e) => switchRole(e.target.value));
  $("language-switch").addEventListener("change", (e) => switchLanguage(e.target.value));

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
  bindSpecDrop();
}

async function init() {
  bindUI();
  applyTheme();
  applyPanelState();
  await refreshSessions();
}

init();
