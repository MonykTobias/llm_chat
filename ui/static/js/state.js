"use strict";

// ── Shared application state ────────────────────────────────────────────
// One mutable object imported (by live binding) across every module. Fields
// that are assigned lazily during a turn are declared up front so there is a
// single place to see everything the UI tracks.
export const state = {
  sessions: [],          // [{id,title,path,language,messages,...}]
  activeId: null,
  languages: [],
  models: [],
  roles: [],
  toolsByRole: {},
  roleModes: {},         // role -> "chat" | "project" | "coder"
  createMode: "project", // which type the new-session form is building
  streaming: false,      // is an agent turn currently running?
  streamingId: null,     // tracks which session is streaming
  queue: [],             // follow-up messages waiting to be sent: {text, attachments}
  attachments: [],       // pending files for the next message: {name,type,size,data}
  specContent: "",       // optional spec/context file text for a new coder session
  specName: "",          // its filename (shown in the drop zone)
  timer: null,           // live response-time interval handle
  turnStart: 0,

  // Set during a streaming turn (see stream.js / tools.js):
  openToolBubbles: {},   // name -> in-flight tool pill awaiting its "end" event
  openToolRow: null,     // current row tool pills append into
  typingEl: null,        // reused blinking-dots bubble
  tokenChunks: 0,        // streamed token chunks (for the live tok/s estimate)
  firstTokenAt: 0,
};

// Per-file size guard so a giant drop can't lock up the browser / blow context.
export const MAX_ATTACH_BYTES = 15 * 1024 * 1024; // 15 MB

// Shared class for assistant text bubbles.
export const AI_BUBBLE_CLS = "ai-bubble bubble-ai";

export function activeSession() {
  return state.sessions.find((s) => s.id === state.activeId);
}
