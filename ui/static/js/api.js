"use strict";

// ── JSON fetch helpers ──────────────────────────────────────────────────
// Thin wrappers around the handful of JSON endpoints the UI talks to. They
// return the parsed body; callers inspect `data.error` exactly as before.
// (The streaming /api/chat endpoint is handled separately in stream.js, since
// it needs the raw response body.)

export async function getJSON(url) {
  const res = await fetch(url);
  return res.json();
}

export async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  return res.json();
}
