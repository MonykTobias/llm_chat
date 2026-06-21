"use strict";

import { state, MAX_ATTACH_BYTES, activeSession } from "./state.js";
import { $, el } from "./dom.js";

// ── Attachments (pending files for the next message) ───────────────────
export function addFiles(fileList) {
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

export function fileIcon(type, name) {
  type = type || "";
  if (type.startsWith("image/")) return "🖼️";
  if (type === "application/pdf" || /\.pdf$/i.test(name || "")) return "📕";
  return "📄";
}

export function renderAttachments() {
  const box = $("attachment-preview");
  box.innerHTML = "";
  if (!state.attachments.length) { box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  state.attachments.forEach((a, i) => {
    const chip = el("div", "chip");
    if ((a.type || "").startsWith("image/") && a.data) {
      const img = el("img", "w-7 h-7 object-cover rounded");
      img.src = a.data;
      chip.appendChild(img);
    } else {
      chip.appendChild(el("span", "text-base", fileIcon(a.type, a.name)));
    }
    chip.appendChild(el("span", "font-mono truncate max-w-[140px]", a.name));
    const x = el("button", "chip-x", "✕");
    x.title = "Remove";
    x.addEventListener("click", () => removeAttachment(i));
    chip.appendChild(x);
    box.appendChild(chip);
  });
}

// One attachment in a sent message: an image thumbnail when we still have the
// data (live turn), otherwise a labelled pill. Re-opened sessions only keep
// {name,type,size} metadata, so they always show the pill.
export function attachmentThumb(a) {
  if ((a.type || "").startsWith("image/") && a.data) {
    const img = el("img",
      "max-w-[180px] max-h-[180px] rounded-lg border border-blue-300 dark:border-blue-700");
    img.src = a.data;
    img.title = a.name;
    return img;
  }
  const pill = el("div", "chip");
  pill.appendChild(el("span", "text-base", fileIcon(a.type, a.name)));
  pill.appendChild(el("span", "font-mono truncate max-w-[180px]", a.name));
  pill.title = a.name;
  return pill;
}

// ── Drag & drop files onto the chat pane ───────────────────────────────
export function bindDragAndDrop() {
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

// ── Coder spec file (optional context loaded into context_store.spec_content) ──
export function bindSpecDrop() {
  const drop = $("spec-drop");
  const input = $("spec-input");
  if (!drop || !input) return;

  drop.addEventListener("click", () => input.click());
  input.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) loadSpecFile(e.target.files[0]);
    e.target.value = "";  // allow re-selecting the same file
  });

  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("dragover"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("dragover"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) loadSpecFile(e.dataTransfer.files[0]);
  });
}

function loadSpecFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    state.specContent = String(reader.result || "");
    state.specName = file.name;
    renderSpecName();
  };
  reader.onerror = () => alert(`Could not read "${file.name}".`);
  reader.readAsText(file);  // markdown / text / yaml -> plain text
}

export function clearSpec() {
  state.specContent = "";
  state.specName = "";
  renderSpecName();
}

function renderSpecName() {
  const drop = $("spec-drop");
  const label = $("spec-name");
  if (!drop || !label) return;
  if (state.specName) {
    label.textContent = `📄 ${state.specName} — click to replace`;
    drop.classList.add("loaded");
  } else {
    label.textContent = "Drop a .md / .txt / .yaml file, or click to choose";
    drop.classList.remove("loaded");
  }
}
