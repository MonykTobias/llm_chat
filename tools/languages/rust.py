"""Rust tools: cargo clippy, cargo test, and cargo check — all run inside the
`crav-rust` container so the host needs no Rust toolchain.

The image (see ``docker/crav-rust.Dockerfile``) ships cargo/rustc plus the clippy
component. Crates declared in `Cargo.toml` are pre-fetched into a derived image
by `_ensure_project_image`: `cargo fetch` is baked into the image layer so the
registry cache is already warm when tool containers start. The derived image tag
is keyed on `Cargo.lock` content (exact versions), falling back to `Cargo.toml`
when no lockfile exists. Repeated reviews of the same project pay the fetch cost
only when the manifest changes.

Type-checking and import-resolution are both done by the compiler, so
run_type_check and check_imports both delegate to `cargo check`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from structured_output import CompileOutput, ErrorOutput
from .base import (
    _compile_path,
    _compile_result,
    _ensure_project_image,
    _run_docker,
    _run_docker_text,
)
from .imports import format_report

_IMAGE = "crav-rust"
_COMPILE_IMAGE = _IMAGE

# rustc/cargo print a severity line (`error[E0425]: msg` / `warning: msg`) then a
# `  --> src/main.rs:3:5` location line; pair them up.
_RS_HEAD_RE = re.compile(r"^(error|warning)(?:\[[^\]]+\])?:\s*(.*)$")
_RS_LOC_RE = re.compile(r"^\s*-->\s*(.+?):(\d+):(\d+)\s*$")


# ── per-project derived image ─────────────────────────────────────────────
# `cargo fetch` downloads all crates declared in Cargo.toml into the local
# registry without compiling anything. We bake it into a RUN layer so the
# registry is already populated when tool containers start — no per-container
# fetch needed. Cache key is Cargo.lock (exact versions) when present, else
# Cargo.toml (may re-fetch if versions resolve differently, but correct).

def _cargo_fetch_cmd(_manifest: str) -> str:
    # --locked: respect the lockfile exactly (no-op when keyed on Cargo.toml).
    # || true: a partial/private dep failure must not abort the image build.
    return "cargo fetch --locked 2>/dev/null || cargo fetch || true"


def _project_image(path: str) -> "tuple[str, str | None]":
    """Return (image_tag, warning) for the Rust project at `path`."""
    return _ensure_project_image(
        base_image=_IMAGE,
        project_path=path,
        # Cargo.lock first: exact reproducible fetch; fall back to Cargo.toml.
        manifests=["Cargo.lock", "Cargo.toml"],
        install_cmd_fn=_cargo_fetch_cmd,
    )


def _prepend_warning(warning: "str | None", result: str) -> str:
    return f"{warning}\n{result}" if warning else result


# ── linter / tests / type check ──────────────────────────────────────────

def run_linter(path: str, language: str) -> str:
    image, warn = _project_image(path)
    result = _run_docker_text(image, "cargo clippy --all-targets --quiet",
                              path, name="cargo clippy", timeout=300)
    return _prepend_warning(warn, result)


def run_tests(path: str, language: str, include_coverage: bool = True) -> str:
    # Stdlib cargo has no built-in coverage; `--nocapture` surfaces test output.
    image, warn = _project_image(path)
    cmd = "cargo test" + (" -- --nocapture" if include_coverage else "")
    result = _run_docker_text(image, cmd, path, name="cargo test", timeout=300)
    return _prepend_warning(warn, result)


def run_type_check(path: str, language: str) -> str:
    image, warn = _project_image(path)
    result = _run_docker_text(image, "cargo check --all-targets",
                              path, name="cargo check", timeout=300)
    return _prepend_warning(warn, result)


def check_imports(path: str, language: str) -> str:
    """Classify `cargo check` JSON diagnostics into the unified import report.

    Unresolved imports surface as E0432/E0433; unused ones as the
    `unused_imports` lint. Rust has no Python-style import cycles, so CIRCULAR
    carries a not-applicable note rather than a misleading "0".
    """
    image, warn = _project_image(path)
    dr = _run_docker(image, "cargo check --all-targets --message-format=json",
                     path, timeout=300)
    if dr.error is not None:
        return _prepend_warning(warn, dr.error)
    broken, unused = _parse_rust_imports(dr.output)
    return _prepend_warning(warn, format_report(
        "rust", broken, unused, [],
        circular_note=("Rust's module system has no import cycles; crate-level "
                       "cycles are prevented by Cargo.")))


_BT_RE = re.compile(r"`([^`]+)`")


def _parse_rust_imports(out: str) -> "tuple[list, list]":
    """Pull (broken, unused) out of `cargo check --message-format=json` output.

    Diagnostics whose primary span points into the crate registry (a dependency,
    not the reviewed code) are skipped so the report stays about this project.
    """
    broken: list[tuple] = []
    unused: list[tuple] = []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if obj.get("reason") != "compiler-message":
            continue
        msg = obj.get("message") or {}
        code = (msg.get("code") or {}).get("code")
        text = msg.get("message", "")
        spans = msg.get("spans") or []
        prim = next((s for s in spans if s.get("is_primary")),
                    spans[0] if spans else None)
        if prim is None:
            continue
        fname = prim.get("file_name", "")
        if (fname.startswith("/usr/local/cargo") or fname.startswith("/root/.cargo")
                or "/registry/" in fname):
            continue
        file = _compile_path(fname)
        lineno = prim.get("line_start", 0)
        if code in ("E0432", "E0433") or text.startswith("unresolved import"):
            m = _BT_RE.search(text)
            disp = f"use {m.group(1)}" if m else "unresolved import"
            broken.append((file, lineno, disp, text))
        elif code == "unused_imports" or text.startswith("unused import"):
            for name in (_BT_RE.findall(text) or [""]):
                unused.append((file, lineno, f"use {name}".strip(), name))
    return broken, unused


def compile_code(path: str, language: str) -> CompileOutput:
    """Compile with `cargo build` (Cargo project) or bare `rustc` for loose files."""
    if (Path(path) / "Cargo.toml").is_file():
        image, warn = _project_image(path)
        dr = _run_docker(image, "cargo build", path)
        errors, warnings = (_parse_rust(dr.output) if dr.error is None else ([], []))
        if warn:
            warnings = [warn] + warnings
        return _compile_result(dr, language, "cargo build", errors, warnings)
    # No manifest: compile each top-level .rs so a loose snippet still checks.
    # No deps to fetch, so use the base image directly.
    cmd = ('rc=0; for f in *.rs; do [ -e "$f" ] || continue; '
           'rustc --emit=metadata "$f" || rc=1; done; exit $rc')
    dr = _run_docker(_COMPILE_IMAGE, cmd, path)
    errors, warnings = (_parse_rust(dr.output) if dr.error is None else ([], []))
    return _compile_result(dr, language, "rustc", errors, warnings)


def _parse_rust(out: str) -> "tuple[list[ErrorOutput], list[str]]":
    """Pair each `error/warning:` head with its following `--> file:line:col`."""
    errors: list[ErrorOutput] = []
    warnings: list[str] = []
    lines = out.splitlines()
    for i, line in enumerate(lines):
        head = _RS_HEAD_RE.match(line.strip())
        if not head:
            continue
        severity, message = head.group(1), head.group(2).strip()
        loc = None
        for nxt in lines[i + 1:i + 4]:
            loc = _RS_LOC_RE.match(nxt)
            if loc:
                break
        if severity == "warning":
            where = (f" ({_compile_path(loc.group(1))}:{loc.group(2)})"
                     if loc else "")
            warnings.append(message + where)
        else:
            errors.append(ErrorOutput(
                file=_compile_path(loc.group(1)) if loc else "",
                line=int(loc.group(2)) if loc else 0,
                column=int(loc.group(3)) if loc else 0,
                message=message))
    return errors, warnings