"""Rust tools: cargo clippy, cargo test, and cargo check.

Required on the host for full results:
    * cargo + rustc          — the toolchain (system-wide, via rustup)
    * the `clippy` component  — for run_linter (`rustup component add clippy`)
Rust has no per-project virtual env: crates declared in `Cargo.toml` are fetched
into a shared registry cache. Type-checking and import-resolution are both done
by the compiler, so run_type_check and check_imports both `cargo check`.

Auto-provisioning: `cargo fetch` (once per project) when a Cargo.toml is present.
"""
from __future__ import annotations

from pathlib import Path

from .base import _provision_once, _run, _tool_or_msg


def _provision(project: str) -> None:
    if not (Path(project) / "Cargo.toml").is_file():
        return
    from .base import _exe
    cargo = _exe("cargo")
    if cargo:
        _run([cargo, "fetch"], cwd=project, timeout=600)


def run_linter(path: str, language: str) -> str:
    _provision_once(path, "rust", _provision)
    cargo, msg = _tool_or_msg("cargo")
    if msg:
        return msg
    # clippy is a cargo subcommand provided by the clippy component; if it's not
    # installed cargo prints a clear "no such subcommand" message, which _run returns.
    return _run([cargo, "clippy", "--all-targets", "--quiet"], cwd=path, timeout=300)


def run_tests(path: str, language: str, include_coverage: bool = True) -> str:
    _provision_once(path, "rust", _provision)
    cargo, msg = _tool_or_msg("cargo")
    if msg:
        return msg
    # Stdlib cargo has no built-in coverage; `--nocapture` surfaces test output.
    cmd = [cargo, "test"]
    if include_coverage:
        cmd += ["--", "--nocapture"]
    return _run(cmd, cwd=path, timeout=300)


def run_type_check(path: str, language: str) -> str:
    _provision_once(path, "rust", _provision)
    cargo, msg = _tool_or_msg("cargo")
    if msg:
        return msg
    return _run([cargo, "check", "--all-targets"], cwd=path, timeout=300)


def check_imports(path: str, language: str) -> str:
    _provision_once(path, "rust", _provision)
    cargo, msg = _tool_or_msg("cargo")
    if msg:
        return msg
    out = _run([cargo, "check", "--all-targets"], cwd=path, timeout=300)
    return ("== IMPORT CHECK (rust via `cargo check`) ==\n"
            "Unresolved imports surface as E0432 'unresolved import' / E0433 errors; "
            "unused ones as the `unused_imports` warning.\n\n" + out)
