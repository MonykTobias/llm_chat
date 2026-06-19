"""Go tools: golangci-lint, go test, and go build — all run inside the `crav-go`
container so the host needs no Go toolchain.

The image (see ``docker/crav-go.Dockerfile``) ships the Go toolchain plus
golangci-lint. Deps declared in `go.mod` are pre-fetched into a derived image by
`_ensure_project_image`: `go mod download` is baked into the image layer so the
module cache is already warm when tool containers start. The derived image tag is
keyed on `go.sum` content (exact module hashes), falling back to `go.mod` when no
sum file exists. Repeated reviews of the same project pay the download cost only
when the manifest changes.

The build cache (compilation artefacts) is not persisted — it is only useful for
incremental rebuilds of the same binary and provides no benefit across ephemeral
containers that each compile from scratch.

Type-checking and import-resolution are both done by the compiler, so
run_type_check and check_imports both delegate to `go build ./...`.
"""
from __future__ import annotations

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

_IMAGE = "crav-go"
_COMPILE_IMAGE = _IMAGE
_COMPILE_CMD = "go build ./..."

# `go build` errors look like `./pkg/x.go:10:6: undefined: foo`.
_GO_ERR_RE = re.compile(r"^(.+?\.go):(\d+):(\d+):\s*(.*)$")


# ── per-project derived image ─────────────────────────────────────────────
# `go mod download` fetches all modules declared in go.mod into the local
# module cache without compiling anything. We bake it into a RUN layer so the
# cache is already populated when tool containers start.
#
# Cache key is go.sum (cryptographic hashes of every module version) when
# present — tighter than go.mod since it pins the exact content of each
# dependency, not just the version range. Falls back to go.mod.

def _go_mod_download_cmd(_manifest: str) -> str:
    # GONOSUMCHECK / GOFLAGS not needed inside the container; plain download.
    # || true: a private/unavailable module must not abort the image build.
    return "go mod download || true"


def _project_image(path: str) -> "tuple[str, str | None]":
    """Return (image_tag, warning) for the Go project at `path`."""
    return _ensure_project_image(
        base_image=_IMAGE,
        project_path=path,
        # go.sum first: cryptographic content hash of every dep version;
        # tighter cache key than go.mod alone.
        manifests=["go.sum", "go.mod"],
        install_cmd_fn=_go_mod_download_cmd,
    )


def _prepend_warning(warning: "str | None", result: str) -> str:
    return f"{warning}\n{result}" if warning else result


# ── linter / tests / type check ──────────────────────────────────────────

def run_linter(path: str, language: str) -> str:
    image, warn = _project_image(path)
    result = _run_docker_text(image, "golangci-lint run ./...",
                              path, name="golangci-lint", timeout=300)
    return _prepend_warning(warn, result)


def run_tests(path: str, language: str, include_coverage: bool = True) -> str:
    image, warn = _project_image(path)
    cmd = "go test ./..." + (" -cover" if include_coverage else "")
    result = _run_docker_text(image, cmd, path, name="go test", timeout=300)
    return _prepend_warning(warn, result)


def run_type_check(path: str, language: str) -> str:
    # Go is compiled — a successful build IS the type check.
    image, warn = _project_image(path)
    result = _run_docker_text(image, "go build ./...",
                              path, name="go build", timeout=300)
    return _prepend_warning(warn, result)


def check_imports(path: str, language: str) -> str:
    """Classify `go build ./...` output into the unified import report.

    Go's compiler is the source of truth: unresolved packages, unused imports and
    import cycles are all hard build errors, so a single build surfaces all three.
    """
    image, warn = _project_image(path)
    dr = _run_docker(image, "go build ./...", path, timeout=300)
    if dr.error is not None:
        return _prepend_warning(warn, dr.error)
    broken, unused, cycles = _parse_go_imports(dr.output)
    return _prepend_warning(warn, format_report("go", broken, unused, cycles))


# ── go build output parsing ───────────────────────────────────────────────

# A located build diagnostic: `./pkg/x.go:10:6: <message>`.
_GO_DIAG_RE = re.compile(r"^(.+?\.go):(\d+):(?:(\d+):)?\s*(.*)$")
# `"fmt" imported and not used`  /  `imported and not used: "fmt"`.
_GO_UNUSED_RE = re.compile(r'"([^"]+)"\s+imported and not used'
                           r'|imported and not used:?\s*"([^"]+)"')
# `package x is not in std` / `cannot find package "x"` / `no required module
# provides package x` — the unresolved-import shapes.
_GO_MISSING_RE = re.compile(
    r'(?:cannot find package "([^"]+)")'
    r'|(?:package ([^\s]+) is not in std)'
    r'|(?:no required module provides package ([^\s;]+))')
# Cycle block:  `package a` / `\timports b` / `\timports a: import cycle not allowed`.
_GO_PKG_RE = re.compile(r"^package (\S+)$")
_GO_IMPORTS_RE = re.compile(r"^\s+imports (\S+?)(: import cycle not allowed)?$")


def _parse_go_imports(out: str) -> "tuple[list, list, list]":
    """Split `go build` output into (broken, unused, cycles) for the report."""
    broken: list[tuple] = []
    unused: list[tuple] = []
    cycles: list[list[str]] = []
    chain: list[str] = []
    for line in out.splitlines():
        # ── import-cycle block (multi-line, no file:line prefix) ──────────
        pkg = _GO_PKG_RE.match(line)
        if pkg:
            chain = [pkg.group(1)]
            continue
        imp = _GO_IMPORTS_RE.match(line)
        if imp and chain:
            chain.append(imp.group(1))
            if imp.group(2):                       # ": import cycle not allowed"
                cycles.append(chain)
                chain = []
            continue

        # ── located diagnostics (file:line:col: message) ─────────────────
        m = _GO_DIAG_RE.match(line.strip())
        if not m:
            continue
        file = _compile_path(m.group(1))
        lineno = int(m.group(2))
        msg = m.group(4).strip()
        um = _GO_UNUSED_RE.search(msg)
        if um:
            pkg_path = um.group(1) or um.group(2)
            unused.append((file, lineno, f'import "{pkg_path}"',
                           pkg_path.rsplit("/", 1)[-1]))
            continue
        mm = _GO_MISSING_RE.search(msg)
        if mm:
            pkg_path = mm.group(1) or mm.group(2) or mm.group(3)
            broken.append((file, lineno, f'import "{pkg_path}"', msg))
    return broken, unused, cycles


def compile_code(path: str, language: str) -> CompileOutput:
    """Compile the whole module tree with `go build ./...` inside a container."""
    image, warn = _project_image(path)
    dr = _run_docker(image, _COMPILE_CMD, path)
    errors = _parse_go_build(dr.output) if dr.error is None else []
    warnings = [warn] if warn else []
    return _compile_result(dr, language, "go build", errors, warnings)


def _parse_go_build(out: str) -> "list[ErrorOutput]":
    errors: list[ErrorOutput] = []
    for line in out.splitlines():
        m = _GO_ERR_RE.match(line.strip())
        if not m:
            continue
        errors.append(ErrorOutput(
            file=_compile_path(m.group(1)), line=int(m.group(2)),
            column=int(m.group(3)), message=m.group(4).strip()))
    return errors