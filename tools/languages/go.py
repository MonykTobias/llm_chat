"""Go tools: golangci-lint / go vet, go test, and go build.

Required on the host for full results:
    * the Go toolchain (`go`) — system-wide; provides build / vet / test
    * golangci-lint           — optional, preferred linter; falls back to `go vet`
Go has no per-project virtual env: deps declared in `go.mod` are fetched into a
shared module cache. Type-checking and import-resolution are both done by the
compiler, so run_type_check and check_imports both `go build ./...`.

Auto-provisioning: `go mod download` (once per project) when a go.mod is present.
"""
from __future__ import annotations

from pathlib import Path

from .base import _exe, _provision_once, _run, _tool_or_msg


def _provision(project: str) -> None:
    if not (Path(project) / "go.mod").is_file():
        return
    go = _exe("go")
    if go:
        _run([go, "mod", "download"], cwd=project, timeout=600)


def run_linter(path: str, language: str) -> str:
    _provision_once(path, "go", _provision)
    golangci = _exe("golangci-lint")
    if golangci:
        return _run([golangci, "run", "./..."], cwd=path, timeout=300)
    # Fall back to the compiler's own vet checks when golangci-lint isn't present.
    go, msg = _tool_or_msg("go")
    if msg:
        return msg
    return ("[note] golangci-lint not found; falling back to `go vet`.\n\n"
            + _run([go, "vet", "./..."], cwd=path, timeout=180))


def run_tests(path: str, language: str, include_coverage: bool = True) -> str:
    _provision_once(path, "go", _provision)
    go, msg = _tool_or_msg("go")
    if msg:
        return msg
    cmd = [go, "test", "./..."]
    if include_coverage:
        cmd.append("-cover")
    return _run(cmd, cwd=path, timeout=300)


def run_type_check(path: str, language: str) -> str:
    _provision_once(path, "go", _provision)
    go, msg = _tool_or_msg("go")
    if msg:
        return msg
    # Go is compiled — a successful build IS the type check.
    return _run([go, "build", "./..."], cwd=path, timeout=300)


def check_imports(path: str, language: str) -> str:
    _provision_once(path, "go", _provision)
    go, msg = _tool_or_msg("go")
    if msg:
        return msg
    out = _run([go, "build", "./..."], cwd=path, timeout=300)
    return ("== IMPORT CHECK (go via `go build ./...`) ==\n"
            "Unresolved/unused imports surface as build errors "
            "(e.g. 'cannot find package', 'imported and not used').\n\n" + out)
