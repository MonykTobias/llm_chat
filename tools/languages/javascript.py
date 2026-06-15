"""JavaScript / TypeScript tools: eslint, the project's test script, and tsc.

Required on the host / in the project for full results:
    * Node.js + npm                — the toolchain (system-wide)
    * the project's `node_modules` — installed deps + local tool binaries
    * eslint                       — linter            (run_linter)
    * a test script in package.json (jest / vitest / mocha …) — run_tests
    * typescript (`tsc`)           — type check + import check for .ts (TS only)
When the project ships no eslint config, the bundled flat config in
`configs/eslint.config.mjs` is used (built-in rules only, no plugins needed).

Auto-provisioning: when `node_modules` is missing we run `npm ci` (if a lockfile
exists) or `npm install`, once per project per server run.
"""
from __future__ import annotations

import json
from pathlib import Path

from .base import (
    CONFIG_DIR,
    _env_msg,
    _exe,
    _project_bin,
    _provision_once,
    _run,
    _tool_or_msg,
)

_ESLINT_CONFIGS = (
    ".eslintrc", ".eslintrc.js", ".eslintrc.cjs", ".eslintrc.json",
    ".eslintrc.yml", ".eslintrc.yaml",
    "eslint.config.js", "eslint.config.mjs", "eslint.config.cjs", "eslint.config.ts",
)


def _provision(project: str) -> None:
    # Only a real Node project (has package.json) and only when deps aren't there.
    if not (Path(project) / "package.json").is_file():
        return
    if (Path(project) / "node_modules").is_dir():
        return
    npm = _exe("npm")
    if not npm:
        return
    has_lock = (Path(project) / "package-lock.json").exists()
    _run([npm, "ci" if has_lock else "install"], cwd=project, timeout=600)


def _read_package_json(project: str) -> "dict | None":
    p = Path(project) / "package.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _has_eslint_config(project: str) -> bool:
    root = Path(project)
    if any((root / name).exists() for name in _ESLINT_CONFIGS):
        return True
    pkg = _read_package_json(project)
    return bool(pkg and pkg.get("eslintConfig"))


def _eslint_cmd(project: str) -> "list[str] | None":
    """Prefer the project-local eslint, then npx, then a global eslint."""
    local = _project_bin(project, "node_modules/.bin/eslint.cmd",
                         "node_modules/.bin/eslint")
    if local:
        return [local]
    npx = _exe("npx")
    if npx:
        return [npx, "--no-install", "eslint"]
    eslint = _exe("eslint")
    return [eslint] if eslint else None


def _tsc_cmd(project: str) -> "list[str] | None":
    local = _project_bin(project, "node_modules/.bin/tsc.cmd",
                         "node_modules/.bin/tsc")
    if local:
        return [local]
    npx = _exe("npx")
    if npx:
        return [npx, "--no-install", "tsc"]
    tsc = _exe("tsc")
    return [tsc] if tsc else None


def run_linter(path: str, language: str) -> str:
    _provision_once(path, "javascript", _provision)
    cmd = _eslint_cmd(path)
    if cmd is None:
        return _env_msg("eslint")
    args = [*cmd, "."]
    if not _has_eslint_config(path):
        # Bundled default flat config — built-in rules only, no plugins required.
        args += ["--config", str(CONFIG_DIR / "eslint.config.mjs")]
    return _run(args, cwd=path, timeout=300)


def run_tests(path: str, language: str, include_coverage: bool = True) -> str:
    _provision_once(path, "javascript", _provision)
    npm, msg = _tool_or_msg("npm")
    if msg:
        return msg
    pkg = _read_package_json(path)
    if not pkg or "test" not in (pkg.get("scripts") or {}):
        return ("No `test` script found in package.json. Add one (e.g. "
                "\"test\": \"jest\" / \"vitest run\") so the suite can run.")
    cmd = [npm, "test"]
    if include_coverage:
        cmd += ["--", "--coverage"]   # jest & vitest both accept --coverage
    return _run(cmd, cwd=path, timeout=300)


def run_type_check(path: str, language: str) -> str:
    if language != "typescript":
        return ("Type checking is not applicable to plain JavaScript. Add TypeScript "
                "(tsc --checkJs with JSDoc types) or rely on run_linter for static "
                "checks.")
    cmd = _tsc_cmd(path)
    if cmd is None:
        return _env_msg("tsc (typescript)")
    return _run([*cmd, "--noEmit", "--pretty", "false"], cwd=path, timeout=300)


def check_imports(path: str, language: str) -> str:
    """Delegate to tsc for TS (it reports unresolved imports); note the JS limit."""
    if language == "typescript":
        cmd = _tsc_cmd(path)
        if cmd is None:
            return _env_msg("tsc (typescript)")
        out = _run([*cmd, "--noEmit", "--pretty", "false"], cwd=path, timeout=300)
        return ("== IMPORT CHECK (typescript via `tsc --noEmit`) ==\n"
                "Unresolved imports surface as TS2307 'Cannot find module' errors.\n\n"
                + out)
    return ("Static import checking for plain JavaScript is not built in. Run "
            "run_linter with eslint-plugin-import (rule `import/no-unresolved`) "
            "configured in the project, or migrate the files to TypeScript and use "
            "check_imports there.")
