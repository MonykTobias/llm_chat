"""JavaScript / TypeScript tools: eslint, the project's test script, and tsc —
all run inside the `crav-node` container so the host needs no Node toolchain.

The image (see ``docker/crav-node.Dockerfile``) ships a global eslint and
typescript so linting and type-checking work even when the project ships
neither; a project-local ``node_modules/.bin`` binary is preferred when present.

Project deps are pre-installed into a derived image by `_ensure_project_image`:
`npm ci` when a lockfile exists (reproducible), else `npm install`. The derived
image tag is keyed on the content of package-lock.json (or package.json when no
lockfile exists), so it auto-invalidates on dependency changes and repeated
reviews of the same project pay the install cost only once.

When the project has no eslint config, the bundled flat config in
``configs/eslint.config.mjs`` is mounted in and used.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from structured_output import CompileOutput, ErrorOutput
from tools._common import _read_text_or_none
from .base import (
    CONFIG_DIR,
    _compile_path,
    _compile_result,
    _ensure_project_image,
    _run_docker,
    _run_docker_text,
)
from .imports import format_report

_IMAGE = "crav-node"
_COMPILE_IMAGE = _IMAGE

# `node --check` syntax-checks one file at a time; loop over the project's .js
# (skipping node_modules) and fail if any file fails.
_JS_COMPILE_CMD = (
    "rc=0; for f in $(find . -name '*.js' -not -path './node_modules/*'); do "
    'node --check "$f" || rc=1; done; exit $rc'
)
# tsc: prefer the project-local binary; otherwise the image's global tsc. With no
# tsconfig, type-check the loose .ts files.
_TS_COMPILE_CMD = (
    'if [ -x node_modules/.bin/tsc ]; then TSC=node_modules/.bin/tsc; '
    'else TSC=tsc; fi; '
    'if [ -f tsconfig.json ]; then FILES=""; '
    "else FILES=\"$(find . -name '*.ts' -not -path './node_modules/*')\"; fi; "
    '"$TSC" --noEmit --pretty false $FILES'
)

# node --check: a `<file>:<line>` header line, then later a `SyntaxError: msg`.
_NODE_HEAD_RE = re.compile(r"^(.+?\.js):(\d+)$")
# tsc --pretty false: `src/a.ts(12,5): error TS2304: Cannot find name 'x'.`
_TSC_RE = re.compile(
    r"^(.+?)\((\d+),(\d+)\):\s*(error|warning)\s+TS\d+:\s*(.*)$")

_ESLINT_CONFIGS = (
    ".eslintrc", ".eslintrc.js", ".eslintrc.cjs", ".eslintrc.json",
    ".eslintrc.yml", ".eslintrc.yaml",
    "eslint.config.js", "eslint.config.mjs", "eslint.config.cjs", "eslint.config.ts",
)

# Shell snippet picking the project-local tsc if present, else the global one.
_PICK_TSC = ('if [ -x node_modules/.bin/tsc ]; then TSC=node_modules/.bin/tsc; '
             'else TSC=tsc; fi; "$TSC"')


# ── helpers ───────────────────────────────────────────────────────────────

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


# ── per-project derived image ─────────────────────────────────────────────
# We key the cache on package-lock.json when it exists (exact reproducible
# install), otherwise on package.json (best-effort install). Either way, the
# install command matches: `npm ci` with a lockfile, `npm install` without.

def _npm_install_cmd(manifest_name: str) -> str:
    """Return the npm install command to bake into the derived image RUN layer."""
    if manifest_name == "package-lock.json":
        # lockfile present → reproducible install; fall back to npm install if
        # ci fails (e.g. lockfile out of sync) so the image still builds.
        return "npm ci || npm install"
    return "npm install"


def _project_image(path: str) -> "tuple[str, str | None]":
    """Return (image_tag, warning) for the JS/TS project at `path`.

    Prefers package-lock.json as the cache key (exact install); falls back to
    package.json. If neither exists the base image is returned as-is.
    """
    return _ensure_project_image(
        base_image=_IMAGE,
        project_path=path,
        # lockfile first: tighter cache key and reproducible install
        manifests=["package-lock.json", "package.json"],
        install_cmd_fn=_npm_install_cmd,
    )


def _prepend_warning(warning: "str | None", result: str) -> str:
    return f"{warning}\n{result}" if warning else result


# ── linter / tests / type check ──────────────────────────────────────────

def run_linter(path: str, language: str) -> str:
    image, warn = _project_image(path)
    flags, mounts = "", None
    if not _has_eslint_config(path):
        # Bundled default flat config — built-in rules only, no plugins required.
        flags = " --config /config/eslint.config.mjs"
        mounts = [(str(CONFIG_DIR), "/config")]
    cmd = (
        'if [ -x node_modules/.bin/eslint ]; then ESLINT=node_modules/.bin/eslint; '
        f'else ESLINT=eslint; fi; "$ESLINT"{flags} .'
    )
    result = _run_docker_text(image, cmd, path, name="eslint", timeout=300,
                              mounts=mounts)
    return _prepend_warning(warn, result)


def run_tests(path: str, language: str, include_coverage: bool = True) -> str:
    pkg = _read_package_json(path)
    if not pkg or "test" not in (pkg.get("scripts") or {}):
        return ("No `test` script found in package.json. Add one (e.g. "
                "\"test\": \"jest\" / \"vitest run\") so the suite can run.")
    image, warn = _project_image(path)
    cov = " -- --coverage" if include_coverage else ""   # jest & vitest both accept it
    result = _run_docker_text(image, f"npm test{cov}", path, name="npm test",
                              timeout=300)
    return _prepend_warning(warn, result)


def run_type_check(path: str, language: str) -> str:
    if language != "typescript":
        return ("Type checking is not applicable to plain JavaScript. Add TypeScript "
                "(tsc --checkJs with JSDoc types) or rely on run_linter for static "
                "checks.")
    image, warn = _project_image(path)
    cmd = f"{_PICK_TSC} --noEmit --pretty false"
    result = _run_docker_text(image, cmd, path, name="tsc", timeout=300)
    return _prepend_warning(warn, result)


def check_imports(path: str, language: str) -> str:
    """Unified import report for JS *and* TS via dependency-cruiser + eslint.

    One container run emits two JSON blobs (split on markers): dependency-cruiser
    gives broken (unresolvable), circular and orphan modules; eslint's
    `no-unused-vars` on import lines gives unused imports. Works the same for plain
    JavaScript and TypeScript (dependency-cruiser is language-agnostic).
    """
    image, warn = _project_image(path)
    has_tsconfig = (Path(path) / "tsconfig.json").is_file()
    tsflag = " --ts-config tsconfig.json" if has_tsconfig else ""
    is_ts = language == "typescript"
    if is_ts:
        # TS: native unused-import detection (tsc TS6133). eslint's default parser
        # can't read TS syntax, so we don't use it here.
        files = ("" if has_tsconfig
                 else "$(find . -name '*.ts' -not -path './node_modules/*')")
        unused_cmd = f"{_PICK_TSC} --noEmit --noUnusedLocals --pretty false {files}"
    else:
        eslint_cfg = ("" if _has_eslint_config(path)
                      else " --config /config/eslint.config.mjs")
        unused_cmd = (
            "if [ -x node_modules/.bin/eslint ]; then ESLINT=node_modules/.bin/eslint; "
            f'else ESLINT=eslint; fi; "$ESLINT" --format json{eslint_cfg} .')
    shell = (
        'echo "<<<DEPCRUISE>>>"; '
        + f"depcruise --config /config/dependency-cruiser.cjs "
          f"--output-type json{tsflag} . 2>/dev/null; "
        + 'echo "<<<UNUSED>>>"; ' + unused_cmd + " 2>/dev/null; true")
    dr = _run_docker(image, shell, path, timeout=300,
                     mounts=[(str(CONFIG_DIR), "/config")])
    if dr.error is not None:
        return _prepend_warning(warn, dr.error)
    dep_blob, _, unused_blob = dr.output.partition("<<<UNUSED>>>")
    dep_blob = dep_blob.partition("<<<DEPCRUISE>>>")[2]
    broken, cycles, orphans = _parse_depcruise(dep_blob)
    unused = (_parse_tsc_unused(unused_blob, path) if is_ts
              else _parse_eslint_unused(unused_blob, path))
    notes: list[str] = []
    if orphans:
        shown = ", ".join(orphans[:20]) + (" …" if len(orphans) > 20 else "")
        notes.append(f"ORPHAN MODULES ({len(orphans)}): {shown}  — files not "
                     "imported anywhere (possible dead code).")
    return _prepend_warning(warn, format_report(language, broken, unused, cycles, notes))


# ── dependency-cruiser output parsing ────────────────────────────────────

def _depcruise_cycle(violation: dict) -> "list[str]":
    """Render a no-circular violation's `cycle` (strings or {name} objects)."""
    out = []
    for c in (violation.get("cycle") or []):
        out.append(c.get("name", "") if isinstance(c, dict) else c)
    return [x for x in out if x]


def _parse_depcruise(blob: str) -> "tuple[list, list, list]":
    """Pull (broken, cycles, orphans) out of dependency-cruiser JSON."""
    broken: list[tuple] = []
    cycles: list[list[str]] = []
    orphans: list[str] = []
    blob = blob.strip()
    if not blob:
        return broken, cycles, orphans
    try:
        data = json.loads(blob)
    except ValueError:
        return broken, cycles, orphans
    for v in ((data.get("summary") or {}).get("violations") or []):
        rule = (v.get("rule") or {}).get("name")
        frm, to = v.get("from", ""), v.get("to", "")
        if rule == "no-unresolvable":
            broken.append((frm, 0, f"import '{to}'", f"unresolvable module '{to}'"))
        elif rule == "no-circular":
            chain = _depcruise_cycle(v)
            cycles.append([frm] + chain if chain else [frm, to])
        elif rule == "no-orphans":
            orphans.append(frm)
    return broken, cycles, orphans


_ESLINT_NAME_RE = re.compile(r"^'([^']+)'")
# A source line that is (the start of) an import / require / re-export.
_IMPORT_LINE_RE = re.compile(r"^\s*import\b|\bfrom\s+['\"]|require\s*\(")


def _parse_eslint_unused(blob: str, project_path: str) -> "list[tuple]":
    """eslint `no-unused-vars` findings whose source line is an import/require."""
    unused: list[tuple] = []
    start = blob.find("[")
    if start == -1:
        return unused
    try:
        data = json.loads(blob[start:])
    except ValueError:
        return unused
    cache: dict[str, list[str]] = {}
    for res in data:
        rel = _compile_path(res.get("filePath", ""))
        for msg in (res.get("messages") or []):
            if msg.get("ruleId") not in ("no-unused-vars",
                                         "@typescript-eslint/no-unused-vars"):
                continue
            line = msg.get("line", 0)
            src = _source_line(cache, project_path, rel, line)
            if src and _IMPORT_LINE_RE.search(src):
                nm = _ESLINT_NAME_RE.search(msg.get("message", ""))
                unused.append((rel, line, src.strip(), nm.group(1) if nm else ""))
    return unused


def _parse_tsc_unused(blob: str, project_path: str) -> "list[tuple]":
    """TS unused-import findings from `tsc --noUnusedLocals` (TS6133), import lines only."""
    unused: list[tuple] = []
    cache: dict[str, list[str]] = {}
    for line in blob.splitlines():
        m = _TSC_RE.match(line.strip())
        if not m:
            continue
        file, ln, _col, _sev, msg = m.groups()
        if "never read" not in msg and "never used" not in msg:
            continue
        rel, line_no = _compile_path(file), int(ln)
        src = _source_line(cache, project_path, rel, line_no)
        if src and _IMPORT_LINE_RE.search(src):
            nm = _ESLINT_NAME_RE.search(msg)
            unused.append((rel, line_no, src.strip(), nm.group(1) if nm else ""))
    return unused


def _source_line(cache: dict, project_path: str, rel: str, line: int) -> str:
    """Read one 1-based source line from the host project (cached per file)."""
    if rel not in cache:
        txt = _read_text_or_none(str(Path(project_path) / rel))
        cache[rel] = txt.splitlines() if txt else []
    lines = cache[rel]
    return lines[line - 1] if 0 < line <= len(lines) else ""


# ── compile ───────────────────────────────────────────────────────────────

def compile_code(path: str, language: str) -> CompileOutput:
    """TypeScript: `tsc --noEmit` (a real compile). Plain JS has no compile step,
    so `node --check` syntax-checks each file instead.

    TypeScript uses the project image so tsc can resolve types from node_modules.
    Plain JS syntax-checking needs no deps, so it runs against the base image.
    """
    if language == "typescript":
        image, warn = _project_image(path)
        dr = _run_docker(image, _TS_COMPILE_CMD, path)
        errors, warnings = (_parse_tsc(dr.output) if dr.error is None else ([], []))
        if warn:
            warnings = [warn] + warnings
        return _compile_result(dr, language, "tsc", errors, warnings)
    # Plain JS: syntax-only, no deps needed.
    dr = _run_docker(_COMPILE_IMAGE, _JS_COMPILE_CMD, path)
    errors = _parse_node_check(dr.output) if dr.error is None else []
    return _compile_result(dr, language, "node --check", errors, [])


def _parse_tsc(out: str) -> "tuple[list[ErrorOutput], list[str]]":
    errors: list[ErrorOutput] = []
    warnings: list[str] = []
    for line in out.splitlines():
        m = _TSC_RE.match(line.strip())
        if not m:
            continue
        file, lineno, col, severity, message = m.groups()
        if severity == "warning":
            warnings.append(f"{_compile_path(file)}:{lineno}: {message}")
        else:
            errors.append(ErrorOutput(
                file=_compile_path(file), line=int(lineno),
                column=int(col), message=message.strip()))
    return errors, warnings


def _parse_node_check(out: str) -> "list[ErrorOutput]":
    """Each failing file emits a `<file>:<line>` header followed by `XxxError: …`."""
    lines = out.splitlines()
    errors: list[ErrorOutput] = []
    for i, line in enumerate(lines):
        m = _NODE_HEAD_RE.match(line.strip())
        if not m:
            continue
        message = "SyntaxError"
        for nxt in lines[i + 1:i + 8]:
            if re.match(r"^\s*\w*Error:", nxt):
                message = nxt.strip()
                break
        errors.append(ErrorOutput(
            file=_compile_path(m.group(1)), line=int(m.group(2)),
            column=0, message=message))
    return errors