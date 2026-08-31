"""
Deterministic, host-side enumeration of which backend modules need an
auto-generated unit test, and where that test file should live.

No LLM, no LangChain — plain Python, in the same spirit as `utils/validation.py`.
The orchestrator calls `derive_test_targets` at the completion gate to enqueue a
test-writing task per untested backend logic module. The test SET is therefore
decided deterministically (truly *auto*-generated), while the test CONTENT is
written by the coder reading the real implementation.

Scope (per project decision): backend logic layers only — services, routes /
controllers, repositories, utils / lib / helpers. Models/schemas, configs,
entrypoints, __init__ files, and the whole frontend are skipped (a weak local
model can't reliably unit-test components without a DOM harness).

Layout (per project decision): Python → `tests/test_<name>.py`; JS/TS → co-located
`<dir>/<name>.test.<ext>`.
"""
import os

# Source extensions we can generate tests for.
_PY_EXTS = {".py"}
_JS_EXTS = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
_TESTABLE_EXTS = _PY_EXTS | _JS_EXTS

# Backend "logic" layers worth unit-testing (path-segment match; singular + plural).
_LOGIC_SEGMENTS = {
    "service", "services",
    "route", "routes",
    "controller", "controllers",
    "repository", "repositories", "repo", "repos",
    "util", "utils",
    "lib",
    "helper", "helpers",
}

# Frontend / non-backend roots we never auto-test.
_FRONTEND_SEGMENTS = {
    "frontend", "client", "web", "components", "pages", "ui", "views",
}

# Basenames (stems) that are never a meaningful unit-test target.
_SKIP_STEMS = {
    "__init__", "__main__", "conftest", "setup", "main", "app", "index",
    "config", "settings", "constants",
}


def _norm(path: str) -> str:
    """Forward slashes, no leading './' or '/'."""
    p = (path or "").replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    return p.lstrip("/")


def _is_test_file(path: str) -> bool:
    base = os.path.basename(_norm(path))
    stem, ext = os.path.splitext(base)
    if ext.lower() in _PY_EXTS:
        return stem.startswith("test_") or stem.endswith("_test")
    return ".test." in base or ".spec." in base


def _is_logic_module(path: str) -> bool:
    """True for a backend logic-layer source file worth an auto-generated test."""
    n = _norm(path)
    dirs = n.split("/")[:-1]
    lowered_dirs = [d.lower() for d in dirs]
    if any(d in _FRONTEND_SEGMENTS for d in lowered_dirs):
        return False
    if not any(d in _LOGIC_SEGMENTS for d in lowered_dirs):
        return False
    stem, ext = os.path.splitext(os.path.basename(n))
    if ext.lower() not in _TESTABLE_EXTS:
        return False
    if stem.lower() in _SKIP_STEMS:
        return False
    return not _is_test_file(n)


def _has_existing_test(impl_path: str, on_disk: set[str]) -> bool:
    """True if SOME test already exists for this module (any layer/convention)."""
    stem, ext = os.path.splitext(os.path.basename(_norm(impl_path)))
    if ext.lower() in _PY_EXTS:
        for p in on_disk:
            s, e = os.path.splitext(os.path.basename(p))
            if e.lower() in _PY_EXTS and s in (f"test_{stem}", f"{stem}_test"):
                return True
        return False
    for p in on_disk:
        b = os.path.basename(p)
        if b.startswith(f"{stem}.test.") or b.startswith(f"{stem}.spec."):
            return True
    return False


def _test_path_for(impl_path: str, used: set[str]) -> str:
    """Convention path for a module's test, disambiguated on stem collision."""
    segs = _norm(impl_path).split("/")
    stem, ext = os.path.splitext(segs[-1])
    if ext.lower() in _PY_EXTS:
        cand = f"tests/test_{stem}.py"
        if cand in used:
            parent = segs[-2] if len(segs) >= 2 else "pkg"
            cand = f"tests/test_{parent}_{stem}.py"
        return cand
    parent = "/".join(segs[:-1])
    return f"{parent}/{stem}.test{ext}" if parent else f"{stem}.test{ext}"


def derive_test_targets(workspace_files, language: str | None = None) -> list[tuple[str, str]]:
    """Map untested backend logic modules to the test files that should cover them.

    Returns a list of (impl_relative_path, test_relative_path). Per-file extension
    drives the convention, so a mixed-language project is handled correctly; the
    `language` argument is accepted for signature symmetry but not required.
    """
    files = [_norm(f) for f in (workspace_files or [])]
    on_disk = set(files)
    targets: list[tuple[str, str]] = []
    used_tests: set[str] = set()
    for f in files:
        if not _is_logic_module(f):
            continue
        if _has_existing_test(f, on_disk):
            continue
        test_path = _test_path_for(f, used_tests)
        if test_path in on_disk:
            continue
        used_tests.add(test_path)
        targets.append((f, test_path))
    return targets
