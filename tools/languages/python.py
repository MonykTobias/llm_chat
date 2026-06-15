"""Python language tools: pylint, pytest, mypy, and a static import check.

Required for full results (installed in the reviewed project's venv, or in
code_review_agent's own .venv as a fallback):
    * pylint   — linter            (run_linter)
    * pytest   — test runner       (run_tests);  pytest-cov for --cov coverage
    * mypy     — static type check (run_type_check)
The import check (check_imports) is pure standard library (ast) — it needs
nothing installed and never executes the project's code.

Dependency auto-provisioning: before linting/testing/type-checking we `pip
install` the project's manifest (requirements.txt / pyproject.toml / setup.py)
into its venv — or into a managed venv we create under code_review_agent/.toolenvs
when the project has none — so mypy/pytest can resolve the project's third-party
imports. This runs at most once per project per server run.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

from tools._common import _IGNORE, _rel, _truncate, _vendor_ignore_regex

from .base import (
    _create_managed_venv,
    _find_project_venv,
    _provision_once,
    _run,
    _run_python_tool,
    _venv_python,
)

# Names that resolve without an installed third-party package: the stdlib plus
# the built-ins. `stdlib_module_names` exists on 3.10+; fall back gracefully.
_STDLIB = frozenset(getattr(sys, "stdlib_module_names", frozenset())) | \
          frozenset(sys.builtin_module_names)

_MANIFESTS = ("requirements.txt", "pyproject.toml", "setup.py", "setup.cfg")


# ── dependency provisioning ──────────────────────────────────────────────

def _python_manifest(project: str) -> "Path | None":
    root = Path(project)
    for name in _MANIFESTS:
        p = root / name
        if p.is_file():
            return p
    return None


def _provision(project: str) -> None:
    """pip-install the project's declared deps into a usable venv (best effort)."""
    manifest = _python_manifest(project)
    if manifest is None:
        return
    # Prefer the project's own venv; otherwise create a managed one so we don't
    # pollute code_review_agent's interpreter with the reviewed project's deps.
    py = _find_project_venv(project) or _create_managed_venv(project)
    cmd = [py, "-m", "pip", "install", "-q"]
    if manifest.name == "requirements.txt":
        cmd += ["-r", str(manifest)]
    else:  # pyproject.toml / setup.py / setup.cfg -> install the project itself
        cmd += [str(project)]
    _run(cmd, timeout=600)


# ── linter / tests / type check ──────────────────────────────────────────

def run_linter(path: str, language: str) -> str:
    """pylint, scoped to skip vendored dirs and the slow duplicate-code checker."""
    _provision_once(path, "python", _provision)
    # Recurse but skip vendored/ignored dirs (.venv etc.), use all cores, and drop
    # duplicate-code — a bare `pylint <root>` otherwise lints thousands of .venv
    # files and times out. Longer budget since real projects can still be large.
    args = [
        "--recursive=y",
        f"--ignore-paths={_vendor_ignore_regex()}",
        "--jobs=0",
        "--disable=duplicate-code",
        path,
    ]
    return _run_python_tool(_venv_python(path), "pylint", args, timeout=300)


def run_tests(path: str, language: str, include_coverage: bool = True) -> str:
    """pytest, scoped out of vendored dirs so it doesn't collect .venv's own tests."""
    _provision_once(path, "python", _provision)
    # -o norecursedirs overrides any project config so collection never descends
    # into .venv/node_modules/build; explicit --ignore for the ones at the root;
    # -q / no cache to trim noise.
    args = ["-q", "-p", "no:cacheprovider",
            f"--override-ini=norecursedirs={' '.join(sorted(_IGNORE))}"]
    root = Path(path)
    for name in sorted(_IGNORE):
        if (root / name).is_dir():
            args.append(f"--ignore={root / name}")
    args.append(path)
    if include_coverage:
        args.append("--cov")
    return _run_python_tool(_venv_python(path), "pytest", args)


def run_type_check(path: str, language: str) -> str:
    """mypy, excluding vendored dirs and silencing missing third-party stubs."""
    _provision_once(path, "python", _provision)
    args = [
        f"--exclude={_vendor_ignore_regex()}",
        "--ignore-missing-imports",
        "--no-error-summary",
        path,
    ]
    return _run_python_tool(_venv_python(path), "mypy", args, timeout=300)


def check_imports(path: str, language: str) -> str:
    """Static import health check — broken / unused / circular. No code executed."""
    return _check_imports_python(path)


# ── static import health check (ast only) ────────────────────────────────
# Deliberately conservative: when in doubt we DON'T flag, because a false
# "broken import" is more harmful to a review than a missed one.

def _py_dotted_for(file: Path, root: Path) -> "tuple[str, bool] | None":
    """Map a .py file to its dotted module path relative to `root`.

    `a/b/c.py` -> ('a.b.c', False); `a/b/__init__.py` -> ('a.b', True). Returns
    None for a root-level __init__.py (no dotted name to give the root itself).
    """
    try:
        parts = list(file.relative_to(root).parts)
    except ValueError:
        return None
    is_pkg = parts[-1] == "__init__.py"
    mod_parts = parts[:-1] if is_pkg else parts[:-1] + [parts[-1][:-3]]
    if not mod_parts:
        return None
    return ".".join(mod_parts), is_pkg


def _py_local_index(files: "list[Path]", root: Path) -> tuple:
    """Build the project's local-module index from the filesystem (no execution).

    Returns (file_to_mod, is_pkg_map, importable, top_segments) where `importable`
    is every local dotted module/package name (incl. all ancestor prefixes) and
    `top_segments` is the set of their first segments.
    """
    file_to_mod: dict[Path, str] = {}
    is_pkg_map: dict[Path, bool] = {}
    importable: set[str] = set()
    for f in files:
        res = _py_dotted_for(f, root)
        if res is None:
            continue
        dotted, is_pkg = res
        file_to_mod[f] = dotted
        is_pkg_map[f] = is_pkg
        parts = dotted.split(".")
        for i in range(1, len(parts) + 1):           # add every ancestor prefix
            importable.add(".".join(parts[:i]))
    top_segments = {name.split(".")[0] for name in importable}
    return file_to_mod, is_pkg_map, importable, top_segments


def _abs_top_ok(top: str, top_segments: "set[str]", spec_cache: dict) -> bool:
    """Does the top-level name resolve as local, stdlib, or an installed package?

    Only ever calls find_spec on the bare top-level name — that performs a finder
    lookup without importing the target, so nothing executes.
    """
    if top in top_segments or top in _STDLIB:
        return True
    if top in spec_cache:
        return spec_cache[top]
    try:
        ok = importlib.util.find_spec(top) is not None
    except (ImportError, ValueError, ModuleNotFoundError, AttributeError):
        ok = False
    spec_cache[top] = ok
    return ok


def _resolve_relative(cur_mod: str, is_pkg: bool, level: int,
                      module: "str | None") -> "tuple[str, str, bool]":
    """Resolve a relative import to (base_pkg, full_target, beyond_top_level)."""
    parts = cur_mod.split(".")
    anchor = parts if is_pkg else parts[:-1]      # the importing module's package
    if level - 1 > len(anchor):
        return "", "", True
    base = anchor[: len(anchor) - (level - 1)]
    target = base + (module.split(".") if module else [])
    return ".".join(base), ".".join(target), False


def _local_target(dotted: str, importable: "set[str]") -> "str | None":
    """Longest importable prefix of `dotted` that is a local module, or None."""
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        cand = ".".join(parts[:i])
        if cand in importable:
            return cand
    return None


def _from_display(node: ast.ImportFrom) -> str:
    """Render an `ast.ImportFrom` back to source-like text for the report."""
    prefix = "." * node.level + (node.module or "")
    names = ", ".join(a.name + (f" as {a.asname}" if a.asname else "")
                      for a in node.names)
    return f"from {prefix} import {names}"


def _py_find_cycles(graph: "dict[str, set[str]]") -> "list[list[str]]":
    """Find import cycles via iterative colored DFS; one cycle per SCC, deduped."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in graph}
    cycles: list[list[str]] = []
    seen: set[frozenset] = set()

    for start in graph:
        if color[start] != WHITE:
            continue
        color[start] = GRAY
        stack = [start]
        work = [(start, iter(sorted(graph[start])))]
        while work:
            node, it = work[-1]
            descended = False
            for nb in it:
                if nb not in graph:
                    continue
                if color[nb] == WHITE:
                    color[nb] = GRAY
                    stack.append(nb)
                    work.append((nb, iter(sorted(graph[nb]))))
                    descended = True
                    break
                if color[nb] == GRAY:                      # back-edge -> cycle
                    idx = stack.index(nb)
                    key = frozenset(stack[idx:])
                    if key not in seen:
                        seen.add(key)
                        cycles.append(stack[idx:] + [nb])
            if not descended:
                color[node] = BLACK
                stack.pop()
                work.pop()
    return cycles


def _check_imports_python(path: str) -> str:
    """Static import health check for a Python project: broken / unused / circular."""
    root = Path(path)
    if not root.exists():
        return f"Directory not found: {path}"

    files = [e for e in root.rglob("*.py")
             if e.is_file() and not (_IGNORE & set(e.relative_to(root).parts))]
    if not files:
        return f"No Python files found under {path}."

    file_to_mod, is_pkg_map, importable, top_segments = _py_local_index(files, root)

    broken: list[tuple] = []        # (relpath, lineno, display, reason)
    unused: list[tuple] = []        # (relpath, lineno, display, bound_name)
    cannot_parse: list[tuple] = []  # (relpath, lineno, message)
    graph: dict[str, set[str]] = {mod: set() for mod in file_to_mod.values()}
    spec_cache: dict[str, bool] = {}
    star_note = False

    for file in files:
        relp = _rel(path, str(file))
        cur_mod = file_to_mod.get(file)
        is_pkg = is_pkg_map.get(file, False)
        try:
            tree = ast.parse(file.read_text(encoding="utf-8", errors="replace"),
                             filename=str(file))
        except SyntaxError as e:
            cannot_parse.append((relp, e.lineno or 0, e.msg))
            continue
        except OSError as e:
            cannot_parse.append((relp, 0, f"could not read file: {e}"))
            continue

        all_imports = [n for n in ast.walk(tree)
                       if isinstance(n, (ast.Import, ast.ImportFrom))]
        top_imports = [n for n in tree.body
                       if isinstance(n, (ast.Import, ast.ImportFrom))]
        used = {n.id for n in ast.walk(tree)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        all_names = _py_dunder_all(tree)

        # ── broken / unresolvable ────────────────────────────────────────
        for node in all_imports:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if not _abs_top_ok(top, top_segments, spec_cache):
                        broken.append((relp, node.lineno, f"import {alias.name}",
                                       f"top-level module '{top}' not found "
                                       "(missing dependency or typo)"))
            elif node.level == 0:                          # absolute from-import
                top = (node.module or "").split(".")[0]
                if node.module and not _abs_top_ok(top, top_segments, spec_cache):
                    broken.append((relp, node.lineno, _from_display(node),
                                   f"top-level module '{top}' not found "
                                   "(missing dependency or typo)"))
            elif cur_mod is not None:                      # relative from-import
                base, target, beyond = _resolve_relative(
                    cur_mod, is_pkg, node.level, node.module)
                if beyond:
                    broken.append((relp, node.lineno, _from_display(node),
                                   "relative import goes beyond the top-level package"))
                else:
                    check = target if node.module else base
                    if check and check not in importable:
                        broken.append((relp, node.lineno, _from_display(node),
                                       "relative import does not resolve to a "
                                       f"project module ('{check}')"))

            # ── circular-import graph edges (local deps only) ────────────
            if cur_mod is not None:
                for t in _py_edge_targets(node, cur_mod, is_pkg, importable):
                    if t != cur_mod:
                        graph[cur_mod].add(t)

        # ── unused (top-level imports only, to avoid local/typing noise) ──
        for node in top_imports:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname and alias.asname == alias.name:
                        continue                            # `import x as x` re-export
                    bound = alias.asname or alias.name.split(".")[0]
                    if bound in used or bound in all_names:
                        continue
                    disp = f"import {alias.name}" + (
                        f" as {alias.asname}" if alias.asname else "")
                    unused.append((relp, node.lineno, disp, bound))
            else:
                if node.module == "__future__":
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        star_note = True
                        continue
                    if alias.asname and alias.asname == alias.name:
                        continue                            # re-export
                    bound = alias.asname or alias.name
                    if bound in used or bound in all_names:
                        continue
                    disp = (f"from {'.' * node.level}{node.module or ''} "
                            f"import {alias.name}"
                            + (f" as {alias.asname}" if alias.asname else ""))
                    unused.append((relp, node.lineno, disp, bound))

    cycles = _py_find_cycles(graph)
    return _format_import_report(broken, unused, cycles, cannot_parse, star_note)


def _py_dunder_all(tree: ast.Module) -> "set[str]":
    """Collect string entries of a module-level `__all__` (treated as 'used')."""
    names: set[str] = set()
    for node in tree.body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AugAssign) else [])
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            continue
        value = node.value
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            for el in value.elts:
                if isinstance(el, ast.Constant) and isinstance(el.value, str):
                    names.add(el.value)
    return names


def _py_edge_targets(node, cur_mod: str, is_pkg: bool,
                     importable: "set[str]") -> "set[str]":
    """Local module(s) an import depends on, for the circular-import graph."""
    targets: set[str] = set()

    def add(dotted: str) -> None:
        t = _local_target(dotted, importable)
        if t:
            targets.add(t)

    if isinstance(node, ast.Import):
        for alias in node.names:
            add(alias.name)
    elif node.level == 0:
        if node.module:
            add(node.module)
            for alias in node.names:
                if alias.name != "*":
                    add(f"{node.module}.{alias.name}")
    else:
        base, target, beyond = _resolve_relative(
            cur_mod, is_pkg, node.level, node.module)
        if not beyond:
            anchor = target if node.module else base
            if anchor:
                add(anchor)
            for alias in node.names:
                if alias.name != "*" and anchor:
                    add(f"{anchor}.{alias.name}")
    return targets


def _format_import_report(broken: list, unused: list, cycles: list,
                          cannot_parse: list, star_note: bool) -> str:
    """Render the sectioned import-check report, bounded by `_truncate`."""
    out = ["== IMPORT CHECK (python) =="]

    if broken:
        out.append(f"\nBROKEN / UNRESOLVABLE ({len(broken)}):")
        for relp, ln, disp, reason in sorted(broken):
            out.append(f"  {relp}:{ln}  {disp}  -> {reason}")
    else:
        out.append("\nBROKEN / UNRESOLVABLE (0): none found.")

    if unused:
        out.append(f"\nUNUSED ({len(unused)}):")
        for relp, ln, disp, bound in sorted(unused):
            out.append(f"  {relp}:{ln}  {disp}  (name '{bound}' never used)")
    else:
        out.append("\nUNUSED (0): none found.")

    if cycles:
        out.append(f"\nCIRCULAR ({len(cycles)}):  (note: Python tolerates many "
                   "import cycles at runtime — treat these as warnings)")
        for cyc in cycles:
            out.append("  " + " -> ".join(cyc))
    else:
        out.append("\nCIRCULAR (0): none found.")

    if cannot_parse:
        out.append(f"\nCANNOT PARSE ({len(cannot_parse)}):")
        for relp, ln, msg in sorted(cannot_parse):
            out.append(f"  {relp}:{ln}  {msg}")

    if star_note:
        out.append("\n(note: `from x import *` lines are skipped for the unused "
                   "check — their names can't be tracked statically.)")

    out.append(f"\nSummary: {len(broken)} broken, {len(unused)} unused, "
               f"{len(cycles)} circular"
               + (f", {len(cannot_parse)} unparseable" if cannot_parse else "")
               + ".")
    return _truncate("\n".join(out))
