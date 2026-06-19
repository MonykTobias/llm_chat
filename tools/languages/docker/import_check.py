#!/usr/bin/env python3
"""Static Python import health check — broken / unused / circular.

Run INSIDE the per-language Python container (``python import_check.py <path>``)
where the reviewed project's dependencies have been pip-installed, so that
``importlib.util.find_spec`` resolves third-party imports correctly — the whole
reason this analysis moved into Docker. It never executes the project's code: it
parses with ``ast`` and only ever calls ``find_spec`` on bare top-level names
(finder lookup, no import).

Self-contained on purpose: zero non-stdlib imports (the few helpers it needs from
``tools/_common.py`` are inlined below) so it runs under any python image without
the code_review_agent package being installed. The logic mirrors the original
``_check_imports_python`` in ``tools/languages/python.py``.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

# ── inlined from tools/_common.py (kept in sync) ─────────────────────────
_IGNORE = {"__pycache__", "node_modules", ".git", ".venv", "venv",
           ".idea", ".mypy_cache", ".pytest_cache", "dist", "build"}

_TOOL_MAX_CHARS = 8_000


def _truncate(text: str, max_chars: int = _TOOL_MAX_CHARS) -> str:
    """Bound `text` to `max_chars`, keeping head + tail with a dropped-middle note."""
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.7)
    tail = max_chars - head
    omitted = len(text) - head - tail
    return (f"{text[:head]}\n\n[… output truncated, {omitted} chars omitted "
            f"to fit the context window …]\n\n{text[-tail:]}")


def _rel(project_path: str, abs_path: str) -> str:
    """Best-effort path relative to the project root, for readable reports."""
    try:
        return str(Path(abs_path).relative_to(Path(project_path).resolve()))
    except ValueError:
        return abs_path


# Names that resolve without an installed third-party package: the stdlib plus
# the built-ins. `stdlib_module_names` exists on 3.10+; fall back gracefully.
_STDLIB = frozenset(getattr(sys, "stdlib_module_names", frozenset())) | \
          frozenset(sys.builtin_module_names)

_MAX_CYCLES = 50   # report at most this many cycles to cap memory/output


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
    """Find import cycles via iterative colored DFS; one cycle per SCC, deduped.

    Uses a stack-position dict for O(1) back-edge resolution (avoids the O(n)
    list.index() scan that blows memory on large graphs) and caps the total
    number of reported cycles so the seen-set and output list stay bounded.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in graph}
    cycles: list[list[str]] = []
    seen: set[frozenset] = set()

    for start in graph:
        if color[start] != WHITE:
            continue
        color[start] = GRAY
        stack: list[str] = [start]
        stack_pos: dict[str, int] = {start: 0}   # node -> index in stack (O(1))
        work = [(start, iter(sorted(graph[start])))]
        while work:
            node, it = work[-1]
            descended = False
            for nb in it:
                if nb not in graph:
                    continue
                if color[nb] == WHITE:
                    color[nb] = GRAY
                    stack_pos[nb] = len(stack)
                    stack.append(nb)
                    work.append((nb, iter(sorted(graph[nb]))))
                    descended = True
                    break
                if color[nb] == GRAY and len(cycles) < _MAX_CYCLES:
                    idx = stack_pos[nb]            # O(1) — was O(n) list.index()
                    key = frozenset(stack[idx:])
                    if key not in seen:
                        seen.add(key)
                        cycles.append(stack[idx:] + [nb])
            if not descended:
                color[node] = BLACK
                stack_pos.pop(node, None)
                stack.pop()
                work.pop()
    return cycles


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


def check_imports(path: str) -> str:
    """Static import health check for a Python project: broken / unused / circular."""
    root = Path(path)
    if not root.exists():
        return f"Directory not found: {path}"

    # Enumerate only the non-ignored top-level entries and rglob from there, so we
    # never descend into .venv/node_modules just to discard the results.
    ignore_roots = {root / name for name in _IGNORE if (root / name).is_dir()}

    def _not_ignored(p: Path) -> bool:
        """True iff no ancestor of `p` (down to `root`) is an ignored dir."""
        return not any(p.is_relative_to(ir) for ir in ignore_roots)

    files = [
        e for e in root.rglob("*.py")
        if e.is_file() and _not_ignored(e)
    ]
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


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    print(check_imports(target))
