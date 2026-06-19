"""Shared rendering for the per-language import checks.

Every language's `check_imports` parses its tool's output into the same four
buckets and hands them to `format_report`, so the model sees one consistent
report shape regardless of language:

    broken  = [(relpath, line, display, reason)]   # unresolvable / missing
    unused  = [(relpath, line, display, name)]     # imported but never used
    cycles  = [[node, node, …]]                    # one chain per cycle
    notes   = [str]                                # extra context lines

The canonical shape (and `find_cycles`) originate from the Python analyzer in
`docker/import_check.py`; that script keeps its own stdlib-only copy because it
runs inside the container, while this module serves the host-side parsers for
go / rust / javascript / java.
"""
from __future__ import annotations

from tools._common import _truncate

# Bucket element type aliases (documentation only).
BrokenItem = "tuple[str, int, str, str]"   # (relpath, line, display, reason)
UnusedItem = "tuple[str, int, str, str]"   # (relpath, line, display, bound_name)

_MAX_CYCLES = 50   # report at most this many cycles to cap memory/output


def find_cycles(graph: "dict[str, set[str]]") -> "list[list[str]]":
    """Find cycles via iterative colored DFS; one cycle per SCC, deduped.

    Language-agnostic: the graph maps a node (module / package) to the set of
    nodes it depends on. Uses a stack-position dict for O(1) back-edge resolution
    and caps the number of reported cycles so the seen-set / output stay bounded.
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


def _dedupe_cycles(cycles: "list[list[str]]") -> "list[list[str]]":
    """Drop cycles that cover the same set of nodes (same cycle, rotated/reversed)."""
    seen: set[frozenset] = set()
    out: list[list[str]] = []
    for cyc in cycles:
        key = frozenset(cyc)
        if key not in seen:
            seen.add(key)
            out.append(cyc)
    return out


def format_report(language: str,
                  broken: "list[tuple]",
                  unused: "list[tuple]",
                  cycles: "list[list[str]]",
                  notes: "list[str] | tuple[str, ...]" = (),
                  *, circular_note: "str | None" = None) -> str:
    """Render the canonical sectioned import-check report, bounded by `_truncate`.

    `circular_note` replaces the "none found" line in the CIRCULAR section when no
    cycles were detected — used by languages where cycles either can't occur or
    aren't analyzed (e.g. Rust), so a clean "0" is not mistaken for "checked".
    """
    # Tools can report the same finding more than once (e.g. cargo --all-targets
    # re-checks a crate per target); dedupe so the report counts each once.
    broken = sorted(set(broken))
    unused = sorted(set(unused))
    cycles = _dedupe_cycles(cycles)

    out = [f"== IMPORT CHECK ({language}) =="]

    if broken:
        out.append(f"\nBROKEN / UNRESOLVABLE ({len(broken)}):")
        for relp, ln, disp, reason in broken:
            out.append(f"  {relp}:{ln}  {disp}  -> {reason}")
    else:
        out.append("\nBROKEN / UNRESOLVABLE (0): none found.")

    if unused:
        out.append(f"\nUNUSED ({len(unused)}):")
        for relp, ln, disp, bound in unused:
            out.append(f"  {relp}:{ln}  {disp}  (name '{bound}' never used)")
    else:
        out.append("\nUNUSED (0): none found.")

    if cycles:
        out.append(f"\nCIRCULAR ({len(cycles)}):")
        for cyc in cycles:
            out.append("  " + " -> ".join(cyc))
    elif circular_note:
        out.append(f"\nCIRCULAR (n/a): {circular_note}")
    else:
        out.append("\nCIRCULAR (0): none found.")

    for note in notes:
        out.append(f"\n{note}")

    out.append(f"\nSummary: {len(broken)} broken, {len(unused)} unused, "
               f"{len(cycles)} circular.")
    return _truncate("\n".join(out))
