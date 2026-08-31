"""Language-agnostic definition extraction for the coder's regression guard.

The multi-step coder rewrites a file in full on every modify step. A blind model
sometimes drops definitions an earlier step added, and the deterministic validator
(lint/type/import/compile) can't see it — the truncated file still compiles. The
coder catches this by comparing the set of definitions BEFORE vs AFTER the rewrite
and refusing to write when something disappeared.

This module supplies that "set of definitions" for every language the pipeline
supports (python, javascript, typescript, go, rust, java) from ONE tree-sitter
parser instead of a hand-written parser per language. It is best-effort by design:
any failure (tree-sitter missing, unknown language, parse error) yields an empty
set so the guard simply never fires there — exactly the graceful degradation the
old Python-only `ast` guard had for non-Python files.

Names are returned UNQUALIFIED (a method and a top-level function with the same
name collapse). That is enough for the guard: a dropped function/method/class makes
its name vanish from the set. We also keep private/unexported names — in the
add-only multi-step build pattern, nothing should be removed regardless of
visibility, and replicating each language's visibility rules would reintroduce the
per-language complexity this module exists to avoid.
"""
from __future__ import annotations

import os

try:  # tree-sitter is optional — degrade to "no guard" if it's not installed.
    from tree_sitter_language_pack import get_parser as _get_parser
except Exception:  # pragma: no cover - import-time environment issue
    _get_parser = None


# Pipeline language -> tree-sitter grammar name.
_PARSER_NAME = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "go": "go",
    "rust": "rust",
    "java": "java",
}

# Node kinds that introduce a named definition, per language. Extra kinds that a
# given grammar never emits are harmless, so the JS/TS set is shared and generous.
_JS_TS_DEFS = {
    "function_declaration", "generator_function_declaration", "method_definition",
    "class_declaration", "abstract_class_declaration", "interface_declaration",
    "type_alias_declaration", "enum_declaration",
}
_DEF_NODES = {
    "python": {"function_definition", "class_definition"},
    "javascript": _JS_TS_DEFS,
    "typescript": _JS_TS_DEFS,
    "go": {"function_declaration", "method_declaration", "type_spec"},
    "rust": {"function_item", "struct_item", "enum_item", "trait_item", "type_item"},
    "java": {
        "method_declaration", "constructor_declaration", "class_declaration",
        "interface_declaration", "enum_declaration", "record_declaration",
    },
}

# File extension -> pipeline language.
_EXT_LANG = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
}


def language_for_path(file_path: str) -> "str | None":
    """Infer a supported pipeline language from a file's extension, or None.

    The coder runs before the validator sets an active language, so the guard
    derives the language from the path it is about to write."""
    if not file_path:
        return None
    return _EXT_LANG.get(os.path.splitext(file_path)[1].lower())


def _v(obj, attr):
    """Read a tree-sitter node/tree attribute, tolerating both the property-style
    and method-style bindings (the bundled grammar pack exposes them as methods)."""
    x = getattr(obj, attr)
    return x() if callable(x) else x


def extract_definitions(content: str, language: "str | None") -> "set[str]":
    """Names of the functions/methods/classes/types defined in `content`.

    Returns an empty set for an unsupported/unknown language, when tree-sitter is
    unavailable, or when parsing fails — so a caller can treat "empty" as "no
    signal, don't fire the guard"."""
    if _get_parser is None or not content or not language:
        return set()
    parser_name = _PARSER_NAME.get(language)
    wanted = _DEF_NODES.get(language)
    if not parser_name or not wanted:
        return set()
    try:
        parser = _get_parser(parser_name)
        try:
            tree = parser.parse(content)              # this build wants str
        except TypeError:
            tree = parser.parse(content.encode("utf-8"))  # other builds want bytes
        data = content.encode("utf-8")  # tree-sitter offsets are utf-8 byte offsets
        names: set[str] = set()
        stack = [_v(tree, "root_node")]
        while stack:
            node = stack.pop()
            if _v(node, "kind") in wanted:
                name_node = node.child_by_field_name("name")
                if name_node is not None:
                    names.add(data[_v(name_node, "start_byte"):_v(name_node, "end_byte")]
                              .decode("utf-8", "replace"))
            for i in range(_v(node, "child_count")):
                stack.append(node.child(i))
        return names
    except Exception:
        return set()
