"""Per-language tool implementations + the dispatch the @tool wrappers call.

`tools/language.py` holds the LangChain `@tool` wrappers (which read the active
language/path off the graph state); it delegates the actual work to the four
dispatch functions here, which route to the right language module:

    python      pylint / pytest / mypy / static-AST import check
    javascript  eslint / npm test / tsc            (handles "typescript" too)
    go          golangci-lint|go vet / go test / go build
    rust        cargo clippy / cargo test / cargo check
    java        Checkstyle / Maven|Gradle test / compile

Each handler module documents the external tools a reviewed project must have
installed for full results. Unknown / non-code languages (e.g. "english") get a
clear "not configured" message rather than an error.
"""
from __future__ import annotations

from structured_output import CompileOutput

from . import go, java, javascript, python, rust

# language name -> handler module exposing run_linter / run_tests /
# run_type_check / check_imports. "typescript" shares the javascript module.
_HANDLERS = {
    "python": python,
    "javascript": javascript,
    "typescript": javascript,
    "go": go,
    "rust": rust,
    "java": java,
}


def _unknown(language: str, kind: str) -> str:
    return (f"No {kind} configured for '{language}'. Supported languages: "
            f"{', '.join(sorted(_HANDLERS))}.")


def run_linter(path: str, language: str) -> str:
    handler = _HANDLERS.get(language)
    if handler is None:
        answer = _unknown(language, "linter")
        return answer
    answer = handler.run_linter(path, language)
    return answer


def run_tests(path: str, language: str, include_coverage: bool = True) -> str:
    handler = _HANDLERS.get(language)
    if handler is None:
        answer = _unknown(language, "test runner")
        return answer
    answer = handler.run_tests(path, language, include_coverage)
    return answer


def run_type_check(path: str, language: str) -> str:
    handler = _HANDLERS.get(language)
    if handler is None:
        answer = _unknown(language, "type checker")
        return answer
    answer = handler.run_type_check(path, language)
    return answer


def check_imports(path: str, language: str) -> str:
    handler = _HANDLERS.get(language)
    if handler is None:
        answer: str = _unknown(language, "import checker")
        return answer
    answer = handler.check_imports(path, language)
    return answer


def compile_code(path: str, language: str) -> CompileOutput:
    """Compile the project in an ephemeral Docker container; return a CompileOutput.

    Non-code / unsupported languages (e.g. "english") get a `status="unavailable"`
    result carrying the not-configured note, mirroring the string dispatchers above.
    """
    handler = _HANDLERS.get(language)
    if handler is None:
        answer = CompileOutput(
            status="unavailable", language=language, compiler="",
            exit_code=-1, errors=[], warnings=[_unknown(language, "compiler")],
            duration_ms=0)
        return answer
    answer = handler.compile_code(path, language)
    return answer
