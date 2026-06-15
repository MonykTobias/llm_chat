"""Backward-compatible facade for the tools package.

The tool suite used to live in this single ~800-line module. It has since been
split by domain into focused submodules:

    _common.py          shared helpers (path confinement, truncation, ignore set)
    filesystem.py       read / write / delete / list files
    language.py         linter / tests / type-check / architecture map
    change_tracking.py  write snapshots, change report, revert
    web.py              DuckDuckGo search + page fetch
    modes.py            language switch + review-stage handoffs

This module re-exports every public name so existing
`from tools.tools import ...` imports keep working unchanged. Prefer importing
from the specific submodule (or from `tools`) in new code.
"""
from __future__ import annotations

from .change_tracking import (
    build_change_report,
    restore_snapshot,
)
from .filesystem import (
    delete_file,
    list_all_files,
    read_file,
    write_file,
)
from .language import (
    analyze_architecture,
    run_linter,
    run_tests,
    run_type_check,
)
from .modes import (
    REVIEW_MODES,
    change_mode_act,
    change_mode_explore,
    change_mode_plan,
    change_mode_verify,
    set_language,
)
from .web import web_browse

__all__ = [
    # filesystem
    "read_file",
    "write_file",
    "delete_file",
    "list_all_files",
    # language tooling
    "run_linter",
    "run_tests",
    "run_type_check",
    "analyze_architecture",
    # change tracking
    "build_change_report",
    "restore_snapshot",
    # web
    "web_browse",
    # modes / language switch
    "set_language",
    "change_mode_explore",
    "change_mode_plan",
    "change_mode_act",
    "change_mode_verify",
    "REVIEW_MODES",
]
