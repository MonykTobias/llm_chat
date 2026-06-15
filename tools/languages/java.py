"""Java tools: build-tool driven compile/test, plus optional Checkstyle linting.

Java has no virtual env. Three layers must be present on the host:
    * a JDK (javac + java)            — the toolchain
    * Maven (`mvn`) or Gradle         — the build tool (we detect which); Gradle is
      driven through the project's `./gradlew` wrapper when present
    * deps declared in pom.xml / build.gradle are fetched into a shared local
      repository (~/.m2 or ~/.gradle) by the build tool
There is no standalone type checker — **compilation is the type check** — and no
default linter: Checkstyle runs only if `checkstyle` is on PATH (we supply the
bundled `configs/checkstyle.xml` when the project has no config of its own).

Auto-provisioning: resolve dependencies once per project (Maven
`dependency:go-offline`, or a Gradle `dependencies` task) so a later
compile/test doesn't pay first-fetch cost mid-run.
"""
from __future__ import annotations

from pathlib import Path

from .base import CONFIG_DIR, _env_msg, _exe, _project_bin, _provision_once, _run


def _build_tool(project: str) -> "str | None":
    root = Path(project)
    if (root / "pom.xml").exists():
        return "maven"
    if (root / "build.gradle").exists() or (root / "build.gradle.kts").exists():
        return "gradle"
    return None


def _gradle_cmd(project: str) -> "list[str] | None":
    wrapper = _project_bin(project, "gradlew.bat", "gradlew")
    if wrapper:
        return [wrapper]
    gradle = _exe("gradle")
    return [gradle] if gradle else None


def _maven_cmd() -> "list[str] | None":
    mvn = _exe("mvn")
    return [mvn] if mvn else None


def _tool_cmd(project: str) -> "tuple[str, list[str]] | tuple[None, str]":
    """Return (build_tool, cmd_prefix) or (None, error_message)."""
    bt = _build_tool(project)
    if bt is None:
        return None, ("No Maven (pom.xml) or Gradle (build.gradle) build file found. "
                      "Java review needs one of these build tools.")
    cmd = _maven_cmd() if bt == "maven" else _gradle_cmd(project)
    if cmd is None:
        return None, _env_msg("mvn" if bt == "maven" else "gradle")
    return bt, cmd


def _provision(project: str) -> None:
    bt, cmd = _tool_cmd(project)
    if bt is None:
        return
    if bt == "maven":
        _run([*cmd, "-q", "-B", "dependency:go-offline"], cwd=project, timeout=600)
    else:
        _run([*cmd, "dependencies", "--quiet"], cwd=project, timeout=600)


def run_linter(path: str, language: str) -> str:
    checkstyle = _exe("checkstyle")
    if not checkstyle:
        return ("[note] No Java linter configured. Checkstyle is not on PATH; install "
                "it, or configure the Checkstyle/PMD/SpotBugs plugin in your build. "
                "(Compilation via run_type_check still catches errors.)")
    # Lint the main source tree if it follows the standard layout, else the root.
    src = Path(path) / "src" / "main" / "java"
    target = str(src) if src.is_dir() else path
    return _run([checkstyle, "-c", str(CONFIG_DIR / "checkstyle.xml"), target],
                cwd=path, timeout=300)


def run_tests(path: str, language: str, include_coverage: bool = True) -> str:
    _provision_once(path, "java", _provision)
    bt, cmd = _tool_cmd(path)
    if bt is None:
        return cmd  # error message
    if bt == "maven":
        return _run([*cmd, "-q", "-B", "test"], cwd=path, timeout=600)
    # Gradle: jacoco coverage is wired in the build, not via a flag.
    return _run([*cmd, "test", "--quiet"], cwd=path, timeout=600)


def run_type_check(path: str, language: str) -> str:
    _provision_once(path, "java", _provision)
    bt, cmd = _tool_cmd(path)
    if bt is None:
        return cmd  # error message
    # Compilation IS the type check in Java.
    if bt == "maven":
        return _run([*cmd, "-q", "-B", "-DskipTests", "compile"], cwd=path, timeout=600)
    return _run([*cmd, "compileJava", "--quiet"], cwd=path, timeout=600)


def check_imports(path: str, language: str) -> str:
    out = run_type_check(path, language)
    return ("== IMPORT CHECK (java via compile) ==\n"
            "Unresolved imports surface as compiler errors "
            "('cannot find symbol' / 'package does not exist'); unused imports are a "
            "Checkstyle/lint concern (see run_linter).\n\n" + out)
