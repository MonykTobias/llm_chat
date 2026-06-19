"""Java tools: build-tool driven compile/test, plus Checkstyle linting — all run
inside a per-build-tool container so the host needs no JDK or build tool.

We detect the build tool and pick the matching image:
    * Maven (pom.xml)            -> ``crav-java-maven``  (Maven + JDK + Checkstyle)
    * Gradle (build.gradle[.kts])-> ``crav-java-gradle`` (Gradle + JDK + Checkstyle;
      driven through the project's `./gradlew` wrapper when present)
    * neither                    -> ``crav-java-jdk``    (plain javac + Checkstyle)

Project deps are pre-fetched into a derived image by `_ensure_project_image`:
    * Maven  → `mvn dependency:resolve` baked into the image layer so the local
               ~/.m2 repo is already warm when the tool containers start.
    * Gradle → `gradle dependencies` does the same for the Gradle cache.
    * javac  → no manifest, no pre-fetch; the base image is used as-is.

The derived image tag is keyed on the manifest content, so it auto-invalidates
when pom.xml / build.gradle changes. Repeated reviews of the same project pay
the dependency-download cost only once.

Checkstyle uses the bundled ``configs/checkstyle.xml`` (mounted at /config).
"""
from __future__ import annotations

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
from .imports import find_cycles, format_report

# Per-build-tool image + compile command. Plain javac handles a project with no
# build file by compiling every .java into a throwaway output dir.
_MAVEN_IMAGE  = "crav-java-maven"
_GRADLE_IMAGE = "crav-java-gradle"
_JAVAC_IMAGE  = "crav-java-jdk"
_CONFIG_MOUNT = (str(CONFIG_DIR), "/config")

_MAVEN_COMPILE  = "mvn -q -B -DskipTests compile"
_GRADLE_COMPILE = ("if [ -x ./gradlew ]; then ./gradlew compileJava --quiet; "
                   "else gradle compileJava --quiet; fi")
_JAVAC_COMPILE  = ("mkdir -p /tmp/out && javac -d /tmp/out "
                   "$(find . -name '*.java' -not -path './build/*')")
_MAVEN_TEST  = "mvn -q -B test"
_GRADLE_TEST = ("if [ -x ./gradlew ]; then ./gradlew test --quiet; "
                "else gradle test --quiet; fi")

# Plain javac:  Foo.java:5: error: ';' expected
_JAVAC_RE = re.compile(r"^(.+?\.java):(\d+):\s*(error|warning):\s*(.*)$")
# Maven compiler plugin:  [ERROR] /path/Foo.java:[5,10] cannot find symbol
_MAVEN_RE = re.compile(
    r"^\[(ERROR|WARNING)\]\s*(.+?\.java):\[(\d+),(\d+)\]\s*(.*)$")

_NO_BUILD_TOOL = ("No Maven (pom.xml) or Gradle (build.gradle) build file found. "
                  "Java tests need one of these build tools.")


# ── build-tool detection ──────────────────────────────────────────────────

def _build_tool(project: str) -> "str | None":
    root = Path(project)
    if (root / "pom.xml").exists():
        return "maven"
    if (root / "build.gradle").exists() or (root / "build.gradle.kts").exists():
        return "gradle"
    return None


# ── per-project derived image ─────────────────────────────────────────────
# Maven/Gradle are their own dependency managers: we pre-fetch deps by running
# the build tool's offline-populate command inside a `docker build` RUN layer.
# The manifest (pom.xml / build.gradle) is COPYed into the build context so
# Docker's layer cache keys on its content — a changed manifest triggers a
# fresh fetch; an unchanged one is a no-op (<1 s).
#
# We copy the *full project* into the build context (not just the manifest) for
# Maven/Gradle because both tools need the full build descriptor to resolve
# multi-module / plugin deps correctly. _ensure_project_image handles the single
# manifest case; for Java we pass the manifest name for tagging but override
# the Dockerfile to COPY the whole project tree.

def _maven_install_cmd(_manifest: str) -> str:
    # Download all declared deps (including plugins) into ~/.m2 inside the image.
    # -q / --batch-mode suppress progress noise; || true so the build doesn't
    # fail if the project has modules that can't resolve without the full source.
    return "mvn -q -B dependency:resolve dependency:resolve-plugins || true"


def _gradle_install_cmd(_manifest: str) -> str:
    # `gradle dependencies` resolves and caches all runtime/test dependency trees.
    return ("if [ -x ./gradlew ]; then ./gradlew dependencies --quiet || true; "
            "else gradle dependencies --quiet || true; fi")


def _project_image(path: str) -> "tuple[str, str | None]":
    """Return (image_tag, warning) for the project at `path`.

    Picks the right base image and install command based on the detected build
    tool, then delegates to `_ensure_project_image`.
    """
    bt = _build_tool(path)
    if bt == "maven":
        return _ensure_project_image(
            base_image=_MAVEN_IMAGE,
            project_path=path,
            manifests=["pom.xml"],
            install_cmd_fn=_maven_install_cmd,
        )
    if bt == "gradle":
        manifest = ("build.gradle.kts"
                    if (Path(path) / "build.gradle.kts").exists()
                    else "build.gradle")
        return _ensure_project_image(
            base_image=_GRADLE_IMAGE,
            project_path=path,
            manifests=[manifest],
            install_cmd_fn=_gradle_install_cmd,
        )
    # Plain javac: no manifest, no pre-fetch needed.
    return _JAVAC_IMAGE, None


def _prepend_warning(warning: "str | None", result: str) -> str:
    return f"{warning}\n{result}" if warning else result


# ── linter / tests / type check ──────────────────────────────────────────

def run_linter(path: str, language: str) -> str:
    """Checkstyle against the standard source tree (or the project root)."""
    image, warn = _project_image(path)
    src = ("src/main/java" if (Path(path) / "src" / "main" / "java").is_dir()
           else ".")
    cmd = f"checkstyle -c /config/checkstyle.xml {src}"
    result = _run_docker_text(image, cmd, path, name="checkstyle", timeout=300,
                              mounts=[_CONFIG_MOUNT])
    return _prepend_warning(warn, result)


def run_tests(path: str, language: str, include_coverage: bool = True) -> str:
    image, warn = _project_image(path)
    bt = _build_tool(path)
    if bt == "maven":
        result = _run_docker_text(image, _MAVEN_TEST, path, name="mvn test",
                                  timeout=600)
    elif bt == "gradle":
        result = _run_docker_text(image, _GRADLE_TEST, path, name="gradle test",
                                  timeout=600)
    else:
        result = _NO_BUILD_TOOL
    return _prepend_warning(warn, result)


def run_type_check(path: str, language: str) -> str:
    """Compilation IS the type check in Java."""
    image, warn = _project_image(path)
    bt = _build_tool(path)
    if bt == "maven":
        result = _run_docker_text(image, _MAVEN_COMPILE, path,
                                  name="mvn compile", timeout=600)
    elif bt == "gradle":
        result = _run_docker_text(image, _GRADLE_COMPILE, path,
                                  name="gradle compileJava", timeout=600)
    else:
        # No build file: compile every .java with plain javac.
        result = _run_docker_text(image, _JAVAC_COMPILE, path,
                                  name="javac", timeout=600)
    return _prepend_warning(warn, result)


def check_imports(path: str, language: str) -> str:
    """Unified import report for Java from three tools in one container run.

    compile → BROKEN (unresolved imports / missing packages); Checkstyle
    (import-only config) → UNUSED; jdeps package graph → CIRCULAR. Falls back to
    plain javac when there's no build file.
    """
    image, warn = _project_image(path)
    bt = _build_tool(path)
    if bt == "maven":
        compile_cmd, classes = _MAVEN_COMPILE, "target/classes"
    elif bt == "gradle":
        compile_cmd, classes = _GRADLE_COMPILE, "build/classes/java/main"
    else:
        compile_cmd, classes = _JAVAC_COMPILE, "/tmp/out"
    src = ("src/main/java" if (Path(path) / "src" / "main" / "java").is_dir()
           else ".")
    shell = (
        'echo "<<<COMPILE>>>"; ' + compile_cmd + " 2>&1; "
        'echo "<<<UNUSED>>>"; '
        f"checkstyle -c /config/checkstyle-imports.xml {src} 2>&1; "
        'echo "<<<CYCLES>>>"; '
        f"jdeps -verbose:package {classes} 2>&1; true")
    dr = _run_docker(image, shell, path, timeout=600, mounts=[_CONFIG_MOUNT])
    if dr.error is not None:
        return _prepend_warning(warn, dr.error)
    compile_out, _, rest = dr.output.partition("<<<UNUSED>>>")
    compile_out = compile_out.partition("<<<COMPILE>>>")[2]
    unused_out, _, cycles_out = rest.partition("<<<CYCLES>>>")
    broken = _parse_java_broken(compile_out, path)
    unused = _parse_checkstyle_unused(unused_out)
    cycles = _parse_jdeps_cycles(cycles_out)
    return _prepend_warning(warn, format_report("java", broken, unused, cycles))


def _is_import_line(project_path: str, rel: str, line: int,
                    cache: dict) -> "str | None":
    """Return the source line at `rel:line` if it's an `import` statement, else None."""
    if rel not in cache:
        txt = _read_text_or_none(str(Path(project_path) / rel))
        cache[rel] = txt.splitlines() if txt else []
    lines = cache[rel]
    src = lines[line - 1] if 0 < line <= len(lines) else ""
    return src if src.strip().startswith("import ") else None


def _parse_java_broken(out: str, project_path: str) -> "list[tuple]":
    """Unresolved imports from compile output: missing packages / import lines."""
    broken: list[tuple] = []
    cache: dict[str, list[str]] = {}
    for line in out.splitlines():
        stripped = line.strip()
        m = _MAVEN_RE.match(stripped)
        if m:
            severity, file, ln, _col, msg = m.groups()
        else:
            m = _JAVAC_RE.match(stripped)
            if not m:
                continue
            file, ln, severity, msg = m.groups()
        if severity.lower() != "error":
            continue
        file, ln, msg = _compile_path(file), int(ln), msg.strip()
        # "package x does not exist" is always an import problem; "cannot find
        # symbol" only counts when the offending line is an import statement.
        if "does not exist" in msg:
            src = _is_import_line(project_path, file, ln, cache)
            broken.append((file, ln, src.strip() if src else msg, msg))
        elif "cannot find symbol" in msg:
            src = _is_import_line(project_path, file, ln, cache)
            if src:
                broken.append((file, ln, src.strip(), msg))
    return broken


# Checkstyle:  [WARN] /work/Foo.java:5:1: Unused import - java.util.List. [UnusedImports]
_CHECKSTYLE_RE = re.compile(
    r"^\[\w+\]\s*(.+?):(\d+)(?::\d+)?:\s*(.*?)\s*(?:\[\w+\])?$")
_CS_IMPORT_RE = re.compile(r"import(?:\s+from\s+\S+)?\s*-\s*([\w.]+)")


def _parse_checkstyle_unused(out: str) -> "list[tuple]":
    """Unused / redundant imports from the import-only Checkstyle run."""
    unused: list[tuple] = []
    for line in out.splitlines():
        m = _CHECKSTYLE_RE.match(line.strip())
        if not m:
            continue
        file, ln, msg = _compile_path(m.group(1)), int(m.group(2)), m.group(3)
        if "import" not in msg.lower():
            continue
        name_m = _CS_IMPORT_RE.search(msg)
        name = name_m.group(1).rstrip(".") if name_m else ""
        unused.append((file, ln, f"import {name};" if name else msg, name))
    return unused


# jdeps -verbose:package:  `   com.example.a -> com.example.b   classes`
_JDEPS_RE = re.compile(r"^\s*([\w.]+)\s*->\s*([\w.]+)\s+\S+\s*$")
# Package prefixes that are JDK / platform, never the reviewed project.
_JDEPS_EXTERNAL = ("java.", "javax.", "jdk.", "sun.", "com.sun.", "org.w3c.",
                   "org.xml.", "org.ietf.")


def _parse_jdeps_cycles(out: str) -> "list[list[str]]":
    """Build the package dependency graph from jdeps output and find cycles."""
    graph: dict[str, set[str]] = {}
    for line in out.splitlines():
        m = _JDEPS_RE.match(line)
        if not m:
            continue
        src, dst = m.group(1), m.group(2)
        if dst.startswith(_JDEPS_EXTERNAL) or src.startswith(_JDEPS_EXTERNAL):
            continue
        graph.setdefault(src, set()).add(dst)
        graph.setdefault(dst, set())
    return find_cycles(graph)


def compile_code(path: str, language: str) -> CompileOutput:
    """Compile via the project's build tool (Maven/Gradle), or plain `javac` when
    there is no build file. Compilation IS the type check in Java."""
    image, warn = _project_image(path)
    bt = _build_tool(path)
    if bt == "maven":
        cmd, compiler = _MAVEN_COMPILE, "mvn compile"
    elif bt == "gradle":
        cmd, compiler = _GRADLE_COMPILE, "gradle compileJava"
    else:
        cmd, compiler = _JAVAC_COMPILE, "javac"
    dr = _run_docker(image, cmd, path)
    errors, warnings = (_parse_java(dr.output) if dr.error is None else ([], []))
    if warn:
        warnings = [warn] + warnings
    return _compile_result(dr, language, compiler, errors, warnings)


def _parse_java(out: str) -> "tuple[list[ErrorOutput], list[str]]":
    """Parse plain javac (`f:line: error: msg`) and Maven (`[ERROR] f:[l,c] msg`)."""
    errors: list[ErrorOutput] = []
    warnings: list[str] = []
    for line in out.splitlines():
        stripped = line.strip()
        m = _MAVEN_RE.match(stripped)
        if m:
            severity, file, ln, col, msg = m.groups()
            file, ln, col = _compile_path(file), int(ln), int(col)
        else:
            m = _JAVAC_RE.match(stripped)
            if not m:
                continue
            file, ln, severity, msg = m.groups()
            file, ln, col = _compile_path(file), int(ln), 0
        if severity.lower() == "warning":
            warnings.append(f"{file}:{ln}: {msg.strip()}")
        else:
            errors.append(ErrorOutput(file=file, line=ln, column=col,
                                      message=msg.strip()))
    return errors, warnings