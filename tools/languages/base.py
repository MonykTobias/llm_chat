"""Shared infrastructure for the per-language tool implementations.

Every language's linter / test runner / type checker / import check now runs in
an ephemeral Docker container built from a fitting per-language image, so a
review never depends on the host's toolchain and never reads from / writes to the
reviewed project's own environment. The per-language modules (`python`,
`javascript`, `go`, `rust`, `java`) build a shell command (provision the
project's deps inside the container, then run the tool) and call the helpers here.

This module centralizes:

* building the per-language images on demand (`_ensure_image`), once per run;
* building project-specific images with deps pre-baked (`_ensure_project_image`),
  cached by Docker's layer cache so repeated reviews of the same project pay the
  install cost only when the manifest changes;
* running a command in a container and capturing it as a structured
  ``DockerResult`` (`_run_docker`) or as plain text (`_run_docker_text`);
* assembling a ``CompileOutput`` from a container run (`_compile_result`).

Design notes
------------
* The ``crav-*`` images are built from the Dockerfiles in
  ``tools/languages/docker/`` and exist only locally (no registry), so
  `_ensure_image` must build them before first use. Public base images are left
  to ``docker run``'s auto-pull.
* Project dependencies are baked into a per-project derived image by
  `_ensure_project_image` (one `docker build` per unique manifest content, then
  Docker's layer cache makes every subsequent build a no-op). Tools then run
  against this derived image with no per-container `pip install`.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from structured_output import CompileOutput, ErrorOutput
from tools._common import _IGNORE, _truncate

# Bundled default lint configs (used only when the reviewed project has none).
CONFIG_DIR = Path(__file__).parent / "configs"

# Dockerfiles for the per-language tool images live here as "<tag>.Dockerfile".
_DOCKER_DIR = Path(__file__).parent / "docker"


def _docker_exe() -> "str | None":
    """Resolve the `docker` binary on PATH (honors PATHEXT on Windows), or None."""
    return shutil.which("docker")


_DOCKER_MISSING = (
    "[environment] Docker is not installed or its daemon is not running. Every "
    "language tool (linter, tests, type check, imports, compile) runs inside a "
    "container; without Docker they cannot run. This is an environment problem, "
    "NOT a defect in the reviewed project — do not report it as a code finding."
)


# ── build the per-language images on demand (once per server run) ─────────
# Local-only images (no registry) must be built before first use. Guarded so a
# given tag builds at most once per server run; the lock also prevents two
# concurrent first-uses from racing the same `docker build`.
_BUILT: set[str] = set()
_BUILD_LOCK = threading.RLock()


def _ensure_image(tag: str) -> "str | None":
    """Ensure the local image `tag` exists, building it from its Dockerfile if not.

    Returns None on success (or when `tag` has no Dockerfile here — then it's a
    public image left to `docker run`'s auto-pull). Returns a clear ``[environment]``
    message when Docker is missing or the build fails, so callers surface it as an
    environment problem rather than a code finding.
    """
    docker = _docker_exe()
    if docker is None:
        return _DOCKER_MISSING
    dockerfile = _DOCKER_DIR / f"{tag}.Dockerfile"
    if not dockerfile.is_file():
        return None  # not one of ours — a pullable public image
    with _BUILD_LOCK:
        if tag in _BUILT:
            return None
        inspect = subprocess.run([docker, "image", "inspect", tag],
                                 capture_output=True, text=True,
                                 encoding="utf-8", errors="replace")
        if inspect.returncode == 0:
            _BUILT.add(tag)
            return None
        print(f"Building docker image {tag} from {dockerfile.name} (first use)...")
        try:
            build = subprocess.run(
                [docker, "build", "-f", str(dockerfile), "-t", tag, str(_DOCKER_DIR)],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=1800)
        except subprocess.TimeoutExpired:
            return (f"[environment] building docker image '{tag}' timed out. This is "
                    f"an environment problem, NOT a defect in the reviewed project.")
        except Exception as e:  # noqa: BLE001
            return f"[environment] could not build docker image '{tag}': {e}"
        if build.returncode != 0:
            return (f"[environment] failed to build docker image '{tag}'. This is an "
                    f"environment problem, NOT a defect in the reviewed project.\n"
                    + _truncate(build.stdout + build.stderr))
        _BUILT.add(tag)
        return None


# ── per-project derived images (deps pre-baked) ───────────────────────────
# For each (base_image, project manifest) pair we build a derived image once.
# Docker's layer cache means the `pip install` RUN layer is only re-executed
# when the manifest content changes — all subsequent builds for the same manifest
# complete in milliseconds. This eliminates repeated per-container pip installs
# in orchestrator workflows where the same project is reviewed many times.
#
# Tag scheme:  <base_image>-proj-<12-char sha1 of manifest content>
# The content hash (not the path) is the cache key so:
#   * moving/renaming the project doesn't invalidate the image;
#   * changing requirements.txt does invalidate it (new tag → new build).
#
# We still guard with _BUILD_LOCK + _BUILT so two concurrent tool calls for the
# same project don't race the same `docker build`.

def _project_image_tag(base_image: str, manifest_content: bytes) -> str:
    """Stable image tag derived from base image name + manifest file content."""
    digest = hashlib.sha1(manifest_content).hexdigest()[:12]
    # Sanitize base_image so the tag is always valid (no slashes/colons in name).
    safe_base = base_image.replace("/", "-").replace(":", "-")
    return f"{safe_base}-proj-{digest}"


def _ensure_project_image(
    base_image: str,
    project_path: str,
    manifests: "list[str]",
    install_cmd_fn: "callable[[str], str]",
) -> "tuple[str, str | None]":
    """Ensure a derived image exists with the project's deps pre-installed.

    Returns ``(image_tag, warning_or_None)``.

    How it works
    ------------
    1. First make sure the base image exists (calls `_ensure_image`).
    2. Find the project's dependency manifest (first match in `manifests`).
       If none exists, return the base image unchanged — nothing to pre-install.
    3. Hash the manifest content to form the derived image tag.  If that tag is
       already in `_BUILT` (or already exists in Docker), we're done instantly.
    4. Otherwise write a minimal Dockerfile into a temp build context:

           FROM <base_image>
           COPY <manifest> /tmp/<manifest>
           RUN <install_cmd>

       and run `docker build -t <tag> <ctx>`.  Docker caches the RUN layer
       against the COPY content, so a repeated build with the same manifest is a
       no-op (completes in <1 s).
    5. On build failure we fall back to the base image and return a warning
       string (non-fatal — the caller will prepend it to the tool output).

    Parameters
    ----------
    base_image:
        The ``crav-*`` base image tag, e.g. ``"crav-python"``.
    project_path:
        Absolute (or resolvable) path to the project root on the host.
    manifests:
        Ordered list of manifest filenames to look for, e.g.
        ``["requirements.txt", "pyproject.toml", "setup.py", "setup.cfg"]``.
        The first one found wins.
    install_cmd_fn:
        Callable that receives the found manifest filename and returns the
        shell command to bake into the ``RUN`` layer, e.g.
        ``lambda m: f"pip install --no-cache-dir -q -r /tmp/{m}"``.
        For a pyproject / setup.py install you'd copy the whole project tree
        instead — see python.py for an example.
    """
    # Step 1 — make sure the base image is available.
    base_err = _ensure_image(base_image)
    if base_err is not None:
        return base_image, base_err

    docker = _docker_exe()
    if docker is None:
        return base_image, _DOCKER_MISSING

    # Step 2 — find the manifest.
    root = Path(project_path).resolve()
    manifest_name: "str | None" = None
    for name in manifests:
        if (root / name).is_file():
            manifest_name = name
            break
    if manifest_name is None:
        return base_image, None  # no manifest → nothing to pre-install

    manifest_path = root / manifest_name
    manifest_content = manifest_path.read_bytes()

    # Step 3 — derive a stable tag from the manifest content.
    tag = _project_image_tag(base_image, manifest_content)

    with _BUILD_LOCK:
        # Fast path: already built this session.
        if tag in _BUILT:
            return tag, None

        # Fast path: image already exists in Docker (survives server restarts).
        inspect = subprocess.run(
            [docker, "image", "inspect", tag],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if inspect.returncode == 0:
            _BUILT.add(tag)
            return tag, None

        # Step 4 — build the derived image.
        install_cmd = install_cmd_fn(manifest_name)
        print(f"Building project image {tag} "
              f"(base={base_image}, manifest={manifest_name})...")

        with tempfile.TemporaryDirectory() as ctx:
            ctx_path = Path(ctx)

            # Copy only the manifest into the build context to keep it tiny.
            shutil.copy2(manifest_path, ctx_path / manifest_name)

            dockerfile = (
                f"FROM {base_image}\n"
                f"COPY {manifest_name} /tmp/{manifest_name}\n"
                f"RUN {install_cmd}\n"
            )
            (ctx_path / "Dockerfile").write_text(dockerfile)

            try:
                build = subprocess.run(
                    [docker, "build", "-t", tag, ctx],
                    capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=1800,
                )
            except subprocess.TimeoutExpired:
                warn = (f"[environment] building project image '{tag}' timed out; "
                        f"falling back to base image (deps will not be pre-installed).")
                return base_image, warn
            except Exception as e:  # noqa: BLE001
                warn = (f"[environment] could not build project image '{tag}': {e}; "
                        f"falling back to base image.")
                return base_image, warn

            # Step 5 — handle build failure (non-fatal fallback).
            if build.returncode != 0:
                warn = (
                    f"[environment] project image build failed for '{tag}'; "
                    f"falling back to base image (deps will not be pre-installed).\n"
                    + _truncate(build.stderr)
                )
                return base_image, warn

        _BUILT.add(tag)
        return tag, None


# ── container runner ──────────────────────────────────────────────────────

def _tar_copy_shell(command: str) -> str:
    """Return a shell fragment that copies /src → /work, skipping _IGNORE dirs.

    Uses a `tar` pipe instead of `cp -a` so vendored directories (.venv,
    node_modules, .git, build, …) are pruned *before* any bytes cross the bind
    mount — the same strategy compile_code already uses with `find -prune`.
    `tar --exclude` matches on the basename of each entry, which is exactly what
    we want: every name in _IGNORE is a plain directory basename.
    """
    excludes = " ".join(f"--exclude=./{name}" for name in sorted(_IGNORE))
    return (
        f"mkdir -p /work && "
        f"tar -C /src {excludes} -cf - . | tar -C /work -xf - && "
        f"cd /work && {command}"
    )


# Each tool runs in an ephemeral container so it never writes to (or executes
# against) the host tree. Isolation: the project is mounted read-only at /src and
# (by default) copied into a throwaway /work the tool can write to; --rm discards
# everything on exit; resource limits cap blast radius. Network is left ON
# (default bridge) so deps can be fetched. Per-ecosystem named volumes
# (`cache_volumes`) persist the dependency cache across runs; extra read-only
# `mounts` carry the bundled configs / the import-check script into the container.
#
# `copy=False` runs directly against the read-only /src (workdir /src) — the
# caller is then responsible for not writing into /src. Python's compile_code uses
# this: it walks /src itself with `find -prune` and redirects bytecode to /tmp.

@dataclass
class DockerResult:
    """Outcome of one `_run_docker` call.

    `error` is set (and the run did not really happen) when Docker is missing, an
    image build failed, the run timed out, or launch failed — callers map that to
    a `status="unavailable"` CompileOutput / an `[environment]` text message
    rather than a code finding.
    """
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    error: "str | None" = None

    @property
    def output(self) -> str:
        """Combined stdout+stderr, the way tools split output across both."""
        return (self.stdout + ("\n" if self.stdout and self.stderr else "")
                + self.stderr)


def _run_docker(image: str, command: str, project_path: str,
                timeout: int = 300, *, copy: bool = True,
                env: "dict[str, str] | None" = None,
                cache_volumes: "list[tuple[str, str]] | None" = None,
                mounts: "list[tuple[str, str]] | None" = None) -> DockerResult:
    """Run `command` inside an ephemeral container; capture exit code + output + timing.

    Builds the image first if it's one of ours (`_ensure_image`). Mounts
    `project_path` read-only at /src. By default copies /src to a writable /work
    and runs there (`copy=False` runs against the read-only /src instead).
    `cache_volumes` are ``(volume_name, container_path)`` named-volume mounts that
    survive ``--rm`` (the dependency cache); `mounts` are ``(host_path,
    container_path)`` extra read-only host mounts (bundled configs, scripts).
    `env` adds container environment variables. On a missing binary / failed build
    / timeout / launch failure it sets `error` instead of raising.
    """
    build_err = _ensure_image(image)
    if build_err is not None:
        return DockerResult(error=build_err)
    docker = _docker_exe()
    if docker is None:
        return DockerResult(error=_DOCKER_MISSING)

    src = str(Path(project_path).resolve())
    if copy:
        workdir = "/work"
        shell = _tar_copy_shell(command)
    else:
        workdir = "/src"
        shell = command
    vol_flags: list[str] = []
    for name, cpath in (cache_volumes or []):
        vol_flags += ["-v", f"{name}:{cpath}"]
    for host, cpath in (mounts or []):
        vol_flags += ["-v", f"{Path(host).resolve()}:{cpath}:ro"]
    env_flags: list[str] = []
    for key, value in (env or {}).items():
        env_flags += ["-e", f"{key}={value}"]
    cmd = [
        docker, "run", "--rm",
        "-v", f"{src}:/src:ro",
        "-w", workdir,
        "--memory=2g", "--cpus=2", "--pids-limit=512",
        "--security-opt", "no-new-privileges",
        *vol_flags, *env_flags,
        image,
        "sh", "-c", shell,
    ]

    start = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                encoding="utf-8", errors="replace",
                                timeout=timeout)
    except FileNotFoundError:
        return DockerResult(error=_DOCKER_MISSING)
    except subprocess.TimeoutExpired:
        elapsed = int((time.perf_counter() - start) * 1000)
        return DockerResult(
            duration_ms=elapsed,
            error=(f"[tool timeout] the run in '{image}' did not finish within "
                   f"{timeout}s (the first run also pulls/builds the image and "
                   f"fetches dependencies). Narrow the project or raise the timeout."))
    except Exception as e:  # noqa: BLE001
        return DockerResult(error=f"[tool error] could not launch docker: {e}")
    elapsed = int((time.perf_counter() - start) * 1000)
    return DockerResult(
        exit_code=result.returncode,
        stdout=_truncate(result.stdout or ""),
        stderr=_truncate(result.stderr or ""),
        duration_ms=elapsed,
    )


def _run_docker_text(image: str, command: str, project_path: str, *,
                     name: str = "tool", timeout: int = 300, copy: bool = True,
                     env: "dict[str, str] | None" = None,
                     cache_volumes: "list[tuple[str, str]] | None" = None,
                     mounts: "list[tuple[str, str]] | None" = None) -> str:
    """Run a tool in a container and return its combined output as text.

    Mirrors the old host `_run` conventions for the linter / test / type-check /
    import tools: a nonzero exit still returns the captured output (that IS the
    result); only a missing Docker / failed build / timeout short-circuits with an
    ``[environment]``/``[tool …]`` message.
    """
    dr = _run_docker(image, command, project_path, timeout=timeout, copy=copy,
                     env=env, cache_volumes=cache_volumes, mounts=mounts)
    if dr.error is not None:
        return dr.error
    return dr.output.strip() or f"'{name}' ran and produced no output."


# ── compile-output assembly (shared by every language's compile_code) ─────

def _compile_path(raw: str) -> str:
    """Normalize a compiler-reported path to project-relative for ErrorOutput.

    Builds happen in the container's /work (a copy of /src), so paths come back as
    ``/work/pkg/x.go`` or ``./pkg/x.go``; strip those prefixes so the file is
    shown the way it sits in the reviewed project.
    """
    p = raw.strip().strip('"')
    for prefix in ("/work/", "/src/"):
        if p.startswith(prefix):
            p = p[len(prefix):]
            break
    while p.startswith("./"):
        p = p[2:]
    return p


def _compile_result(dr: DockerResult, language: str, compiler: str,
                    errors: "list[ErrorOutput]",
                    warnings: "list[str]") -> CompileOutput:
    """Assemble a CompileOutput from a DockerResult + parsed errors/warnings.

    Centralizes the status convention so every language agrees: a missing/timed-out
    Docker (or failed image build) is `unavailable` (an environment problem,
    surfaced via warnings), a nonzero exit or any parsed error is `error`,
    otherwise `success`. Warnings never flip the status on their own.
    """
    if dr.error is not None:
        return CompileOutput(
            status="unavailable", language=language, compiler=compiler,
            exit_code=-1, errors=[], warnings=[dr.error],
            duration_ms=dr.duration_ms)
    status = "error" if (dr.exit_code != 0 or errors) else "success"
    return CompileOutput(
        status=status, language=language, compiler=compiler,
        exit_code=dr.exit_code, errors=errors, warnings=warnings,
        duration_ms=dr.duration_ms)