"""Docker-isolated execution and evidence reporting for testing handoffs."""

from __future__ import annotations

import json
import os
import re
import selectors
import shutil
import shlex
import signal
import subprocess
import tarfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Callable, Protocol

from testing_agent.contract import (
    HandoffError,
    Snapshot,
    ValidatedHandoff,
    create_snapshot,
    new_testing_run_id,
    validate_handoff,
)


LOG_LIMIT_BYTES = 64 * 1024
DOCKER_IMAGE_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._/:@-]{0,254}$")
SECRET_PATTERNS = (
    re.compile(r"(?i)(https?://)[^/@\s]+@"),
    re.compile(r"(?i)\b(authorization|proxy-authorization|cookie|set-cookie)(\s*[:=]\s*)[^\r\n,]+"),
    re.compile(r"(?i)\b(password|passwd|token|secret|api[_-]?key|credential)(\s*[=:]\s*)[^,\s\"']+"),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})\b"),
)
SUPPORTED_TEST_FILES = {".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact(value: str) -> str:
    value = SECRET_PATTERNS[0].sub(r"\1[REDACTED]@", value)
    value = SECRET_PATTERNS[1].sub(r"\1\2[REDACTED]", value)
    value = SECRET_PATTERNS[2].sub(r"\1\2[REDACTED]", value)
    value = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [REDACTED]", value)
    return SECRET_PATTERNS[3].sub("[REDACTED]", value)


@dataclass(frozen=True)
class AgentSettings:
    python_image: str = "python:3.12.11-slim-bookworm"
    node_image: str = "node:22.16.0-bookworm-slim"
    uv_version: str = "0.8.15"
    setup_timeout_seconds: int = 900
    test_timeout_seconds: int = 900
    docker_timeout_seconds: int = 900
    memory_limit: str = "2g"
    cpu_limit: float = 2.0
    pids_limit: int = 256

    def validate(self) -> None:
        for name, image in (("python_image", self.python_image), ("node_image", self.node_image)):
            if not isinstance(image, str) or not DOCKER_IMAGE_PATTERN.fullmatch(image) or image.startswith("-"):
                raise HandoffError(f"configured {name} is not a valid Docker image reference")
        if not self.python_image.startswith(("python:", "docker.io/library/python:")):
            raise HandoffError("python_image must use the official Docker Hub Python image")
        if not self.node_image.startswith(("node:", "docker.io/library/node:")):
            raise HandoffError("node_image must use the official Docker Hub Node image")
        if not re.search(r"(?:^|/)python:\d+\.\d+\.\d+(?:[-@]|$)", self.python_image):
            raise HandoffError("python_image must use an explicit versioned tag, not latest")
        if not re.search(r"(?:^|/)node:\d+\.\d+\.\d+(?:[-@]|$)", self.node_image):
            raise HandoffError("node_image must use an explicit versioned tag, not latest")
        if not isinstance(self.uv_version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", self.uv_version):
            raise HandoffError("configured uv_version must be an exact x.y.z version")
        for name, value in (
            ("setup_timeout_seconds", self.setup_timeout_seconds),
            ("test_timeout_seconds", self.test_timeout_seconds),
            ("docker_timeout_seconds", self.docker_timeout_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 900:
                raise HandoffError(f"configured {name} must be between 1 and 900 seconds")
        if isinstance(self.cpu_limit, bool) or not isinstance(self.cpu_limit, (int, float)) or not 0.1 <= self.cpu_limit <= 64:
            raise HandoffError("configured cpu_limit must be between 0.1 and 64")
        if isinstance(self.pids_limit, bool) or not isinstance(self.pids_limit, int) or not 16 <= self.pids_limit <= 4096:
            raise HandoffError("configured pids_limit must be between 16 and 4096")
        if not isinstance(self.memory_limit, str) or not re.fullmatch(r"\d+(?:[kmg])?", self.memory_limit.lower()):
            raise HandoffError("configured memory_limit must be a Docker memory size such as 2g")


@dataclass(frozen=True)
class CommandResult:
    name: str
    kind: str
    argv: tuple[str, ...]
    status: str
    exit_code: int | None
    duration_seconds: float
    output: str
    timeout_seconds: int


@dataclass(frozen=True)
class TestPlan:
    runtime: str
    runtime_requirement: str | None
    runtime_requirement_verified: bool
    image: str
    package_manager: str
    package_manager_version: str | None
    workdir: str
    setup_commands: tuple[tuple[str, tuple[str, ...], int], ...]
    test_command: tuple[str, ...] | None
    build_command: tuple[str, ...] | None
    test_framework: str | None
    test_files: tuple[str, ...]
    build_workdir: str


@dataclass
class RunEvidence:
    outcome: str
    setup_status: str
    setup_details: str
    commands: list[CommandResult] = field(default_factory=list)
    test_count: int | None = None
    passed_count: int | None = None
    failed_count: int | None = None
    test_evidence: str = "not_run"
    limitations: list[str] = field(default_factory=list)


class DockerFailure(RuntimeError):
    """A Docker daemon or CLI operation failed."""


class DockerBackend(Protocol):
    def check(self, timeout: int, cancel_event: Event) -> str: ...

    def start(self, image: str, settings: AgentSettings, cancel_event: Event) -> str: ...

    def copy_into(self, container_id: str, source: Path, cancel_event: Event) -> None: ...

    def disconnect_network(self, container_id: str, cancel_event: Event) -> None: ...

    def execute(
        self,
        container_id: str,
        argv: tuple[str, ...],
        workdir: str,
        timeout: int,
        cancel_event: Event,
    ) -> tuple[str, int | None, float, str]: ...

    def remove(self, container_id: str) -> str | None: ...


class DockerCli:
    """Small Docker CLI adapter; generated code never receives Docker access."""

    def __init__(self, settings: AgentSettings) -> None:
        self.settings = settings

    def _invoke(
        self,
        argv: list[str],
        timeout: int,
        cancel_event: Event,
        *,
        output_limit: int = 4096,
    ) -> tuple[int, str]:
        if cancel_event.is_set():
            raise InterruptedError("cancelled")
        try:
            process = subprocess.Popen(
                ["docker", *argv],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
            )
        except OSError as error:
            raise DockerFailure(f"Docker CLI is unavailable: {error}") from error
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        collected = bytearray()
        started = time.monotonic()
        try:
            while True:
                if cancel_event.is_set():
                    self._kill_process(process)
                    raise InterruptedError("cancelled")
                if time.monotonic() - started > timeout:
                    self._kill_process(process)
                    raise DockerFailure(f"Docker operation timed out after {timeout} seconds")
                for key, _ in selector.select(timeout=0.1):
                    chunk = os.read(key.fd, 8192)
                    if not chunk:
                        selector.unregister(key.fileobj)
                    elif len(collected) < output_limit:
                        collected.extend(chunk[: output_limit - len(collected)])
                code = process.poll()
                if code is not None and not selector.get_map():
                    return code, collected.decode("utf-8", errors="replace")
                if code is not None:
                    try:
                        chunk = os.read(process.stdout.fileno(), 8192)
                    except OSError:
                        chunk = b""
                    if not chunk:
                        if selector.get_map():
                            selector.unregister(process.stdout)
                    elif len(collected) < output_limit:
                        collected.extend(chunk[: output_limit - len(collected)])
        finally:
            selector.close()
            if process.poll() is None:
                self._kill_process(process)

    @staticmethod
    def _kill_process(process: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                process.kill()
            process.wait()

    def _checked(self, argv: list[str], cancel_event: Event, timeout: int | None = None) -> str:
        code, output = self._invoke(
            argv,
            timeout or self.settings.docker_timeout_seconds,
            cancel_event,
        )
        if code != 0:
            raise DockerFailure(_redact(output.strip()) or f"docker {' '.join(argv[:2])} failed ({code})")
        return output

    def check(self, timeout: int, cancel_event: Event) -> str:
        return self._checked(["version", "--format", "{{.Server.Version}}"], cancel_event, timeout).strip()

    def start(self, image: str, settings: AgentSettings, cancel_event: Event) -> str:
        name = f"agents-test-{new_testing_run_id()}"
        try:
            output = self._checked(
                [
                    "run", "--detach", "--rm", "--name", name,
                    "--network", "bridge",
                    "--memory", settings.memory_limit,
                    "--memory-swap", settings.memory_limit,
                    "--cpus", str(settings.cpu_limit),
                    "--pids-limit", str(settings.pids_limit),
                    "--cap-drop=ALL",
                    "--security-opt", "no-new-privileges:true",
                    "--user", "1000:1000",
                    "--read-only",
                    "--tmpfs", "/tmp:rw,nosuid,nodev,size=2g",
                    "--workdir", "/",
                    "--env", "HOME=/tmp/home",
                    image,
                    "sh", "-c", "mkdir -p /tmp/workspace /tmp/home && chmod 700 /tmp/home && sleep infinity",
                ],
                cancel_event,
                timeout=settings.docker_timeout_seconds,
            )
        except Exception:
            try:
                self._invoke(["rm", "--force", name], 10, Event(), output_limit=1024)
            except Exception:
                pass
            raise
        container_id = output.strip().splitlines()[-1]
        if not re.fullmatch(r"[a-fA-F0-9]{12,64}", container_id):
            raise DockerFailure("Docker did not return a valid disposable container ID")
        return container_id

    def copy_into(self, container_id: str, source: Path, cancel_event: Event) -> None:
        try:
            process = subprocess.Popen(
                ["docker", "exec", "--interactive", container_id, "tar", "-xf", "-", "-C", "/tmp/workspace"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                close_fds=True,
                start_new_session=True,
            )
        except OSError as error:
            raise DockerFailure(f"Docker CLI is unavailable: {error}") from error
        write_errors: list[Exception] = []
        error_output = bytearray()

        def drain_errors() -> None:
            assert process.stderr is not None
            while chunk := process.stderr.read(8192):
                if len(error_output) < 4096:
                    error_output.extend(chunk[: 4096 - len(error_output)])

        def write_archive() -> None:
            try:
                assert process.stdin is not None
                with process.stdin, tarfile.open(fileobj=process.stdin, mode="w|") as archive:
                    for entry in sorted(source.rglob("*"), key=lambda path: path.relative_to(source).as_posix()):
                        if entry.is_symlink():
                            raise DockerFailure("snapshot unexpectedly contains a symbolic link")
                        info = archive.gettarinfo(
                            str(entry),
                            arcname=entry.relative_to(source).as_posix(),
                        )
                        info.mode |= 0o200
                        if entry.is_dir():
                            info.mode |= 0o100
                            archive.addfile(info)
                        else:
                            with entry.open("rb") as source_file:
                                archive.addfile(info, source_file)
            except Exception as error:
                write_errors.append(error)

        writer = threading.Thread(target=write_archive, daemon=True)
        error_reader = threading.Thread(target=drain_errors, daemon=True)
        writer.start()
        error_reader.start()
        started = time.monotonic()
        while writer.is_alive():
            if cancel_event.is_set():
                self._kill_process(process)
                writer.join(timeout=2)
                raise InterruptedError("cancelled")
            if time.monotonic() - started > self.settings.docker_timeout_seconds:
                self._kill_process(process)
                writer.join(timeout=2)
                raise DockerFailure("Docker source transfer timed out")
            if process.poll() is not None:
                writer.join(timeout=0.2)
                break
            writer.join(timeout=0.1)
        writer.join()
        code = process.wait()
        error_reader.join()
        if write_errors:
            raise DockerFailure(f"Unable to transfer the sanitized source snapshot: {write_errors[0]}")
        if code != 0:
            details = _redact(error_output.decode("utf-8", errors="replace").strip())
            raise DockerFailure(f"Docker could not extract the sanitized source snapshot (exit {code}): {details}")
        if (source / "product").is_dir() and (source / "tests").is_dir():
            self._checked(
                [
                    "exec", container_id, "sh", "-c",
                    "mkdir -p /tmp/workspace/product/tests && cp -a /tmp/workspace/tests/. /tmp/workspace/product/tests/",
                ],
                cancel_event,
            )
        self._checked(
            ["exec", container_id, "chmod", "-R", "u+rwX", "/tmp/workspace"],
            cancel_event,
        )

    def disconnect_network(self, container_id: str, cancel_event: Event) -> None:
        self._checked(["network", "disconnect", "bridge", container_id], cancel_event)

    def execute(
        self,
        container_id: str,
        argv: tuple[str, ...],
        workdir: str,
        timeout: int,
        cancel_event: Event,
    ) -> tuple[str, int | None, float, str]:
        started = time.monotonic()
        process: subprocess.Popen[bytes] | None = None
        try:
            process = subprocess.Popen(
                ["docker", "exec", "--env", "HOME=/tmp/home", "--env", "TMPDIR=/tmp", "--workdir", workdir, container_id, *argv],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
            )
        except OSError as error:
            raise DockerFailure(f"Docker CLI is unavailable: {error}") from error
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        collected = bytearray()
        timed_out = False
        try:
            while True:
                if cancel_event.is_set():
                    self._kill_process(process)
                    return "cancelled", None, time.monotonic() - started, _redact(collected.decode("utf-8", "replace"))
                if time.monotonic() - started > timeout:
                    timed_out = True
                    self._kill_process(process)
                    break
                for key, _ in selector.select(timeout=0.1):
                    chunk = os.read(key.fd, 8192)
                    if not chunk:
                        selector.unregister(key.fileobj)
                    elif len(collected) < LOG_LIMIT_BYTES:
                        collected.extend(chunk[: LOG_LIMIT_BYTES - len(collected)])
                code = process.poll()
                if code is not None and not selector.get_map():
                    break
                if code is not None:
                    try:
                        chunk = os.read(process.stdout.fileno(), 8192)
                    except OSError:
                        chunk = b""
                    if not chunk and selector.get_map():
                        selector.unregister(process.stdout)
                    elif chunk and len(collected) < LOG_LIMIT_BYTES:
                        collected.extend(chunk[: LOG_LIMIT_BYTES - len(collected)])
            code = process.poll()
        finally:
            selector.close()
            if process.poll() is None:
                self._kill_process(process)
        output = collected.decode("utf-8", errors="replace")
        if len(collected) >= LOG_LIMIT_BYTES:
            output += "\n[log truncated at 64 KiB]\n"
        if timed_out:
            status = "timed_out"
        elif code == 0:
            status = "passed"
        else:
            status = "failed"
        return status, code, time.monotonic() - started, _redact(output)

    def remove(self, container_id: str) -> str | None:
        try:
            code, output = self._invoke(["rm", "--force", container_id], 10, Event(), output_limit=1024)
            if code != 0:
                return _redact(output.strip()) or f"docker rm exited with status {code}"
        except Exception as error:
            return _redact(str(error))
        return None


class SetupIssue(RuntimeError):
    pass


def _files(root: Path) -> list[Path]:
    found: list[Path] = []
    if not root.is_dir():
        return found
    for directory, directories, names in os.walk(root, followlinks=False):
        directories[:] = sorted(
            name for name in directories
            if name not in {".git", "node_modules", "venv", ".venv", "__pycache__", ".pytest_cache"}
            and not (Path(directory) / name).is_symlink()
        )
        for name in sorted(names):
            path = Path(directory) / name
            if path.is_symlink() or not path.is_file():
                continue
            found.append(path)
    return found


def _load_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _node_test_files(*roots: Path) -> tuple[str, ...]:
    result = []
    for root in roots:
        for path in _files(root):
            suffix = path.suffix.lower()
            parts = {part.lower() for part in path.relative_to(root).parts[:-1]}
            name = path.name.lower()
            if suffix in SUPPORTED_TEST_FILES and (
                ".test." in name or ".spec." in name or bool(parts & {"test", "tests", "__tests__", "e2e"})
            ):
                result.append(f"{root.name}/{path.relative_to(root).as_posix()}")
    return tuple(result)


def _python_test_files(*roots: Path) -> tuple[str, ...]:
    result: list[str] = []
    for root in roots:
        for path in _files(root):
            if path.suffix == ".py" and (path.name.startswith("test_") or path.name.endswith("_test.py")):
                result.append(path.as_posix())
    return tuple(sorted(set(result)))


def _declared_dependency_names(handoff: ValidatedHandoff) -> set[str]:
    entries = handoff.data.get("declared_dependencies", [])
    if not isinstance(entries, list):
        return set()
    return {
        str(item.get("name", "")).lower().replace("_", "-")
        for item in entries
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }


def _root_manifest_paths(handoff: ValidatedHandoff, snapshot: Path) -> tuple[Path, Path]:
    if handoff.project_mode == "existing_project":
        return snapshot / "project", snapshot / "project"
    return snapshot / "product", snapshot / "tests"


def _make_plan(handoff: ValidatedHandoff, snapshot: Path, settings: AgentSettings) -> TestPlan:
    project_root, tests_root = _root_manifest_paths(handoff, snapshot)
    package = _load_json(project_root / "package.json")
    python_manifest = any(
        (project_root / name).is_file()
        for name in ("pyproject.toml", "requirements.txt", "Pipfile", "setup.py")
    )
    if package is not None and python_manifest:
        raise SetupIssue("mixed Python and Node project roots are not supported in one test run")

    if package is not None:
        manager_value = package.get("packageManager")
        if manager_value is not None and not isinstance(manager_value, str):
            raise SetupIssue("package.json packageManager must be a string")
        manager_version: str | None = None
        if manager_value == "npm":
            pass
        elif isinstance(manager_value, str):
            version_match = re.fullmatch(r"npm@(\d+\.\d+\.\d+)(?:\+.*)?", manager_value)
            if not version_match:
                raise SetupIssue("packageManager must be npm or declare an exact npm x.y.z version")
            manager_version = version_match.group(1)
        if not (project_root / "package-lock.json").is_file():
            found = next(
                (name for name in ("pnpm-lock.yaml", "yarn.lock", "bun.lock", "bun.lockb") if (project_root / name).exists()),
                None,
            )
            detail = f"found {found}" if found else "no package lockfile was found"
            raise SetupIssue(f"npm ci requires package-lock.json; {detail}")
        node_version = _node_version_requirement(project_root, package)
        runtime_verified = _check_runtime_compatibility(node_version, settings.node_image, "Node.js")
        scripts = package.get("scripts", {})
        if not isinstance(scripts, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in scripts.items()):
            raise SetupIssue("package.json scripts have an unsupported shape")
        test_files = _node_test_files(project_root, tests_root) if handoff.project_mode == "new_project" else _node_test_files(project_root)
        if handoff.project_mode == "new_project":
            overlaps = [
                path.relative_to(tests_root).as_posix()
                for path in _files(tests_root)
                if (project_root / "tests" / path.relative_to(tests_root)).exists()
            ]
            if overlaps:
                raise SetupIssue(f"generated tests collide with product/tests files: {overlaps[0]}")
        test_script = scripts.get("test")
        if test_files and not isinstance(test_script, str):
            raise SetupIssue("Node.js test files were found, but package.json has no test script")
        if test_files:
            parsed_test = _node_test_command(test_script)
            if parsed_test is None:
                raise SetupIssue(
                    "package.json test script must directly invoke vitest, jest, mocha, tap, or node --test; shell wrappers are not executed"
                )
            test_command, framework = parsed_test
        else:
            test_command = None
            framework = None
        build_command = ("npm", "run", "build") if isinstance(scripts.get("build"), str) else None
        npm_setup: list[tuple[str, tuple[str, ...], int]] = []
        if manager_version is not None:
            npm_setup.append(("verify-package-manager", ("npm", "--version"), settings.docker_timeout_seconds))
        npm_setup.append(("install", ("npm", "ci", "--no-audit", "--no-fund"), settings.setup_timeout_seconds))
        return TestPlan(
            runtime="node",
            runtime_requirement=node_version,
            runtime_requirement_verified=runtime_verified,
            image=settings.node_image,
            package_manager="npm",
            package_manager_version=manager_version,
            workdir="/tmp/workspace/product" if handoff.project_mode == "new_project" else "/tmp/workspace/project",
            setup_commands=tuple(npm_setup),
            test_command=test_command,
            build_command=build_command,
            test_framework=framework,
            test_files=test_files,
            build_workdir="/tmp/workspace/product" if handoff.project_mode == "new_project" else "/tmp/workspace/project",
        )

    has_python = python_manifest or any(
        path.suffix == ".py"
        for root in (project_root, tests_root)
        for path in _files(root)
    )
    if not has_python:
        raise SetupIssue("no supported Python or npm project manifest was found")
    python_version = _python_version_requirement(project_root)
    runtime_verified = _check_runtime_compatibility(python_version, settings.python_image, "Python")
    unsupported_locks = [name for name in ("poetry.lock", "Pipfile.lock") if (project_root / name).exists()]
    uv_locked = (project_root / "uv.lock").is_file() and (project_root / "pyproject.toml").is_file()
    if (project_root / "Pipfile").is_file() and not (project_root / "Pipfile.lock").is_file():
        raise SetupIssue("Pipfile requires Pipfile.lock; Pipenv installs are not run without its lockfile")
    if unsupported_locks:
        raise SetupIssue(f"Python lockfile manager is unsupported: {', '.join(unsupported_locks)}")
    if (project_root / "uv.lock").exists() and not uv_locked:
        raise SetupIssue("uv.lock requires a root pyproject.toml")
    setup: list[tuple[str, tuple[str, ...], int]] = []
    if uv_locked:
        setup.extend([
            ("install-test-tooling", ("python", "-m", "pip", "install", "--disable-pip-version-check", f"uv=={settings.uv_version}"), settings.setup_timeout_seconds),
            ("install-project", ("uv", "sync", "--locked", "--all-groups", "--all-extras"), settings.setup_timeout_seconds),
        ])
        python_command = ("uv", "run", "--locked", "python")
    else:
        requirement_files: list[str] = []
        for root in dict.fromkeys((project_root, tests_root)):
            if not root.is_dir():
                continue
            prefix = "" if root == project_root else "../tests/"
            for filename in ("requirements.txt", "requirements-dev.txt", "requirements-test.txt"):
                if (root / filename).is_file():
                    requirement_files.append(f"{prefix}{filename}")
        for requirement_file in requirement_files:
            setup.append((
                f"install-{requirement_file}",
                ("python", "-m", "pip", "install", "--disable-pip-version-check", "-r", requirement_file),
                settings.setup_timeout_seconds,
            ))
        if (project_root / "pyproject.toml").is_file() or (project_root / "setup.py").is_file():
            no_deps = ["--no-deps"] if requirement_files else []
            setup.append(("install-project", ("python", "-m", "pip", "install", "--disable-pip-version-check", *no_deps, "-e", "."), settings.setup_timeout_seconds))
        python_command = ("python",)

    test_files = _python_test_files(project_root, tests_root)
    dependencies = _declared_dependency_names(handoff)
    pytest_declared = "pytest" in dependencies or any(
        re.match(r"\s*pytest(?:\s|[<=>~!;\[])", line)
        for name in ("requirements.txt", "requirements-dev.txt", "requirements-test.txt")
        if (project_root / name).is_file()
        for line in (project_root / name).read_text(encoding="utf-8", errors="replace").splitlines()
    )
    unittest_declared = any(
        "unittest.TestCase" in path.read_text(encoding="utf-8", errors="replace")
        or re.search(r"class\s+\w+\s*\(\s*TestCase\s*\)", path.read_text(encoding="utf-8", errors="replace"))
        for path_string in test_files
        for path in [Path(path_string)]
        if path.is_file()
    )
    if test_files and pytest_declared:
        test_paths = ("/tmp/workspace/project",) if handoff.project_mode == "existing_project" else ("/tmp/workspace/product", "/tmp/workspace/tests")
        test_command = (*python_command, "-m", "pytest", "-q", *test_paths)
        framework = "pytest"
    elif test_files and unittest_declared:
        if handoff.project_mode == "new_project":
            has_tests_root = any(Path(path).is_relative_to(tests_root) for path in test_files)
            has_product_tests = any(Path(path).is_relative_to(project_root) for path in test_files)
            if has_tests_root and has_product_tests:
                raise SetupIssue("unittest suites split across product and tests roots are unsupported")
            unittest_root = "/tmp/workspace/tests" if has_tests_root else "/tmp/workspace/product"
            test_command = (*python_command, "-m", "unittest", "discover", "-v", "-s", unittest_root)
        else:
            test_command = (*python_command, "-m", "unittest", "discover", "-v")
        framework = "unittest"
    elif test_files:
        raise SetupIssue("Python test files were found, but no declared pytest dependency or unittest.TestCase suite was recognized")
    else:
        test_command = None
        framework = None
    if project_root == tests_root:
        build_workdir = "/tmp/workspace/project" if handoff.project_mode == "existing_project" else "/tmp/workspace/product"
    else:
        build_workdir = "/tmp/workspace/product"
    return TestPlan(
        runtime="python",
        runtime_requirement=python_version,
        runtime_requirement_verified=runtime_verified,
        image=settings.python_image,
        package_manager="uv" if uv_locked else "pip",
        package_manager_version=settings.uv_version if uv_locked else None,
        workdir="/tmp/workspace/project" if handoff.project_mode == "existing_project" else "/tmp/workspace/product",
        setup_commands=tuple(setup),
        test_command=test_command,
        build_command=None,
        test_framework=framework,
        test_files=test_files,
        build_workdir=build_workdir,
    )


def _node_test_command(script: str) -> tuple[tuple[str, ...], str] | None:
    try:
        tokens = shlex.split(script)
    except ValueError:
        return None
    if not tokens or any(character in script for character in (";", "&", "|", ">", "<", "`", "$", "\n", "\r")):
        return None
    if len(tokens) >= 2 and tokens[0] == "node" and tokens[1] == "--test":
        return tuple(tokens), "node --test"
    if tokens[0] == "npx" and len(tokens) >= 2:
        tokens = tokens[1:]
    framework = tokens[0]
    if framework not in {"vitest", "jest", "mocha", "tap"}:
        return None
    return ("npm", "exec", "--no", "--", *tokens), framework


def _node_version_requirement(project_root: Path, package: dict[str, object]) -> str | None:
    for filename in (".nvmrc", ".node-version"):
        path = project_root / filename
        if path.is_file():
            content = path.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                return content
    engines = package.get("engines", {})
    if isinstance(engines, dict) and isinstance(engines.get("node"), str):
        return engines["node"]
    return None


def _python_version_requirement(project_root: Path) -> str | None:
    for filename in (".python-version",):
        path = project_root / filename
        if path.is_file():
            content = path.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                return content
    pyproject = project_root / "pyproject.toml"
    if pyproject.is_file():
        match = re.search(r"(?m)^requires-python\s*=\s*[\"']([^\"']+)", pyproject.read_text(encoding="utf-8", errors="replace"))
        if match:
            return match.group(1)
    return None


def _check_runtime_compatibility(requirement: str | None, image: str, runtime: str) -> bool:
    if requirement is None:
        return True
    image_runtime = "node" if runtime == "Node.js" else runtime.lower()
    image_version = re.search(rf"{image_runtime}:(\d+)\.(\d+)(?:\.(\d+))?", image)
    if image_version is None:
        raise SetupIssue(f"configured {runtime} image must include an explicit numeric version")
    image_tuple = tuple(int(part or 0) for part in image_version.groups())
    exact = re.fullmatch(r"v?(\d+)\.(\d+)(?:\.(\d+))?", requirement)
    if exact:
        required = tuple(int(part or 0) for part in exact.groups())
        if image_tuple[:2] != required[:2] or required[2] and image_tuple[2] != required[2]:
            raise SetupIssue(f"declared {runtime} version {requirement} does not match configured image {image}")
        return True
    if runtime == "Node.js":
        match = re.fullmatch(r"(?:\^|~|>=?)?v?(\d+)(?:\.\d+(?:\.\d+)?)?", requirement)
        if match and int(match.group(1)) != image_tuple[0]:
            raise SetupIssue(f"declared Node.js range {requirement} does not match configured image {image}")
        return match is not None
    minimum = re.fullmatch(r">=\s*(\d+)\.(\d+)(?:\.(\d+))?", requirement)
    if minimum:
        required = tuple(int(part or 0) for part in minimum.groups())
        if image_tuple < required:
            raise SetupIssue(f"configured Python image {image} does not satisfy {requirement}")
        return True
    return False


def _test_counts(framework: str | None, output: str) -> tuple[int | None, int | None, int | None, str]:
    if framework == "pytest":
        matches = re.findall(r"(\d+)\s+(passed|failed|error|skipped|xfailed|xpassed)", output)
        if matches:
            counts = {name: 0 for name in ("passed", "failed", "error", "skipped", "xfailed", "xpassed")}
            for count, name in matches:
                counts[name] += int(count)
            total = sum(counts.values())
            return total, counts["passed"] + counts["xpassed"], counts["failed"] + counts["error"], "counted"
        if "no tests ran" in output.lower():
            return 0, 0, 0, "no_tests"
    if framework == "unittest":
        match = re.search(r"Ran\s+(\d+)\s+tests?", output)
        if match:
            total = int(match.group(1))
            failures = sum(int(value) for value in re.findall(r"\b(?:failures|errors)=([0-9]+)", output))
            return total, total - failures, failures, "counted"
    if framework in {"vitest", "jest"}:
        matches = re.findall(r"Tests?:?\s*(\d+)\s+(passed|failed|skipped|todo)", output, flags=re.IGNORECASE)
        if matches:
            counts = {"passed": 0, "failed": 0, "skipped": 0, "todo": 0}
            for count, name in matches:
                counts[name.lower()] += int(count)
            return sum(counts.values()), counts["passed"], counts["failed"], "counted"
    if framework == "mocha":
        passing = re.findall(r"(\d+)\s+passing", output, flags=re.IGNORECASE)
        failing = re.findall(r"(\d+)\s+failing", output, flags=re.IGNORECASE)
        if passing or failing:
            passed = sum(int(value) for value in passing)
            failed = sum(int(value) for value in failing)
            return passed + failed, passed, failed, "counted"
    if framework == "node --test":
        total = re.findall(r"(?:ℹ\s*)?tests\s+(\d+)", output, flags=re.IGNORECASE)
        passed = re.findall(r"(?:ℹ\s*)?pass\s+(\d+)", output, flags=re.IGNORECASE)
        failed = re.findall(r"(?:ℹ\s*)?fail\s+(\d+)", output, flags=re.IGNORECASE)
        if total and passed:
            return int(total[-1]), int(passed[-1]), int(failed[-1]) if failed else 0, "counted"
    if framework == "tap":
        total = re.findall(r"#\s*tests\s+(\d+)", output, flags=re.IGNORECASE)
        passed = re.findall(r"#\s*pass\s+(\d+)", output, flags=re.IGNORECASE)
        failed = re.findall(r"#\s*fail\s+(\d+)", output, flags=re.IGNORECASE)
        if total and passed:
            return int(total[-1]), int(passed[-1]), int(failed[-1]) if failed else 0, "counted"
    return None, None, None, "execution_count_unrecognized"


def _test_runner_missing(output: str) -> bool:
    lowered = output.lower()
    return (
        "no module named" in lowered
        or "cannot find module" in lowered
        or "command not found" in lowered
        or re.search(r"\b(?:pytest|vitest|jest|mocha)\b.*\bnot found\b", lowered) is not None
    )


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(f".{new_testing_run_id()}.tmp")
    with temporary.open("x", encoding="utf-8") as output:
        output.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_report(directory: Path, result: dict[str, object]) -> None:
    evidence = result.get("test_evidence", {})
    snapshot = result.get("source_snapshot")
    snapshot_hash = snapshot.get("sha256", "unavailable") if isinstance(snapshot, dict) else "unavailable"
    lines = [
        "# External testing report",
        "",
        f"- Outcome: **{result['outcome']}**",
        f"- Pipeline run: `{result['run_id']}`",
        f"- Testing run: `{result['testing_run_id']}`",
        f"- Tested source SHA-256: `{snapshot_hash}`",
        f"- Runtime: `{result.get('runtime', 'unavailable')}`",
        f"- Setup: `{result['environment_setup']['status']}` — {result['environment_setup']['details']}",
        "",
        "## Test evidence",
        "",
        f"- Status: `{evidence.get('status', 'not_run')}`",
        f"- Count: `{evidence.get('test_count', 'unknown')}`",
        f"- Passed: `{evidence.get('passed_count', 'unknown')}`",
        f"- Failed: `{evidence.get('failed_count', 'unknown')}`",
        "",
        "## Commands",
        "",
    ]
    commands = result.get("commands", [])
    if isinstance(commands, list) and commands:
        for command in commands:
            if not isinstance(command, dict):
                continue
            argv = " ".join(str(token) for token in command.get("argv", []))
            lines.append(f"- `{command.get('kind')}` `{argv}`: **{command.get('status')}** (exit {command.get('exit_code')}) — `{command.get('log_file')}`")
    else:
        lines.append("- No commands were run.")
    limitations = result.get("limitations", [])
    if isinstance(limitations, list) and limitations:
        lines.extend(["", "## Limitations", ""])
        lines.extend(f"- {item}" for item in limitations)
    lines.extend([
        "",
        "The result is separate from pipeline static-validation state. No pipeline manifest or handoff status was modified.",
        "",
    ])
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _command_record(command: CommandResult, log_file: str) -> dict[str, object]:
    return {
        "name": command.name,
        "kind": command.kind,
        "argv": list(command.argv),
        "status": command.status,
        "exit_code": command.exit_code,
        "duration_seconds": round(command.duration_seconds, 3),
        "timeout_seconds": command.timeout_seconds,
        "log_file": log_file,
    }


def _ensure_output_path(
    handoff: ValidatedHandoff,
    output_dir: str | os.PathLike[str] | None,
    testing_run_id: str,
) -> Path:
    is_default = output_dir is None
    requested = Path(output_dir) if output_dir is not None else handoff.run_dir / "external-testing" / f"{handoff.run_id}-{testing_run_id}"
    if requested.is_symlink():
        raise HandoffError("testing output directory cannot be a symbolic link")
    for parent in requested.parents:
        if parent == parent.parent:
            break
        if parent.is_symlink():
            raise HandoffError("testing output path cannot contain symbolic links")
    target = requested
    if not target.is_absolute():
        raise HandoffError("output directory must be absolute")
    target = target.resolve(strict=False)
    for source in handoff.source_roots:
        try:
            if os.path.commonpath((str(target), str(source.path))) == str(source.path):
                raise HandoffError("testing output directory must not be inside the source project")
        except ValueError as error:
            raise HandoffError("testing output directory is invalid") from error
    if target.exists() or target.is_symlink():
        raise HandoffError("testing output directory already exists")
    if is_default:
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if target.parent.is_symlink():
            raise HandoffError("testing output parent cannot be a symbolic link")
        target.parent.chmod(0o700)
    target.mkdir(mode=0o700, parents=True, exist_ok=False)
    return target


def _save_command_log(output_dir: Path, command: CommandResult) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", command.name).strip("-.") or "command"
    relative = f"logs/{len(list((output_dir / 'logs').glob('*.log'))) + 1:02d}-{name}.log"
    log_path = output_dir / relative
    log_path.parent.mkdir(exist_ok=True)
    content = command.output[:LOG_LIMIT_BYTES]
    if len(command.output.encode("utf-8", errors="replace")) > LOG_LIMIT_BYTES:
        content += "\n[log truncated at 64 KiB]\n"
    log_path.write_text(_redact(content), encoding="utf-8")
    return relative


def _run_command(
    docker: DockerBackend,
    container_id: str,
    name: str,
    kind: str,
    argv: tuple[str, ...],
    workdir: str,
    timeout: int,
    cancel_event: Event,
) -> CommandResult:
    status, exit_code, duration, output = docker.execute(
        container_id, argv, workdir, timeout, cancel_event
    )
    command = CommandResult(name, kind, argv, status, exit_code, duration, output, timeout)
    return command


def run_external_tests(
    handoff_path: str | os.PathLike[str],
    *,
    authorized_project_root: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    settings: AgentSettings | None = None,
    docker: DockerBackend | None = None,
    cancel_event: Event | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Execute supported tests in a disposable, network-restricted Docker container."""
    settings = settings or AgentSettings()
    settings.validate()
    cancel_event = cancel_event or Event()
    notify = progress or (lambda _message: None)
    handoff = validate_handoff(handoff_path, authorized_project_root)
    testing_run_id = new_testing_run_id()
    artifact_dir = _ensure_output_path(handoff, output_dir, testing_run_id)
    snapshot: Snapshot | None = None
    container_id: str | None = None
    started_at = _now()
    evidence = RunEvidence("setup_failed", "not_attempted", "Testing did not start.")
    plan: TestPlan | None = None
    backend = docker or DockerCli(settings)
    result: dict[str, object] = {
        "schema_version": 1,
        "testing_run_id": testing_run_id,
        "run_id": handoff.run_id,
        "project_mode": handoff.project_mode,
        "pipeline_status": handoff.pipeline_status,
        "outcome": "setup_failed",
        "started_at": started_at,
        "completed_at": None,
        "result_path": str(artifact_dir / "testing-result.json"),
        "runtime": None,
        "runtime_requirement": None,
        "package_manager_version": None,
        "source_snapshot": None,
        "environment_setup": {"status": "not_attempted", "details": "Testing did not start.", "docker_server_version": None},
        "commands": [],
        "test_evidence": {"status": "not_run", "test_count": None, "passed_count": None, "failed_count": None},
        "limitations": [],
        "container_cleanup": {"status": "not_required", "details": None},
    }
    try:
        notify("Creating a sanitized, immutable source snapshot.")
        snapshot = create_snapshot(handoff, artifact_dir / "source-snapshot")
        result["source_snapshot"] = {
            "path": "source-snapshot/",
            "sha256": snapshot.sha256,
            "file_count": snapshot.file_count,
            "byte_count": snapshot.byte_count,
            "omitted_entries": len(snapshot.omitted_entries),
        }
        if snapshot.omitted_entries:
            evidence.limitations.append(
                f"{len(snapshot.omitted_entries)} symlink, cache, sensitive-name, or non-regular entries were omitted from the snapshot."
            )
        if cancel_event.is_set():
            evidence.outcome = "cancelled"
            evidence.setup_status = "not_attempted"
            evidence.setup_details = "Cancelled before runtime discovery."
        else:
            plan = _make_plan(handoff, snapshot.path, settings)
            result["runtime"] = plan.runtime
            result["runtime_requirement"] = plan.runtime_requirement
            result["package_manager"] = plan.package_manager
            result["package_manager_version"] = plan.package_manager_version
            if plan.runtime_requirement and not plan.runtime_requirement_verified:
                evidence.limitations.append(
                    f"Declared {plan.runtime} runtime range {plan.runtime_requirement!r} could not be fully checked against {plan.image}."
                )
            if plan.test_command is None and plan.build_command is None:
                evidence.outcome = "no_tests"
                evidence.setup_status = "not_attempted"
                evidence.setup_details = "No recognized tests or supported build script were found; no container was started."
                evidence.test_evidence = "no_tests"
            else:
                notify(f"Starting disposable {plan.runtime} container for dependency setup.")
                docker_version = backend.check(settings.docker_timeout_seconds, cancel_event)
                result["environment_setup"]["docker_server_version"] = docker_version
                container_id = backend.start(plan.image, settings, cancel_event)
                backend.copy_into(container_id, snapshot.path, cancel_event)
                setup_success = True
                setup_messages: list[str] = []
                for name, argv, timeout in plan.setup_commands:
                    if cancel_event.is_set():
                        raise InterruptedError("cancelled")
                    notify(f"Installing declared dependencies using {plan.package_manager}.")
                    command = _run_command(
                        backend, container_id, name, "setup", argv,
                        plan.workdir, timeout, cancel_event,
                    )
                    if (
                        name == "verify-package-manager"
                        and plan.package_manager_version is not None
                        and command.status == "passed"
                        and command.output.strip() != plan.package_manager_version
                    ):
                        command = CommandResult(
                            command.name,
                            command.kind,
                            command.argv,
                            "failed",
                            command.exit_code,
                            command.duration_seconds,
                            f"Expected npm {plan.package_manager_version}; runtime provides {command.output.strip()}.",
                            command.timeout_seconds,
                        )
                    evidence.commands.append(command)
                    summary = f"{name}: {command.status}"
                    if command.status != "passed" and command.output:
                        summary += f" ({command.output.strip()[:256]})"
                    setup_messages.append(summary)
                    if command.status != "passed":
                        setup_success = False
                        break
                if not setup_success:
                    evidence.outcome = "cancelled" if cancel_event.is_set() else "setup_failed"
                    evidence.setup_status = "failed" if not cancel_event.is_set() else "not_attempted"
                    evidence.setup_details = "; ".join(setup_messages) or "Dependency setup was interrupted."
                else:
                    evidence.setup_status = "succeeded"
                    evidence.setup_details = "; ".join(setup_messages) if setup_messages else "No third-party dependency installation was required."
                    if cancel_event.is_set():
                        evidence.outcome = "cancelled"
                    else:
                        notify("Dependency setup finished; disabling container networking before checks.")
                        backend.disconnect_network(container_id, cancel_event)
                        if plan.build_command is not None:
                            build = _run_command(
                                backend, container_id, "build", "build",
                                plan.build_command, plan.build_workdir,
                                settings.test_timeout_seconds, cancel_event,
                            )
                            evidence.commands.append(build)
                        if plan.test_command is None:
                            evidence.outcome = "no_tests"
                            evidence.test_evidence = "no_tests"
                            evidence.limitations.append("A supported build command ran, but no test files and test command were recognized; build success is not a test pass.")
                        elif cancel_event.is_set():
                            evidence.outcome = "cancelled"
                        else:
                            notify("Running the declared test entry point with container networking disabled.")
                            test_command = _run_command(
                                backend, container_id, "tests", "test",
                                plan.test_command, plan.workdir,
                                settings.test_timeout_seconds, cancel_event,
                            )
                            evidence.commands.append(test_command)
                            count, passed, failed, count_status = _test_counts(
                                plan.test_framework, test_command.output
                            )
                            evidence.test_count = count
                            evidence.passed_count = passed
                            evidence.failed_count = failed
                            evidence.test_evidence = count_status
                            if test_command.status == "cancelled" or cancel_event.is_set():
                                evidence.outcome = "cancelled"
                            elif test_command.status == "failed" and _test_runner_missing(test_command.output):
                                evidence.outcome = "setup_failed"
                                evidence.setup_status = "failed"
                                evidence.setup_details = "Test command could not start because a module or executable is missing. Check declared test dependencies and lockfiles."
                                evidence.test_evidence = "runner_unavailable"
                            elif test_command.status in {"failed", "timed_out"} or (failed or 0) > 0:
                                evidence.outcome = "tests_failed"
                            elif test_command.status == "passed" and (passed or 0) > 0:
                                evidence.outcome = "passed"
                                evidence.test_evidence = "verified"
                            else:
                                evidence.outcome = "no_tests"
                                evidence.test_evidence = "execution_count_unrecognized"
                                evidence.limitations.append(
                                    "The test command exited successfully, but no supported runner reported a positive executed-test count; pass was not asserted."
                                )
                        if any(command.kind == "build" and command.status != "passed" for command in evidence.commands):
                            evidence.limitations.append("The build command did not pass; its status is reported separately from test evidence.")
                            if evidence.outcome == "passed":
                                evidence.outcome = "tests_failed"
    except InterruptedError:
        evidence.outcome = "cancelled"
        evidence.setup_status = "not_attempted" if evidence.setup_status == "not_attempted" else evidence.setup_status
        evidence.setup_details = "Testing was cancelled; the active container was stopped."
    except (HandoffError, SetupIssue, DockerFailure) as error:
        evidence.outcome = "setup_failed"
        evidence.setup_status = "failed"
        evidence.setup_details = _redact(str(error))
        notify(f"External testing setup failed: {evidence.setup_details}")
    except Exception as error:  # Keep a structured failure artifact for unexpected runner/infra errors.
        evidence.outcome = "setup_failed"
        evidence.setup_status = "failed"
        evidence.setup_details = f"External runner infrastructure error: {_redact(str(error))}"
        notify(evidence.setup_details)
    finally:
        if container_id is not None:
            try:
                cleanup_error = backend.remove(container_id)
            except Exception as error:
                cleanup_error = _redact(str(error))
            if cleanup_error:
                evidence.outcome = "setup_failed"
                evidence.setup_status = "failed"
                evidence.setup_details = f"Disposable container cleanup failed: {cleanup_error}"
                evidence.limitations.append("The disposable container could not be confirmed removed; inspect Docker manually.")
                result["container_cleanup"] = {"status": "failed", "details": cleanup_error}
            else:
                result["container_cleanup"] = {"status": "removed", "details": None}

    evidence.limitations = list(dict.fromkeys(evidence.limitations))
    result["outcome"] = evidence.outcome
    result["runtime"] = plan.runtime if plan else result["runtime"]
    result["runtime_requirement"] = plan.runtime_requirement if plan else result["runtime_requirement"]
    result["package_manager_version"] = plan.package_manager_version if plan else result.get("package_manager_version")
    result["environment_setup"] = {
        **result["environment_setup"],
        "status": evidence.setup_status,
        "details": evidence.setup_details,
        "image": plan.image if plan else None,
        "package_manager": plan.package_manager if plan else None,
            "network_policy": "bridge only during dependency setup; disconnected before build/test commands" if container_id else "not_started",
    }
    result["commands"] = []
    for command in evidence.commands:
        log_name = _save_command_log(artifact_dir, command)
        result["commands"].append(_command_record(command, log_name))
    result["test_evidence"] = {
        "status": evidence.test_evidence,
        "framework": plan.test_framework if plan else None,
        "test_files_detected": len(plan.test_files) if plan else 0,
        "test_count": evidence.test_count,
        "passed_count": evidence.passed_count,
        "failed_count": evidence.failed_count,
    }
    result["limitations"] = evidence.limitations
    result["completed_at"] = _now()
    _write_json_exclusive(artifact_dir / "testing-result.json", result)
    _write_report(artifact_dir, result)
    notify(f"External testing outcome: {evidence.outcome}. Result: {artifact_dir / 'testing-result.json'}")
    return result
