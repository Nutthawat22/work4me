"""Validation and source snapshot handling for testing handoffs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4


SCHEMA_VERSION = 1
MAX_HANDOFF_BYTES = 4 * 1024 * 1024
MAX_SOURCE_FILES = 20_000
MAX_SOURCE_BYTES = 1024 * 1024 * 1024
MAX_SOURCE_FILE_BYTES = 64 * 1024 * 1024
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
IGNORED_DIRECTORIES = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
    ".next", ".nuxt", ".turbo", "coverage", "dist", "build", "target",
    ".idea", ".vscode", ".cache", ".ssh", ".aws", ".azure", ".kube", "vendor",
}
SAFE_ENV_TEMPLATES = {".env.example", ".env.sample", ".env.template"}
SENSITIVE_BASENAMES = {
    ".netrc", ".npmrc", ".pypirc", ".git-credentials", "id_rsa", "id_ed25519",
    "credentials.json", "service-account.json", "dockerconfig.json",
}
SENSITIVE_NAME = re.compile(
    r"(?i)(?:^\.env(?:\.|$)|(?:^|[-_.])(?:secret|credentials?|private[-_.]?key)(?:[-_.]|$)|"
    r"\.(?:pem|key|p12|pfx|keystore)$)"
)


class HandoffError(ValueError):
    """The handoff or an authorized input path is invalid."""


@dataclass(frozen=True)
class SourceRoot:
    label: str
    path: Path


@dataclass(frozen=True)
class ValidatedHandoff:
    handoff_path: Path
    run_dir: Path
    run_id: str
    pipeline_status: str
    project_mode: str
    source_roots: tuple[SourceRoot, ...]
    data: dict[str, Any]


@dataclass(frozen=True)
class Snapshot:
    path: Path
    sha256: str
    file_count: int
    byte_count: int
    omitted_entries: tuple[str, ...]


def _canonical_directory(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value or not os.path.isabs(value):
        raise HandoffError(f"{field} must be an absolute path")
    path = Path(value)
    try:
        if path.is_symlink() or not path.is_dir():
            raise HandoffError(f"{field} must be an existing non-symlink directory")
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise HandoffError(f"{field} cannot be accessed: {error}") from error
    return resolved


def validate_handoff(
    handoff_path: str | os.PathLike[str],
    authorized_project_root: str | os.PathLike[str] | None = None,
) -> ValidatedHandoff:
    requested = Path(handoff_path)
    if not requested.is_absolute():
        raise HandoffError("handoff path must be absolute")
    if requested.name != "testing-handoff.json" or requested.is_symlink():
        raise HandoffError("expected a regular testing-handoff.json file")
    try:
        handoff = requested.resolve(strict=True)
        if handoff.stat().st_size > MAX_HANDOFF_BYTES or not handoff.is_file():
            raise HandoffError("handoff must be a regular file no larger than 4 MiB")
        data = json.loads(handoff.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HandoffError(f"handoff cannot be read as JSON: {error}") from error
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise HandoffError("unsupported testing handoff schema")
    run_id = data.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise HandoffError("handoff run_id is missing or invalid")
    external = data.get("external_testing")
    if not isinstance(external, dict) or external.get("status") != "pending":
        raise HandoffError("handoff external-testing status is not pending")
    pipeline_status = data.get("pipeline_status")
    if pipeline_status != "validation_passed_external_testing_pending":
        raise HandoffError("pipeline static validation did not pass; external testing was not started")
    paths = data.get("paths")
    if not isinstance(paths, dict):
        raise HandoffError("handoff paths must be an object")
    run_dir = _canonical_directory(paths.get("run_dir"), "paths.run_dir")
    if handoff.parent.resolve(strict=True) != run_dir:
        raise HandoffError("handoff must be stored directly in its declared run directory")
    try:
        manifest_path = run_dir / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise HandoffError("run manifest must be a regular file inside the run directory")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HandoffError(f"run manifest cannot be read: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("run_id") != run_id:
        raise HandoffError("handoff run_id does not match the run manifest")
    if manifest.get("status") != pipeline_status:
        raise HandoffError("handoff pipeline status does not match the run manifest")

    mode = data.get("project_mode")
    roots: list[SourceRoot] = []
    if mode == "new_project":
        if authorized_project_root is not None:
            raise HandoffError("--project-root is only valid for existing-project handoffs")
        product = _canonical_directory(paths.get("product_dir"), "paths.product_dir")
        tests = _canonical_directory(paths.get("tests_dir"), "paths.tests_dir")
        if product == tests or product in tests.parents or tests in product.parents:
            raise HandoffError("new-project product_dir and tests_dir must be separate roots")
        for label, source in (("product", product), ("tests", tests)):
            if source != run_dir and run_dir not in source.parents:
                raise HandoffError(f"paths.{label}_dir is outside the declared run directory")
            roots.append(SourceRoot(label, source))
        project_root = _canonical_directory(paths.get("project_root"), "paths.project_root")
        if project_root != product:
            raise HandoffError("new-project project_root must match product_dir")
    elif mode == "existing_project":
        if authorized_project_root is None:
            raise HandoffError("existing-project handoffs require explicit --project-root authorization")
        requested_root = Path(authorized_project_root)
        if not requested_root.is_absolute():
            raise HandoffError("--project-root must be absolute")
        authorized = _canonical_directory(str(requested_root), "--project-root")
        project_root = _canonical_directory(paths.get("project_root"), "paths.project_root")
        product = _canonical_directory(paths.get("product_dir"), "paths.product_dir")
        tests = _canonical_directory(paths.get("tests_dir"), "paths.tests_dir")
        if authorized != project_root or product != project_root or tests != project_root:
            raise HandoffError("--project-root must exactly match the handoff's existing project root")
        changes_path = paths.get("changes_json")
        if not isinstance(changes_path, str) or Path(changes_path).resolve() != run_dir / "changes.json":
            raise HandoffError("existing-project handoff changes_json is outside its run directory")
        try:
            changes_file = run_dir / "changes.json"
            if changes_file.is_symlink() or not changes_file.is_file():
                raise HandoffError("changes.json must be a regular file inside the run directory")
            changes = json.loads(changes_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HandoffError(f"existing-project change record cannot be read: {error}") from error
        if not isinstance(changes, dict) or Path(str(changes.get("project_path", ""))).resolve() != project_root:
            raise HandoffError("authorized project root does not match changes.json")
        roots.append(SourceRoot("project", project_root))
    else:
        raise HandoffError("project_mode must be new_project or existing_project")

    for root in roots:
        if root.path.is_symlink() or not root.path.is_dir():
            raise HandoffError(f"source root {root.label!r} is unavailable or unsafe")
    return ValidatedHandoff(
        handoff_path=handoff,
        run_dir=run_dir,
        run_id=run_id,
        pipeline_status=pipeline_status,
        project_mode=mode,
        source_roots=tuple(roots),
        data=data,
    )


def _excluded(name: str, is_directory: bool) -> bool:
    if is_directory and name in IGNORED_DIRECTORIES:
        return True
    if name.lower() in SENSITIVE_BASENAMES:
        return True
    return name not in SAFE_ENV_TEMPLATES and bool(SENSITIVE_NAME.search(name))


def create_snapshot(handoff: ValidatedHandoff, destination: Path) -> Snapshot:
    """Copy approved source roots, omitting links, caches, and sensitive files."""
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    omitted: list[str] = []
    file_count = 0
    byte_count = 0
    entry_count = 0

    try:
        for source_root in handoff.source_roots:
            target_root = destination / source_root.label
            target_root.mkdir(mode=0o700)
            pending: list[tuple[Path, Path]] = [(source_root.path, target_root)]
            while pending:
                source_dir, target_dir = pending.pop()
                with os.scandir(source_dir) as entries:
                    for entry in sorted(entries, key=lambda item: item.name):
                        entry_count += 1
                        if entry_count > MAX_SOURCE_FILES * 2:
                            raise HandoffError("source exceeds the snapshot entry limit")
                        source = Path(entry.path)
                        relative = source.relative_to(source_root.path).as_posix()
                        is_link = entry.is_symlink()
                        if is_link:
                            omitted.append(f"{source_root.label}/{relative}: symbolic link")
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            if _excluded(entry.name, True):
                                omitted.append(f"{source_root.label}/{relative}: excluded directory")
                                continue
                            target = target_dir / entry.name
                            target.mkdir(mode=0o700)
                            pending.append((source, target))
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            omitted.append(f"{source_root.label}/{relative}: non-regular file")
                            continue
                        if _excluded(entry.name, False):
                            omitted.append(f"{source_root.label}/{relative}: sensitive filename")
                            continue
                        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                        try:
                            source_fd = os.open(source, flags)
                            with os.fdopen(source_fd, "rb") as source_file:
                                metadata = os.fstat(source_file.fileno())
                                if not stat.S_ISREG(metadata.st_mode):
                                    omitted.append(f"{source_root.label}/{relative}: changed during snapshot")
                                    continue
                                if metadata.st_size > MAX_SOURCE_FILE_BYTES:
                                    raise HandoffError(f"source file exceeds 64 MiB snapshot limit: {relative}")
                                file_count += 1
                                if file_count > MAX_SOURCE_FILES:
                                    raise HandoffError("source exceeds the snapshot file or size limit")
                                target = target_dir / entry.name
                                with target.open("xb") as target_file:
                                    while chunk := source_file.read(1024 * 1024):
                                        byte_count += len(chunk)
                                        if byte_count > MAX_SOURCE_BYTES:
                                            raise HandoffError("source exceeds the snapshot file or size limit")
                                        target_file.write(chunk)
                                target.chmod(0o555 if metadata.st_mode & 0o111 else 0o444)
                        except OSError as error:
                            raise HandoffError(f"source changed or could not be snapshotted: {relative}: {error}") from error

        directories = sorted(
            (path for path in destination.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for directory in directories:
            directory.chmod(0o555)
        destination.chmod(0o555)
        digest = hashlib.sha256()
        for file_path in sorted(path for path in destination.rglob("*") if path.is_file()):
            relative = file_path.relative_to(destination).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            with file_path.open("rb") as source_file:
                while chunk := source_file.read(1024 * 1024):
                    digest.update(len(chunk).to_bytes(8, "big"))
                    digest.update(chunk)
        return Snapshot(destination, digest.hexdigest(), file_count, byte_count, tuple(omitted))
    except Exception:
        for path in destination.rglob("*"):
            try:
                path.chmod(0o700 if path.is_dir() else 0o600)
            except OSError:
                pass
        try:
            destination.chmod(0o700)
            shutil.rmtree(destination)
        except OSError:
            pass
        raise


def new_testing_run_id() -> str:
    return uuid4().hex
