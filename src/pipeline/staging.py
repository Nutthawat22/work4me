"""Per-work-item source staging and best-effort publication."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import asdict, dataclass
from typing import Any

from pipeline.project_context import IGNORED_DIRECTORIES, snapshot_project_files
from pipeline.state import OwnedFile, WorkItem
from pipeline.task_validation import normalize_relative_path

STAGE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StageLayout:
    identity: str
    directory: str
    manifest_path: str
    files_directory: str


def planned_output_path(owned_file: OwnedFile, config: dict[str, Any]) -> str:
    pipeline = config["pipeline"]
    existing_project = bool(pipeline.get("_existing_project_path"))
    area_key = "tests_dir" if owned_file.area == "tests" else "output_dir"
    area = normalize_relative_path(pipeline[area_key])
    path = normalize_relative_path(owned_file.path)
    if existing_project:
        return path
    if path.startswith(area + "/"):
        path = path[len(area) + 1 :]
    return f"{area}/{path}"


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _task_snapshot(work_item: WorkItem) -> dict[str, object]:
    return {
        "id": work_item.id,
        "category": work_item.category,
        "language": work_item.language,
        "module": work_item.module,
        "scope": work_item.scope,
        "title": work_item.title,
        "description": work_item.description,
        "acceptance_criteria": work_item.acceptance_criteria,
        "requirement_refs": work_item.requirement_refs,
        "owned_files": [asdict(item) for item in work_item.owned_files],
        "depends_on": work_item.depends_on,
        "expected_interfaces": work_item.expected_interfaces,
    }


def _existing_source_snapshot(work_item: WorkItem, config: dict[str, Any]) -> str:
    pipeline = config["pipeline"]
    baseline = pipeline.get("_existing_project_baseline")
    if isinstance(baseline, dict):
        return _digest(baseline)
    root_value = pipeline.get("_existing_project_path")
    if not root_value:
        return _digest({})
    root = os.path.realpath(root_value)
    files: dict[str, str] = {}
    for owned_file in work_item.owned_files:
        relative = normalize_relative_path(owned_file.path)
        path = os.path.join(root, relative)
        if os.path.isfile(path) and not os.path.islink(path):
            with open(path, "rb") as source:
                files[relative] = hashlib.sha256(source.read()).hexdigest()
        else:
            files[relative] = "<missing>"
    return _digest(files)


def _owned_source_files(work_item: WorkItem, config: dict[str, Any]) -> dict[str, str]:
    root_value = config["pipeline"].get("_existing_project_path")
    if not root_value:
        return {}
    root = os.path.realpath(root_value)
    files: dict[str, str] = {}
    for owned_file in work_item.owned_files:
        relative = normalize_relative_path(owned_file.path)
        path = os.path.join(root, relative)
        if os.path.isfile(path) and not os.path.islink(path):
            with open(path, "rb") as source:
                files[relative] = hashlib.sha256(source.read()).hexdigest()
        else:
            files[relative] = "<missing>"
    return files


def _current_project_snapshot(
    run_dir: str,
    config: dict[str, Any],
) -> tuple[dict[str, str], dict[str, str]] | None:
    pipeline = config["pipeline"]
    root_value = pipeline.get("_existing_project_path")
    baseline = pipeline.get("_existing_project_baseline")
    if not root_value or not isinstance(baseline, dict):
        return None
    expected = dict(baseline)
    staging_root = os.path.join(os.path.realpath(run_dir), "staging")
    if os.path.isdir(staging_root):
        for current, directories, filenames in os.walk(staging_root, followlinks=False):
            directories[:] = [name for name in directories if not os.path.islink(os.path.join(current, name))]
            if "stage.json" not in filenames:
                continue
            try:
                with open(os.path.join(current, "stage.json"), "r", encoding="utf-8") as source:
                    stage = json.load(source)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(stage, dict) or stage.get("status") != "published":
                continue
            targets = stage.get("published_target_digests")
            if not isinstance(targets, dict):
                continue
            for path, digest in targets.items():
                if not isinstance(path, str) or not isinstance(digest, str):
                    continue
                try:
                    relative = normalize_relative_path(path)
                except ValueError:
                    continue
                if digest == "<missing>":
                    expected.pop(relative, None)
                else:
                    expected[relative] = digest
    protected = pipeline.get("_protected_run_root")
    excluded = (protected,) if isinstance(protected, str) and protected else ()
    return snapshot_project_files(os.path.realpath(root_value), excluded), expected


def owned_source_matches(
    layout: StageLayout,
    work_item: WorkItem,
    config: dict[str, Any],
    run_dir: str,
) -> bool:
    if not config["pipeline"].get("_existing_project_path"):
        return True
    manifest = read_stage(layout)
    live = _owned_source_files(work_item, config)
    project_snapshot = _current_project_snapshot(run_dir, config)
    return (project_snapshot is None or project_snapshot[0] == project_snapshot[1]) and live in (
        manifest.get("source_owned_snapshot"),
        manifest.get("published_target_digests"),
    )


def open_stage(work_item: WorkItem, config: dict[str, Any], run_dir: str) -> StageLayout:
    pipeline = config["pipeline"]
    common_identity_data = {
        "schema_version": STAGE_SCHEMA_VERSION,
        "plan_digest": pipeline.get("_stage_plan_digest", ""),
        "task": _task_snapshot(work_item),
        "source_snapshot": _existing_source_snapshot(work_item, config),
        "output_dir": pipeline.get("output_dir"),
        "tests_dir": pipeline.get("tests_dir"),
        "existing_project": bool(pipeline.get("_existing_project_path")),
    }
    common_identity = _digest(common_identity_data)
    live_owned_sources = _owned_source_files(work_item, config)
    live_snapshot_data = _current_project_snapshot(run_dir, config)
    live_snapshot_digest = _digest(live_snapshot_data[0]) if live_snapshot_data else ""
    project_source_matches = (
        live_snapshot_data is None or live_snapshot_data[0] == live_snapshot_data[1]
    )
    task_key = hashlib.sha256(work_item.id.encode("utf-8")).hexdigest()[:24]
    directory = os.path.join(os.path.realpath(run_dir), "staging", task_key)
    manifest_path = os.path.join(directory, "stage.json")
    files_directory = os.path.join(directory, "files")

    previous: dict[str, Any] | None = None
    try:
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            loaded = json.load(manifest_file)
            previous = loaded if isinstance(loaded, dict) else None
    except (OSError, json.JSONDecodeError):
        previous = None
    can_reuse = (
        previous is not None
        and previous.get("common_identity") == common_identity
        and project_source_matches
        and live_owned_sources in (
            previous.get("source_owned_snapshot"),
            previous.get("published_target_digests"),
        )
    )
    identity = (
        str(previous.get("identity"))
        if can_reuse and previous is not None
        else _digest({
            "common_identity": common_identity,
            "owned_source_snapshot": live_owned_sources,
            "live_project_snapshot": live_snapshot_digest,
        })
    )
    if not can_reuse:
        if os.path.lexists(directory):
            shutil.rmtree(directory)
        os.makedirs(files_directory, exist_ok=True)
        _write_manifest(
            manifest_path,
            {
                "schema_version": STAGE_SCHEMA_VERSION,
                "identity": identity,
                "common_identity": common_identity,
                "task_id": work_item.id,
                "plan_digest": common_identity_data["plan_digest"],
                "source_snapshot_digest": common_identity_data["source_snapshot"],
                "live_project_snapshot_digest": live_snapshot_digest,
                "source_owned_snapshot": live_owned_sources,
                "published_target_digests": {},
                "files": {},
                "consistency_feedback": "",
            },
        )
    else:
        os.makedirs(files_directory, exist_ok=True)
    return StageLayout(identity, directory, manifest_path, files_directory)


def read_stage(layout: StageLayout) -> dict[str, Any]:
    try:
        with open(layout.manifest_path, "r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, json.JSONDecodeError):
        return {"schema_version": STAGE_SCHEMA_VERSION, "identity": layout.identity, "files": {}}
    if not isinstance(manifest, dict) or manifest.get("identity") != layout.identity:
        return {"schema_version": STAGE_SCHEMA_VERSION, "identity": layout.identity, "files": {}}
    return manifest


def write_stage_content(layout: StageLayout, relative_path: str, content: str) -> str:
    relative = normalize_relative_path(relative_path)
    target = os.path.realpath(os.path.join(layout.files_directory, relative))
    files_root = os.path.realpath(layout.files_directory)
    if os.path.commonpath((files_root, target)) != files_root:
        raise ValueError(f"staged path escapes feature workspace: {relative}")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    temporary = target + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="") as staged_file:
        staged_file.write(content)
    os.replace(temporary, target)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def stage_file_content(layout: StageLayout, relative_path: str) -> str:
    relative = normalize_relative_path(relative_path)
    path = os.path.join(layout.files_directory, relative)
    with open(path, "r", encoding="utf-8", newline="") as staged_file:
        return staged_file.read()


def persist_stage(layout: StageLayout, manifest: dict[str, Any]) -> None:
    _write_manifest(layout.manifest_path, manifest)


def _write_manifest(path: str, manifest: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")
    os.replace(temporary, path)


def publish_stage(
    layout: StageLayout,
    config: dict[str, Any],
    run_dir: str,
    file_actions: list[tuple[str, str]],
) -> tuple[list[str], list[str], str]:
    """Publish all staged writes/deletes with per-file backups and rollback.

    Each file replacement is atomic on its filesystem. The feature-wide set is
    not atomic; on failure the function best-effort restores each touched path
    and returns any paths whose rollback failed.
    """
    pipeline = config["pipeline"]
    project_root_value = pipeline.get("_existing_project_path")
    project_mode = bool(project_root_value)
    root = os.path.realpath(project_root_value if project_mode else run_dir)
    changed: list[tuple[str, str, bytes | None, int | None]] = []
    written: list[str] = []
    deleted: list[str] = []

    def checked_target(relative: str) -> str:
        normalized = normalize_relative_path(relative)
        target = os.path.join(root, normalized)
        resolved = os.path.realpath(target)
        if os.path.commonpath((root, resolved)) != root:
            raise OSError(f"publish path escapes its root: {normalized}")
        current = root
        for component in normalized.split("/"):
            current = os.path.join(current, component)
            if os.path.islink(current):
                raise OSError(f"publish path contains a symbolic link: {normalized}")
        if project_mode:
            protected = pipeline.get("_protected_run_root")
            if protected and os.path.commonpath((os.path.realpath(protected), resolved)) == os.path.realpath(protected):
                raise OSError(f"publish path is reserved for run artifacts: {normalized}")
            if any(part.casefold() in IGNORED_DIRECTORIES for part in normalized.split("/")):
                raise OSError(f"publish path is inside an excluded directory: {normalized}")
        else:
            area = normalize_relative_path(pipeline["tests_dir"] if normalized.startswith(
                normalize_relative_path(pipeline["tests_dir"]) + "/"
            ) else pipeline["output_dir"])
            area_root = os.path.realpath(os.path.join(root, area))
            if os.path.commonpath((area_root, resolved)) != area_root:
                raise OSError(f"publish path escapes its configured area: {normalized}")
        return target

    try:
        for action, relative in file_actions:
            target = checked_target(relative)
            if os.path.lexists(target) and not os.path.isfile(target):
                raise OSError(f"publish target is not a regular file: {relative}")
            previous: bytes | None = None
            mode: int | None = None
            if os.path.isfile(target):
                with open(target, "rb") as original:
                    previous = original.read()
                mode = os.stat(target, follow_symlinks=False).st_mode
            changed.append((relative, target, previous, mode))
            if action == "delete":
                if os.path.lexists(target):
                    os.remove(target)
                deleted.append(relative)
                continue
            if action != "write":
                raise ValueError(f"unknown publish action: {action}")
            content = stage_file_content(layout, relative)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            target_parent = os.path.realpath(os.path.dirname(target))
            if os.path.commonpath((root, target_parent)) != root:
                raise OSError(f"publish parent escapes its root: {relative}")
            descriptor, temporary = tempfile.mkstemp(prefix=".stage-", dir=os.path.dirname(target))
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
                    output.write(content)
                os.chmod(temporary, stat.S_IMODE(mode) if mode is not None else 0o644)
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.remove(temporary)
            written.append(relative)
    except (OSError, ValueError) as error:
        rollback_failures: list[str] = []
        for relative, target, previous, mode in reversed(changed):
            try:
                if previous is None:
                    if os.path.lexists(target):
                        os.remove(target)
                else:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    descriptor, temporary = tempfile.mkstemp(prefix=".rollback-", dir=os.path.dirname(target))
                    try:
                        with os.fdopen(descriptor, "wb") as output:
                            output.write(previous)
                        if mode is not None:
                            os.chmod(temporary, stat.S_IMODE(mode))
                        os.replace(temporary, target)
                    finally:
                        if os.path.exists(temporary):
                            os.remove(temporary)
            except OSError:
                rollback_failures.append(relative)
        detail = f"Feature publication failed; best-effort rollback attempted: {error}"
        if rollback_failures:
            detail += "; rollback incomplete for: " + ", ".join(rollback_failures)
        return [], [], detail
    return written, deleted, ""
