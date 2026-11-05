"""Preflight and provenance for safe, selective repair child runs."""

from __future__ import annotations

import dataclasses
import contextlib
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:
    fcntl = None

from pipeline.master import MasterAgent, MasterPlanError
from pipeline.project_context import IGNORED_DIRECTORIES, snapshot_project_files
from pipeline.project_paths import resolve_project_path
from pipeline.state import AgentResult, FileAttempt, OwnedFile, TaskAttempt, ValidationIssue, WorkItem
from pipeline.task_validation import normalize_relative_path

PIPELINE_SCHEMA_VERSION = 3
TYPESCRIPT_VERSION = "6.0.3"
MAX_PLAN_BYTES = 4 * 1024 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_RESUME_FILE_BYTES = 8 * 1024 * 1024
MAX_RESUME_SOURCE_BYTES = 128 * 1024 * 1024


class ResumeError(ValueError):
    """A source run is not safe or compatible to resume."""


@contextlib.contextmanager
def resume_source_lock(config: dict[str, Any], source_run_dir: str):
    runs_root = os.path.realpath(resolve_project_path(config["pipeline"]["runs_dir"]))
    lock_root = os.path.join(runs_root, ".resume-locks")
    if os.path.islink(lock_root):
        raise ResumeError("Resume lock directory is unsafe")
    os.makedirs(lock_root, mode=0o700, exist_ok=True)
    lock_info = os.stat(lock_root, follow_symlinks=False)
    owner = getattr(os, "getuid", lambda: None)()
    if not stat.S_ISDIR(lock_info.st_mode) or (owner is not None and lock_info.st_uid != owner):
        raise ResumeError("Resume lock directory is not owned by this pipeline user")
    os.chmod(lock_root, 0o700)
    lock_name = hashlib.sha256(os.path.realpath(source_run_dir).encode()).hexdigest() + ".lock"
    lock_path = os.path.join(lock_root, lock_name)
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or (owner is not None and info.st_uid != owner):
            raise ResumeError("Resume source lock is not a regular file owned by this pipeline user")
        os.fchmod(descriptor, 0o600)
        if fcntl is None:
            raise ResumeError("This platform does not provide the required resume concurrency lock")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ResumeError("Another resume is already active for this source run") from error
        yield
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@dataclass
class ResumeRequest:
    source_run_dir: str
    source_run_id: str
    work_items: list[WorkItem]
    requested_task_ids: list[str]
    selected_task_ids: list[str]
    project_root: str | None
    published_files: dict[str, str]
    successful_results: list[AgentResult]
    unselected_results: list[AgentResult]
    retry_context: dict[str, list[ValidationIssue]]
    stage_manifests: dict[str, dict[str, Any]]
    provenance: dict[str, Any]


def compatibility_digest(config: dict[str, Any], project_root: str | None = None) -> str:
    pipeline = config.get("pipeline", {})
    payload = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "typescript_version": TYPESCRIPT_VERSION,
        "output_dir": pipeline.get("output_dir"),
        "tests_dir": pipeline.get("tests_dir"),
        "max_retries": pipeline.get("max_retries"),
        "languages": config.get("languages", {}),
        "specialist_model": config.get("models", {}).get("specialist", {}),
        "project_mode": bool(project_root),
        "project_path": os.path.realpath(project_root) if project_root else None,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_json(path: str, limit: int) -> dict[str, Any]:
    if os.path.islink(path) or not os.path.isfile(path):
        raise ResumeError(f"Resume artifact is missing or unsafe: {os.path.basename(path)}")
    if os.path.getsize(path) > limit:
        raise ResumeError(f"Resume artifact exceeds its size limit: {os.path.basename(path)}")
    try:
        with open(path, "r", encoding="utf-8") as source:
            result = json.load(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResumeError(f"Resume artifact is unreadable: {os.path.basename(path)}") from error
    if not isinstance(result, dict):
        raise ResumeError(f"Resume artifact has an invalid shape: {os.path.basename(path)}")
    return result


def _task_plan_digest(work_items: list[WorkItem]) -> str:
    identity_fields = (
        "id", "category", "language", "module", "scope", "title", "description",
        "acceptance_criteria", "requirement_refs", "owned_files", "depends_on",
        "expected_interfaces", "design_contracts",
    )
    stable_plan = [
        {
            **{field: getattr(item, field) for field in identity_fields},
            **({"design_contract_inventory": item.design_contract_inventory}
               if item.design_contract_inventory else {}),
        }
        for item in work_items
    ]
    encoded = json.dumps(
        stable_plan, sort_keys=True, default=dataclasses.asdict, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _existing_sources(project_root: str, runs_root: str) -> dict[str, str]:
    return snapshot_project_files(project_root, (runs_root,))


def _generated_sources(run_dir: str, config: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    root = os.path.realpath(run_dir)
    for key in ("output_dir", "tests_dir"):
        area = normalize_relative_path(config["pipeline"][key])
        area_root = os.path.join(root, area)
        if not os.path.isdir(area_root):
            continue
        for current, directories, filenames in os.walk(area_root, followlinks=False):
            directories[:] = sorted(
                name for name in directories
                if name not in IGNORED_DIRECTORIES
                and not os.path.islink(os.path.join(current, name))
            )
            for name in sorted(filenames):
                path = os.path.join(current, name)
                if os.path.islink(path) or not os.path.isfile(path):
                    raise ResumeError("Generated source inventory contains an unsafe non-regular file")
                relative = os.path.relpath(path, root).replace(os.sep, "/")
                digest = hashlib.sha256()
                with open(path, "rb") as source:
                    for chunk in iter(lambda: source.read(65536), b""):
                        digest.update(chunk)
                result[relative] = digest.hexdigest()
    return result


def _resolve_source_run(config: dict[str, Any], run_reference: str) -> tuple[str, str]:
    runs_root = os.path.realpath(resolve_project_path(config["pipeline"]["runs_dir"]))
    candidate = run_reference if os.path.isabs(run_reference) else os.path.join(runs_root, run_reference)
    if os.path.islink(candidate):
        raise ResumeError("Resume source run directory may not be a symbolic link")
    resolved = os.path.realpath(candidate)
    if os.path.commonpath((runs_root, resolved)) != runs_root or resolved == runs_root:
        raise ResumeError("Resume source run must be inside configured runs storage")
    if not os.path.isdir(resolved):
        raise ResumeError("Resume source run directory does not exist")
    return runs_root, resolved


def resolve_resume_source(config: dict[str, Any], run_reference: str) -> str:
    return _resolve_source_run(config, run_reference)[1]


def _work_items(plan: dict[str, Any], config: dict[str, Any], project_root: str | None) -> list[WorkItem]:
    if plan.get("format_version") != 2 or not isinstance(plan.get("tasks"), list):
        raise ResumeError("Legacy or unsupported task-plan schema; an authorized plan amendment is required before resume")
    validation_config = dict(config)
    validation_config["pipeline"] = dict(config.get("pipeline", {}))
    if project_root:
        validation_config["pipeline"]["_existing_project_path"] = project_root
    try:
        items = MasterAgent._validate_and_build(plan["tasks"], set(config["languages"]))
        for item, encoded in zip(items, plan["tasks"], strict=True):
            item.design_contracts = encoded.get("design_contracts", [])
            item.design_contract_inventory = encoded.get("design_contract_inventory", [])
        MasterAgent(validation_config)._validate_task_plan(items)
        for item, encoded in zip(items, plan["tasks"], strict=True):
            if encoded.get("status") not in {"blocked", "ready", "running", "passed", "failed"}:
                raise ResumeError(f"Task {item.id} has an unsupported status in the source plan")
            item.status = encoded["status"]
            item.attempts = [
                TaskAttempt(**attempt)
                for attempt in encoded.get("attempts", [])
                if isinstance(attempt, dict)
            ]
            item.file_progress = [
                FileAttempt(**attempt)
                for attempt in encoded.get("file_progress", [])
                if isinstance(attempt, dict)
            ]
    except (MasterPlanError, TypeError, ValueError, KeyError) as error:
        if isinstance(error, ResumeError):
            raise
        raise ResumeError(f"Source task plan is invalid under the current schema: {error}") from error
    return items


def _load_stage(
    source_run_dir: str,
    item: WorkItem,
    expected_plan_digest: str,
    config: dict[str, Any],
    existing_project: bool,
) -> dict[str, Any] | None:
    task_key = hashlib.sha256(item.id.encode("utf-8")).hexdigest()[:24]
    task_dir = os.path.join(source_run_dir, "staging", task_key)
    if os.path.islink(os.path.join(source_run_dir, "staging")) or os.path.islink(task_dir):
        raise ResumeError(f"Task {item.id} staging path is unsafe")
    stage_path = os.path.join(task_dir, "stage.json")
    if not os.path.exists(stage_path):
        return None
    stage = _read_json(stage_path, MAX_MANIFEST_BYTES)
    if (
        stage.get("schema_version") != 1
        or stage.get("task_id") != item.id
        or stage.get("plan_digest") != expected_plan_digest
        or not isinstance(stage.get("files"), dict)
    ):
        raise ResumeError(f"Task {item.id} stage manifest does not match the source plan")
    planned_paths = {
        _planned_path(owned, config, existing_project)
        for owned in item.owned_files
    }
    for path, record in stage["files"].items():
        if path not in planned_paths or not isinstance(record, dict):
            raise ResumeError(f"Task {item.id} stage contains an unowned path")
        attempts = record.get("attempts", [])
        if not isinstance(attempts, list) or any(not isinstance(attempt, dict) for attempt in attempts):
            raise ResumeError(f"Task {item.id} stage has malformed attempt history")
        plan_attempts = [
            attempt.number for attempt in item.file_progress if attempt.path == path
        ]
        stage_attempts = [
            attempt.get("number") for attempt in attempts if isinstance(attempt.get("number"), int)
        ]
        if plan_attempts and stage_attempts and max(plan_attempts) != max(stage_attempts):
            raise ResumeError(f"Task {item.id} file-attempt history is inconsistent for {path}")
    return stage


def _latest_validation_issues(source_run_dir: str) -> list[ValidationIssue]:
    results_dir = os.path.join(source_run_dir, "validation", "results")
    if not os.path.isdir(results_dir) or os.path.islink(results_dir):
        return []
    attempts = sorted(
        (name for name in os.listdir(results_dir) if name.startswith("attempt-") and name.endswith(".json")),
        key=lambda name: int("".join(character for character in name if character.isdigit()) or 0),
    )
    if not attempts:
        return []
    data = _read_json(os.path.join(results_dir, attempts[-1]), MAX_MANIFEST_BYTES)
    raw = data.get("errors", [])
    return [
        ValidationIssue(
            str(issue.get("file_path", "")),
            str(issue.get("work_item_id", "")),
            str(issue.get("error_output", "")),
            issue.get("severity", "error"),
        )
        for issue in raw if isinstance(issue, dict)
    ]


def load_resume_request(
    config: dict[str, Any],
    run_reference: str,
    task_ids: list[str],
    *,
    project_root: str | None = None,
) -> ResumeRequest:
    if not task_ids or any(not isinstance(task_id, str) or not task_id.strip() for task_id in task_ids):
        raise ResumeError("Resume requires one or more exact task ids")
    if len(task_ids) != len(set(task_ids)):
        raise ResumeError("Resume task ids must be unique")
    runs_root, source_run_dir = _resolve_source_run(config, run_reference)
    manifest = _read_json(os.path.join(source_run_dir, "manifest.json"), MAX_MANIFEST_BYTES)
    if manifest.get("status") not in {
        "failed", "validation_failed", "validation_passed_external_testing_pending"
    }:
        raise ResumeError("Only finalized source runs with a complete status can be resumed")
    if manifest.get("pipeline_schema_version") != PIPELINE_SCHEMA_VERSION:
        raise ResumeError(
            "Source run predates the current pipeline/tooling schema; missing tooling metadata requires an authorized plan amendment"
        )
    if "published_files_error" in manifest:
        raise ResumeError("Source run has no trustworthy published-file snapshot")
    plan_path = os.path.join(source_run_dir, "design", "plan.json")
    plan = _read_json(plan_path, MAX_PLAN_BYTES)
    source_project = manifest.get("project_path")
    if source_project is not None and not isinstance(source_project, str):
        raise ResumeError("Source project-path metadata is invalid")
    if source_project:
        source_project = os.path.realpath(source_project)
        if not project_root or os.path.realpath(project_root) != source_project:
            raise ResumeError("Existing-project resume requires the exact source project root")
        project_root = source_project
    elif project_root:
        raise ResumeError("A generated-project run cannot be resumed against an existing project root")
    if manifest.get("pipeline_config_digest") != compatibility_digest(config, project_root):
        raise ResumeError("Current runtime/tooling/output configuration is incompatible with the source run")
    items = _work_items(plan, config, project_root)
    if any(
        owned.language in {"typescript", "javascript"}
        or owned.path.casefold().endswith((".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"))
        for item in items for owned in item.owned_files
    ):
        toolchain_manifest = (
            Path(__file__).resolve().parents[2]
            / "tools" / "typescript-checker" / "node_modules" / "typescript" / "package.json"
        )
        installed = _read_json(str(toolchain_manifest), MAX_MANIFEST_BYTES)
        if installed.get("version") != TYPESCRIPT_VERSION:
            raise ResumeError("Pinned trusted TypeScript checker is missing or has an incompatible version")
    expected_plan_digest = _task_plan_digest(items)
    if manifest.get("stage_plan_digest") != expected_plan_digest:
        raise ResumeError("Source plan digest does not match the recorded plan identity")
    if plan.get("format_version") != 2:
        raise ResumeError("Source plan schema is incompatible; an authorized plan amendment is required")
    by_id = {item.id: item for item in items}
    unknown = sorted(set(task_ids) - set(by_id))
    if unknown:
        raise ResumeError("Unknown task id(s): " + ", ".join(unknown))

    published = manifest.get("published_files")
    if not isinstance(published, dict) or any(
        not isinstance(path, str) or not isinstance(digest, str)
        for path, digest in published.items()
    ):
        raise ResumeError("Source run has no complete published-file digest snapshot")
    for relative in published:
        try:
            normalize_relative_path(relative)
        except ValueError as error:
            raise ResumeError(f"Published path is unsafe in source run: {relative!r}") from error
    current = (
        _existing_sources(project_root, runs_root)
        if project_root
        else _generated_sources(source_run_dir, config)
    )
    if current != published:
        raise ResumeError("Published source changed since the source run; refusing stale-source resume")

    source_stages: dict[str, dict[str, Any]] = {}
    for item in items:
        stage = _load_stage(
            source_run_dir, item, expected_plan_digest, config, project_root is not None
        )
        if stage is not None:
            source_stages[item.id] = stage
        if item.status == "passed":
            if stage is None or stage.get("status") != "published":
                raise ResumeError(f"Passed task {item.id} has no verified published stage history")
            for owned in item.owned_files:
                path = _planned_path(owned, config, project_root is not None)
                record = stage["files"].get(path)
                if not isinstance(record, dict):
                    raise ResumeError(f"Passed task {item.id} has no stage digest for {path}")
                action = record.get("action")
                if action == "delete":
                    if path in published:
                        raise ResumeError(f"Deleted output {path} is present in the published snapshot")
                elif action == "write":
                    if record.get("digest") != published.get(path):
                        raise ResumeError(f"Published output digest does not match task {item.id}: {path}")
                else:
                  raise ResumeError(f"Passed task {item.id} has an invalid publish action for {path}")

    descendants: dict[str, set[str]] = {item.id: set() for item in items}
    for item in items:
        for dependency_id in item.depends_on:
            if dependency_id in descendants:
                descendants[dependency_id].add(item.id)
    for task_id, stage in source_stages.items():
        scope = stage.get("repair_scope")
        if scope is None:
            continue
        item = by_id[task_id]
        paths = scope.get("paths") if isinstance(scope, dict) else None
        affected_ids = scope.get("work_item_ids") if isinstance(scope, dict) else None
        reachable = set()
        pending = list(descendants[task_id])
        while pending:
            current = pending.pop()
            if current in reachable:
                continue
            reachable.add(current)
            pending.extend(descendants[current])
        if (
            not isinstance(paths, list)
            or not all(isinstance(path, str) for path in paths)
            or not set(paths).issubset({
                _planned_path(owned, config, project_root is not None)
                for owned in item.owned_files
            })
            or not isinstance(affected_ids, list)
            or not all(isinstance(value, str) for value in affected_ids)
            or not set(affected_ids).issubset(reachable)
        ):
            conservative_paths = [
                path for path, record in stage["files"].items()
                if isinstance(record, dict) and record.get("status") in {"staged", "rejected"}
            ]
            if task_id in task_ids and not conservative_paths:
                conservative_paths = [
                    _planned_path(owned, config, project_root is not None)
                    for owned in item.owned_files
                ]
            stage["repair_scope"] = {
                "kind": "conservative_feature_fallback",
                "paths": conservative_paths,
                "work_item_ids": [],
            }

    selected = set(task_ids)
    max_attempts = config["pipeline"].get("max_retries", 1)
    for task_id in selected:
        if len(by_id[task_id].attempts) >= max_attempts:
            raise ResumeError(f"Task retry budget is exhausted for {task_id}")
    allowed_ids = set(by_id)
    changed = set(selected)
    while changed:
        current_id = changed.pop()
        stage = source_stages.get(current_id, {})
        scope = stage.get("repair_scope", {})
        affected_ids = scope.get("work_item_ids", []) if isinstance(scope, dict) else []
        if isinstance(affected_ids, list):
            for affected_id in affected_ids:
                if affected_id in allowed_ids and affected_id not in selected:
                    selected.add(affected_id)
                    changed.add(affected_id)
        for item in items:
            if item.id in selected or current_id not in item.depends_on:
                continue
            if item.status == "blocked":
                selected.add(item.id)
                changed.add(item.id)

    for task_id in selected:
        item = by_id[task_id]
        if len(item.attempts) >= max_attempts:
            raise ResumeError(f"Task retry budget is exhausted for {task_id}")
        unavailable = [
            dependency for dependency in item.depends_on
            if dependency not in selected and by_id[dependency].status != "passed"
        ]
        if unavailable:
            raise ResumeError(
                f"Task {task_id} depends on tasks that are neither selected nor verified passed: "
                + ", ".join(unavailable)
            )

    retry_context: dict[str, list[ValidationIssue]] = {task_id: [] for task_id in selected}
    for issue in _latest_validation_issues(source_run_dir):
        if issue.work_item_id in selected:
            retry_context[issue.work_item_id].append(issue)

    successful_results: list[AgentResult] = []
    unselected_results: list[AgentResult] = []
    for item in items:
        if item.id in selected:
            continue
        if item.status == "passed":
            last = item.attempts[-1] if item.attempts else None
            successful_results.append(AgentResult(
                item.id,
                f"{item.category.title()}Agent",
                True,
                list(last.files_written) if last else [],
                files_deleted=list(last.files_deleted) if last else [],
            ))
        elif item.status in {"failed", "blocked", "running", "ready"}:
            unselected_results.append(AgentResult(
                item.id,
                f"{item.category.title()}Agent",
                False,
                [],
                f"Task not selected for resume; source status was {item.status}",
            ))

    provenance = {
        "source_run_id": manifest.get("run_id", os.path.basename(source_run_dir)),
        "source_run_dir": source_run_dir,
        "source_plan_digest": expected_plan_digest,
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "typescript_version": TYPESCRIPT_VERSION,
        "requested_task_ids": list(task_ids),
        "selected_task_ids": sorted(selected),
        "published_files_sha256": hashlib.sha256(
            json.dumps(published, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    return ResumeRequest(
        source_run_dir=source_run_dir,
        source_run_id=str(manifest.get("run_id", os.path.basename(source_run_dir))),
        work_items=items,
        requested_task_ids=list(task_ids),
        selected_task_ids=sorted(selected),
        project_root=project_root,
        published_files=dict(published),
        successful_results=successful_results,
        unselected_results=unselected_results,
        retry_context=retry_context,
        stage_manifests=source_stages,
        provenance=provenance,
    )


def _planned_path(owned_file: OwnedFile, config: dict[str, Any], existing_project: bool) -> str:
    path = normalize_relative_path(owned_file.path)
    if existing_project:
        return path
    area_key = "tests_dir" if owned_file.area == "tests" else "output_dir"
    area = normalize_relative_path(config["pipeline"][area_key])
    if path.startswith(area + "/"):
        path = path[len(area) + 1:]
    return f"{area}/{path}"


def import_resume_sources(request: ResumeRequest, config: dict[str, Any], child_run_dir: str) -> None:
    """Copy verified published files and stage history needed by the child plan."""
    if not request.project_root:
        total_imported = 0
        for relative, expected_digest in request.published_files.items():
            normalize_relative_path(relative)
            source_path = os.path.join(request.source_run_dir, *relative.split("/"))
            target_path = os.path.join(child_run_dir, *relative.split("/"))
            if os.path.islink(source_path) or not os.path.isfile(source_path):
                raise ResumeError(f"Published source became a symbolic link: {relative}")
            size = os.path.getsize(source_path)
            if size > MAX_RESUME_FILE_BYTES or total_imported + size > MAX_RESUME_SOURCE_BYTES:
                raise ResumeError("Published source snapshot exceeds the resume file/source safety budget")
            with open(source_path, "rb") as source:
                content = source.read(MAX_RESUME_FILE_BYTES + 1)
            total_imported += len(content)
            if hashlib.sha256(content).hexdigest() != expected_digest:
                raise ResumeError(f"Published source changed during resume snapshot import: {relative}")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, "xb") as target:
                target.write(content)

    from pipeline.staging import open_stage, persist_stage, read_stage, write_stage_content

    selected = set(request.selected_task_ids)
    for item in request.work_items:
        task_id = item.id
        is_selected = task_id in selected
        if not is_selected and item.status != "passed":
            continue
        parent = request.stage_manifests.get(task_id)
        if parent is None:
            if item.status == "passed":
                raise ResumeError(f"Passed task {task_id} has no verified stage to import")
            continue
        layout = open_stage(item, config, child_run_dir)
        child = read_stage(layout)
        child_files = child.setdefault("files", {})
        for relative, record in parent.get("files", {}).items():
            if not isinstance(record, dict) or relative not in {
                _planned_path(owned, config, request.project_root is not None)
                for owned in item.owned_files
            }:
                continue
            imported = dict(record)
            attempts = record.get("attempts", [])
            if not isinstance(attempts, list):
                raise ResumeError(f"Task {task_id} has malformed file-attempt history")
            imported["attempts"] = [dict(attempt) for attempt in attempts if isinstance(attempt, dict)]
            if record.get("digest"):
                source_stage_root = os.path.join(
                    request.source_run_dir, "staging", hashlib.sha256(task_id.encode()).hexdigest()[:24]
                )
                source_files_root = os.path.join(source_stage_root, "files")
                if os.path.islink(source_stage_root) or os.path.islink(source_files_root):
                    raise ResumeError(f"Task {task_id} stage source path is unsafe")
                staged_source = os.path.join(source_files_root, *relative.split("/"))
                if not os.path.isfile(staged_source) or os.path.islink(staged_source):
                    if record.get("status") in {"staged", "rejected"}:
                        raise ResumeError(f"Task {task_id} candidate source is missing for {relative}")
                else:
                    with open(staged_source, "r", encoding="utf-8", newline="") as source:
                        content = source.read(256 * 1024 + 1)
                    if len(content) > 256 * 1024 or hashlib.sha256(content.encode("utf-8")).hexdigest() != record["digest"]:
                        raise ResumeError(f"Task {task_id} rejected-stage digest failed verification for {relative}")
                    write_stage_content(layout, relative, content)
            child_files[relative] = imported

        repair_scope = parent.get("repair_scope", {})
        repair_paths = set(repair_scope.get("paths", [])) if isinstance(repair_scope, dict) else set()
        if item.status == "passed" and not repair_paths:
            repair_paths = {
                _planned_path(owned, config, request.project_root is not None)
                for owned in item.owned_files
            }
        if is_selected:
            for relative in repair_paths:
                record = child_files.get(relative)
                if isinstance(record, dict) and record.get("status") == "staged":
                    record["status"] = "rejected"
                    record["failure_details"] = "Explicitly selected for resume repair"
            child["status"] = "in_progress"
            child["repair_scope"] = {
                "kind": "resume_import",
                "paths": sorted(repair_paths),
                "work_item_ids": list(repair_scope.get("work_item_ids", [])) if isinstance(repair_scope, dict) else [],
            }
        else:
            child["status"] = "published"
            child["repair_scope"] = repair_scope
        child["consistency_feedback"] = str(parent.get("consistency_feedback", ""))[:1200]
        persist_stage(layout, child)


def write_resume_history(request: ResumeRequest, child_run_dir: str) -> None:
    entries = []
    for task_id in request.selected_task_ids:
        item = next(item for item in request.work_items if item.id == task_id)
        entries.append({
            "task_id": task_id,
            "prior_task_status": item.status,
            "prior_attempt_count": len(item.attempts),
            "prior_file_attempt_count": len(item.file_progress),
        })
    with open(os.path.join(child_run_dir, "resume-history.json"), "x", encoding="utf-8") as output:
        json.dump({"provenance": request.provenance, "selected_tasks": entries}, output, indent=2)
        output.write("\n")
