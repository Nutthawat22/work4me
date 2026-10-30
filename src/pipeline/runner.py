"""
src/pipeline/runner.py

REPL entrypoint for the agentic pipeline (Phase 1: Master + Design only).
Replaces the old main.py keyword-routing orchestrator.

Usage: python src/pipeline/runner.py
"""

import copy
import dataclasses
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Callable

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = str(_REPO_ROOT / "src")
sys.path.insert(0, _SRC_ROOT)

from pipeline.project_paths import PROJECT_ROOT, resolve_project_path
from pipeline.master import MasterAgent, MasterPlanError
from pipeline.config_validation import validate_config
from pipeline.dispatch import dispatch_work_items, CycleError
from pipeline.instructions import write_instructions_md
from pipeline.run_paths import append_index, create_run_dir, write_manifest
from pipeline.project_context import (
    build_project_context,
    snapshot_project_files,
    summarize_project_changes,
    validate_project_path,
)
from pipeline.state import AgentResult, FailReport, TaskAttempt, ValidationResult, WorkItem
from pipeline.static_validator import StaticValidator
from pipeline.testing_handoff import write_testing_handoff
from specialists.providers import acp_client


def load_config() -> tuple[dict, str]:
    """Load configuration from ~/.config/Agents/config.json, falling back to
    project root config.json. Returns (config, path_loaded_from)."""
    config_home = os.path.expanduser("~/.config/Agents/config.json")
    local = os.path.join(PROJECT_ROOT, "config.json")
    path = config_home if os.path.exists(config_home) else local
    with open(path, "r") as f:
        return json.load(f), path


def print_work_item(item: WorkItem) -> None:
    print(f"  [{item.id}] ({item.category}) {item.title}")
    print(f"    description: {item.description}")
    print(f"    acceptance_criteria:")
    for criterion in item.acceptance_criteria:
        print(f"      - {criterion}")
    print(f"    owned_files: {[f'{file.area}/{file.path}' for file in item.owned_files]}")
    print(f"    depends_on: {item.depends_on}")


def print_agent_result(result: AgentResult) -> None:
    status = "✅" if result.success else "❌"
    print(f"  [{result.work_item_id}] {result.agent_name} {status}")
    print(f"    files_written: {result.files_written}")
    if result.files_deleted:
        print(f"    files_deleted: {result.files_deleted}")
    if result.notes:
        print(f"    notes: {result.notes}")


def _print_dispatch_progress(item: WorkItem, index: int, total: int) -> None:
    print(f"  ⏳ [{index}/{total}] {item.id} ({item.category}) {item.title} — working...")


def print_specialist_failures(results: list[AgentResult]) -> None:
    failures = [r for r in results if not r.success]
    if not failures:
        return
    print("⚠️  SPECIALIST FAILURE(S) DETECTED:")
    for r in failures:
        print(f"  [{r.work_item_id}] {r.agent_name} failed")
        print(f"    notes: {r.notes}")


def _final_agent_results_by_work_item(
    agent_results: list[AgentResult],
) -> list[AgentResult]:
    """
    Reduce a chronologically-ordered list of AgentResults (initial
    dispatch + every retry round, in call order) down to one
    AgentResult per work_item_id: whichever result for that id was
    produced LAST. A WorkItem that failed on an early attempt but
    succeeded on a later retry round is represented by its (successful)
    later result here, not its earlier failure — this is what lets the
    dispatch-success check below correctly reflect "final state after
    all retries", not just the very first dispatch round.
    """
    by_id: dict[str, AgentResult] = {}
    for result in agent_results:
        by_id[result.work_item_id] = result
    return list(by_id.values())


def _sync_task_results(
    work_items: list[WorkItem],
    results: list[AgentResult],
    attempts_before: dict[str, int],
) -> None:
    """Keep task artifacts correct when dispatch is replaced by a test double."""
    by_id = {item.id: item for item in work_items}
    for result in results:
        item = by_id.get(result.work_item_id)
        if item is None or len(item.attempts) != attempts_before.get(item.id, 0):
            continue
        if not result.success and result.notes.startswith("Task blocked by dependencies:"):
            item.status = "blocked"
            continue
        item.status = "passed" if result.success else "failed"
        item.attempts.append(
            TaskAttempt(
                number=len(item.attempts) + 1,
                status="passed" if result.success else "failed",
                files_written=list(result.files_written),
                result_summary=(
                    f"Wrote {len(result.files_written)} file(s); deleted {len(result.files_deleted)} file(s)"
                    if result.success
                    else ""
                ),
                failure_details=result.notes if not result.success else "",
                files_deleted=list(result.files_deleted),
            )
        )


def print_fail_report(report: FailReport) -> None:
    print("═" * 50)
    print(f"PIPELINE FAILED after {report.run_number} attempts")
    print("═" * 50)
    print()
    print(f"Unresolved tasks: {', '.join(report.unresolved_items) or '(none resolved)'}")
    print()
    print("Failures:")
    for failure in report.failures:
        print(f"  ✗ {failure.file_path}")
        if failure.error_output:
            for line in failure.error_output.splitlines():
                print(f"    {line}")
        print()
    print("Summary:")
    print(f"  {report.summary}")
    print("═" * 50)


def _write_testing_handoff_safely(config: dict, run_dir: str) -> bool:
    try:
        write_testing_handoff(config, run_dir)
        return True
    except Exception:
        logging.exception(
            "Testing handoff artifact generation failed for run %s",
            os.path.basename(run_dir),
        )
        for filename in ("testing-handoff.json", "testing-handoff.md"):
            artifact_path = os.path.join(run_dir, filename)
            try:
                if os.path.lexists(artifact_path):
                    os.unlink(artifact_path)
            except OSError:
                logging.exception(
                    "Could not remove incomplete testing handoff artifact %s",
                    filename,
                )
        return False


def _finalize_pipeline_run(
    config: dict,
    run_dir: str,
    *,
    project_root: str | None,
    baseline: dict[str, str] | None,
    run_failed: bool,
    result: bool,
    pipeline_exception: tuple[BaseException, TracebackType | None] | None,
    progress: Callable[[dict[str, object]], None] | None,
) -> tuple[bool, tuple[BaseException, TracebackType | None] | None]:
    pipeline_config = config["pipeline"]
    if project_root is not None and baseline is not None:
        try:
            _finalize_existing_project_changes(
                run_dir,
                project_root,
                baseline,
                (pipeline_config.get("_protected_run_root", ""),),
                failed=run_failed,
            )
        except Exception as error:
            run_failed = True
            result = False
            logging.exception(
                "Existing-project change finalization failed for run %s",
                os.path.basename(run_dir),
            )
            if pipeline_exception is None:
                pipeline_exception = (error, error.__traceback__)

    manifest_path = os.path.join(run_dir, "manifest.json")
    manifest: dict[str, object] = {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            loaded_manifest = json.load(manifest_file)
        if isinstance(loaded_manifest, dict):
            manifest = loaded_manifest
    except Exception as error:
        run_failed = True
        result = False
        logging.exception(
            "Could not read run manifest while finalizing run %s",
            os.path.basename(run_dir),
        )
        if pipeline_exception is None:
            pipeline_exception = (error, error.__traceback__)

    if pipeline_exception is not None:
        manifest.update(
            status="failed",
            failure_reason="pipeline_exception",
            validation_status=manifest.get("validation_status", "not_completed"),
            external_testing_status="pending",
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
    elif manifest.get("status") == "in_progress":
        manifest.update(
            status="failed",
            failure_reason="pipeline_exception",
            validation_status="not_completed",
            external_testing_status="pending",
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        run_failed = True
        result = False

    candidate_status = str(manifest.get("status", "failed"))
    if result and candidate_status == "validation_passed_external_testing_pending":
        manifest["status"] = "finalizing"
        try:
            write_manifest(run_dir, manifest)
        except Exception as error:
            run_failed = True
            result = False
            candidate_status = "failed"
            manifest.update(
                status="failed",
                failure_reason="manifest_finalization_failure",
            )
            logging.exception(
                "Could not stage final run manifest for run %s",
                os.path.basename(run_dir),
            )
            if pipeline_exception is None:
                pipeline_exception = (error, error.__traceback__)

    pipeline_config["_testing_handoff_pipeline_status"] = candidate_status
    handoff_written = _write_testing_handoff_safely(config, run_dir)
    pipeline_config.pop("_testing_handoff_pipeline_status", None)
    manifest["testing_handoff_status"] = "written" if handoff_written else "failed"
    if not handoff_written:
        run_failed = True
        if candidate_status == "validation_passed_external_testing_pending":
            result = False
            manifest.update(
                status="failed",
                failure_reason="testing_handoff_failure",
            )
        manifest["testing_handoff_warning"] = (
            "Required testing handoff artifacts are incomplete; see pipeline logs."
        )
        print(
            "⚠️ Required testing handoff artifacts are incomplete; "
            "the run is not reported as successful.",
            file=sys.stderr,
        )
    else:
        if candidate_status == "validation_passed_external_testing_pending":
            manifest["status"] = candidate_status
        manifest.pop("testing_handoff_warning", None)

    finalization_failed = False
    try:
        write_manifest(run_dir, manifest)
    except Exception:
        run_failed = True
        result = False
        finalization_failed = True
        manifest["status"] = "failed"
        manifest.setdefault("failure_reason", "manifest_finalization_failure")
        logging.exception(
            "Could not write finalized run manifest for run %s",
            os.path.basename(run_dir),
        )

    try:
        runs_dir = resolve_project_path(pipeline_config["runs_dir"])
        append_index(runs_dir, manifest)
    except Exception:
        run_failed = True
        result = False
        finalization_failed = True
        manifest["status"] = "failed"
        manifest.setdefault("failure_reason", "run_index_finalization_failure")
        logging.exception(
            "Could not append finalized run index entry for run %s",
            os.path.basename(run_dir),
        )
        try:
            write_manifest(run_dir, manifest)
        except Exception:
            logging.exception(
                "Could not correct run manifest after index failure for run %s; "
                "the on-disk manifest may still report a completed workflow",
                os.path.basename(run_dir),
            )

    if progress is not None:
        try:
            progress({
                "run_id": manifest.get("run_id", os.path.basename(run_dir)),
                "phase": "teardown",
                "outcome": (
                    "failed"
                    if finalization_failed
                    or (
                        not result
                        and candidate_status == "validation_passed_external_testing_pending"
                    )
                    else candidate_status
                ),
                "validation_status": manifest.get("validation_status", "not_completed"),
                "external_testing_status": manifest.get("external_testing_status", "pending"),
                "testing_handoff_status": manifest["testing_handoff_status"],
            })
        except Exception as error:
            logging.exception(
                "Could not publish final pipeline progress for run %s",
                os.path.basename(run_dir),
            )

    return result, pipeline_exception


def run_pipeline(
    config: dict,
    *,
    user_input: str | None = None,
    work_items: list[WorkItem] | None = None,
    label: str | None = None,
    progress: Callable[[dict[str, object]], None] | None = None,
    existing_project_path: str | None = None,
) -> bool:
    """
    Run one full pipeline pass end-to-end (run-dir creation -> decompose
    -> dispatch -> static validation/repair -> manifest finalize), driven either by a
    raw free-text prompt (REPL path, routed through
    MasterAgent.decompose) or a pre-built WorkItem list (e.g. an
    ingestion tool — pipeline/ingest.py — that already called
    MasterAgent.plan_handoff()). Exactly one of user_input / work_items
    must be given.

    Free-text planning, specialist dispatch, and deterministic static
    validation run here. Generated-project tests/builds remain external.

    Args:
        config: Loaded pipeline config (see load_config()).
        user_input: Raw free-text prompt. Triggers master.decompose().
        work_items: Pre-built WorkItem list. Skips decompose() entirely
            and dispatches these directly.
        label: Human-readable label used for the run dir slug
            (run_paths.create_run_dir) and the manifest's "prompt"
            field. Defaults to user_input when user_input is given.
            Required when work_items is given, since there's no raw
            prompt string to derive it from.
        existing_project_path: Optional absolute server-side project
            directory. When set, specialists edit its owned files in place;
            run artifacts remain under pipeline.runs_dir.
        progress: Optional structured progress callback. Completed task
            counts represent specialist attempts; retries add to the count.

    Returns:
        True if planning, specialist dispatch, static validation, and both
        required testing-handoff artifacts completed successfully. False
        indicates pipeline, validation, or handoff artifact failure.
    """
    if (user_input is None) == (work_items is None):
        raise ValueError(
            "run_pipeline requires exactly one of user_input or work_items"
        )

    if label is None:
        if user_input is None:
            raise ValueError("label is required when work_items is given")
        label = user_input

    run_config = config
    project_root: str | None = None
    baseline: dict[str, str] | None = None
    run_failed = False
    result = False
    pipeline_exception: tuple[BaseException, TracebackType | None] | None = None
    try:
        if existing_project_path is not None:
            run_config = copy.deepcopy(config)
            pipeline_config = run_config["pipeline"]
            runs_root = pipeline_config["runs_dir"]
            runs_root = os.path.realpath(resolve_project_path(runs_root))
            project_root = validate_project_path(existing_project_path, runs_root)
            context, baseline = build_project_context(project_root, (runs_root,))
            pipeline_config["_existing_project_path"] = project_root
            pipeline_config["_existing_project_context"] = context
            pipeline_config["_protected_run_root"] = runs_root
            pipeline_config["_existing_project_baseline"] = baseline
        result = _run_pipeline_body(run_config, user_input, work_items, label, progress)
        if not result:
            run_failed = True
    except BaseException as error:
        run_failed = True
        pipeline_exception = (error, error.__traceback__)

    try:
        active_run_dir = run_config["pipeline"].get("_active_run_dir")
        if active_run_dir:
            try:
                result, pipeline_exception = _finalize_pipeline_run(
                    run_config,
                    active_run_dir,
                    project_root=project_root,
                    baseline=baseline,
                    run_failed=run_failed,
                    result=result,
                    pipeline_exception=pipeline_exception,
                    progress=progress,
                )
            except BaseException as error:
                logging.exception(
                    "Unexpected run finalization failure for %s",
                    os.path.basename(active_run_dir),
                )
                if pipeline_exception is None:
                    pipeline_exception = (error, error.__traceback__)
            finally:
                run_config["pipeline"].pop("_testing_handoff_pipeline_status", None)
    finally:
        try:
            acp_client.teardown()
        except BaseException as error:
            logging.exception("ACP teardown failed")
            if pipeline_exception is None:
                pipeline_exception = (error, error.__traceback__)

    if pipeline_exception is not None:
        error, traceback = pipeline_exception
        raise error.with_traceback(traceback)
    return result


def _finalize_existing_project_changes(
    run_dir: str,
    project_root: str,
    baseline: dict[str, str],
    excluded_roots: tuple[str, ...],
    *,
    failed: bool,
) -> None:
    current = snapshot_project_files(project_root, tuple(path for path in excluded_roots if path))
    changes = summarize_project_changes(baseline, current)
    warning = (
        "The pipeline did not complete successfully. Existing-project edits are in place; "
        "partial changes may remain and were not rolled back."
        if failed
        else None
    )
    manifest_path = os.path.join(run_dir, "manifest.json")
    manifest: dict | None = None
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        manifest["project_path"] = project_root
        manifest["changes"] = changes
        manifest["partial_edits_possible"] = failed
        if failed and manifest.get("status") == "in_progress":
            manifest["status"] = "failed"
            manifest["failure_reason"] = "pipeline_exception"
            manifest["validation_status"] = "not_completed"
            manifest["external_testing_status"] = "pending"
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()

    summary = {
        "project_path": project_root,
        "status": manifest.get("status", "failed" if failed else "completed") if manifest else (
            "failed" if failed else "completed"
        ),
        **changes,
        "partial_edits_possible": failed,
        "warning": warning,
    }
    with open(os.path.join(run_dir, "changes.json"), "w", encoding="utf-8") as change_file:
        json.dump(summary, change_file, indent=2)

    if manifest is not None:
        write_manifest(run_dir, manifest)


def _run_pipeline_body(
    config: dict,
    user_input: str | None,
    work_items: list[WorkItem] | None,
    label: str,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> bool:
    """
    The work body writes run artifacts and candidate status. The public
    wrapper finalizes handoff artifacts, manifest, run index, progress, and
    ACP teardown before returning.
    """
    master = MasterAgent(config)
    master.set_intent(label)

    runs_dir_abs = resolve_project_path(config["pipeline"]["runs_dir"])
    run_dir = create_run_dir(runs_dir_abs, label)
    config["pipeline"]["_active_run_dir"] = run_dir
    run_id = os.path.basename(run_dir)
    print(f"\n📁 Run dir: {run_dir}\n")

    def report(**updates: object) -> None:
        if progress is not None:
            progress({"run_id": run_id, **updates})

    report(phase="planning")

    manifest = {
        "run_id": os.path.basename(run_dir),
        "prompt": label,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "work_item_count": None,
        "status": "in_progress",
        "validation_status": "pending",
        "external_testing_status": "pending",
    }
    existing_project_path = config["pipeline"].get("_existing_project_path")
    if existing_project_path:
        manifest["project_path"] = existing_project_path
    write_manifest(run_dir, manifest)

    if work_items is None:
        print("🧠 Planning... (MasterAgent LLM call in progress, this can take a while)\n")
        try:
            project_context = config["pipeline"].get("_existing_project_context")
            if project_context:
                work_items = master.decompose(user_input, project_context=project_context)
            else:
                work_items = master.decompose(user_input)
        except MasterPlanError as e:
            print(f"\n⚠️  MasterAgent failed to produce a valid plan: {e}\n")
            manifest["status"] = "failed"
            manifest["failure_reason"] = "design_parse_error"
            manifest["validation_status"] = "not_completed"
            manifest["external_testing_status"] = "pending"
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_manifest(run_dir, manifest)
            write_instructions_md(
                run_dir, config, [], [], None, "design_parse_error"
            )
            return False

    try:
        master._validate_task_plan(work_items)
    except MasterPlanError as e:
        print(f"\n⚠️  Invalid task plan: {e}\n")
        manifest["status"] = "failed"
        manifest["failure_reason"] = "task_plan_invalid"
        manifest["validation_status"] = "not_completed"
        manifest["external_testing_status"] = "pending"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(run_dir, manifest)
        write_instructions_md(run_dir, config, [], [], None, "task_plan_invalid")
        return False

    manifest["work_item_count"] = len(work_items)
    write_manifest(run_dir, manifest)

    plan_path = os.path.join(run_dir, "design", "plan.json")
    reported_task_statuses: dict[str, str] = {}

    def persist_plan(_item: WorkItem | None = None) -> None:
        with open(plan_path, "w") as plan_file:
            json.dump(
                {"format_version": 2, "tasks": [dataclasses.asdict(wi) for wi in work_items]},
                plan_file,
                indent=2,
            )
        if _item is not None:
            reported_task_statuses[_item.id] = _item.status
            report(
                phase="dispatch",
                tasks_total=len(work_items),
                tasks_started=sum(len(item.attempts) for item in work_items)
                + sum(item.status == "running" for item in work_items),
                tasks_completed=sum(len(item.attempts) for item in work_items),
                tasks_blocked=sum(status == "blocked" for status in reported_task_statuses.values()),
                task={
                    "id": _item.id,
                    "category": _item.category,
                    "status": _item.status,
                    "attempt": len(_item.attempts) + (1 if _item.status == "running" else 0),
                },
            )

    persist_plan()

    print(f"✅ Plan ready: {len(work_items)} task(s) (see design/plan.json for full detail)\n")

    print(f"🚀 Dispatching {len(work_items)} task(s)...\n")
    report(
        phase="dispatch",
        tasks_total=len(work_items),
        tasks_started=0,
        tasks_completed=0,
        tasks_blocked=0,
        task=None,
        attempt_round=1,
        max_attempts=config["pipeline"]["max_retries"],
    )
    attempts_before = {item.id: len(item.attempts) for item in work_items}
    try:
        agent_results = dispatch_work_items(
            work_items, config, run_dir, progress=_print_dispatch_progress,
            task_update=persist_plan,
        )
    except CycleError as e:
        print(f"\n⚠️  Dispatch failed — dependency cycle: {e}\n")
        manifest["status"] = "failed"
        manifest["failure_reason"] = "dispatch_cycle_error"
        manifest["validation_status"] = "not_completed"
        manifest["external_testing_status"] = "pending"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(run_dir, manifest)
        write_instructions_md(
            run_dir, config, work_items, [], None, "dispatch_cycle_error"
        )
        return False

    _sync_task_results(work_items, agent_results, attempts_before)
    persist_plan()

    print(f"Dispatch results ({len(agent_results)} agent(s) ran):\n")
    for result in agent_results:
        print_agent_result(result)
    print_specialist_failures(agent_results)
    print()

    max_retries = config["pipeline"]["max_retries"]
    validator = StaticValidator()

    current_work_items = work_items
    all_agent_results: list[AgentResult] = list(agent_results)
    report(phase="static_validation", task=None, attempt_round=1)
    validation_result: ValidationResult = validator.run(
        config, work_items, run_dir, all_agent_results
    )

    attempt = 1
    with open(
        os.path.join(run_dir, "validation", "results", f"attempt-{attempt}.json"), "w"
    ) as f:
        json.dump(dataclasses.asdict(validation_result), f, indent=2)

    print(
        f"Static validation (attempt {attempt}): "
        f"{validation_result.checked_files} file(s) checked, "
        f"{len(validation_result.errors)} error(s), "
        f"{len(validation_result.warnings)} warning(s)"
    )
    for warning in validation_result.warnings:
        print(f"  ⚠️ {warning.file_path}: {warning.error_output}")
    for limitation in validation_result.limitations:
        print(f"  ℹ️ Validation limitation: {limitation}")
    print()

    while attempt < max_retries:
        unresolved_specialist_ids = {
            r.work_item_id
            for r in _final_agent_results_by_work_item(all_agent_results)
            if not r.success
        }
        should_retry = not validation_result.passed or bool(unresolved_specialist_ids)

        if not should_retry:
            break

        retry_context = master.map_validation_issues_to_work_items(
            validation_result, all_agent_results, work_items
        )
        if not retry_context:
            print("⚠️  Could not map any failures to WorkItems — stopping retries.\n")
            break

        current_work_items = [wi for wi in work_items if wi.id in retry_context]
        attempt += 1
        report(phase="retrying", task=None, attempt_round=attempt)

        print(f"Retrying attempt {attempt} for tasks: "
              f"{[wi.id for wi in current_work_items]}\n")

        attempts_before = {item.id: len(item.attempts) for item in current_work_items}
        try:
            retry_results = dispatch_work_items(
                current_work_items, config, run_dir,
                retry_context=retry_context, strict=False,
                known_results=all_agent_results,
                progress=_print_dispatch_progress,
                task_update=persist_plan,
            )
        except CycleError as e:
            print(f"\n⚠️  Retry dispatch failed — dependency cycle: {e}\n")
            break

        _sync_task_results(current_work_items, retry_results, attempts_before)
        persist_plan()

        print(f"Dispatch results ({len(retry_results)} agent(s) ran):\n")
        for result in retry_results:
            print_agent_result(result)
        print_specialist_failures(retry_results)
        print()

        all_agent_results.extend(retry_results)

        report(phase="static_validation", task=None, attempt_round=attempt)
        validation_result = validator.run(
            config, work_items, run_dir, all_agent_results
        )
        with open(
            os.path.join(run_dir, "validation", "results", f"attempt-{attempt}.json"), "w"
        ) as f:
            json.dump(dataclasses.asdict(validation_result), f, indent=2)

        print(
            f"Static validation (attempt {attempt}): "
            f"{validation_result.checked_files} file(s) checked, "
            f"{len(validation_result.errors)} error(s), "
            f"{len(validation_result.warnings)} warning(s)"
        )
        for warning in validation_result.warnings:
            print(f"  ⚠️ {warning.file_path}: {warning.error_output}")
        for limitation in validation_result.limitations:
            print(f"  ℹ️ Validation limitation: {limitation}")
        print()

      # Validation success alone is insufficient when a specialist failed;
      # require both the planned files to validate and every task to finish.
    final_agent_results = _final_agent_results_by_work_item(all_agent_results)
    failed_work_item_ids = sorted(
        r.work_item_id for r in final_agent_results if not r.success
    )

    if validation_result.passed and not failed_work_item_ids:
        print("✅ Static validation passed; external runtime testing is pending.\n")
        manifest["status"] = "validation_passed_external_testing_pending"
        manifest["validation_status"] = "passed"
        manifest["external_testing_status"] = "pending"
        manifest["validation_warnings"] = [dataclasses.asdict(item) for item in validation_result.warnings]
        manifest["validation_limitations"] = validation_result.limitations
        manifest["attempts"] = attempt
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(run_dir, manifest)
        write_instructions_md(
            run_dir, config, work_items, all_agent_results, validation_result,
            "validation_passed_external_testing_pending",
        )
        return True

    if failed_work_item_ids:
        # A dispatch failure remains blocking even if the generated files
        # happen to pass static checks.
        print(
            f"⚠️  {len(failed_work_item_ids)} of {len(final_agent_results)} "
            "task(s) failed specialist dispatch; the run did not pass.\n"
        )
        fail_report = FailReport(
            run_number=attempt,
            failures=validation_result.errors,
            unresolved_items=failed_work_item_ids,
            summary=(
                f"{len(failed_work_item_ids)} of {len(final_agent_results)} "
                f"task(s) failed specialist dispatch after {attempt} attempt(s)."
            ),
        )
        with open(
            os.path.join(run_dir, "validation", "results", "fail_report.json"), "w"
        ) as f:
            json.dump(dataclasses.asdict(fail_report), f, indent=2)
        print_fail_report(fail_report)
        print()

        manifest["status"] = "failed"
        manifest["failure_reason"] = "specialist_dispatch_failure"
        manifest["validation_status"] = "passed" if validation_result.passed else "failed"
        manifest["external_testing_status"] = "pending"
        manifest["validation_warnings"] = [dataclasses.asdict(item) for item in validation_result.warnings]
        manifest["validation_limitations"] = validation_result.limitations
        manifest["attempts"] = attempt
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(run_dir, manifest)
        write_instructions_md(
            run_dir, config, work_items, all_agent_results, validation_result, "failed"
        )
        return False

    # Recompute the mapping against the final failing validation result so
    # "unresolved" reflects the WorkItems actually implicated by the
    # last failure — not just whatever subset happened to be
    # in-flight when the retry budget/mapping ran out (relevant
    # when max_retries==1 and no retry loop iteration ever ran).
    final_retry_context = master.map_validation_issues_to_work_items(
        validation_result, all_agent_results, work_items
    )
    unresolved = sorted(final_retry_context) or [
        wi.id for wi in current_work_items
    ]
    fail_report = master.build_fail_report(attempt, validation_result, unresolved)
    with open(
        os.path.join(run_dir, "validation", "results", "fail_report.json"), "w"
    ) as f:
        json.dump(dataclasses.asdict(fail_report), f, indent=2)
    print_fail_report(fail_report)
    print()

    manifest["status"] = "validation_failed"
    manifest["validation_status"] = "failed"
    manifest["failure_reason"] = "static_validation_failed"
    manifest["external_testing_status"] = "pending"
    manifest["validation_warnings"] = [dataclasses.asdict(item) for item in validation_result.warnings]
    manifest["validation_limitations"] = validation_result.limitations
    manifest["attempts"] = attempt
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_manifest(run_dir, manifest)
    write_instructions_md(
        run_dir, config, work_items, all_agent_results, validation_result, "validation_failed"
    )
    return False


def main():
    # Force line-buffered stdout: when stdout isn't a real TTY (piped,
    # redirected, or run inside some wrapper/subprocess), Python defaults
    # to full block-buffering, so progress prints (Planning.../dispatch
    # "working..." lines) don't actually reach the terminal until the
    # buffer fills or the process exits — they appear to "batch dump" at
    # the end instead of streaming live. Forcing line-buffering here
    # fixes that without needing flush=True on every individual print().
    sys.stdout.reconfigure(line_buffering=True)

    config, config_path = load_config()

    errors = validate_config(config)
    if errors:
        print(f"⚠️  Config validation failed (path: {config_path}):")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    print("=" * 50)
    print("🤖 Agentic Pipeline Ready")
    print("=" * 50)
    print("Type your request and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nShutting down...")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not user_input:
            continue

        run_pipeline(config, user_input=user_input)


if __name__ == "__main__":
    main()
