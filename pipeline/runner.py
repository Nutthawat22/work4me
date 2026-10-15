"""
pipeline/runner.py

REPL entrypoint for the agentic pipeline (Phase 1: Master + Design only).
Replaces the old main.py keyword-routing orchestrator.

Usage: python pipeline/runner.py
"""

import dataclasses
import json
import os
import sys
from datetime import datetime, timezone
from typing import Callable

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pipeline.master import MasterAgent, MasterPlanError
from pipeline.config_validation import validate_config
from pipeline.dispatch import dispatch_work_items, CycleError
from pipeline.instructions import write_instructions_md
from pipeline.run_paths import append_index, create_run_dir, write_manifest
from pipeline.state import AgentResult, FailReport, TaskAttempt, TestResult, WorkItem
from pipeline.test_runner import TestRunner
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
                result_summary=f"Wrote {len(result.files_written)} file(s)" if result.success else "",
                failure_details=result.notes if not result.success else "",
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
        print(f"  ✗ {failure.test_name}")
        if failure.error_output:
            for line in failure.error_output.splitlines():
                print(f"    {line}")
        print()
    print("Summary:")
    print(f"  {report.summary}")
    print("═" * 50)


def run_pipeline(
    config: dict,
    *,
    user_input: str | None = None,
    work_items: list[WorkItem] | None = None,
    label: str | None = None,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> bool:
    """
    Run one full pipeline pass end-to-end (run-dir creation -> decompose
    -> dispatch -> test/retry -> manifest finalize), driven either by a
    raw free-text prompt (REPL path, routed through
    MasterAgent.decompose) or a pre-built WorkItem list (e.g. an
    ingestion tool — pipeline/ingest.py — that already called
    MasterAgent.plan_handoff()). Exactly one of user_input / work_items
    must be given.

    This is a pure extraction of the former REPL while-loop body in
    main() — no behavioral change for the REPL path.

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
        progress: Optional structured progress callback. Completed task
            counts represent specialist attempts; retries add to the count.

    Returns:
        True if the run's tests ultimately passed, False otherwise
        (MasterPlanError, a dependency-cycle CycleError, or test
        failures surviving all retries all count as False) — callers
        can use this directly as a process exit-code signal.
    """
    if (user_input is None) == (work_items is None):
        raise ValueError(
            "run_pipeline requires exactly one of user_input or work_items"
        )

    if label is None:
        if user_input is None:
            raise ValueError("label is required when work_items is given")
        label = user_input

    try:
        return _run_pipeline_body(config, user_input, work_items, label, progress)
    finally:
        acp_client.teardown()


def _run_pipeline_body(
    config: dict,
    user_input: str | None,
    work_items: list[WorkItem] | None,
    label: str,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> bool:
    """
    The actual run_pipeline() body, extracted so run_pipeline() can wrap
    it in a try/finally that unconditionally tears down the ACP session
    pool (specialists.providers.acp_client) on every exit path — success,
    test-failure, MasterPlanError, or CycleError — without needing to
    duplicate the teardown call at each early return. See run_pipeline()
    for the public contract; this function's behavior/return value is
    identical to the pre-extraction run_pipeline() body.
    """
    master = MasterAgent(config)
    master.set_intent(label)

    runs_dir_abs = os.path.join(PROJECT_ROOT, config["pipeline"]["runs_dir"])
    run_dir = create_run_dir(runs_dir_abs, label)
    run_id = os.path.basename(run_dir)
    print(f"\n📁 Run dir: {run_dir}\n")

      # tests_blocking defaults to True (absent key == today's behavior):
      # explicit False makes test-*content* failures advisory-only for the
      # run's final pass/fail decision and the retry loop's test-driven
      # trigger — it does NOT touch the vacuous-pass guard further down
      # (specialist dispatch failures always block, regardless of this
      # flag). See _run_pipeline_body's retry loop and final decision
      # block below for where this is actually consumed.
    tests_blocking = config["pipeline"].get("tests_blocking", True)

    def report(**updates: object) -> None:
        if progress is not None:
            progress({"run_id": run_id, "tests_blocking": tests_blocking, **updates})

    report(phase="planning")

    manifest = {
        "run_id": os.path.basename(run_dir),
        "prompt": label,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "work_item_count": None,
        "status": "in_progress",
        "tests_blocking": tests_blocking,
    }
    write_manifest(run_dir, manifest)

    if work_items is None:
        print("🧠 Planning... (MasterAgent LLM call in progress, this can take a while)\n")
        try:
            work_items = master.decompose(user_input)
        except MasterPlanError as e:
            print(f"\n⚠️  MasterAgent failed to produce a valid plan: {e}\n")
            manifest["status"] = "failed"
            manifest["failure_reason"] = "design_parse_error"
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_manifest(run_dir, manifest)
            append_index(runs_dir_abs, manifest)
            write_instructions_md(
                run_dir, config, [], [], None, "design_parse_error"
            )
            report(phase="teardown", outcome="failed")
            return False

    try:
        master._validate_task_plan(work_items)
    except MasterPlanError as e:
        print(f"\n⚠️  Invalid task plan: {e}\n")
        manifest["status"] = "failed"
        manifest["failure_reason"] = "task_plan_invalid"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(run_dir, manifest)
        append_index(runs_dir_abs, manifest)
        write_instructions_md(run_dir, config, [], [], None, "task_plan_invalid")
        report(phase="teardown", outcome="failed")
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
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(run_dir, manifest)
        append_index(runs_dir_abs, manifest)
        write_instructions_md(
            run_dir, config, work_items, [], None, "dispatch_cycle_error"
        )
        report(phase="teardown", outcome="failed")
        return False

    _sync_task_results(work_items, agent_results, attempts_before)
    persist_plan()

    print(f"Dispatch results ({len(agent_results)} agent(s) ran):\n")
    for result in agent_results:
        print_agent_result(result)
    print_specialist_failures(agent_results)
    print()

    max_retries = config["pipeline"]["max_retries"]
    test_runner = TestRunner()

    current_work_items = work_items
    all_agent_results: list[AgentResult] = list(agent_results)
    report(phase="testing", task=None, attempt_round=1)
    test_result: TestResult = test_runner.run(config, work_items, run_dir)

    attempt = 1
    with open(
        os.path.join(run_dir, "tests", "results", f"attempt-{attempt}.json"), "w"
    ) as f:
        json.dump(dataclasses.asdict(test_result), f, indent=2)

    print(
        f"Test run (attempt {attempt}): "
        f"{test_result.total - test_result.failed}/{test_result.total} passed"
    )
    print()

    while attempt < max_retries:
        # Test-content failures trigger retries only when blocking is enabled;
        # failed task executions always retry regardless of that setting.
        unresolved_specialist_ids = {
            r.work_item_id
            for r in _final_agent_results_by_work_item(all_agent_results)
            if not r.success
        }
        should_retry = (
            (not test_result.passed or bool(unresolved_specialist_ids))
            if tests_blocking
            else bool(unresolved_specialist_ids)
        )

        if not should_retry:
            break

        retry_context = master.map_failures_to_work_items(
            test_result, all_agent_results, work_items
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

        report(phase="testing", task=None, attempt_round=attempt)
        test_result = test_runner.run(config, work_items, run_dir)
        with open(
            os.path.join(run_dir, "tests", "results", f"attempt-{attempt}.json"), "w"
        ) as f:
            json.dump(dataclasses.asdict(test_result), f, indent=2)

        print(
            f"Test run (attempt {attempt}): "
            f"{test_result.total - test_result.failed}/{test_result.total} passed"
        )
        print()

    # test_result.passed alone isn't sufficient evidence of a real pass:
    # TestRunner.run() returns passed=True, total=0 both when a run
    # legitimately needed no tests (fine) AND when every specialist
    # failed before writing anything, so no test files existed to run
    # (NOT fine — see _final_agent_results_by_work_item's docstring).
    # TestResult has no visibility into AgentResults, so this
    # combination has to be checked here, where both are in scope —
    # reduce all_agent_results (initial dispatch + every retry round)
    # down to one final AgentResult per work_item_id and confirm every
    # one of them actually succeeded before trusting test_result.passed.
    final_agent_results = _final_agent_results_by_work_item(all_agent_results)
    failed_work_item_ids = sorted(
        r.work_item_id for r in final_agent_results if not r.success
    )

    if test_result.passed and not failed_work_item_ids:
        print("✅ All tests passed.\n")
        manifest["status"] = "passed"
        manifest["attempts"] = attempt
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(run_dir, manifest)
        append_index(runs_dir_abs, manifest)
        write_instructions_md(
            run_dir, config, work_items, all_agent_results, test_result, "passed"
        )
        report(phase="teardown", outcome="passed", attempt_round=attempt)
        return True

    if test_result.passed and failed_work_item_ids:
        # Vacuous pass: test_result.passed is True (0 tests ran) only
        # because the failed WorkItems above never got far enough to
        # write test files — not because there was genuinely nothing to
        # test. Report this as a failed run rather than "✅ All tests
        # passed.". This guard is NOT affected by tests_blocking: a
        # specialist dispatch failure is a specialist-failure signal, not
        # a test-content signal, and always blocks.
        print(
            f"⚠️  {len(failed_work_item_ids)} of {len(final_agent_results)} "
            "task(s) failed dispatch and produced no output — the "
            "0/0 vacuous test pass does not mean this run succeeded.\n"
        )
        fail_report = FailReport(
            run_number=attempt,
            failures=[],
            unresolved_items=failed_work_item_ids,
            summary=(
                f"{len(failed_work_item_ids)} of {len(final_agent_results)} "
                f"task(s) failed dispatch (specialist errors) after "
                f"{attempt} attempt(s); no test files were produced for "
                "them, so the 0/0 vacuous test pass cannot be treated as "
                "a real pass."
            ),
        )
        with open(
            os.path.join(run_dir, "tests", "results", "fail_report.json"), "w"
        ) as f:
            json.dump(dataclasses.asdict(fail_report), f, indent=2)
        print_fail_report(fail_report)
        print()

        manifest["status"] = "failed"
        manifest["failure_reason"] = "specialist_dispatch_failure"
        manifest["attempts"] = attempt
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(run_dir, manifest)
        append_index(runs_dir_abs, manifest)
        write_instructions_md(
            run_dir, config, work_items, all_agent_results, test_result, "failed"
        )
        report(phase="teardown", outcome="failed", attempt_round=attempt)
        return False

    if not tests_blocking and not failed_work_item_ids:
        # Non-blocking mode: test *content* failed, but every specialist's
        # final dispatch succeeded — the vacuous-pass guard above doesn't
        # apply (failed_work_item_ids is empty), so this is a genuine
        # test-content failure, not a specialist failure. Per this run's
        # explicit config, that's advisory-only: don't fail the run, but
        # don't silently report "✅ passed" either — a distinct status
        # string keeps this traceable/queryable separately from a real
        # "passed" run.
        print(
            "⚠️  Tests failed but tests_blocking=false — treating as "
            "advisory-only. Specialist dispatch succeeded for all "
            "tasks; automated test results should be treated as "
            "unverified.\n"
        )
        manifest["status"] = "passed_with_test_failures"
        manifest["attempts"] = attempt
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(run_dir, manifest)
        append_index(runs_dir_abs, manifest)
        write_instructions_md(
            run_dir, config, work_items, all_agent_results, test_result,
            "passed_with_test_failures",
        )
        report(
            phase="teardown",
            outcome="passed_with_test_failures",
            attempt_round=attempt,
        )
        return True

    # Recompute the mapping against the final failing test_result so
    # "unresolved" reflects the WorkItems actually implicated by the
    # last failure — not just whatever subset happened to be
    # in-flight when the retry budget/mapping ran out (relevant
    # when max_retries==1 and no retry loop iteration ever ran).
    final_retry_context = master.map_failures_to_work_items(
        test_result, all_agent_results, work_items
    )
    unresolved = sorted(final_retry_context) or [
        wi.id for wi in current_work_items
    ]
    fail_report = master.build_fail_report(attempt, test_result, unresolved)
    with open(
        os.path.join(run_dir, "tests", "results", "fail_report.json"), "w"
    ) as f:
        json.dump(dataclasses.asdict(fail_report), f, indent=2)
    print_fail_report(fail_report)
    print()

    manifest["status"] = "failed"
    manifest["attempts"] = attempt
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_manifest(run_dir, manifest)
    append_index(runs_dir_abs, manifest)
    write_instructions_md(
        run_dir, config, work_items, all_agent_results, test_result, "failed"
    )
    report(phase="teardown", outcome="failed", attempt_round=attempt)
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
