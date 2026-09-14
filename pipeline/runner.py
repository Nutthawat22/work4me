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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pipeline.master import MasterAgent, MasterPlanError
from pipeline.config_validation import validate_config
from pipeline.dispatch import dispatch_work_items, CycleError
from pipeline.instructions import write_instructions_md
from pipeline.run_paths import append_index, create_run_dir, write_manifest
from pipeline.state import AgentResult, FailReport, TestResult, WorkItem
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
    print(f"  [{item.id}] ({item.type}) {item.title}")
    print(f"    description: {item.description}")
    print(f"    acceptance_criteria:")
    for criterion in item.acceptance_criteria:
        print(f"      - {criterion}")
    print(f"    output_path: {item.output_path}")
    print(f"    depends_on: {item.depends_on}")


def print_agent_result(result: AgentResult) -> None:
    status = "✅" if result.success else "❌"
    print(f"  [{result.work_item_id}] {result.agent_name} {status}")
    print(f"    files_written: {result.files_written}")
    if result.notes:
        print(f"    notes: {result.notes}")


def _print_dispatch_progress(item: WorkItem, index: int, total: int) -> None:
    print(f"  ⏳ [{index}/{total}] {item.id} ({item.type}) {item.title} — working...")


def print_specialist_failures(results: list[AgentResult]) -> None:
    failures = [r for r in results if not r.success]
    if not failures:
        return
    print("⚠️  SPECIALIST FAILURE(S) DETECTED:")
    for r in failures:
        print(f"  [{r.work_item_id}] {r.agent_name} failed")
        print(f"    notes: {r.notes}")


def print_fail_report(report: FailReport) -> None:
    print("═" * 50)
    print(f"PIPELINE FAILED after {report.run_number} attempts")
    print("═" * 50)
    print()
    print(f"Unresolved work items: {', '.join(report.unresolved_items) or '(none resolved)'}")
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
        return _run_pipeline_body(config, user_input, work_items, label)
    finally:
        acp_client.teardown()


def _run_pipeline_body(
    config: dict,
    user_input: str | None,
    work_items: list[WorkItem] | None,
    label: str,
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
    print(f"\n📁 Run dir: {run_dir}\n")

    manifest = {
        "run_id": os.path.basename(run_dir),
        "prompt": label,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "work_item_count": None,
        "status": "in_progress",
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
            return False

    manifest["work_item_count"] = len(work_items)
    write_manifest(run_dir, manifest)

    with open(os.path.join(run_dir, "design", "plan.json"), "w") as f:
        json.dump([dataclasses.asdict(wi) for wi in work_items], f, indent=2)

    print(f"✅ Plan ready: {len(work_items)} WorkItem(s) (see design/plan.json for full detail)\n")

    print(f"🚀 Dispatching {len(work_items)} WorkItem(s)...\n")
    try:
        agent_results = dispatch_work_items(
            work_items, config, run_dir, progress=_print_dispatch_progress
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
        return False

    print(f"Dispatch results ({len(agent_results)} agent(s) ran):\n")
    for result in agent_results:
        print_agent_result(result)
    print_specialist_failures(agent_results)
    print()

    max_retries = config["pipeline"]["max_retries"]
    test_runner = TestRunner()

    current_work_items = work_items
    all_agent_results: list[AgentResult] = list(agent_results)
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

    while not test_result.passed and attempt < max_retries:
        retry_context = master.map_failures_to_work_items(
            test_result, all_agent_results, work_items
        )
        if not retry_context:
            print("⚠️  Could not map any failures to WorkItems — stopping retries.\n")
            break

        current_work_items = [wi for wi in work_items if wi.id in retry_context]
        attempt += 1

        print(f"Retrying attempt {attempt} for WorkItems: "
              f"{[wi.id for wi in current_work_items]}\n")

        try:
            retry_results = dispatch_work_items(
                current_work_items, config, run_dir,
                retry_context=retry_context, strict=False,
                known_results=all_agent_results,
                progress=_print_dispatch_progress,
            )
        except CycleError as e:
            print(f"\n⚠️  Retry dispatch failed — dependency cycle: {e}\n")
            break

        print(f"Dispatch results ({len(retry_results)} agent(s) ran):\n")
        for result in retry_results:
            print_agent_result(result)
        print_specialist_failures(retry_results)
        print()

        all_agent_results.extend(retry_results)

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

    if test_result.passed:
        print("✅ All tests passed.\n")
        manifest["status"] = "passed"
        manifest["attempts"] = attempt
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(run_dir, manifest)
        append_index(runs_dir_abs, manifest)
        write_instructions_md(
            run_dir, config, work_items, all_agent_results, test_result, "passed"
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
