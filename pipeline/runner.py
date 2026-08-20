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

from pipeline.master import MasterAgent
from pipeline.config_validation import validate_config
from pipeline.design import DesignAgent, DesignParseError
from pipeline.dispatch import dispatch_work_items, CycleError
from pipeline.run_paths import append_index, create_run_dir, write_manifest
from pipeline.state import AgentResult, FailReport, TestResult, WorkItem
from pipeline.test_runner import TestRunner


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


def main():
    config, config_path = load_config()

    errors = validate_config(config)
    if errors:
        print(f"⚠️  Config validation failed (path: {config_path}):")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    master = MasterAgent(config)
    design_agent = DesignAgent(config)

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

        master.set_intent(user_input)

        runs_dir_abs = os.path.join(PROJECT_ROOT, config["pipeline"]["runs_dir"])
        run_dir = create_run_dir(runs_dir_abs, user_input)
        print(f"\n📁 Run dir: {run_dir}\n")

        manifest = {
            "run_id": os.path.basename(run_dir),
            "prompt": user_input,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "work_item_count": None,
            "status": "in_progress",
        }
        write_manifest(run_dir, manifest)

        try:
            work_items = design_agent.decompose(user_input)
        except DesignParseError as e:
            print(f"\n⚠️  DesignAgent failed to produce a valid plan: {e}\n")
            manifest["status"] = "failed"
            manifest["failure_reason"] = "design_parse_error"
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_manifest(run_dir, manifest)
            append_index(runs_dir_abs, manifest)
            continue

        manifest["work_item_count"] = len(work_items)
        write_manifest(run_dir, manifest)

        with open(os.path.join(run_dir, "design", "plan.json"), "w") as f:
            json.dump([dataclasses.asdict(wi) for wi in work_items], f, indent=2)

        print(f"\nDesignAgent produced {len(work_items)} WorkItem(s):\n")
        for item in work_items:
            print_work_item(item)
        print()

        try:
            agent_results = dispatch_work_items(work_items, config, run_dir)
        except CycleError as e:
            print(f"\n⚠️  Dispatch failed — dependency cycle: {e}\n")
            manifest["status"] = "failed"
            manifest["failure_reason"] = "dispatch_cycle_error"
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_manifest(run_dir, manifest)
            append_index(runs_dir_abs, manifest)
            continue

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
        else:
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


if __name__ == "__main__":
    main()
