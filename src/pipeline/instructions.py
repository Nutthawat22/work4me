"""
src/pipeline/instructions.py

Generates deterministic pipeline status and points external testers to the
testing handoff. This file does not execute or suggest project commands.

Tasks own multiple declared files; run instructions inspect the complete
successful output set instead of relying on single-file task labels.
"""

import os

from pipeline.state import AgentResult, ValidationResult, WorkItem

def _collect_languages(work_items: list[WorkItem]) -> list[str]:
    seen: list[str] = []
    for item in work_items:
        for owned_file in item.owned_files:
            if owned_file.language not in seen:
                seen.append(owned_file.language)
    return seen


def _successful_files(agent_results: list[AgentResult]) -> list[str]:
    files: list[str] = []
    for result in agent_results:
        if result.success:
            files.extend(result.files_written)
    return sorted(set(files))


def generate_instructions_md(
    config: dict,
    work_items: list[WorkItem],
    agent_results: list[AgentResult],
    validation_result: ValidationResult | None,
    run_status: str,
    run_id: str,
    run_dir: str | None = None,
) -> str:
    """
    Build the instructions.md content as a string (caller writes it to
    disk — see write_instructions_md).

    Args:
          config: Pipeline configuration associated with this run.
        work_items: The full WorkItem plan for this run.
        agent_results: All AgentResults produced across every dispatch
            round (initial + retries).
        validation_result: The latest deterministic static validation result,
            or None if planning/dispatch failed before validation.
        run_status: Mirrors manifest status. A successful pipeline status means
            static validation passed; external runtime testing remains pending.
        run_id: The run directory's basename, for the header.
          run_dir: Absolute run-artifact path, retained for the caller API.

    Returns:
        Markdown content as a string.
    """
    lines: list[str] = []
    lines.append(f"# How to Run This Product — {run_id}")
    lines.append("")

    lines.append("## Status")
    lines.append("")
    if run_status in {"passed", "validation_passed_external_testing_pending"}:
        lines.append("✅ Static validation passed. External runtime tests/builds are pending; no runtime test was executed by this pipeline.")
    elif run_status == "validation_failed":
        count = len(validation_result.errors) if validation_result else 0
        lines.append(f"❌ Static validation found {count} blocking error(s); see validation/results/fail_report.json. External runtime testing remains pending.")
    elif run_status == "failed":
        lines.append("❌ Specialist dispatch or pipeline execution failed. Files may be partial; external runtime testing remains pending.")
    elif run_status == "design_parse_error":
        lines.append("❌ Planning failed — no code was generated this run.")
    elif run_status == "dispatch_cycle_error":
        lines.append(
            "❌ Dispatch failed (dependency cycle) — some files below may "
            "have been written before the error."
        )
    elif run_status == "task_plan_invalid":
        lines.append("❌ Planning produced an invalid task/file ownership model; no code was generated.")
    else:
        lines.append(f"Status: {run_status}")
    lines.append("")

    successful_files = _successful_files(agent_results)
    lines.append("## External Testing")
    lines.append("")
    lines.append("No generated-project command, dependency installation, runtime test, or build was executed by the pipeline.")
    lines.append("Review `testing-handoff.md` and `testing-handoff.json`, then test an isolated copy of the project. For existing-project mode, do not run setup or tests against the original project directory.")
    lines.append("")

    lines.append("## Files Written")
    lines.append("")
    if successful_files:
        for f in successful_files:
            lines.append(f"- `{f}`")
    else:
        lines.append("None.")
    lines.append("")

    languages = _collect_languages(work_items)
    if languages:
        lines.append("Planned code languages: " + ", ".join(languages))
    if validation_result and validation_result.warnings:
        lines.append("")
        lines.append("Static validation warnings:")
        lines.extend(
            f"- `{issue.file_path}`: {issue.error_output}"
            for issue in validation_result.warnings
        )
    if validation_result and validation_result.limitations:
        lines.append("")
        lines.append("Static validation limitations:")
        lines.extend(f"- {limitation}" for limitation in validation_result.limitations)
    lines.append("")

    return "\n".join(lines)


def write_instructions_md(
    run_dir: str,
    config: dict,
    work_items: list[WorkItem],
    agent_results: list[AgentResult],
    validation_result: ValidationResult | None,
    run_status: str,
) -> str:
    """
    Generate and write instructions.md to run_dir. Returns the full path
    written.
    """
    run_id = os.path.basename(run_dir.rstrip(os.sep))
    content = generate_instructions_md(
        config, work_items, agent_results, validation_result, run_status, run_id,
        run_dir=run_dir,
    )
    path = os.path.join(run_dir, "instructions.md")
    with open(path, "w") as f:
        f.write(content)
    return path
