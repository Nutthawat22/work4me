"""
pipeline/instructions.py

Generates a deterministic, templated instructions.md file at the end of
every pipeline run (pass or fail) explaining how to actually run the
product this run produced. Deliberately NOT an LLM call — this inspects
the actual files written to run_dir/{output_dir} (looking for a
recognized dependency manifest and reading its "scripts"/entry point) so
the commands given are concrete and copy-pasteable, not a generic report.

UPDATE (2026-09-04, post-run-0008 finding): the planning system prompt's
(then pipeline/design.py's, now pipeline/master.py's) RUNNABILITY rule
used to ask for a single "[ASSEMBLY]" WorkItem to
produce BOTH a dependency manifest (package.json) AND an entry-point
source file. That's impossible — every WorkItem in this pipeline writes
to exactly one output_path (see specialists/base.py's run_specialist),
so the LLM could only pick one file to actually write and a package.json
never got generated (run 0008: 18/18 WorkItems succeeded but the
"[ASSEMBLY]" WorkItem only produced src/server/main.ts, no manifest).
The rule now asks for TWO WorkItems — "[ASSEMBLY-MANIFEST]" for the
dependency manifest and "[ASSEMBLY-ENTRYPOINT]" for the entry-point file
— and this module looks for both markers plus known manifest filenames
directly, so it can report accurately even if the LLM's titling drifts.
"""

import json
import os

from pipeline.state import AgentResult, TestResult, WorkItem

ASSEMBLY_TITLE_MARKERS = ("[ASSEMBLY-MANIFEST]", "[ASSEMBLY-ENTRYPOINT]", "[ASSEMBLY]")

# Recognized dependency manifest filenames, in priority order, mapped to
# the (install_command, run_command_builder) needed to use them. Node/Bun
# manifests are checked for an actual "scripts" entry so the run command
# reflects what was really generated rather than guessing "npm start".
NODE_MANIFEST_FILENAMES = ("package.json",)
PYTHON_MANIFEST_FILENAMES = ("requirements.txt", "pyproject.toml")


def _collect_languages(work_items: list[WorkItem]) -> list[str]:
    seen: list[str] = []
    for item in work_items:
        if item.language not in seen:
            seen.append(item.language)
    return seen


def _find_assembly_work_items(work_items: list[WorkItem]) -> list[WorkItem]:
    return [
        wi for wi in work_items
        if any(marker in wi.title for marker in ASSEMBLY_TITLE_MARKERS)
    ]


def _successful_files(agent_results: list[AgentResult]) -> list[str]:
    files: list[str] = []
    for result in agent_results:
        if result.success:
            files.extend(result.files_written)
    return sorted(set(files))


def _find_manifest_file(files: list[str], filenames: tuple[str, ...]) -> str | None:
    """Return the first successfully-written file (relative to run_dir)
    whose basename matches one of filenames, or None if none was
    produced."""
    for candidate in filenames:
        for f in files:
            if os.path.basename(f) == candidate:
                return f
    return None


def _read_npm_scripts(run_dir: str, package_json_relpath: str) -> dict[str, str]:
    """Read run_dir/package_json_relpath and return its "scripts" dict,
    or {} if the file is missing/unreadable/malformed — callers should
    fall back to a generic "npm start" suggestion in that case rather
    than crash on a best-effort instructions file."""
    full_path = os.path.join(run_dir, package_json_relpath)
    try:
        with open(full_path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    scripts = data.get("scripts")
    return scripts if isinstance(scripts, dict) else {}


def _pick_run_script(scripts: dict[str, str]) -> str | None:
    """Prefer "dev", then "start", then the first script defined —
    matches how most Node/Bun projects name their primary run command."""
    for preferred in ("dev", "start"):
        if preferred in scripts:
            return preferred
    return next(iter(scripts), None)


def _build_run_steps(
    run_dir: str, config: dict, work_items: list[WorkItem], successful_files: list[str]
) -> list[str]:
    """
    Build a concrete, copy-pasteable list of shell steps to install and
    run whatever this run produced, by inspecting the actual manifest
    file on disk rather than assuming one exists. Returns an empty list
    if no recognized manifest was found (caller reports this as a gap,
    not silently).
    """
    manifest_path = _find_manifest_file(successful_files, NODE_MANIFEST_FILENAMES)
    if manifest_path is not None:
        directory = os.path.dirname(manifest_path)
        scripts = _read_npm_scripts(run_dir, manifest_path)
        run_script = _pick_run_script(scripts)
        uses_bun = any(
            "bun" in (wi.description + wi.title).lower() for wi in work_items
        )
        install_cmd = "bun install" if uses_bun else "npm install"
        run_cmd = (
            f"{'bun run' if uses_bun else 'npm run'} {run_script}"
            if run_script
            else f"{'bun run' if uses_bun else 'npm start'}"
        )
        steps = []
        if directory:
            steps.append(f"cd {directory}")
        steps.append(install_cmd)
        steps.append(run_cmd)
        return steps

    manifest_path = _find_manifest_file(successful_files, PYTHON_MANIFEST_FILENAMES)
    if manifest_path is not None:
        directory = os.path.dirname(manifest_path)
        steps = []
        if directory:
            steps.append(f"cd {directory}")
        if os.path.basename(manifest_path) == "requirements.txt":
            steps.append("pip install -r requirements.txt")
        else:
            steps.append("pip install .")
        entrypoint = next(
            (f for f in successful_files if os.path.basename(f) in ("main.py", "app.py")),
            None,
        )
        if entrypoint:
            entry_relative = os.path.relpath(entrypoint, directory) if directory else entrypoint
            steps.append(f"python {entry_relative}")
        else:
            steps.append("python main.py  # adjust to the actual entry-point file above")
        return steps

    return []


def generate_instructions_md(
    config: dict,
    work_items: list[WorkItem],
    agent_results: list[AgentResult],
    test_result: TestResult | None,
    run_status: str,
    run_id: str,
    run_dir: str | None = None,
) -> str:
    """
    Build the instructions.md content as a string (caller writes it to
    disk — see write_instructions_md).

    Args:
        config: Loaded pipeline config — used to list per-language test
            commands actually configured (not every language that
            exists in config, only ones this run's work_items used).
        work_items: The full WorkItem plan for this run.
        agent_results: All AgentResults produced across every dispatch
            round (initial + retries).
        test_result: The final TestResult, or None if the run never
            reached the test phase (e.g. MasterPlanError/CycleError).
        run_status: One of "passed", "passed_with_test_failures", "failed",
            "design_parse_error", "dispatch_cycle_error" — mirrors
            manifest["status"]/manifest["failure_reason"].
            "passed_with_test_failures" is the tests_blocking=false
            non-blocking-test-failure case (see pipeline/runner.py).
        run_id: The run directory's basename, for the header.
        run_dir: Absolute path to the run directory, needed to actually
            read a written package.json's "scripts" off disk. If None
            (e.g. a design_parse_error/dispatch_cycle_error run with no
            files written yet), manifest detection is skipped and the
            "no manifest found" path is reported instead.

    Returns:
        Markdown content as a string.
    """
    lines: list[str] = []
    lines.append(f"# How to Run This Product — {run_id}")
    lines.append("")

    lines.append("## Status")
    lines.append("")
    if run_status == "passed":
        lines.append("✅ All configured tests passed.")
    elif run_status == "passed_with_test_failures":
        total = test_result.total if test_result else 0
        failed = test_result.failed if test_result else 0
        lines.append(
            f"⚠️ Tests failed but were not enforced this run "
            f"(tests_blocking=false) — {failed} of {total} test(s) "
            "failed. Files were written and specialist dispatch "
            "succeeded, but automated test results should be treated "
            "as unverified/advisory only."
        )
    elif run_status == "failed":
        total = test_result.total if test_result else 0
        failed = test_result.failed if test_result else 0
        lines.append(
            f"❌ {failed} of {total} test(s) failed after exhausting "
            "retries (see tests/results/fail_report.json). Files below "
            "were still written — treat as unverified."
        )
    elif run_status == "design_parse_error":
        lines.append("❌ Planning failed — no code was generated this run.")
    elif run_status == "dispatch_cycle_error":
        lines.append(
            "❌ Dispatch failed (dependency cycle) — some files below may "
            "have been written before the error."
        )
    else:
        lines.append(f"Status: {run_status}")
    lines.append("")

    successful_files = _successful_files(agent_results)
    run_steps = (
        _build_run_steps(run_dir, config, work_items, successful_files)
        if run_dir is not None
        else []
    )

    lines.append("## Run It")
    lines.append("")
    if run_steps:
        lines.append("```bash")
        lines.extend(run_steps)
        lines.append("```")
    else:
        assembly_items = _find_assembly_work_items(work_items)
        if assembly_items:
            lines.append(
                "⚠️ No dependency manifest (package.json / requirements.txt "
                "/ pyproject.toml) was found among the files this run "
                "actually wrote, so there's no install/run command to "
                "give — even though the plan included an assembly step:"
            )
            lines.append("")
            for wi in assembly_items:
                lines.append(f"- {wi.title} → `{wi.output_path}`")
            lines.append("")
            lines.append(
                "Check whether that WorkItem's specialist run failed or "
                "wrote to an unexpected path (see files list below)."
            )
        else:
            lines.append(
                "⚠️ This run's plan had no assembly/entry-point WorkItem "
                "at all — the files below are individual modules, not a "
                "runnable application. There is no single install/run "
                "command."
            )
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
    languages_config = config.get("languages", {})
    if languages:
        lines.append("## Running Tests")
        lines.append("")
        for language in languages:
            lang_config = languages_config.get(language)
            if lang_config is None:
                continue
            command = " ".join(lang_config.get("test_command", []))
            framework = lang_config.get("test_framework_name", "?")
            lines.append(f"- **{language}** (`{framework}`): `{command}`")
        lines.append("")

    return "\n".join(lines)


def write_instructions_md(
    run_dir: str,
    config: dict,
    work_items: list[WorkItem],
    agent_results: list[AgentResult],
    test_result: TestResult | None,
    run_status: str,
) -> str:
    """
    Generate and write instructions.md to run_dir. Returns the full path
    written.
    """
    run_id = os.path.basename(run_dir.rstrip(os.sep))
    content = generate_instructions_md(
        config, work_items, agent_results, test_result, run_status, run_id,
        run_dir=run_dir,
    )
    path = os.path.join(run_dir, "instructions.md")
    with open(path, "w") as f:
        f.write(content)
    return path
