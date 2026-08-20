"""
specialists/base.py

Shared specialist execution helper. All four specialists (Logic, UI,
Config, Test) delegate to run_specialist() instead of duplicating the
LLM-call + parse + write logic.
"""

import os
import re

from pipeline.state import AgentResult, TestFailure, WorkItem
from specialists.llm_client import call_llm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAX_RETRY_FAILURES_SHOWN = 5
MAX_ERROR_OUTPUT_CHARS_IN_PROMPT = 500
MAX_DEPENDENCY_FILES_SHOWN = 5
MAX_DEPENDENCY_CHARS_IN_PROMPT = 3000


def _strip_code_fences(raw: str) -> str:
    """Strip a leading/trailing markdown code fence line if present, keeping
    everything between. Handles ```python ... ``` and ``` ... ```."""
    stripped = raw.strip()

    fence_match = re.match(r"^```[a-zA-Z0-9_+-]*\s*\n(.*?)\n?```$", stripped, re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    return stripped


def _build_retry_section(retry_failures: list[TestFailure]) -> str:
    """Build the '\\n\\nThis is a retry...' prompt section from the
    failures that implicated this WorkItem. Truncates each failure's
    error_output and caps the number of failures shown, to avoid
    bloating context."""
    shown = retry_failures[:MAX_RETRY_FAILURES_SHOWN]
    lines = [
        f"- {failure.test_name}: "
        f"{failure.error_output[:MAX_ERROR_OUTPUT_CHARS_IN_PROMPT]}"
        for failure in shown
    ]
    return (
        "\n\nThis is a retry. The previous implementation failed these "
        "tests:\n"
        + "\n".join(lines)
        + "\n\nFix the implementation so these tests pass, while still "
        "satisfying all acceptance criteria above."
    )


def _build_dependency_section(dependency_context: dict[str, str]) -> str:
    """Build the '\\n\\nThis WorkItem depends on...' prompt section from
    already-implemented dependency file contents. Caps the number of
    files shown and truncates each to avoid bloating context."""
    shown = list(dependency_context.items())[:MAX_DEPENDENCY_FILES_SHOWN]
    blocks = [
        f"--- {path} ---\n{content[:MAX_DEPENDENCY_CHARS_IN_PROMPT]}"
        for path, content in shown
    ]
    return (
        "\n\nThis WorkItem depends on the following already-implemented "
        "files. Use their actual function/class names, signatures, and "
        "exports — do not invent different ones:\n\n"
        + "\n\n".join(blocks)
    )


def run_specialist(
    work_item: WorkItem,
    config: dict,
    run_dir: str,
    system_prompt: str,
    agent_name: str,
    subdir_key: str,
    retry_failures: list[TestFailure] | None = None,
    dependency_context: dict[str, str] | None = None,
) -> AgentResult:
    """
    Call the LLM for a single WorkItem, write its output file, and return
    an AgentResult describing what happened.

    Args:
        work_item: The WorkItem to implement.
        config: Loaded config dict.
        run_dir: Path to this pipeline run's output directory (see
            pipeline/run_paths.py). Output files are written under
            run_dir/{subdir}/... instead of PROJECT_ROOT.
        system_prompt: Specialist-specific system prompt template. May
            contain a `{language}` placeholder, which is filled in with
            work_item.language. If the prompt has already been fully
            formatted (no placeholders left, e.g. TestAgent pre-formats
            both {language} and {test_framework} itself), formatting here
            is a no-op.
        agent_name: Name to record in the returned AgentResult.
        subdir_key: Which config["pipeline"][...] key to resolve the output
            directory from ("output_dir" or "tests_dir").
        retry_failures: TestFailures (if any) that implicated this
            WorkItem on a previous attempt. When non-empty, a labeled
            section is appended to the user prompt describing what
            broke, so the LLM has retry context. None/empty for the
            initial (non-retry) dispatch.
        dependency_context: Optional map of output_path -> file content
            for this WorkItem's already-implemented dependencies (see
            pipeline/dispatch.py's build_dependency_context()). When
            non-empty, a separate labeled section is appended to the
            user prompt so the LLM writes against the real dependency
            API instead of inventing one. Composed independently of the
            retry-failure section above — both may be present at once.

    Returns:
        AgentResult with success=True and files_written populated on
        success, or success=False and notes describing the failure.
    """
    model = config["models"]["specialist"]
    formatted_system_prompt = system_prompt.format(language=work_item.language)
    user_content = (
        f"Title: {work_item.title}\n"
        f"Language: {work_item.language}\n"
        f"Description: {work_item.description}\n"
        f"Acceptance criteria:\n"
        + "\n".join(f"- {c}" for c in work_item.acceptance_criteria)
        + f"\nOutput path: {work_item.output_path}"
    )
    if retry_failures:
        user_content += _build_retry_section(retry_failures)
    if dependency_context:
        user_content += _build_dependency_section(dependency_context)
    messages = [
        {"role": "system", "content": formatted_system_prompt},
        {"role": "user", "content": user_content},
    ]

    raw = call_llm(messages, model, config)

    if raw.startswith(f"[{model}] Error:"):
        return AgentResult(
            work_item_id=work_item.id,
            agent_name=agent_name,
            success=False,
            files_written=[],
            notes=raw,
        )

    content = _strip_code_fences(raw)

    subdir = config["pipeline"][subdir_key]
    relative_path = os.path.join(subdir, work_item.output_path)
    full_path = os.path.join(run_dir, subdir, work_item.output_path)

    try:
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
    except OSError as e:
        return AgentResult(
            work_item_id=work_item.id,
            agent_name=agent_name,
            success=False,
            files_written=[],
            notes=f"Failed to write file at {full_path}: {e}",
        )

    return AgentResult(
        work_item_id=work_item.id,
        agent_name=agent_name,
        success=True,
        files_written=[relative_path],
        notes="",
    )
