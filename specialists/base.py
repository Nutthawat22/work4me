"""
specialists/base.py

Shared specialist execution helper. All eight specialists (Logic, UI,
Config, Test, Scaffold, Integrate, Feature, Shared) delegate to
run_specialist()/run_multifile_specialist() instead of duplicating the
LLM-call + parse + write logic.

Self-check loop: after generating and writing its output, a specialist
verifies its OWN work against the WorkItem's acceptance_criteria (and
dependency_context, if given) via a SEPARATE self-check LLM call in the
same pooled session. On rejection, the specialist regenerates with the
self-check's issues appended as feedback, up to SELF_CHECK_MAX_ATTEMPTS
extra attempts. This is entirely contained within one run_specialist()/
run_multifile_specialist() call — callers (specialists/*.py's execute(),
and in turn pipeline/dispatch.py) see a single call in, a single
AgentResult out, same as before this loop existed.
"""

import json
import os
import re

from pipeline.state import AgentResult, ReviewResult, TestFailure, WorkItem
from specialists.llm_client import call_llm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAX_RETRY_FAILURES_SHOWN = 5
MAX_ERROR_OUTPUT_CHARS_IN_PROMPT = 500
MAX_DEPENDENCY_FILES_SHOWN = 5
MAX_DEPENDENCY_CHARS_IN_PROMPT = 3000

# Timeout for every call_llm() in this module (generation AND self-check
# — see _self_check's own call below). The old implicit 60s default
# (specialists/llm_client.py's call_llm(timeout: int = 60)) is far too
# short for real code-generation: a live production handoff-mode run hit
# ALL 25 work items timing out identically at 60s, writing zero files.
# 180s mirrors pipeline/master.py's own PLAN_HANDOFF_TIMEOUT_SECONDS /
# DECOMPOSE_FEATURES_TIMEOUT_SECONDS (240s) and pipeline/file_manifest.py's
# PLAN_FILE_MANIFEST_TIMEOUT_SECONDS (240s) precedent of hardcoded
# module-level timeout constants rather than config-driven values — kept
# slightly lower than those since a single-file/small-multi-file
# specialist call is normally cheaper than whole-project planning, but
# still 3x the old default. Self-check reuses this same constant rather
# than a shorter one: its output (a small JSON verdict) is cheap, but its
# input is NOT (full acceptance criteria + dependency context + the
# specialist's own written files, same order of magnitude as a
# generation call) and LLM latency here is dominated by upstream
# think-time, not output size — a too-short self-check timeout would
# silently fail open (see _self_check's None-on-error handling) and
# disable the review loop under the same load that caused this bug in
# the first place. Revisit if self-check timeouts turn out to be rare
# in practice.
SPECIALIST_TIMEOUT_SECONDS = 180

# Self-check attempt cap: up to this many EXTRA generation attempts
# beyond the first on self-check rejection, so 3 total generation
# attempts max per work item — mirrors the bounded-retry spirit of
# MAX_RETRY_FAILURES_SHOWN etc. above (small and capped, never
# unbounded).
SELF_CHECK_MAX_ATTEMPTS = 2

# Caps for what gets embedded in the self-check prompt — mirrors
# MAX_DEPENDENCY_CHARS_IN_PROMPT's truncation policy, applied to the
# specialist's OWN just-written output instead.
MAX_SELF_CHECK_OUTPUT_CHARS = 6000
MAX_SELF_CHECK_FILES_SHOWN = 8
MAX_SELF_CHECK_ISSUES_SHOWN = 10

SELF_CHECK_SYSTEM_PROMPT = (
    "You are reviewing your OWN previous output for a work item, before "
    "it is finalized. This is a fast sanity-and-interoperability check, "
    "not a full test run. Check exactly two things: (1) does the output "
    "plausibly satisfy the stated acceptance criteria, and (2) if "
    "dependency files are shown, does the output correctly use what they "
    "actually expose (do imports/function names/class names/exported "
    "symbols/API endpoints match, or does the output invent names that "
    "don't exist in the dependencies shown)? Do not reject for stylistic "
    "preferences or issues unrelated to these two checks. Output ONLY a "
    'JSON object of this exact shape: {"accepted": true|false, "issues": '
    '["specific issue 1", ...], "reasoning": "one paragraph explaining '
    'the verdict"} — no markdown fences, no other text. "issues" must be '
    "empty if accepted, and each issue must be concrete/actionable if not."
)


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


def _build_self_check_feedback_section(self_check: ReviewResult) -> str:
    """Build the '\\n\\nYour own self-check...' prompt section from a
    self-check rejection, to append when regenerating. Mirrors
    _build_retry_section's shape/tone, but for a self-check rejection
    instead of a post-test failure."""
    shown = self_check.issues[:MAX_SELF_CHECK_ISSUES_SHOWN]
    lines = [f"- {issue}" for issue in shown]
    body = "\n".join(lines) if lines else f"- {self_check.reasoning}"
    return (
        "\n\nYour own self-check of the previous attempt rejected it. "
        "Specific issues:\n"
        + body
        + f"\n\nSelf-check reasoning: {self_check.reasoning}"
        + "\n\nFix the implementation to address these issues, while "
        "still satisfying all acceptance criteria above."
    )


def _build_self_check_user_content(
    work_item: WorkItem,
    written_content: dict[str, str],
    dependency_context: dict[str, str] | None,
) -> str:
    """Build the self-check call's user-turn prompt: the WorkItem's own
    spec, the dependency context the specialist was given (if any), and
    what the specialist just wrote — everything the two self-check
    questions (acceptance criteria, interoperability) need to judge
    against."""
    lines = [
        f"WorkItem: {work_item.id} ({work_item.type})",
        f"Title: {work_item.title}",
        f"Description: {work_item.description}",
        "Acceptance criteria:",
    ]
    lines.extend(f"- {c}" for c in work_item.acceptance_criteria)

    lines.append(
        "\n--- Dependency context (already-implemented files this WorkItem depends on) ---"
    )
    if dependency_context:
        for path, content in dependency_context.items():
            lines.append(f"\n--- {path} ---\n{content[:MAX_DEPENDENCY_CHARS_IN_PROMPT]}")
    else:
        lines.append("(none — this WorkItem has no dependencies)")

    lines.append("\n--- What you just wrote (under self-check) ---")
    shown = list(written_content.items())[:MAX_SELF_CHECK_FILES_SHOWN]
    for path, content in shown:
        lines.append(f"\n--- {path} ---\n{content[:MAX_SELF_CHECK_OUTPUT_CHARS]}")

    return "\n".join(lines)


def _build_self_check_response_schema() -> dict:
    """Build a prompt-embedded structured-output schema for the
    self-check's accept/reject verdict. Deliberately simple, same shape
    as ReviewResult (accepted/issues/reasoning)."""
    return {
        "name": "self_check",
        "schema": {
            "type": "object",
            "properties": {
                "accepted": {"type": "boolean"},
                "issues": {"type": "array", "items": {"type": "string"}},
                "reasoning": {"type": "string"},
            },
            "required": ["accepted", "issues", "reasoning"],
            "additionalProperties": False,
        },
    }


def _parse_self_check_json(raw: str) -> dict:
    """Parse raw into a self-check verdict dict, stripping a ```json
    fence if present. Raises ValueError (not MasterPlanError — this
    module has no dependency on pipeline.master) on any parse problem;
    _self_check's caller treats a ValueError here as fail-open."""
    stripped = raw.strip()

    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fence_match:
        stripped = fence_match.group(1).strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse self-check response as JSON: {e}\nRaw response: {raw}")

    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object for a self-check verdict, got: {type(parsed).__name__}")

    return parsed


def _validate_and_build_self_check(parsed: dict) -> ReviewResult:
    """Validate a parsed self-check verdict dict's required keys/types
    and construct a ReviewResult, raising ValueError on problems."""
    required = {"accepted", "issues", "reasoning"}
    missing = required - parsed.keys()
    if missing:
        raise ValueError(f"Self-check verdict missing required fields: {sorted(missing)}")

    if not isinstance(parsed["accepted"], bool):
        raise ValueError("Self-check verdict 'accepted' must be a boolean")

    if not isinstance(parsed["issues"], list) or not all(
        isinstance(i, str) for i in parsed["issues"]
    ):
        raise ValueError("Self-check verdict 'issues' must be a list of strings")

    if not isinstance(parsed["reasoning"], str):
        raise ValueError("Self-check verdict 'reasoning' must be a string")

    return ReviewResult(
        accepted=parsed["accepted"],
        issues=parsed["issues"],
        reasoning=parsed["reasoning"],
    )


def _self_check(
    work_item: WorkItem,
    written_content: dict[str, str],
    dependency_context: dict[str, str] | None,
    config: dict,
    run_dir: str,
) -> ReviewResult | None:
    """
    Ask the specialist role to verify its OWN just-written output against
    work_item.acceptance_criteria and dependency_context, in a SEPARATE
    LLM call reusing the SAME pooled session as generation (same
    session_scope: {"project": ..., "role": "specialist"}).

    Returns:
        A ReviewResult with the accept/reject verdict, or None
        (fail-open) if the self-check call itself errors out or its
        response can't be parsed as a valid verdict — a broken
        self-check call shouldn't block an otherwise-successful
        generation from being accepted.
    """
    model_cfg = config["models"]["specialist"]
    model = model_cfg["model"]
    provider = model_cfg["provider"]

    user_content = _build_self_check_user_content(work_item, written_content, dependency_context)
    messages = [
        {"role": "system", "content": SELF_CHECK_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    response_schema = _build_self_check_response_schema()

    raw = call_llm(
        messages, model, config, provider=provider,
        timeout=SPECIALIST_TIMEOUT_SECONDS,
        response_schema=response_schema,
        session_scope={"project": os.path.basename(run_dir.rstrip(os.sep)), "role": "specialist"},
    )

    if raw.startswith(f"[{model}] Error:"):
        return None

    try:
        return _validate_and_build_self_check(_parse_self_check_json(raw))
    except ValueError:
        return None


def _build_base_user_content(work_item: WorkItem) -> str:
    return (
        f"Title: {work_item.title}\n"
        f"Language: {work_item.language}\n"
        f"Description: {work_item.description}\n"
        f"Acceptance criteria:\n"
        + "\n".join(f"- {c}" for c in work_item.acceptance_criteria)
        + f"\nOutput path: {work_item.output_path}"
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
    Call the LLM for a single WorkItem, write its output file, self-check
    the result against the WorkItem's acceptance_criteria/
    dependency_context, and return an AgentResult describing what
    happened.

    On self-check rejection, regenerates (re-running the same LLM call
    with the self-check's issues appended as feedback) up to
    SELF_CHECK_MAX_ATTEMPTS extra times before giving up and returning
    success=False with the self-check's final rejection reasoning in
    notes.

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
            WorkItem on a previous attempt (from the SEPARATE post-test
            retry loop in pipeline/runner.py — orthogonal to this
            function's own internal self-check loop). When non-empty, a
            labeled section is appended to the user prompt describing
            what broke, so the LLM has retry context. None/empty for the
            initial (non-retry) dispatch.
        dependency_context: Optional map of output_path -> file content
            for this WorkItem's already-implemented dependencies (see
            pipeline/dispatch.py's build_dependency_context()). When
            non-empty, a separate labeled section is appended to the
            user prompt so the LLM writes against the real dependency
            API instead of inventing one, and the self-check loop also
            uses it to check interoperability. Composed independently of
            the retry-failure section above — both may be present at
            once.

    Returns:
        AgentResult with success=True and files_written populated once
        self-checked and accepted, or success=False and notes describing
        the failure (LLM/write error, or self-check rejection exhausting
        all attempts).
    """
    model_cfg = config["models"]["specialist"]
    model = model_cfg["model"]
    provider = model_cfg["provider"]
    formatted_system_prompt = system_prompt.format(language=work_item.language)

    base_user_content = _build_base_user_content(work_item)
    if retry_failures:
        base_user_content += _build_retry_section(retry_failures)
    if dependency_context:
        base_user_content += _build_dependency_section(dependency_context)

    subdir = config["pipeline"][subdir_key]
    output_path = work_item.output_path
    if subdir_key == "tests_dir":
        # TestAgent (the only subdir_key="tests_dir" caller) prompts the
        # LLM with the WorkItem's output_path but no explicit instruction
        # on whether it's relative to run_dir or to tests_dir itself —
        # in practice the LLM (mirroring how plan.json's own test
        # WorkItems name their output_path, e.g. "tests/foo.test.js")
        # consistently includes a leading "{tests_dir}/" prefix. Since
        # this path is then joined with subdir=tests_dir below, keeping
        # that prefix would double it up into
        # run_dir/tests/tests/foo.test.js. Strip one matching leading
        # prefix here so output_path is always treated as relative to
        # tests_dir directly, same convention as output_dir-relative
        # WorkItems (ui/logic/config/scaffold/integrate) already use.
        normalized = output_path.replace("\\", "/")
        prefix = subdir.rstrip("/") + "/"
        if normalized.startswith(prefix):
            output_path = normalized[len(prefix):]
    relative_path = os.path.join(subdir, output_path)
    full_path = os.path.join(run_dir, subdir, output_path)

    self_check_feedback: ReviewResult | None = None

    for attempt in range(SELF_CHECK_MAX_ATTEMPTS + 1):
        user_content = base_user_content
        if self_check_feedback is not None:
            user_content += _build_self_check_feedback_section(self_check_feedback)

        messages = [
            {"role": "system", "content": formatted_system_prompt},
            {"role": "user", "content": user_content},
        ]

        raw = call_llm(
            messages, model, config, provider=provider,
            timeout=SPECIALIST_TIMEOUT_SECONDS,
            session_scope={"project": os.path.basename(run_dir.rstrip(os.sep)), "role": "specialist"},
        )

        if raw.startswith(f"[{model}] Error:"):
            return AgentResult(
                work_item_id=work_item.id,
                agent_name=agent_name,
                success=False,
                files_written=[],
                notes=raw,
            )

        content = _strip_code_fences(raw)

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

        check = _self_check(work_item, {relative_path: content}, dependency_context, config, run_dir)

        if check is None or check.accepted:
            return AgentResult(
                work_item_id=work_item.id,
                agent_name=agent_name,
                success=True,
                files_written=[relative_path],
                notes="",
            )

        if attempt == SELF_CHECK_MAX_ATTEMPTS:
            issues_text = "; ".join(check.issues) if check.issues else check.reasoning
            return AgentResult(
                work_item_id=work_item.id,
                agent_name=agent_name,
                success=False,
                files_written=[],
                notes=(
                    f"Self-check rejected after {attempt + 1} attempt(s): "
                    f"{issues_text}. Reasoning: {check.reasoning}"
                ),
            )

        print(
            f"  🔎 [{work_item.id}] self-check rejected — regenerating "
            f"(attempt {attempt + 2}/{SELF_CHECK_MAX_ATTEMPTS + 1})..."
        )
        self_check_feedback = check

    # Unreachable: the loop above always returns on its final iteration.
    raise AssertionError("run_specialist's self-check loop exited without returning")


def run_multifile_specialist(
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
    Multi-file variant of run_specialist for the scaffold/integrate/
    feature/shared specialists.

    Unlike run_specialist (one WorkItem -> exactly one output file), the
    scaffold/integrate/feature/shared stages each need to emit a whole
    set of files at once. Rather than forcing one file per WorkItem, this
    helper expects the LLM to return a JSON object mapping relative file
    paths to complete file contents:

        {"files": {"package.json": "...", "src/server/index.ts": "..."}}

    A bare {path: content} object (no "files" wrapper) is also accepted.
    Every file is written under run_dir/{subdir}/{path}, and all relative
    paths are collected into the returned AgentResult.files_written.

    Same self-check loop as run_specialist: after a successful write, a
    separate self-check LLM call verifies the written files against
    acceptance_criteria/dependency_context, regenerating (all files, from
    the same original prompt plus self-check feedback) up to
    SELF_CHECK_MAX_ATTEMPTS extra times on rejection.

    Args mirror run_specialist exactly (minus review_feedback, which
    never existed on this function). The system_prompt is expected to
    already instruct the model to output the {"files": {...}} shape; it
    is still `.format(language=...)`-ed here for consistency (a no-op if
    the prompt is already fully formatted).

    Returns:
        AgentResult with success=True and files_written populated once
        self-checked and accepted, or success=False and notes describing
        the failure (LLM error, JSON parse error, wrong shape, write
        error, or self-check rejection exhausting all attempts).
    """
    model_cfg = config["models"]["specialist"]
    model = model_cfg["model"]
    provider = model_cfg["provider"]
    # .replace (not .format): the prompt contains literal JSON braces ({"files": ...}) that str.format would choke on.
    formatted_system_prompt = system_prompt.replace("{language}", work_item.language)

    base_user_content = _build_base_user_content(work_item)
    if retry_failures:
        base_user_content += _build_retry_section(retry_failures)
    if dependency_context:
        base_user_content += _build_dependency_section(dependency_context)

    subdir = config["pipeline"][subdir_key]

    self_check_feedback: ReviewResult | None = None

    for attempt in range(SELF_CHECK_MAX_ATTEMPTS + 1):
        user_content = base_user_content
        if self_check_feedback is not None:
            user_content += _build_self_check_feedback_section(self_check_feedback)

        messages = [
            {"role": "system", "content": formatted_system_prompt},
            {"role": "user", "content": user_content},
        ]

        raw = call_llm(
            messages, model, config, provider=provider,
            timeout=SPECIALIST_TIMEOUT_SECONDS,
            session_scope={"project": os.path.basename(run_dir.rstrip(os.sep)), "role": "specialist"},
        )

        if raw.startswith(f"[{model}] Error:"):
            return AgentResult(
                work_item_id=work_item.id,
                agent_name=agent_name,
                success=False,
                files_written=[],
                notes=raw,
            )

        content = _strip_code_fences(raw)

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            return AgentResult(
                work_item_id=work_item.id,
                agent_name=agent_name,
                success=False,
                files_written=[],
                notes=f"Failed to parse multi-file JSON response: {e}\nRaw response: {raw}",
            )

        if isinstance(parsed, dict) and "files" in parsed:
            files = parsed["files"]
        else:
            files = parsed

        if not isinstance(files, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in files.items()
        ):
            return AgentResult(
                work_item_id=work_item.id,
                agent_name=agent_name,
                success=False,
                files_written=[],
                notes=(
                    "Multi-file response has wrong shape — expected "
                    '{"files": {path: content}} or {path: content} with string '
                    f"values.\nRaw response: {raw}"
                ),
            )

        files_written: list[str] = []
        written_content: dict[str, str] = {}
        write_error: AgentResult | None = None
        for path, file_content in files.items():
            relative_path = os.path.join(subdir, path)
            full_path = os.path.join(run_dir, subdir, path)
            try:
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w") as f:
                    f.write(file_content)
            except OSError as e:
                write_error = AgentResult(
                    work_item_id=work_item.id,
                    agent_name=agent_name,
                    success=False,
                    files_written=[],
                    notes=f"Failed to write file at {full_path}: {e}",
                )
                break
            files_written.append(relative_path)
            written_content[relative_path] = file_content

        if write_error is not None:
            return write_error

        check = _self_check(work_item, written_content, dependency_context, config, run_dir)

        if check is None or check.accepted:
            return AgentResult(
                work_item_id=work_item.id,
                agent_name=agent_name,
                success=True,
                files_written=files_written,
                notes="",
            )

        if attempt == SELF_CHECK_MAX_ATTEMPTS:
            issues_text = "; ".join(check.issues) if check.issues else check.reasoning
            return AgentResult(
                work_item_id=work_item.id,
                agent_name=agent_name,
                success=False,
                files_written=[],
                notes=(
                    f"Self-check rejected after {attempt + 1} attempt(s): "
                    f"{issues_text}. Reasoning: {check.reasoning}"
                ),
            )

        print(
            f"  🔎 [{work_item.id}] self-check rejected — regenerating "
            f"(attempt {attempt + 2}/{SELF_CHECK_MAX_ATTEMPTS + 1})..."
        )
        self_check_feedback = check

    # Unreachable: the loop above always returns on its final iteration.
    raise AssertionError("run_multifile_specialist's self-check loop exited without returning")
