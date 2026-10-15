"""
specialists/base.py

Shared multi-file task execution helper. Category workers delegate to
run_multifile_specialist() so output ownership and validation stay uniform.

Self-check loop: after generating and writing its output, a specialist
verifies its OWN work against the WorkItem's acceptance_criteria (and
dependency_context, if given) via a SEPARATE self-check LLM call in the
same pooled session. On rejection, the specialist regenerates with the
    self-check's issues appended as feedback, up to SELF_CHECK_MAX_ATTEMPTS
extra attempts. This is entirely contained within one run_multifile_specialist()
call — callers (specialists/*.py's execute(),
and in turn pipeline/dispatch.py) see a single call in, a single
AgentResult out, same as before this loop existed.
"""

import json
import os
import re

from pipeline.state import AgentResult, ReviewResult, TestFailure, WorkItem
from pipeline.task_validation import normalize_relative_path, truncate_with_marker
from specialists.llm_client import call_llm

MAX_RETRY_FAILURES_SHOWN = 5
MAX_ERROR_OUTPUT_CHARS_IN_PROMPT = 500
MAX_DEPENDENCY_FILES_SHOWN = 20
MAX_DEPENDENCY_CHARS_IN_PROMPT = 3000
MAX_DEPENDENCY_TOTAL_CHARS_IN_PROMPT = 30000

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


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _has_symlink_component(root: str, relative_path: str) -> bool:
    current = root
    for component in relative_path.split(os.sep):
        current = os.path.join(current, component)
        if os.path.islink(current):
            return True
    return False

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
    blocks: list[str] = []
    omitted: list[str] = []
    total_chars = 0
    for path, content in dependency_context.items():
        if len(blocks) >= MAX_DEPENDENCY_FILES_SHOWN or total_chars >= MAX_DEPENDENCY_TOTAL_CHARS_IN_PROMPT:
            omitted.append(path)
            continue
        remaining = MAX_DEPENDENCY_TOTAL_CHARS_IN_PROMPT - total_chars
        shown_content = truncate_with_marker(
            content,
            min(MAX_DEPENDENCY_CHARS_IN_PROMPT, remaining),
            "\n... (truncated by dependency context budget)",
        )
        blocks.append(f"--- {path} ---\n{shown_content}")
        total_chars += len(shown_content)
    if omitted:
        blocks.append(
            f"({len(omitted)} dependency file(s) omitted by prompt budget: "
            + ", ".join(omitted)
            + ")"
        )
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
        f"Task: {work_item.id} ({work_item.category})",
        f"Module: {work_item.module}",
        f"Scope: {work_item.scope}",
        f"Title: {work_item.title}",
        f"Description: {work_item.description}",
        "Acceptance criteria:",
    ]
    lines.extend(f"- {c}" for c in work_item.acceptance_criteria)
    lines.append(f"Requirement references: {', '.join(work_item.requirement_refs) or '(none)'}")
    lines.extend(f"Expected interface: {contract}" for contract in work_item.expected_interfaces)

    lines.append(
        "\n--- Dependency context (already-implemented files this WorkItem depends on) ---"
    )
    if dependency_context:
        total_chars = 0
        shown_count = 0
        omitted: list[str] = []
        for path, content in dependency_context.items():
            if shown_count >= MAX_DEPENDENCY_FILES_SHOWN or total_chars >= MAX_DEPENDENCY_TOTAL_CHARS_IN_PROMPT:
                omitted.append(path)
                continue
            remaining = MAX_DEPENDENCY_TOTAL_CHARS_IN_PROMPT - total_chars
            shown = content[:min(MAX_DEPENDENCY_CHARS_IN_PROMPT, remaining)]
            if len(shown) < len(content):
                shown += "\n... (truncated by self-check dependency budget)"
            lines.append(f"\n--- {path} ---\n{shown}")
            total_chars += len(shown)
            shown_count += 1
        if omitted:
            lines.append("\nDependency files omitted by self-check budget: " + ", ".join(omitted))
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


def _build_base_user_content(work_item: WorkItem, config: dict) -> str:
    files = []
    for owned_file in work_item.owned_files:
        area_key = "tests_dir" if owned_file.area == "tests" else "output_dir"
        configured_dir = normalize_relative_path(config["pipeline"][area_key])
        path = normalize_relative_path(owned_file.path)
        prefix = configured_dir + "/"
        if path.startswith(prefix):
            path = path[len(prefix):]
        files.append(f"- {owned_file.area}/{path} ({owned_file.language})")
    return (
        f"Module: {work_item.module}\n"
        f"Scope: {work_item.scope}\n"
        f"Title: {work_item.title}\n"
        f"Primary language: {work_item.language}\n"
        f"Description: {work_item.description}\n"
        f"Requirement references: {', '.join(work_item.requirement_refs) or '(none)'}\n"
        f"Acceptance criteria:\n"
        + "\n".join(f"- {c}" for c in work_item.acceptance_criteria)
        + f"\nExpected interfaces:\n"
        + ("\n".join(f"- {contract}" for contract in work_item.expected_interfaces) or "- (none specified)")
        + f"\nOwned files (create exactly this set; do not repeat the area directory prefix):\n"
        + "\n".join(files)
    )


def run_multifile_specialist(
    work_item: WorkItem,
    config: dict,
    run_dir: str,
    system_prompt: str,
    agent_name: str,
    retry_failures: list[TestFailure] | None = None,
    dependency_context: dict[str, str] | None = None,
) -> AgentResult:
    """
    Generate the exact set of files owned by one category task.

    The LLM returns a JSON object mapping relative file paths to complete
    file contents:

        {"files": {"package.json": "...", "src/server/index.ts": "..."}}

    Every response path must exactly match one of the WorkItem's owned
    files. Each file is written under its configured product/tests area.

    After a successful write, a separate self-check LLM call verifies the written files against
    acceptance_criteria/dependency_context, regenerating (all files, from
    the same original prompt plus self-check feedback) up to
    SELF_CHECK_MAX_ATTEMPTS extra times on rejection.

    The system_prompt instructs the model to output the `{"files": {...}}` shape.

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

    try:
        base_user_content = _build_base_user_content(work_item, config)
    except (KeyError, TypeError, ValueError) as exc:
        return AgentResult(
            work_item_id=work_item.id,
            agent_name=agent_name,
            success=False,
            files_written=[],
            notes=f"Invalid task file ownership: {exc}",
        )
    if retry_failures:
        base_user_content += _build_retry_section(retry_failures)
    if dependency_context:
        base_user_content += _build_dependency_section(dependency_context)

    pipeline_config = config["pipeline"]
    run_root = os.path.realpath(run_dir)
    owned_paths: dict[str, tuple[str, str, str]] = {}
    output_aliases: dict[str, str] = {}
    for owned_file in work_item.owned_files:
        area_key = "tests_dir" if owned_file.area == "tests" else "output_dir"
        try:
            configured_dir = normalize_relative_path(pipeline_config[area_key])
            path = normalize_relative_path(owned_file.path)
        except (KeyError, TypeError, ValueError) as exc:
            return AgentResult(
                work_item_id=work_item.id,
                agent_name=agent_name,
                success=False,
                files_written=[],
                notes=f"Invalid output ownership configuration: {exc}",
            )
        prefix = configured_dir + "/"
        if path.startswith(prefix):
            path = path[len(prefix):]
        try:
            path = normalize_relative_path(path)
        except ValueError as exc:
            return AgentResult(
                work_item_id=work_item.id,
                agent_name=agent_name,
                success=False,
                files_written=[],
                notes=f"Invalid owned path {owned_file.path!r}: {exc}",
            )
        key = path.casefold()
        if key in owned_paths:
            return AgentResult(
                work_item_id=work_item.id,
                agent_name=agent_name,
                success=False,
                files_written=[],
                notes=f"Duplicate normalized path in task ownership: {owned_file.path!r}",
            )
        area_dir = os.path.realpath(os.path.join(run_root, configured_dir))
        if (
            os.path.commonpath((run_root, area_dir)) != run_root
            or _has_symlink_component(run_root, configured_dir)
        ):
            return AgentResult(
                work_item_id=work_item.id,
                agent_name=agent_name,
                success=False,
                files_written=[],
                notes=f"Configured output directory escapes the run: {configured_dir!r}",
            )
        owned_paths[key] = (
            os.path.join(configured_dir, path),
            os.path.join(area_dir, path),
            area_dir,
        )
        output_aliases[normalize_relative_path(owned_file.path).casefold()] = key
        output_aliases[(configured_dir + "/" + path).casefold()] = key

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
            parsed = json.loads(content, object_pairs_hook=_unique_json_object)
        except (json.JSONDecodeError, ValueError) as e:
            return AgentResult(
                work_item_id=work_item.id,
                agent_name=agent_name,
                success=False,
                files_written=[],
                notes=f"Failed to parse multi-file JSON response: {e}\nRaw response: {raw}",
            )

        files = parsed.get("files") if isinstance(parsed, dict) and set(parsed) == {"files"} else None
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
                    '{"files": {path: content}} with string '
                    f"values.\nRaw response: {raw}"
                ),
            )

        normalized_files: dict[str, str] = {}
        for path, file_content in files.items():
            try:
                normalized = normalize_relative_path(path)
            except ValueError as exc:
                return AgentResult(
                    work_item_id=work_item.id,
                    agent_name=agent_name,
                    success=False,
                    files_written=[],
                    notes=f"Unsafe output path {path!r}: {exc}",
                )
            key = output_aliases.get(normalized.casefold(), normalized.casefold())
            if key in normalized_files:
                return AgentResult(
                    work_item_id=work_item.id,
                    agent_name=agent_name,
                    success=False,
                    files_written=[],
                    notes=f"Duplicate normalized output path: {path!r}",
                )
            normalized_files[key] = file_content

        expected = set(owned_paths)
        actual = set(normalized_files)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra or not files:
            return AgentResult(
                work_item_id=work_item.id,
                agent_name=agent_name,
                success=False,
                files_written=[],
                notes=f"Output file set mismatch: missing={missing}, extra={extra}",
            )
        empty = sorted(path for path, value in normalized_files.items() if not value.strip())
        if empty:
            return AgentResult(
                work_item_id=work_item.id,
                agent_name=agent_name,
                success=False,
                files_written=[],
                notes=f"Output contains empty file content: {empty}",
            )

        files_written: list[str] = []
        written_content: dict[str, str] = {}
        write_targets: list[tuple[str, str, str, str]] = []
        for key, file_content in normalized_files.items():
            relative_path, full_path, area_root = owned_paths[key]
            resolved_target = os.path.realpath(full_path)
            area_relative_path = os.path.relpath(full_path, area_root)
            if (
                os.path.commonpath((area_root, resolved_target)) != area_root
                or _has_symlink_component(area_root, area_relative_path)
            ):
                return AgentResult(
                    work_item_id=work_item.id,
                    agent_name=agent_name,
                    success=False,
                    files_written=[],
                    notes=f"Output path escapes or aliases its configured area: {relative_path}",
                )
            write_targets.append((relative_path, full_path, area_root, file_content))
            written_content[relative_path] = file_content

        try:
            for relative_path, full_path, area_root, file_content in write_targets:
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                resolved_parent = os.path.realpath(os.path.dirname(full_path))
                area_relative_path = os.path.relpath(full_path, area_root)
                if (
                    os.path.commonpath((area_root, resolved_parent)) != area_root
                    or _has_symlink_component(area_root, area_relative_path)
                ):
                    raise OSError(f"Output parent escapes or aliases configured area: {relative_path}")
                with open(full_path, "w") as output_file:
                    output_file.write(file_content)
                files_written.append(relative_path)
        except OSError as exc:
            return AgentResult(
                work_item_id=work_item.id,
                agent_name=agent_name,
                success=False,
                files_written=files_written,
                notes=f"Failed to write owned output: {exc}",
            )

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
