"""
spike/acp_json_compliance_spike.py

Phase 0 GATING spike (see design doc:
dev-plans/agents/features/2026-09-14-acp-native-master-architecture.md).
Standalone, throwaway. Does NOT modify acp_client.py, design.py, or any
other spike file.

Measures the REAL JSON schema-compliance rate of ACP's prompt-only
schema mechanism (specialists/providers/acp_client.py's acp_call() +
_build_schema_instruction() -- there is no token-constrained
structured-output enforcement over ACP, unlike chat_completions/
responses providers' response_format=json_schema,strict=true).

Target model: claude-sonnet-5. NOTE -- the design doc's own text
mentions "kimi-k2.7-code, gpt-5.6-luna" as example role models, but
those are placeholder names that only exist in config.example.json.
The REAL config that actually loads at runtime
(~/.config/Agents/config.json, which pipeline/runner.py's
load_config() checks before falling back to the repo's config.json)
has every role (code/default/master/design/specialist) pointed at
claude-sonnet-5. That's the model actually in production use, so
that's what this spike targets.

Reuses, unmodified:
  - acp_call() / AcpSessionPool (_POOL) / teardown() from
    specialists/providers/acp_client.py.
  - pipeline.design's REAL WorkItem/Feature response schemas
    (_build_work_items_response_schema, _build_features_response_schema),
    system prompt templates (SYSTEM_PROMPT_TEMPLATE,
    FEATURES_SYSTEM_PROMPT_TEMPLATE), and parse/validate static methods
    (DesignAgent._parse_json/_validate_and_build,
    DesignAgent._parse_features_json/_validate_and_build_features) --
    imported directly since pipeline/design.py imports cleanly without
    needing a live config at import time. (Its import chain does trigger
    specialists/providers/acp_client.py's `import acp`, so this script
    must be run with spike/venv's interpreter, which has
    agent-client-protocol installed -- see Run instructions below.)

For each of 2 representative schemas (WorkItem array via
DesignAgent.decompose()'s schema, Feature array via
DesignAgent.decompose_features()'s schema), runs N_TRIALS trials
against claude-sonnet-5 over ACP:

  1st attempt: acp_call(..., response_schema=<real schema>) with a
  realistic small prompt. Records: did the raw response parse as JSON
  at all, did it pass full schema/field validation, latency, and (on
  failure) the raw failure reason.

  On ANY 1st-attempt failure (parse OR validation), immediately
  re-invokes acp_call EXACTLY ONCE in the SAME pooled ACP session (same
  session_scope, so AcpSessionPool reuses the session) with the
  original prompt + schema instruction (added automatically by
  acp_call whenever response_schema= is passed) + the specific
  parse/validation error appended as an additional turn. Records
  whether that single retry recovers a valid parse+validation. This is
  an empirical test of the design doc's Option A (retry-with-error-
  feedback) recommendation.

  Each trial uses a unique session_scope role key so trials don't
  contaminate each other's conversation context, and the pooled
  session is torn down (via acp_client.teardown()) after each trial
  completes (including its retry, if any) to keep resource usage
  bounded to one live `opencode acp` subprocess at a time.

Prints a per-schema summary table: first-attempt parse-success rate,
first-attempt schema-validation-success rate, retry-recovery rate (of
trials that failed on the first attempt, what fraction did the single
retry-with-error-feedback fix), and average first-attempt latency.

Run:
    cd spike && venv/bin/python3 acp_json_compliance_spike.py
    # or: SPIKE_N_TRIALS=30 venv/bin/python3 acp_json_compliance_spike.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field

# pipeline/design.py and specialists/providers/acp_client.py live at the
# repo root (one level up from spike/); add it to sys.path so this
# script can `import pipeline.design` / `import specialists.providers...`
# regardless of the cwd it's invoked from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline.design import (  # noqa: E402
    DesignAgent,
    DesignParseError,
    FEATURES_SYSTEM_PROMPT_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE,
    _build_features_response_schema,
    _build_work_items_response_schema,
)
from specialists.providers.acp_client import acp_call, teardown as acp_teardown  # noqa: E402

REAL_CONFIG_PATH = os.path.expanduser("~/.config/Agents/config.json")
MODEL = "claude-sonnet-5"
N_TRIALS = int(os.environ.get("SPIKE_N_TRIALS", "20"))
TIMEOUT_SECONDS = 90

WORKITEM_USER_PROMPT = (
    "Decompose this small feature request into 2-3 WorkItems: "
    "'Add a health-check endpoint that returns server uptime'"
)

FEATURES_USER_PROMPT = (
    "Decompose the following small approved technical design excerpt into "
    "2-3 Features. (This is a condensed excerpt standing in for a full "
    "design document, for spike purposes.)\n\n"
    "System: a URL shortener service.\n"
    "- REQ-001: Users can submit a long URL and receive a short code.\n"
    "- REQ-002: Visiting a short code redirects to the original long URL.\n"
    "- REQ-003: The system tracks a visit count per short code.\n"
    "- WP-001: Implement the create-short-link API endpoint and storage.\n"
    "- WP-002: Implement the redirect-by-short-code endpoint with visit "
    "count increment.\n"
    "- TEST-001: Creating a link with a valid URL returns a unique short "
    "code.\n"
    "- TEST-002: Visiting a valid short code redirects with HTTP 302 and "
    "increments the visit count.\n"
)


def _load_real_config() -> dict:
    if os.path.isfile(REAL_CONFIG_PATH):
        with open(REAL_CONFIG_PATH) as f:
            return json.load(f)
    return {
        "litellm_url": os.environ.get("LITELLM_BASE_URL", "https://api.rctz.online/v1"),
        "litellm_key": os.environ.get("LITELLM_API_KEY", "sk-REPLACE-ME"),
        "languages": {"python": {}, "javascript": {}, "typescript": {}},
    }


@dataclass
class TrialResult:
    schema_name: str
    trial_idx: int
    first_latency: float
    first_parse_ok: bool
    first_validate_ok: bool
    first_error: str | None
    first_raw: str
    retried: bool = False
    retry_latency: float | None = None
    retry_parse_ok: bool | None = None
    retry_validate_ok: bool | None = None
    retry_error: str | None = None
    retry_raw: str | None = None


def _attempt_work_items(raw: str, valid_languages: set[str]) -> tuple[bool, bool, str | None]:
    try:
        parsed = DesignAgent._parse_json(raw)
    except DesignParseError as e:
        return False, False, f"PARSE_ERROR: {e}"
    try:
        DesignAgent._validate_and_build(parsed, valid_languages)
    except DesignParseError as e:
        return True, False, f"VALIDATE_ERROR: {e}"
    except Exception as e:  # defensive -- unexpected shape (e.g. non-dict items)
        return True, False, f"VALIDATE_ERROR (unexpected {type(e).__name__}): {e}"
    return True, True, None


def _attempt_features(raw: str, valid_languages: set[str]) -> tuple[bool, bool, str | None]:
    try:
        parsed = DesignAgent._parse_features_json(raw)
    except DesignParseError as e:
        return False, False, f"PARSE_ERROR: {e}"
    try:
        DesignAgent._validate_and_build_features(parsed, valid_languages)
    except DesignParseError as e:
        return True, False, f"VALIDATE_ERROR: {e}"
    except Exception as e:
        return True, False, f"VALIDATE_ERROR (unexpected {type(e).__name__}): {e}"
    return True, True, None


def run_schema_trials(
    schema_name: str,
    config: dict,
    valid_languages: set[str],
    system_prompt: str,
    user_prompt: str,
    response_schema: dict,
    attempt_fn,
    n_trials: int,
) -> list[TrialResult]:
    results: list[TrialResult] = []

    for i in range(n_trials):
        session_scope = {"project": "acp_json_compliance_spike", "role": f"{schema_name}_trial_{i}"}
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        t0 = time.monotonic()
        raw = acp_call(
            messages, MODEL, config, timeout=TIMEOUT_SECONDS,
            response_schema=response_schema, session_scope=session_scope,
        )
        first_latency = time.monotonic() - t0

        if raw.startswith(f"[{MODEL}] Error:"):
            parse_ok, validate_ok, err = False, False, f"ACP_CALL_ERROR: {raw}"
        else:
            parse_ok, validate_ok, err = attempt_fn(raw, valid_languages)

        result = TrialResult(
            schema_name=schema_name, trial_idx=i, first_latency=first_latency,
            first_parse_ok=parse_ok, first_validate_ok=validate_ok,
            first_error=err, first_raw=raw,
        )

        if not (parse_ok and validate_ok):
            retry_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {
                    "role": "user",
                    "content": (
                        "Your previous response failed with this error:\n"
                        f"{err}\n\n"
                        "Please provide a corrected response. Output ONLY "
                        "the corrected valid JSON, no other text."
                    ),
                },
            ]
            t1 = time.monotonic()
            retry_raw = acp_call(
                retry_messages, MODEL, config, timeout=TIMEOUT_SECONDS,
                response_schema=response_schema, session_scope=session_scope,
            )
            retry_latency = time.monotonic() - t1

            if retry_raw.startswith(f"[{MODEL}] Error:"):
                retry_parse_ok, retry_validate_ok, retry_err = False, False, f"ACP_CALL_ERROR: {retry_raw}"
            else:
                retry_parse_ok, retry_validate_ok, retry_err = attempt_fn(retry_raw, valid_languages)

            result.retried = True
            result.retry_latency = retry_latency
            result.retry_parse_ok = retry_parse_ok
            result.retry_validate_ok = retry_validate_ok
            result.retry_error = retry_err
            result.retry_raw = retry_raw

        # Tear down the pooled subprocess/session for this trial before
        # moving to the next one, so we never hold more than one live
        # `opencode acp` subprocess at a time across the whole spike run.
        acp_teardown()

        retry_note = ""
        if result.retried:
            retry_note = (
                f" | retry: parse={result.retry_parse_ok} "
                f"validate={result.retry_validate_ok} latency={result.retry_latency:.2f}s"
            )
        print(
            f"  [{schema_name}] trial {i + 1}/{n_trials}: "
            f"parse={parse_ok} validate={validate_ok} latency={first_latency:.2f}s"
            f"{retry_note}"
        )
        if err:
            print(f"      first_error: {err[:300]}")
        if result.retried and result.retry_error:
            print(f"      retry_error: {result.retry_error[:300]}")

        results.append(result)

    return results


def summarize(schema_name: str, results: list[TrialResult]) -> dict:
    n = len(results)
    first_parse_ok = sum(1 for r in results if r.first_parse_ok)
    first_validate_ok = sum(1 for r in results if r.first_validate_ok)
    failed_first = [r for r in results if not r.first_validate_ok]
    retry_recovered = sum(1 for r in failed_first if r.retry_validate_ok)
    avg_latency = sum(r.first_latency for r in results) / n if n else 0.0

    return {
        "schema_name": schema_name,
        "n_trials": n,
        "first_parse_rate": first_parse_ok / n if n else 0.0,
        "first_validate_rate": first_validate_ok / n if n else 0.0,
        "n_failed_first": len(failed_first),
        "retry_recovery_rate": (retry_recovered / len(failed_first)) if failed_first else None,
        "avg_first_latency_s": avg_latency,
    }


def print_summary_table(summaries: list[dict]) -> None:
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    header = (
        f"{'schema':<28}{'n':>4}{'parse_ok':>10}{'valid_ok':>10}"
        f"{'n_failed':>10}{'retry_recov':>16}{'avg_lat(s)':>12}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        retry_str = (
            f"{s['retry_recovery_rate']:.0%}" if s["retry_recovery_rate"] is not None else "n/a (0 failed)"
        )
        print(
            f"{s['schema_name']:<28}{s['n_trials']:>4}"
            f"{s['first_parse_rate']:>10.0%}{s['first_validate_rate']:>10.0%}"
            f"{s['n_failed_first']:>10}{retry_str:>16}{s['avg_first_latency_s']:>12.2f}"
        )
    print("=" * 100)


def main() -> None:
    config = _load_real_config()
    valid_languages = set(config["languages"].keys())

    print(f"Model: {MODEL}")
    print(f"N_TRIALS per schema: {N_TRIALS}")
    print(f"Config source: {REAL_CONFIG_PATH if os.path.isfile(REAL_CONFIG_PATH) else '(env var fallback)'}")
    print(f"Valid languages: {sorted(valid_languages)}")
    print()

    work_items_schema = _build_work_items_response_schema("decompose_work_items_spike", valid_languages)
    work_items_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        valid_languages=", ".join(sorted(valid_languages))
    )

    features_schema = _build_features_response_schema(valid_languages)
    features_system_prompt = FEATURES_SYSTEM_PROMPT_TEMPLATE.format(
        valid_languages=", ".join(sorted(valid_languages))
    )

    all_results: dict[str, list[TrialResult]] = {}

    print("--- Schema 1: WorkItem array (DesignAgent.decompose()'s schema) ---")
    all_results["work_items"] = run_schema_trials(
        "work_items", config, valid_languages,
        work_items_system_prompt, WORKITEM_USER_PROMPT,
        work_items_schema, _attempt_work_items, N_TRIALS,
    )

    print()
    print("--- Schema 2: Feature array (DesignAgent.decompose_features()'s schema) ---")
    all_results["features"] = run_schema_trials(
        "features", config, valid_languages,
        features_system_prompt, FEATURES_USER_PROMPT,
        features_schema, _attempt_features, N_TRIALS,
    )

    summaries = [summarize(name, results) for name, results in all_results.items()]
    print_summary_table(summaries)

    print()
    print("Raw per-trial results (JSON) below, for inspection of any failures:")
    dumped = {
        name: [
            {
                "trial_idx": r.trial_idx,
                "first_parse_ok": r.first_parse_ok,
                "first_validate_ok": r.first_validate_ok,
                "first_latency": round(r.first_latency, 3),
                "first_error": r.first_error,
                "retried": r.retried,
                "retry_parse_ok": r.retry_parse_ok,
                "retry_validate_ok": r.retry_validate_ok,
                "retry_latency": round(r.retry_latency, 3) if r.retry_latency is not None else None,
                "retry_error": r.retry_error,
            }
            for r in results
        ]
        for name, results in all_results.items()
    }
    print(json.dumps(dumped, indent=2))


if __name__ == "__main__":
    main()
