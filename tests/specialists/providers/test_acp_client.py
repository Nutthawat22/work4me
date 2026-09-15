"""
tests/specialists/providers/test_acp_client.py

Unit tests for specialists/providers/acp_client.py. Uses a STUBBED/mocked
ACP transport (a fake acp.spawn_agent_process + fake connection) -- no
real `opencode acp` subprocess is spawned here. See spike/acp_spike*.py
for real-subprocess verification.
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO_ROOT)

from specialists.providers import acp_client
from specialists.providers.acp_client import (
    AcpSessionPool,
    _build_minimal_opencode_config,
    _build_retry_feedback_prompt,
    acp_call,
    acp_call_with_retry,
)


# ── Fake ACP transport ───────────────────────────────────────────────────────

class FakeProcess:
    """Stand-in for asyncio.subprocess.Process. returncode is None while
    'alive'; setting it to a non-None value simulates a dead subprocess."""

    def __init__(self):
        self.returncode = None


class FakePromptResponse:
    def __init__(self, stop_reason="end_turn"):
        self.stopReason = stop_reason

    def model_dump(self, by_alias=True):
        return {"usage": {}, "stopReason": self.stopReason}


class FakeConnection:
    def __init__(self, client, prompt_behavior, init_behavior=None):
        self._client = client
        self._prompt_behavior = prompt_behavior
        self._init_behavior = init_behavior

    async def initialize(self, protocol_version, client_capabilities):
        if self._init_behavior is not None:
            await self._init_behavior()
        return SimpleNamespace()

    async def new_session(self, cwd, mcp_servers):
        return SimpleNamespace(session_id="fake-session-1")

    async def prompt(self, session_id, prompt):
        return await self._prompt_behavior(self, session_id)

    async def close(self):
        return None


class FakeAcpBackend:
    """
    Configurable stand-in for the whole acp.spawn_agent_process surface.
    Each call to the produced factory ("spawn") increments spawn_count and
    records the FakeProcess created, so tests can assert on spawn
    frequency (pool reuse) and simulate a subprocess dying between calls
    by mutating a previously-returned FakeProcess.returncode.
    """

    def __init__(self, prompt_behavior=None, init_behavior=None, spawn_should_fail=False):
        self.spawn_count = 0
        self.processes: list[FakeProcess] = []
        self.prompt_behavior = prompt_behavior or _default_success_behavior
        self.init_behavior = init_behavior
        self.spawn_should_fail = spawn_should_fail

    def spawn_agent_process(self, client, *args, cwd=None, env=None, **kwargs):
        backend = self

        @asynccontextmanager
        async def _cm():
            backend.spawn_count += 1
            if backend.spawn_should_fail:
                raise FileNotFoundError("opencode: command not found (mocked)")
            process = FakeProcess()
            backend.processes.append(process)
            conn = FakeConnection(client, backend.prompt_behavior, backend.init_behavior)
            yield conn, process

        return _cm()


async def _default_success_behavior(conn, session_id):
    await conn._client.session_update(
        session_id,
        SimpleNamespace(sessionUpdate="agent_message_chunk", content=SimpleNamespace(text="mocked response text")),
    )
    return FakePromptResponse()


async def _timeout_behavior(conn, session_id):
    await asyncio.sleep(999)


async def _no_text_behavior(conn, session_id):
    # Resolves normally but never emits an agent_message_chunk -- exercises
    # the "no agent_message_chunk text received" failure path.
    return FakePromptResponse()


async def _init_failure_behavior():
    raise RuntimeError("handshake failed (mocked)")


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def fresh_pool(monkeypatch):
    """Give each test its own AcpSessionPool + background loop, torn down
    after the test so no threads/loops leak across tests."""
    pool = AcpSessionPool()
    monkeypatch.setattr(acp_client, "_POOL", pool)
    yield pool
    pool.teardown_all()


def _patch_backend(monkeypatch, backend: FakeAcpBackend):
    monkeypatch.setattr(acp_client.acp, "spawn_agent_process", backend.spawn_agent_process)


BASE_CONFIG = {"litellm_url": "https://litellm.example.test/v1", "litellm_key": "sk-test-key"}
MESSAGES = [{"role": "user", "content": "hello"}]


# ── _build_minimal_opencode_config: pure function tests ──────────────────────

def test_build_minimal_opencode_config_shape():
    cfg = _build_minimal_opencode_config(BASE_CONFIG, "claude-sonnet-5")

    assert cfg["default_agent"] == "minimal"
    assert cfg["model"] == "litellm/claude-sonnet-5"

    provider = cfg["provider"]["litellm"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["apiKey"] == BASE_CONFIG["litellm_key"]
    assert provider["options"]["baseURL"] == BASE_CONFIG["litellm_url"]
    assert provider["models"] == {"claude-sonnet-5": {"name": "claude-sonnet-5"}}

    agent = cfg["agent"]["minimal"]
    assert agent["tools"] == {"*": False}
    assert all(v == "deny" for v in agent["permission"].values())
    assert set(agent["permission"].keys()) >= {"edit", "bash", "read", "glob", "grep", "task", "webfetch"}


def test_build_minimal_opencode_config_uses_passed_model_not_config_model():
    cfg = _build_minimal_opencode_config(BASE_CONFIG, "some-other-model")
    assert cfg["model"] == "litellm/some-other-model"
    assert "some-other-model" in cfg["provider"]["litellm"]["models"]


# ── acp_call: success path ────────────────────────────────────────────────────

def test_acp_call_success_returns_accumulated_text(monkeypatch, fresh_pool):
    backend = FakeAcpBackend()
    _patch_backend(monkeypatch, backend)

    result = acp_call(MESSAGES, "claude-sonnet-5", BASE_CONFIG, timeout=5)

    assert result == "mocked response text"


# ── acp_call: failure paths (must return error strings, never raise) ────────

def test_acp_call_spawn_failure_returns_error_string(monkeypatch, fresh_pool):
    backend = FakeAcpBackend(spawn_should_fail=True)
    _patch_backend(monkeypatch, backend)

    result = acp_call(MESSAGES, "claude-sonnet-5", BASE_CONFIG, timeout=5)

    assert result.startswith("[claude-sonnet-5] Error:")
    assert "opencode" in result.lower()


def test_acp_call_prompt_timeout_returns_error_string(monkeypatch, fresh_pool):
    backend = FakeAcpBackend(prompt_behavior=_timeout_behavior)
    _patch_backend(monkeypatch, backend)

    result = acp_call(MESSAGES, "claude-sonnet-5", BASE_CONFIG, timeout=1)

    assert result == "[claude-sonnet-5] Error: ACP prompt timed out after 1s."


def test_acp_call_handshake_failure_returns_error_string(monkeypatch, fresh_pool):
    backend = FakeAcpBackend(init_behavior=_init_failure_behavior)
    _patch_backend(monkeypatch, backend)

    result = acp_call(MESSAGES, "claude-sonnet-5", BASE_CONFIG, timeout=5)

    assert result.startswith("[claude-sonnet-5] Error:")
    assert "handshake failed" in result


def test_acp_call_no_text_received_returns_error_string(monkeypatch, fresh_pool):
    backend = FakeAcpBackend(prompt_behavior=_no_text_behavior)
    _patch_backend(monkeypatch, backend)

    result = acp_call(MESSAGES, "claude-sonnet-5", BASE_CONFIG, timeout=5)

    assert result.startswith("[claude-sonnet-5] Error:")
    assert "no agent_message_chunk text received" in result


# ── Session pool reuse ────────────────────────────────────────────────────────

def test_acp_call_reuses_pooled_session_for_same_role(monkeypatch, fresh_pool):
    backend = FakeAcpBackend()
    _patch_backend(monkeypatch, backend)

    r1 = acp_call(MESSAGES, "claude-sonnet-5", BASE_CONFIG, timeout=5)
    r2 = acp_call(MESSAGES, "claude-sonnet-5", BASE_CONFIG, timeout=5)

    assert r1 == "mocked response text"
    assert r2 == "mocked response text"
    assert backend.spawn_count == 1  # only spawned once -- second call reused the pooled session


# ── Crash recovery ────────────────────────────────────────────────────────────

def test_acp_call_respawns_after_dead_session_detected(monkeypatch, fresh_pool):
    backend = FakeAcpBackend()
    _patch_backend(monkeypatch, backend)

    r1 = acp_call(MESSAGES, "claude-sonnet-5", BASE_CONFIG, timeout=5)
    assert r1 == "mocked response text"
    assert backend.spawn_count == 1

    # Simulate the pooled subprocess dying between calls.
    backend.processes[0].returncode = 1

    r2 = acp_call(MESSAGES, "claude-sonnet-5", BASE_CONFIG, timeout=5)

    assert r2 == "mocked response text"
    assert backend.spawn_count == 2  # respawned once after detecting the dead session


def test_acp_call_respawn_failure_returns_error_string(monkeypatch, fresh_pool):
    backend = FakeAcpBackend()
    _patch_backend(monkeypatch, backend)

    r1 = acp_call(MESSAGES, "claude-sonnet-5", BASE_CONFIG, timeout=5)
    assert r1 == "mocked response text"

    # Kill the session AND make every subsequent spawn attempt fail, so the
    # respawn-on-dead-session path itself fails.
    backend.processes[0].returncode = 1
    backend.spawn_should_fail = True

    r2 = acp_call(MESSAGES, "claude-sonnet-5", BASE_CONFIG, timeout=5)

    assert r2.startswith("[claude-sonnet-5] Error:")


# ── session_scope pooling key regression tests ───────────────────────────────
#
# Guards against the bug session_scope was added to fix: before
# session_scope existed, acp_call pooled sessions by (model, litellm_url)
# ONLY, so two different roles (e.g. "design" and "specialist") sharing the
# same model would incorrectly share one ACP conversation/session — and,
# more broadly, nothing distinguished separate pipeline runs at all. These
# tests assert the fixed behavior: session_scope={"project", "role"} fully
# determines the pooling key (project+role identical -> reuse; either
# differs -> separate sessions), and omitting session_scope entirely still
# falls back to the original (model, litellm_url) behavior unchanged.

def test_acp_call_different_session_scope_uses_different_sessions(monkeypatch, fresh_pool):
    backend = FakeAcpBackend()
    _patch_backend(monkeypatch, backend)

    r1 = acp_call(
        MESSAGES, "claude-sonnet-5", BASE_CONFIG, timeout=5,
        session_scope={"project": "0001-project-a", "role": "specialist"},
    )
    r2 = acp_call(
        MESSAGES, "claude-sonnet-5", BASE_CONFIG, timeout=5,
        session_scope={"project": "0002-project-b", "role": "specialist"},
    )

    assert r1 == "mocked response text"
    assert r2 == "mocked response text"
    assert backend.spawn_count == 2  # different projects -- two separate pooled sessions


def test_acp_call_same_session_scope_reuses_session(monkeypatch, fresh_pool):
    backend = FakeAcpBackend()
    _patch_backend(monkeypatch, backend)

    scope = {"project": "0001-project-a", "role": "specialist"}

    r1 = acp_call(MESSAGES, "claude-sonnet-5", BASE_CONFIG, timeout=5, session_scope=scope)
    r2 = acp_call(MESSAGES, "claude-sonnet-5", BASE_CONFIG, timeout=5, session_scope=scope)

    assert r1 == "mocked response text"
    assert r2 == "mocked response text"
    assert backend.spawn_count == 1  # same project+role -- second call reused the pooled session


def test_acp_call_same_model_url_different_role_uses_different_sessions(monkeypatch, fresh_pool):
    """
    The actual bug being fixed: two different roles ("design" and
    "specialist") using the SAME model+litellm_url must NOT share a pooled
    session. Before session_scope, _role_key_for only used
    f"{model}::{litellm_url}", so this would incorrectly reuse one session
    (spawn_count == 1). With session_scope, project+role fully disambiguates.
    """
    backend = FakeAcpBackend()
    _patch_backend(monkeypatch, backend)

    r1 = acp_call(
        MESSAGES, "claude-sonnet-5", BASE_CONFIG, timeout=5,
        session_scope={"project": "0001-project-a", "role": "design"},
    )
    r2 = acp_call(
        MESSAGES, "claude-sonnet-5", BASE_CONFIG, timeout=5,
        session_scope={"project": "0001-project-a", "role": "specialist"},
    )

    assert r1 == "mocked response text"
    assert r2 == "mocked response text"
    assert backend.spawn_count == 2  # same model+url, different role -- must NOT share a session


def test_acp_call_no_session_scope_falls_back_to_model_url_key(monkeypatch, fresh_pool):
    """No session_scope given -> pooling falls back to the pre-existing
    (model, litellm_url) key, so repeated calls with no session_scope still
    reuse one pooled session (backward-compat path exercised by every other
    test in this file)."""
    backend = FakeAcpBackend()
    _patch_backend(monkeypatch, backend)

    r1 = acp_call(MESSAGES, "claude-sonnet-5", BASE_CONFIG, timeout=5)
    r2 = acp_call(MESSAGES, "claude-sonnet-5", BASE_CONFIG, timeout=5)

    assert r1 == "mocked response text"
    assert r2 == "mocked response text"
    assert backend.spawn_count == 1  # no session_scope -- reused via model+url fallback key


# ── acp_call_with_retry: Option A (same-session retry-with-error-feedback) ──

def _sequenced_behavior(*texts):
    """Build a prompt_behavior that emits `texts[i]` on the i-th call
    (0-indexed), for tests asserting a specific sequence of ACP responses
    across successive prompt() calls against the SAME pooled session."""
    call_count = {"n": 0}

    async def _behavior(conn, session_id):
        idx = min(call_count["n"], len(texts) - 1)
        call_count["n"] += 1
        await conn._client.session_update(
            session_id,
            SimpleNamespace(sessionUpdate="agent_message_chunk", content=SimpleNamespace(text=texts[idx])),
        )
        return FakePromptResponse()

    return _behavior


def _validate_json_object(raw: str) -> dict:
    """Minimal parse_and_validate_fn for tests: raises on non-JSON or
    missing "ok" key, mirroring the shape of design.py's
    _parse_json/_validate_and_build chain (parse then validate, raise a
    descriptive exception on either failure)."""
    import json as _json
    parsed = _json.loads(raw)
    if not isinstance(parsed, dict) or "ok" not in parsed:
        raise ValueError(f"missing required field 'ok' in {parsed!r}")
    return parsed


def test_acp_call_with_retry_success_first_attempt_no_retry(monkeypatch, fresh_pool):
    backend = FakeAcpBackend(prompt_behavior=_sequenced_behavior('{"ok": true}'))
    _patch_backend(monkeypatch, backend)

    result = acp_call_with_retry(
        MESSAGES, "claude-sonnet-5", BASE_CONFIG, timeout=5,
        parse_and_validate_fn=_validate_json_object,
        session_scope={"project": "0001-project-a", "role": "design"},
    )

    assert result == '{"ok": true}'
    assert backend.spawn_count == 1  # only ever needed one pooled session


def test_acp_call_with_retry_failure_then_success_reuses_same_session(monkeypatch, fresh_pool):
    backend = FakeAcpBackend(
        prompt_behavior=_sequenced_behavior("not json at all", '{"ok": true}'),
    )
    _patch_backend(monkeypatch, backend)

    scope = {"project": "0001-project-a", "role": "design"}
    result = acp_call_with_retry(
        MESSAGES, "claude-sonnet-5", BASE_CONFIG, timeout=5,
        parse_and_validate_fn=_validate_json_object,
        session_scope=scope,
        max_attempts=3,
    )

    assert result == '{"ok": true}'
    # Same-session retry: still exactly ONE pooled subprocess/session spawned
    # across both attempts -- this is what makes it "same-session" per
    # Option A, not a fresh session per retry.
    assert backend.spawn_count == 1


def test_acp_call_with_retry_second_attempt_sees_error_feedback_in_prompt(monkeypatch, fresh_pool):
    """Assert the retry's prompt actually carries the failure reason
    (not just that a second call happened) -- the core of Option A's
    self-correction signal."""
    seen_prompts = []
    behavior = _sequenced_behavior("not json at all", '{"ok": true}')

    async def _capturing_behavior(conn, session_id):
        return await behavior(conn, session_id)

    class _CapturingConnection(FakeConnection):
        async def prompt(self, session_id, prompt):
            seen_prompts.append(prompt[0].text if hasattr(prompt[0], "text") else str(prompt[0]))
            return await self._prompt_behavior(self, session_id)

    class _CapturingBackend(FakeAcpBackend):
        def spawn_agent_process(self, client, *args, cwd=None, env=None, **kwargs):
            backend = self

            @asynccontextmanager
            async def _cm():
                backend.spawn_count += 1
                process = FakeProcess()
                backend.processes.append(process)
                conn = _CapturingConnection(client, backend.prompt_behavior, backend.init_behavior)
                yield conn, process

            return _cm()

    backend = _CapturingBackend(prompt_behavior=_capturing_behavior)
    _patch_backend(monkeypatch, backend)

    result = acp_call_with_retry(
        MESSAGES, "claude-sonnet-5", BASE_CONFIG, timeout=5,
        parse_and_validate_fn=_validate_json_object,
        session_scope={"project": "0001-project-a", "role": "design"},
        max_attempts=3,
    )

    assert result == '{"ok": true}'
    assert len(seen_prompts) == 2
    assert "hello" in seen_prompts[0]  # first attempt sent the original message
    assert "invalid" in seen_prompts[1].lower()
    assert "Expecting value" in seen_prompts[1]  # json.JSONDecodeError message surfaced as feedback


def test_acp_call_with_retry_all_attempts_fail_returns_final_error_with_count(monkeypatch, fresh_pool):
    backend = FakeAcpBackend(prompt_behavior=_sequenced_behavior("still not json"))
    _patch_backend(monkeypatch, backend)

    result = acp_call_with_retry(
        MESSAGES, "claude-sonnet-5", BASE_CONFIG, timeout=5,
        parse_and_validate_fn=_validate_json_object,
        session_scope={"project": "0001-project-a", "role": "design"},
        max_attempts=3,
    )

    assert result.startswith("[claude-sonnet-5] Error:")
    assert "failed after 3 attempts" in result
    assert "last error" in result
    assert backend.spawn_count == 1  # still same session across every failed attempt


def test_acp_call_with_retry_acp_error_string_triggers_retry(monkeypatch, fresh_pool):
    """An acp_call-level error (e.g. transient timeout) should also
    trigger a retry, not just parse/validate failures."""
    backend = FakeAcpBackend()  # default success behavior once actually invoked
    _patch_backend(monkeypatch, backend)

    call_count = {"n": 0}
    real_acp_call = acp_client.acp_call

    def _flaky_acp_call(messages, model, config, timeout=60, response_schema=None, session_scope=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return f"[{model}] Error: ACP prompt timed out after {timeout}s."
        return real_acp_call(messages, model, config, timeout=timeout, response_schema=response_schema, session_scope=session_scope)

    monkeypatch.setattr(acp_client, "acp_call", _flaky_acp_call)

    result = acp_client.acp_call_with_retry(
        MESSAGES, "claude-sonnet-5", BASE_CONFIG, timeout=5,
        session_scope={"project": "0001-project-a", "role": "design"},
        max_attempts=3,
    )

    assert result == "mocked response text"
    assert call_count["n"] == 2


def test_acp_call_with_retry_max_attempts_less_than_one_raises():
    with pytest.raises(ValueError):
        acp_call_with_retry(MESSAGES, "claude-sonnet-5", BASE_CONFIG, max_attempts=0)


# ── _build_retry_feedback_prompt: forbids repeating/concatenating attempts ──

def test_retry_feedback_prompt_forbids_repeating_previous_attempt_with_schema():
    prompt = _build_retry_feedback_prompt(
        {"name": "x", "schema": {"type": "object"}}, "some failure reason"
    )
    assert "some failure reason" in prompt
    assert "exactly one" in prompt.lower()
    assert "repeat" in prompt.lower()
    assert "previous attempt" in prompt.lower()


def test_retry_feedback_prompt_forbids_repeating_previous_attempt_without_schema():
    prompt = _build_retry_feedback_prompt(None, "some failure reason")
    assert "some failure reason" in prompt
    assert "exactly one" in prompt.lower()
    assert "repeat" in prompt.lower()
    assert "previous attempt" in prompt.lower()
