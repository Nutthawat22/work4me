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
    acp_call,
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
