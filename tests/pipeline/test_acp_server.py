"""
tests/pipeline/test_acp_server.py

Unit tests for pipeline/acp_server.py (the ACP Agent/server entrypoint —
Phase 5 of the ACP-native rearchitecture). No real LLM calls, no real
ACP subprocess spawned: pipeline.acp_server.run_pipeline is monkeypatched
directly (same convention as tests/pipeline/test_runner_acp_teardown.py).

No pytest-asyncio plugin is installed in this project's venv, so async
Agent methods are driven directly via asyncio.run(...) inside otherwise
ordinary sync test functions — matching how
specialists/providers/acp_client.py's own sync/async bridge is tested
elsewhere in this repo.
"""

import asyncio
import os
import sys
import threading
import time

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import acp

from pipeline import acp_server
from pipeline.acp_server import PipelineAgent, _extract_prompt_text, _slugify_label


def _fake_config():
    return (
        {
            "litellm_url": "https://litellm.example.test/v1",
            "litellm_key": "sk-test-key",
            "models": {
                "design": {"model": "m", "provider": "acp"},
                "specialist": {"model": "m", "provider": "acp"},
            },
            "pipeline": {"max_retries": 1, "runs_dir": "runs/", "output_dir": "product", "tests_dir": "tests"},
            "languages": {"python": {}},
        },
        "/fake/config.json",
    )


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setattr(acp_server, "load_config", _fake_config)
    # Re-install the module's stdout proxy wrapping whatever sys.stdout
    # currently is (pytest's own per-test capture object, when running
    # under pytest) -- the proxy was originally installed once at
    # import time, before pytest's capture fixture had replaced
    # sys.stdout for THIS test, so without this it would silently wrap
    # a stale/already-replaced object instead of the live one.
    acp_server._install_stdout_proxy()
    a = PipelineAgent()
    return a


class _RecordingConn:
    """Fake acp.Client-shaped connection: records every session_update
    call instead of sending anything over a real transport."""

    def __init__(self):
        self.updates: list[tuple[str, str]] = []

    async def session_update(self, session_id, update):
        text = getattr(update.content, "text", None)
        self.updates.append((session_id, text))


def _text_block(text: str):
    return acp.text_block(text)


# ── initialize() ─────────────────────────────────────────────────────────────

def test_initialize_returns_well_formed_response(agent):
    resp = asyncio.run(agent.initialize(protocol_version=1))

    assert resp.protocol_version == 1
    assert resp.agent_capabilities is not None
    assert resp.agent_info.name == "agents-pipeline"


# ── new_session() ────────────────────────────────────────────────────────────

def test_new_session_returns_tracked_session_id(agent):
    resp = asyncio.run(agent.new_session(cwd="/tmp/some-cwd"))

    assert resp.session_id in agent._sessions
    assert agent._sessions[resp.session_id].cwd == "/tmp/some-cwd"
    assert agent._sessions[resp.session_id].cancelled is False


def test_new_session_ids_are_unique(agent):
    r1 = asyncio.run(agent.new_session(cwd="/tmp/a"))
    r2 = asyncio.run(agent.new_session(cwd="/tmp/a"))

    assert r1.session_id != r2.session_id


# ── prompt(): unknown session ────────────────────────────────────────────────

def test_prompt_unknown_session_raises_request_error(agent):
    with pytest.raises(acp.RequestError):
        asyncio.run(agent.prompt(session_id="does-not-exist", prompt=[_text_block("hello")]))


# ── prompt(): text extraction + run_pipeline invocation ─────────────────────

def test_prompt_extracts_text_and_calls_run_pipeline(agent, monkeypatch):
    agent._conn = _RecordingConn()
    session_resp = asyncio.run(agent.new_session(cwd="/tmp/a"))

    captured_calls = []

    def fake_run_pipeline(config, *, user_input=None, work_items=None, label=None):
        captured_calls.append({"user_input": user_input, "label": label})
        return True

    monkeypatch.setattr(acp_server, "run_pipeline", fake_run_pipeline)

    result = asyncio.run(
        agent.prompt(session_id=session_resp.session_id, prompt=[_text_block("build a hello world page")])
    )

    assert len(captured_calls) == 1
    assert captured_calls[0]["user_input"] == "build a hello world page"
    assert captured_calls[0]["label"]  # non-empty slug derived from the prompt
    assert result.stop_reason == "end_turn"


def test_prompt_multiple_text_blocks_are_joined(agent, monkeypatch):
    agent._conn = _RecordingConn()
    session_resp = asyncio.run(agent.new_session(cwd="/tmp/a"))

    captured = []
    monkeypatch.setattr(
        acp_server, "run_pipeline",
        lambda config, **kwargs: (captured.append(kwargs["user_input"]) or True),
    )

    asyncio.run(
        agent.prompt(
            session_id=session_resp.session_id,
            prompt=[_text_block("part one"), _text_block("part two")],
        )
    )

    assert captured == ["part one part two"]


def test_prompt_non_text_blocks_ignored(agent, monkeypatch):
    agent._conn = _RecordingConn()
    session_resp = asyncio.run(agent.new_session(cwd="/tmp/a"))

    captured = []
    monkeypatch.setattr(
        acp_server, "run_pipeline",
        lambda config, **kwargs: (captured.append(kwargs["user_input"]) or True),
    )

    image_block = acp.image_block(data="base64stuff", mime_type="image/png")
    asyncio.run(
        agent.prompt(
            session_id=session_resp.session_id,
            prompt=[image_block, _text_block("only this counts")],
        )
    )

    assert captured == ["only this counts"]


def test_prompt_reuse_of_same_session_id_rejected(agent, monkeypatch):
    agent._conn = _RecordingConn()
    session_resp = asyncio.run(agent.new_session(cwd="/tmp/a"))
    monkeypatch.setattr(acp_server, "run_pipeline", lambda config, **kwargs: True)

    asyncio.run(agent.prompt(session_id=session_resp.session_id, prompt=[_text_block("first")]))

    with pytest.raises(acp.RequestError):
        asyncio.run(agent.prompt(session_id=session_resp.session_id, prompt=[_text_block("second")]))


# ── prompt(): success / failure stop_reason mapping ─────────────────────────

def test_prompt_run_pipeline_success_returns_end_turn(agent, monkeypatch):
    agent._conn = _RecordingConn()
    session_resp = asyncio.run(agent.new_session(cwd="/tmp/a"))
    monkeypatch.setattr(acp_server, "run_pipeline", lambda config, **kwargs: True)

    result = asyncio.run(agent.prompt(session_id=session_resp.session_id, prompt=[_text_block("x")]))

    assert result.stop_reason == "end_turn"
    sent_texts = " ".join(t for _, t in agent._conn.updates if t)
    assert "✅ Pipeline passed" in sent_texts


def test_prompt_run_pipeline_failure_still_end_turn_with_failure_message(agent, monkeypatch):
    """
    StopReason has no explicit failure value (see acp/schema.py:
    Literal["end_turn", "max_tokens", "max_turn_requests", "refusal",
    "cancelled"]) — a pipeline run that completes but fails its tests is
    still a normal completed turn from ACP's perspective, not a
    protocol-level refusal/cancellation. The failure is communicated via
    the streamed session_update text ("❌ Pipeline failed"), not via
    stop_reason.
    """
    agent._conn = _RecordingConn()
    session_resp = asyncio.run(agent.new_session(cwd="/tmp/a"))
    monkeypatch.setattr(acp_server, "run_pipeline", lambda config, **kwargs: False)

    result = asyncio.run(agent.prompt(session_id=session_resp.session_id, prompt=[_text_block("x")]))

    assert result.stop_reason == "end_turn"
    sent_texts = " ".join(t for _, t in agent._conn.updates if t)
    assert "❌ Pipeline failed" in sent_texts


def test_prompt_run_pipeline_unexpected_exception_does_not_crash(agent, monkeypatch):
    agent._conn = _RecordingConn()
    session_resp = asyncio.run(agent.new_session(cwd="/tmp/a"))

    def _boom(config, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(acp_server, "run_pipeline", _boom)

    result = asyncio.run(agent.prompt(session_id=session_resp.session_id, prompt=[_text_block("x")]))

    # Must not raise -- returns a PromptResponse instead of propagating.
    assert result.stop_reason == "refusal"


def test_prompt_no_text_content_returns_refusal(agent, monkeypatch):
    agent._conn = _RecordingConn()
    session_resp = asyncio.run(agent.new_session(cwd="/tmp/a"))
    called = []
    monkeypatch.setattr(acp_server, "run_pipeline", lambda config, **kwargs: called.append(1) or True)

    image_block = acp.image_block(data="base64stuff", mime_type="image/png")
    result = asyncio.run(agent.prompt(session_id=session_resp.session_id, prompt=[image_block]))

    assert result.stop_reason == "refusal"
    assert called == []  # run_pipeline never invoked


# ── cancel() ─────────────────────────────────────────────────────────────────

def test_cancel_sets_flag_on_existing_session(agent):
    session_resp = asyncio.run(agent.new_session(cwd="/tmp/a"))

    asyncio.run(agent.cancel(session_id=session_resp.session_id))

    assert agent._sessions[session_resp.session_id].cancelled is True


def test_cancel_unknown_session_id_is_a_silent_noop(agent):
    # CancelNotification is a notification, not a request -- there is no
    # error response channel, so an unknown session_id must not raise.
    asyncio.run(agent.cancel(session_id="does-not-exist"))  # should not raise


def test_prompt_reports_cancelled_stop_reason_when_flag_set_before_completion(agent, monkeypatch):
    agent._conn = _RecordingConn()
    session_resp = asyncio.run(agent.new_session(cwd="/tmp/a"))

    def fake_run_pipeline(config, **kwargs):
        # Simulate cancel() arriving while the (mocked, instant) pipeline
        # run is "in flight" -- best-effort cancellation only checks the
        # flag AFTER run_pipeline returns (see module docstring), so this
        # sets the flag directly on the session state to simulate that.
        agent._sessions[session_resp.session_id].cancelled = True
        return True

    monkeypatch.setattr(acp_server, "run_pipeline", fake_run_pipeline)

    result = asyncio.run(agent.prompt(session_id=session_resp.session_id, prompt=[_text_block("x")]))

    assert result.stop_reason == "cancelled"


# ── helper functions ─────────────────────────────────────────────────────────

def test_extract_prompt_text_joins_text_blocks():
    blocks = [_text_block("hello"), _text_block("world")]
    assert _extract_prompt_text(blocks) == "hello world"


def test_extract_prompt_text_ignores_non_text_blocks():
    blocks = [acp.image_block(data="x", mime_type="image/png"), _text_block("only text")]
    assert _extract_prompt_text(blocks) == "only text"


def test_slugify_label_nonempty_for_blank_input():
    assert _slugify_label("   ") == "acp-prompt"


def test_slugify_label_truncates_long_text():
    long_text = "word " * 50
    label = _slugify_label(long_text, max_len=20)
    assert len(label) <= 20


# ── stub methods for optional Agent protocol surface ────────────────────────

def test_load_session_raises_method_not_found(agent):
    with pytest.raises(acp.RequestError):
        asyncio.run(agent.load_session(cwd="/tmp", session_id="x"))


def test_authenticate_raises_method_not_found(agent):
    with pytest.raises(acp.RequestError):
        asyncio.run(agent.authenticate(method_id="whatever"))


def test_close_session_removes_session(agent):
    session_resp = asyncio.run(agent.new_session(cwd="/tmp/a"))
    asyncio.run(agent.close_session(session_id=session_resp.session_id))
    assert session_resp.session_id not in agent._sessions


def test_list_sessions_returns_tracked_sessions(agent):
    r1 = asyncio.run(agent.new_session(cwd="/tmp/a"))
    r2 = asyncio.run(agent.new_session(cwd="/tmp/b"))

    resp = asyncio.run(agent.list_sessions())
    ids = {s.session_id for s in resp.sessions}
    assert ids == {r1.session_id, r2.session_id}


def test_ext_method_raises_method_not_found(agent):
    with pytest.raises(acp.RequestError):
        asyncio.run(agent.ext_method("custom/thing", {}))


def test_ext_notification_is_a_noop(agent):
    asyncio.run(agent.ext_notification("custom/thing", {}))  # should not raise


# ── Concurrency: the most important test given the stdout capture hazard ───

def test_concurrent_prompts_do_not_cross_contaminate_captured_stdout(agent, monkeypatch):
    """
    Two concurrent prompt() calls, each backed by a fake run_pipeline
    that prints distinguishable dummy output from a background thread
    (mirroring how the REAL run_pipeline()/dispatch.py print() from
    inside a blocking call offloaded via asyncio.to_thread), must not
    leak any of session A's lines into session B's captured
    session_update stream, or vice versa.
    """
    agent._conn = _RecordingConn()
    session_a = asyncio.run(agent.new_session(cwd="/tmp/a"))
    session_b = asyncio.run(agent.new_session(cwd="/tmp/b"))

    def fake_run_pipeline(config, *, user_input=None, work_items=None, label=None):
        # Distinguish which session this call belongs to via the label
        # (derived from user_input by _slugify_label in prompt()) since
        # run_pipeline itself has no session concept.
        tag = "A" if "session-a" in (user_input or "") else "B"
        for i in range(15):
            print(f"[{tag}] line {i}")
            time.sleep(0.002)
        return True

    monkeypatch.setattr(acp_server, "run_pipeline", fake_run_pipeline)

    async def _drive_both():
        await asyncio.gather(
            agent.prompt(session_id=session_a.session_id, prompt=[_text_block("session-a task")]),
            agent.prompt(session_id=session_b.session_id, prompt=[_text_block("session-b task")]),
        )

    asyncio.run(_drive_both())

    text_a = "".join(t for sid, t in agent._conn.updates if sid == session_a.session_id and t)
    text_b = "".join(t for sid, t in agent._conn.updates if sid == session_b.session_id and t)

    lines_a = [l for l in text_a.splitlines() if l.strip()]
    lines_b = [l for l in text_b.splitlines() if l.strip()]

    contaminated_a = [l for l in lines_a if "[A]" not in l and "Pipeline" not in l]
    contaminated_b = [l for l in lines_b if "[B]" not in l and "Pipeline" not in l]

    assert contaminated_a == [], f"session A captured contaminated lines: {contaminated_a}"
    assert contaminated_b == [], f"session B captured contaminated lines: {contaminated_b}"

    assert any("[A] line" in l for l in lines_a)
    assert any("[B] line" in l for l in lines_b)


def test_debug_stdout_type(agent):
    import sys
    print("TYPE:", type(sys.stdout), file=sys.stderr)
    print("REAL TYPE:", type(sys.stdout._real) if hasattr(sys.stdout, "_real") else None, file=sys.stderr)
