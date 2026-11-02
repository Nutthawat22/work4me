import asyncio
import sys
from pathlib import Path

import acp
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from testing_agent.acp_server import PROMPT_USAGE, TestingAgent as AgentUnderTest
from testing_agent.runner import AgentSettings


class _Connection:
    def __init__(self):
        self.messages = []

    async def session_update(self, session_id, update):
        self.messages.append((session_id, update))


def test_acp_initialize_advertises_manual_testing_status():
    agent = AgentUnderTest(AgentSettings())

    response = asyncio.run(agent.initialize(acp.PROTOCOL_VERSION))

    metadata = response.agent_capabilities.field_meta["agents-testing-agent"]
    assert metadata["runStatus"]["method"] == "_agents-testing-agent/run/status"
    assert metadata["manualDelegation"] is True


def test_prompt_delegates_once_and_reports_artifact_path(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    handoff = run_dir / "testing-handoff.json"
    handoff.write_text("{}")
    agent = AgentUnderTest(AgentSettings())
    conn = _Connection()
    agent.on_connect(conn)

    def run(handoff_path, **kwargs):
        assert handoff_path == handoff
        kwargs["progress"]("snapshot created")
        return {
            "outcome": "passed",
            "testing_run_id": "testing-1",
            "result_path": str(run_dir / "external-testing" / "testing-1" / "testing-result.json"),
            "source_snapshot": {"sha256": "abc123"},
        }

    monkeypatch.setattr("testing_agent.acp_server.run_external_tests", run)

    async def exercise():
        session = await agent.new_session(cwd=str(run_dir))
        response = await agent.prompt(
            session.session_id,
            [acp.text_block(f'run "{handoff}"')],
        )
        status = await agent.ext_method("agents-testing-agent/run/status", {"sessionId": session.session_id})
        return response, status

    response, status = asyncio.run(exercise())

    assert response.stop_reason == "end_turn"
    assert status["outcome"] == "passed"
    assert status["status"] == "completed"
    assert status["result_path"].endswith("testing-result.json")
    assert conn.messages


def test_acp_rejects_relative_paths_extra_commands_and_wrong_session_cwd(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    agent = AgentUnderTest(AgentSettings())
    session = asyncio.run(agent.new_session(cwd=str(run_dir)))

    with pytest.raises(ValueError, match="absolute"):
        AgentUnderTest._parse_prompt("run relative/testing-handoff.json")
    with pytest.raises(ValueError, match="Use:"):
        AgentUnderTest._parse_prompt("run /tmp/testing-handoff.json && docker run")

    response = asyncio.run(agent.prompt(session.session_id, [acp.text_block("run /tmp/testing-handoff.json")]))
    status = asyncio.run(agent.ext_method("agents-testing-agent/run/status", {"sessionId": session.session_id}))

    assert response.stop_reason == "refusal"
    assert status["outcome"] == "invalid_prompt"
    assert PROMPT_USAGE in status["error"] or "handoff must be directly" in status["error"]


def test_existing_project_prompt_requires_explicit_root_argument():
    handoff, project = AgentUnderTest._parse_prompt(
        'run "/tmp/run with spaces/testing-handoff.json" --project-root "/tmp/project with spaces"'
    )

    assert str(handoff) == "/tmp/run with spaces/testing-handoff.json"
    assert str(project) == "/tmp/project with spaces"
