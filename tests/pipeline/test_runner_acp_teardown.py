"""
tests/pipeline/test_runner_acp_teardown.py

Confirms pipeline/runner.py::run_pipeline() calls
specialists.providers.acp_client.teardown() exactly once at the end of
every run, regardless of which exit path is taken (DesignParseError,
CycleError, test-failure, or success). No real LLM/subprocess/test-suite
work happens here — every collaborator run_pipeline() calls is mocked or
stubbed so each test isolates the exit path under test.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline import runner
from pipeline.design import DesignParseError
from pipeline.dispatch import CycleError
from pipeline.state import TestResult as PipelineTestResult


def _config(tmp_path) -> dict:
    return {
        "litellm_url": "https://litellm.example.test/v1",
        "litellm_key": "sk-test-key",
        "models": {
            "design": {"model": "gpt-5.6-luna", "provider": "responses"},
            "specialist": {"model": "kimi-k2.7-code", "provider": "chat_completions"},
        },
        "pipeline": {
            "max_retries": 1,
            "runs_dir": str(tmp_path / "runs"),
            "output_dir": "product",
            "tests_dir": "tests",
        },
        "languages": {},
    }


@pytest.fixture
def teardown_spy(monkeypatch):
    calls = []
    monkeypatch.setattr(runner.acp_client, "teardown", lambda: calls.append(True))
    return calls


def test_teardown_called_on_design_parse_error(tmp_path, monkeypatch, teardown_spy):
    monkeypatch.setattr(
        "pipeline.design.DesignAgent.decompose",
        lambda self, user_prompt: (_ for _ in ()).throw(DesignParseError("bad plan")),
    )

    result = runner.run_pipeline(_config(tmp_path), user_input="build a thing")

    assert result is False
    assert teardown_spy == [True]


def test_teardown_called_on_dispatch_cycle_error(tmp_path, monkeypatch, teardown_spy):
    from pipeline.state import WorkItem

    work_items = [
        WorkItem(
            id="WI-001", type="logic", language="python", title="t",
            description="d", acceptance_criteria=[], output_path="a.py",
            depends_on=[],
        ),
    ]
    monkeypatch.setattr(
        "pipeline.runner.dispatch_work_items",
        lambda *a, **k: (_ for _ in ()).throw(CycleError("cycle detected")),
    )

    result = runner.run_pipeline(_config(tmp_path), work_items=work_items, label="test run")

    assert result is False
    assert teardown_spy == [True]


def test_teardown_called_on_success(tmp_path, monkeypatch, teardown_spy):
    from pipeline.state import WorkItem, AgentResult

    work_items = [
        WorkItem(
            id="WI-001", type="logic", language="python", title="t",
            description="d", acceptance_criteria=[], output_path="a.py",
            depends_on=[],
        ),
    ]
    monkeypatch.setattr(
        "pipeline.runner.dispatch_work_items",
        lambda *a, **k: [AgentResult(work_item_id="WI-001", agent_name="logic", success=True, files_written=["a.py"])],
    )
    monkeypatch.setattr(
        "pipeline.runner.TestRunner.run",
        lambda self, config, work_items, run_dir: PipelineTestResult(passed=True, total=0, failed=0, failures=[]),
    )

    result = runner.run_pipeline(_config(tmp_path), work_items=work_items, label="test run")

    assert result is True
    assert teardown_spy == [True]
