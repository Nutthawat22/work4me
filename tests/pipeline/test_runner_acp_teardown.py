"""
tests/pipeline/test_runner_acp_teardown.py

Confirms pipeline/runner.py::run_pipeline() calls
specialists.providers.acp_client.teardown() exactly once at the end of
every run, regardless of which exit path is taken (MasterPlanError,
CycleError, test-failure, or success). No real LLM/subprocess/test-suite
work happens here — every collaborator run_pipeline() calls is mocked or
stubbed so each test isolates the exit path under test.
"""

import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline import runner
from pipeline.master import MasterPlanError
from pipeline.dispatch import CycleError
from pipeline.state import TestResult as PipelineTestResult


def _config(tmp_path) -> dict:
    return {
        "litellm_url": "https://litellm.example.test/v1",
        "litellm_key": "sk-test-key",
        "models": {
            "design": {"model": "gpt-5.6-luna", "provider": "acp"},
            "specialist": {"model": "kimi-k2.7-code", "provider": "acp"},
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
        "pipeline.master.MasterAgent.decompose",
        lambda self, user_prompt: (_ for _ in ()).throw(MasterPlanError("bad plan")),
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


def test_all_specialists_failed_reports_failure_despite_vacuous_test_pass(
    tmp_path, monkeypatch, teardown_spy, capsys
):
    """
    The exact bug scenario: every AgentResult has success=False (e.g. all
    specialists timed out), files_written is [] for all of them, so no
    test files exist and TestRunner.run()'s no-tests-ran fallback returns
    a vacuous TestResult(passed=True, total=0, ...). The run must be
    reported as FAILED, not "✅ All tests passed.".
    """
    from pipeline.state import WorkItem, AgentResult

    work_items = [
        WorkItem(
            id="WI-001", type="logic", language="python", title="t1",
            description="d", acceptance_criteria=[], output_path="a.py",
            depends_on=[],
        ),
        WorkItem(
            id="WI-002", type="logic", language="python", title="t2",
            description="d", acceptance_criteria=[], output_path="b.py",
            depends_on=[],
        ),
    ]
    monkeypatch.setattr(
        "pipeline.runner.dispatch_work_items",
        lambda *a, **k: [
            AgentResult(
                work_item_id="WI-001", agent_name="logic", success=False,
                files_written=[], notes="Error: ACP prompt timed out after 60s.",
            ),
            AgentResult(
                work_item_id="WI-002", agent_name="logic", success=False,
                files_written=[], notes="Error: ACP prompt timed out after 60s.",
            ),
        ],
    )
    monkeypatch.setattr(
        "pipeline.runner.TestRunner.run",
        lambda self, config, work_items, run_dir: PipelineTestResult(passed=True, total=0, failed=0, failures=[]),
    )

    result = runner.run_pipeline(_config(tmp_path), work_items=work_items, label="test run")

    assert result is False
    assert teardown_spy == [True]

    captured = capsys.readouterr()
    assert "✅ All tests passed." not in captured.out

    runs_dir = tmp_path / "runs"
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    assert len(run_dirs) == 1
    manifest_path = run_dirs[0] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "failed"
    assert manifest["failure_reason"] == "specialist_dispatch_failure"


def test_legitimate_no_tests_needed_still_reports_passed(tmp_path, monkeypatch, teardown_spy, capsys):
    """
    The legitimate case the TestRunner fallback exists for: every
    specialist succeeds, but the run genuinely has no test files for any
    language (e.g. a tiny config-only task) — TestResult(passed=True,
    total=0, ...) is a correct trivial pass here and must NOT be flagged
    as a dispatch failure.
    """
    from pipeline.state import WorkItem, AgentResult

    work_items = [
        WorkItem(
            id="WI-001", type="config", language="python", title="t1",
            description="d", acceptance_criteria=[], output_path="config.json",
            depends_on=[],
        ),
    ]
    monkeypatch.setattr(
        "pipeline.runner.dispatch_work_items",
        lambda *a, **k: [
            AgentResult(
                work_item_id="WI-001", agent_name="config", success=True,
                files_written=["config.json"],
            ),
        ],
    )
    monkeypatch.setattr(
        "pipeline.runner.TestRunner.run",
        lambda self, config, work_items, run_dir: PipelineTestResult(passed=True, total=0, failed=0, failures=[]),
    )

    result = runner.run_pipeline(_config(tmp_path), work_items=work_items, label="test run")

    assert result is True
    assert teardown_spy == [True]

    captured = capsys.readouterr()
    assert "✅ All tests passed." in captured.out

    runs_dir = tmp_path / "runs"
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    assert len(run_dirs) == 1
    manifest_path = run_dirs[0] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "passed"


def test_dispatch_failure_only_in_later_retry_round_still_fails(
    tmp_path, monkeypatch, teardown_spy, capsys
):
    """
    Point 4 of the bug fix: dispatch failures surviving only into a
    LATER retry round (not the very first dispatch) must also be
    caught, even if the retry loop exits (max_retries exhausted) with
    test_result still vacuously passed=True/total=0. Simulates: round 1
    has a real test failure (non-vacuous) triggering a retry; the
    retried WorkItem fails dispatch entirely on the retry round, and by
    the time the retry budget is exhausted, test_result has gone back
    to a vacuous 0/0 pass (e.g. the previously-failing test file is no
    longer resolvable) — this must still be reported as failed, not
    passed.
    """
    from pipeline.state import WorkItem, AgentResult, TestFailure

    work_items = [
        WorkItem(
            id="WI-001", type="test", language="python", title="t1",
            description="d", acceptance_criteria=[], output_path="test_a.py",
            depends_on=[],
        ),
    ]

    dispatch_call_count = {"n": 0}

    def fake_dispatch(*args, **kwargs):
        dispatch_call_count["n"] += 1
        if dispatch_call_count["n"] == 1:
            return [
                AgentResult(
                    work_item_id="WI-001", agent_name="test", success=True,
                    files_written=["test_a.py"],
                )
            ]
        return [
            AgentResult(
                work_item_id="WI-001", agent_name="test", success=False,
                files_written=[], notes="Error: ACP prompt timed out after 60s.",
            )
        ]

    monkeypatch.setattr("pipeline.runner.dispatch_work_items", fake_dispatch)

    test_run_call_count = {"n": 0}

    def fake_test_run(self, config, work_items, run_dir):
        test_run_call_count["n"] += 1
        if test_run_call_count["n"] == 1:
            return PipelineTestResult(
                passed=False, total=1, failed=1,
                failures=[
                    TestFailure(
                        test_name="test_a.py::Foo::test_bar",
                        work_item_id="",
                        error_output="assertion failed",
                    )
                ],
            )
        return PipelineTestResult(passed=True, total=0, failed=0, failures=[])

    monkeypatch.setattr("pipeline.runner.TestRunner.run", fake_test_run)

    config = _config(tmp_path)
    config["pipeline"]["max_retries"] = 2

    result = runner.run_pipeline(config, work_items=work_items, label="test run")

    assert result is False
    assert teardown_spy == [True]

    captured = capsys.readouterr()
    assert "✅ All tests passed." not in captured.out

    runs_dir = tmp_path / "runs"
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    assert len(run_dirs) == 1
    manifest_path = run_dirs[0] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "failed"
    assert manifest["failure_reason"] == "specialist_dispatch_failure"
