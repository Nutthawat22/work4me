"""
tests/pipeline/test_tests_blocking.py

Tests for config["pipeline"]["tests_blocking"] (default True) — see
pipeline/runner.py's _run_pipeline_body. When explicitly set to False,
a failing test_result should not by itself fail the run (advisory-only),
but the vacuous-pass guard (specialist dispatch failures) must still
block regardless. No real LLM/subprocess/test-suite work happens here —
every collaborator run_pipeline() calls is mocked or stubbed, matching
tests/pipeline/test_runner_acp_teardown.py's conventions.
"""

import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline import runner
from pipeline.state import AgentResult, TestFailure, TestResult as PipelineTestResult, WorkItem


def _config(tmp_path, *, tests_blocking: bool | None, max_retries: int = 3) -> dict:
    pipeline_cfg = {
        "max_retries": max_retries,
        "runs_dir": str(tmp_path / "runs"),
        "output_dir": "product",
        "tests_dir": "tests",
    }
    if tests_blocking is not None:
        pipeline_cfg["tests_blocking"] = tests_blocking

    return {
        "litellm_url": "https://litellm.example.test/v1",
        "litellm_key": "sk-test-key",
        "models": {
            "design": {"model": "gpt-5.6-luna", "provider": "acp"},
            "specialist": {"model": "kimi-k2.7-code", "provider": "acp"},
        },
        "pipeline": pipeline_cfg,
        "languages": {},
    }


def _work_item(item_id: str = "WI-001") -> WorkItem:
    return WorkItem(
        id=item_id, type="logic", language="python", title="t",
        description="d", acceptance_criteria=[], output_path=f"{item_id}.py",
        depends_on=[],
    )


def _read_manifest(tmp_path) -> dict:
    runs_dir = tmp_path / "runs"
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    assert len(run_dirs) == 1
    return json.loads((run_dirs[0] / "manifest.json").read_text())


@pytest.fixture(autouse=True)
def _no_real_teardown(monkeypatch):
    monkeypatch.setattr(runner.acp_client, "teardown", lambda: None)


def _failing_test_result() -> PipelineTestResult:
    return PipelineTestResult(
        passed=False, total=1, failed=1,
        failures=[
            TestFailure(
                test_name="WI-001.py::Foo::test_bar",
                work_item_id="",
                error_output="assertion failed",
            )
        ],
    )


# ── tests_blocking=false, all specialists succeeded, tests failed ──────────

def test_non_blocking_failed_tests_all_specialists_ok_passes_run(tmp_path, monkeypatch, capsys):
    work_items = [_work_item()]
    monkeypatch.setattr(
        "pipeline.runner.dispatch_work_items",
        lambda *a, **k: [
            AgentResult(work_item_id="WI-001", agent_name="logic", success=True,
                        files_written=["WI-001.py"]),
        ],
    )
    monkeypatch.setattr(
        "pipeline.runner.TestRunner.run",
        lambda self, config, work_items, run_dir: _failing_test_result(),
    )

    config = _config(tmp_path, tests_blocking=False)
    result = runner.run_pipeline(config, work_items=work_items, label="test run")

    assert result is True

    captured = capsys.readouterr()
    assert "✅ All tests passed." not in captured.out
    assert "tests_blocking=false" in captured.out

    manifest = _read_manifest(tmp_path)
    assert manifest["status"] == "passed_with_test_failures"
    assert manifest["tests_blocking"] is False


# ── tests_blocking=false, specialist failed, tests failed → still failed ──

def test_non_blocking_failed_tests_specialist_failure_still_fails_run(tmp_path, monkeypatch, capsys):
    work_items = [_work_item("WI-001"), _work_item("WI-002")]
    monkeypatch.setattr(
        "pipeline.runner.dispatch_work_items",
        lambda *a, **k: [
            AgentResult(work_item_id="WI-001", agent_name="logic", success=True,
                        files_written=["WI-001.py"]),
            AgentResult(work_item_id="WI-002", agent_name="logic", success=False,
                        files_written=[], notes="Error: ACP prompt timed out."),
        ],
    )
    monkeypatch.setattr(
        "pipeline.runner.TestRunner.run",
        lambda self, config, work_items, run_dir: _failing_test_result(),
    )
    monkeypatch.setattr(
        "pipeline.master.MasterAgent.map_failures_to_work_items",
        lambda self, test_result, agent_results, work_items: {},
    )

    config = _config(tmp_path, tests_blocking=False, max_retries=1)
    result = runner.run_pipeline(config, work_items=work_items, label="test run")

    assert result is False

    manifest = _read_manifest(tmp_path)
    assert manifest["status"] == "failed"
    assert manifest["tests_blocking"] is False


# ── tests_blocking=true (explicit) behavior unchanged from today ──────────

def test_blocking_true_explicit_failed_tests_fails_run_after_retries(tmp_path, monkeypatch):
    work_items = [_work_item()]
    monkeypatch.setattr(
        "pipeline.runner.dispatch_work_items",
        lambda *a, **k: [
            AgentResult(work_item_id="WI-001", agent_name="logic", success=True,
                        files_written=["WI-001.py"]),
        ],
    )
    monkeypatch.setattr(
        "pipeline.runner.TestRunner.run",
        lambda self, config, work_items, run_dir: _failing_test_result(),
    )
    monkeypatch.setattr(
        "pipeline.master.MasterAgent.map_failures_to_work_items",
        lambda self, test_result, agent_results, work_items: {},
    )

    config = _config(tmp_path, tests_blocking=True, max_retries=1)
    result = runner.run_pipeline(config, work_items=work_items, label="test run")

    assert result is False

    manifest = _read_manifest(tmp_path)
    assert manifest["status"] == "failed"
    assert manifest["tests_blocking"] is True


# ── tests_blocking absent key defaults to True (today's behavior) ─────────

def test_tests_blocking_absent_key_defaults_to_blocking(tmp_path, monkeypatch):
    work_items = [_work_item()]
    monkeypatch.setattr(
        "pipeline.runner.dispatch_work_items",
        lambda *a, **k: [
            AgentResult(work_item_id="WI-001", agent_name="logic", success=True,
                        files_written=["WI-001.py"]),
        ],
    )
    monkeypatch.setattr(
        "pipeline.runner.TestRunner.run",
        lambda self, config, work_items, run_dir: _failing_test_result(),
    )
    monkeypatch.setattr(
        "pipeline.master.MasterAgent.map_failures_to_work_items",
        lambda self, test_result, agent_results, work_items: {},
    )

    config = _config(tmp_path, tests_blocking=None, max_retries=1)
    assert "tests_blocking" not in config["pipeline"]
    result = runner.run_pipeline(config, work_items=work_items, label="test run")

    assert result is False

    manifest = _read_manifest(tmp_path)
    assert manifest["status"] == "failed"
    assert manifest["tests_blocking"] is True


# ── retry loop: tests_blocking=false, test-content failure alone, no ──────
# ── unresolved specialist work → retry loop should NOT burn a retry ───────

def test_non_blocking_pure_test_failure_does_not_trigger_retry(tmp_path, monkeypatch):
    work_items = [_work_item()]

    dispatch_call_count = {"n": 0}

    def fake_dispatch(*args, **kwargs):
        dispatch_call_count["n"] += 1
        return [
            AgentResult(work_item_id="WI-001", agent_name="logic", success=True,
                        files_written=["WI-001.py"]),
        ]

    monkeypatch.setattr("pipeline.runner.dispatch_work_items", fake_dispatch)
    monkeypatch.setattr(
        "pipeline.runner.TestRunner.run",
        lambda self, config, work_items, run_dir: _failing_test_result(),
    )

    config = _config(tmp_path, tests_blocking=False, max_retries=3)
    result = runner.run_pipeline(config, work_items=work_items, label="test run")

    assert result is True
    # Only the initial dispatch call — no retry dispatch should have
    # happened since there was no unresolved specialist work, and the
    # test-content failure alone must not burn a retry in non-blocking
    # mode.
    assert dispatch_call_count["n"] == 1

    manifest = _read_manifest(tmp_path)
    assert manifest["status"] == "passed_with_test_failures"


# ── retry loop: tests_blocking=false, but specialist dispatch fails ───────
# ── → retries should still happen (unresolved specialist work) ───────────

def test_non_blocking_unresolved_specialist_failure_still_retries(tmp_path, monkeypatch):
    work_items = [_work_item()]

    dispatch_call_count = {"n": 0}

    def fake_dispatch(*args, **kwargs):
        dispatch_call_count["n"] += 1
        if dispatch_call_count["n"] == 1:
            return [
                AgentResult(work_item_id="WI-001", agent_name="logic", success=False,
                            files_written=[], notes="Error: ACP prompt timed out."),
            ]
        return [
            AgentResult(work_item_id="WI-001", agent_name="logic", success=True,
                        files_written=["WI-001.py"]),
        ]

    monkeypatch.setattr("pipeline.runner.dispatch_work_items", fake_dispatch)

    test_run_call_count = {"n": 0}

    def fake_test_run(self, config, work_items, run_dir):
        test_run_call_count["n"] += 1
        if test_run_call_count["n"] == 1:
            # Vacuous pass: nothing written yet since dispatch failed.
            return PipelineTestResult(passed=True, total=0, failed=0, failures=[])
        return PipelineTestResult(passed=True, total=0, failed=0, failures=[])

    monkeypatch.setattr("pipeline.runner.TestRunner.run", fake_test_run)
    monkeypatch.setattr(
        "pipeline.master.MasterAgent.map_failures_to_work_items",
        lambda self, test_result, agent_results, work_items: {"WI-001": []},
    )

    config = _config(tmp_path, tests_blocking=False, max_retries=2)
    result = runner.run_pipeline(config, work_items=work_items, label="test run")

    # Retry loop's should_retry check only fires on `not test_result.passed`
    # — a vacuous passed=True/total=0 result technically satisfies
    # test_result.passed, so this scenario is actually caught by the
    # vacuous-pass guard rather than the retry loop's own trigger. Confirm
    # dispatch was called only once here (the vacuous-pass guard fires
    # immediately without retrying) and the run is reported failed.
    assert dispatch_call_count["n"] == 1
    assert result is False

    manifest = _read_manifest(tmp_path)
    assert manifest["status"] == "failed"
    assert manifest["failure_reason"] == "specialist_dispatch_failure"


def test_non_blocking_real_test_failure_with_unresolved_specialist_retries(tmp_path, monkeypatch):
    """
    The genuine retry-loop-should-still-fire case: a real (non-vacuous)
    test failure AND the failing WorkItem's specialist dispatch itself
    failed on round 1 — should_retry must be True here (unresolved
    specialist work), even in tests_blocking=false mode.
    """
    work_items = [_work_item()]

    dispatch_call_count = {"n": 0}

    def fake_dispatch(*args, **kwargs):
        dispatch_call_count["n"] += 1
        if dispatch_call_count["n"] == 1:
            return [
                AgentResult(work_item_id="WI-001", agent_name="logic", success=False,
                            files_written=[], notes="Error: ACP prompt timed out."),
            ]
        return [
            AgentResult(work_item_id="WI-001", agent_name="logic", success=True,
                        files_written=["WI-001.py"]),
        ]

    monkeypatch.setattr("pipeline.runner.dispatch_work_items", fake_dispatch)

    test_run_call_count = {"n": 0}

    def fake_test_run(self, config, work_items, run_dir):
        test_run_call_count["n"] += 1
        if test_run_call_count["n"] == 1:
            return _failing_test_result()
        return PipelineTestResult(passed=True, total=1, failed=0, failures=[])

    monkeypatch.setattr("pipeline.runner.TestRunner.run", fake_test_run)
    monkeypatch.setattr(
        "pipeline.master.MasterAgent.map_failures_to_work_items",
        lambda self, test_result, agent_results, work_items: {"WI-001": []},
    )

    config = _config(tmp_path, tests_blocking=False, max_retries=2)
    result = runner.run_pipeline(config, work_items=work_items, label="test run")

    # Round 1: real test failure + WI-001 dispatch failed -> should_retry
    # True (unresolved specialist work) -> retry dispatched -> round 2
    # succeeds and tests pass.
    assert dispatch_call_count["n"] == 2
    assert result is True

    manifest = _read_manifest(tmp_path)
    assert manifest["status"] == "passed"
