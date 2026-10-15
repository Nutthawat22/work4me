import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline import runner
from pipeline.state import (
    AgentResult,
    OwnedFile,
    TaskAttempt,
    TestResult as PipelineTestResult,
    WorkItem,
)


def _config(tmp_path, *, tests_blocking=True):
    return {
        "litellm_url": "https://litellm.example.test/v1",
        "litellm_key": "sk-test-key",
        "models": {
            "design": {"model": "test", "provider": "acp"},
            "specialist": {"model": "test", "provider": "acp"},
        },
        "pipeline": {
            "max_retries": 1,
            "runs_dir": str(tmp_path / "runs"),
            "output_dir": "product",
            "tests_dir": "tests",
            "tests_blocking": tests_blocking,
        },
        "languages": {"python": {}},
    }


def _work_item():
    return WorkItem(
        id="TASK-one",
        category="feature",
        language="python",
        module="one",
        scope="Implement one feature",
        title="One feature",
        description="Implement one feature.",
        acceptance_criteria=["The feature works."],
        requirement_refs=[],
        owned_files=[OwnedFile("one.py", "python")],
    )


def _install_successful_dispatch(monkeypatch):
    def dispatch(work_items, config, run_dir, *, task_update=None, **kwargs):
        item = work_items[0]
        item.status = "ready"
        task_update(item)
        item.status = "running"
        task_update(item)
        item.status = "passed"
        item.attempts.append(
            TaskAttempt(number=1, status="passed", files_written=["one.py"])
        )
        task_update(item)
        return [AgentResult("TASK-one", "FeatureAgent", True, ["one.py"])]

    monkeypatch.setattr(runner, "dispatch_work_items", dispatch)
    monkeypatch.setattr(
        runner.TestRunner,
        "run",
        lambda self, config, work_items, run_dir: PipelineTestResult(True, 1, 0, []),
    )
    monkeypatch.setattr(runner.acp_client, "teardown", lambda: None)


def test_run_pipeline_reports_monotonic_task_attempt_progress(tmp_path, monkeypatch):
    _install_successful_dispatch(monkeypatch)
    updates = []

    result = runner.run_pipeline(
        _config(tmp_path),
        work_items=[_work_item()],
        label="progress run",
        progress=updates.append,
    )

    running = next(
        update for update in updates
        if isinstance(update.get("task"), dict) and update["task"].get("status") == "running"
    )
    completed = next(
        update for update in updates
        if isinstance(update.get("task"), dict) and update["task"].get("status") == "passed"
    )
    assert result is True
    assert running["tasks_total"] == 1
    assert running["tasks_started"] == 1
    assert running["tasks_completed"] == 0
    assert completed["tasks_started"] == 1
    assert completed["tasks_completed"] == 1
    assert completed["task"]["attempt"] == 1
    assert updates[-1]["phase"] == "teardown"
    assert updates[-1]["outcome"] == "passed"


def test_run_pipeline_reports_advisory_test_failures_as_distinct_outcome(
    tmp_path, monkeypatch
):
    _install_successful_dispatch(monkeypatch)
    monkeypatch.setattr(
        runner.TestRunner,
        "run",
        lambda self, config, work_items, run_dir: PipelineTestResult(False, 1, 1, []),
    )
    updates = []

    result = runner.run_pipeline(
        _config(tmp_path, tests_blocking=False),
        work_items=[_work_item()],
        label="advisory progress run",
        progress=updates.append,
    )

    assert result is True
    assert updates[-1]["tests_blocking"] is False
    assert updates[-1]["outcome"] == "passed_with_test_failures"
