import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline import runner
from pipeline.state import AgentResult, OwnedFile, TaskAttempt, WorkItem


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
        acceptance_criteria=["The feature has a valid Python source file."],
        requirement_refs=[],
        owned_files=[OwnedFile("one.py", "python")],
    )


def _install_successful_dispatch(monkeypatch):
    def dispatch(work_items, config, run_dir, *, task_update=None, **kwargs):
        item = work_items[0]
        with open(os.path.join(run_dir, "product", "one.py"), "w", encoding="utf-8") as output:
            output.write("value = 1\n")
        item.status = "ready"
        task_update(item)
        item.status = "running"
        task_update(item)
        item.status = "passed"
        item.attempts.append(
            TaskAttempt(number=1, status="passed", files_written=["product/one.py"])
        )
        task_update(item)
        return [AgentResult("TASK-one", "FeatureAgent", True, ["product/one.py"])]

    monkeypatch.setattr(runner, "dispatch_work_items", dispatch)
    monkeypatch.setattr(runner.acp_client, "teardown", lambda: None)


def test_run_pipeline_reports_attempt_progress_and_pending_external_tests(tmp_path, monkeypatch):
    _install_successful_dispatch(monkeypatch)
    updates = []
    original_write_handoff = runner.write_testing_handoff

    def observe_handoff(config, run_dir):
        assert not (tmp_path / "runs" / "index.jsonl").exists()
        with open(os.path.join(run_dir, "manifest.json"), encoding="utf-8") as manifest_file:
            staged_manifest = json.load(manifest_file)
        assert staged_manifest["status"] == "finalizing"
        paths = original_write_handoff(config, run_dir)
        assert all(os.path.isfile(path) for path in paths)
        return paths

    monkeypatch.setattr(runner, "write_testing_handoff", observe_handoff)

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
    assert updates[-1]["outcome"] == "validation_passed_external_testing_pending"
    assert updates[-1]["validation_status"] == "passed"
    assert updates[-1]["external_testing_status"] == "pending"

    run_dir = next((tmp_path / "runs").iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "validation_passed_external_testing_pending"
    assert manifest["testing_handoff_status"] == "written"
    assert (run_dir / "testing-handoff.md").is_file()
    assert (run_dir / "testing-handoff.json").is_file()
    handoff = json.loads((run_dir / "testing-handoff.json").read_text())
    assert handoff["pipeline_status"] == manifest["status"]
    assert updates[-1]["testing_handoff_status"] == manifest["testing_handoff_status"]


def test_legacy_tests_blocking_flag_does_not_turn_static_validation_into_runtime_pass(
    tmp_path, monkeypatch
):
    _install_successful_dispatch(monkeypatch)
    updates = []

    result = runner.run_pipeline(
        _config(tmp_path, tests_blocking=False),
        work_items=[_work_item()],
        label="legacy config",
        progress=updates.append,
    )

    assert result is True
    assert "tests_blocking" not in updates[-1]
    assert updates[-1]["outcome"] == "validation_passed_external_testing_pending"
