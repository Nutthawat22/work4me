import json
import logging
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline import runner
from pipeline.dispatch import CycleError
from pipeline.master import MasterPlanError
from pipeline.state import AgentResult, OwnedFile, WorkItem


def _config(tmp_path) -> dict:
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
        },
        "languages": {"python": {}},
    }


def _work_item() -> WorkItem:
    return WorkItem(
        id="WI-001", category="feature", language="python", module="a",
        scope="Implement a", title="A", description="Implement a",
        acceptance_criteria=["a works"], requirement_refs=[],
        owned_files=[OwnedFile("a.py", "python")],
    )


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

    assert runner.run_pipeline(_config(tmp_path), user_input="build a thing") is False
    assert teardown_spy == [True]
    run_dir = next(path for path in (tmp_path / "runs").iterdir() if path.is_dir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["validation_status"] == "not_completed"
    assert manifest["external_testing_status"] == "pending"


def test_teardown_called_on_dispatch_cycle_error(tmp_path, monkeypatch, teardown_spy):
    monkeypatch.setattr(
        "pipeline.runner.dispatch_work_items",
        lambda *args, **kwargs: (_ for _ in ()).throw(CycleError("cycle detected")),
    )

    assert runner.run_pipeline(
        _config(tmp_path), work_items=[_work_item()], label="cycle"
    ) is False
    assert teardown_spy == [True]


def test_teardown_called_after_static_validation_success(tmp_path, monkeypatch, teardown_spy):
    def dispatch(work_items, config, run_dir, **kwargs):
        path = os.path.join(run_dir, "product", "a.py")
        with open(path, "w", encoding="utf-8") as output:
            output.write("value = 1\n")
        return [AgentResult("WI-001", "FeatureAgent", True, ["product/a.py"])]

    monkeypatch.setattr("pipeline.runner.dispatch_work_items", dispatch)

    assert runner.run_pipeline(
        _config(tmp_path), work_items=[_work_item()], label="static pass"
    ) is True
    assert teardown_spy == [True]
    run_dir = next(path for path in (tmp_path / "runs").iterdir() if path.is_dir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "validation_passed_external_testing_pending"
    assert (run_dir / "testing-handoff.md").is_file()


def test_validation_failure_is_recorded_and_still_tears_down(tmp_path, monkeypatch, teardown_spy):
    def dispatch(work_items, config, run_dir, **kwargs):
        with open(os.path.join(run_dir, "product", "a.py"), "w", encoding="utf-8") as output:
            output.write("def broken(:\n    pass\n")
        return [AgentResult("WI-001", "FeatureAgent", True, ["product/a.py"])]

    monkeypatch.setattr("pipeline.runner.dispatch_work_items", dispatch)

    assert runner.run_pipeline(
        _config(tmp_path), work_items=[_work_item()], label="invalid syntax"
    ) is False
    assert teardown_spy == [True]
    run_dir = next((tmp_path / "runs").iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    result = json.loads((run_dir / "validation/results/attempt-1.json").read_text())
    assert manifest["status"] == "validation_failed"
    assert result["errors"][0]["error_output"].startswith("Python syntax error")
    assert json.loads((run_dir / "testing-handoff.json").read_text())["external_testing"]["status"] == "pending"


def test_unexpected_exception_still_writes_handoff_and_tears_down(tmp_path, monkeypatch, teardown_spy):
    monkeypatch.setattr(
        "pipeline.runner.dispatch_work_items",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("dispatch broke")),
    )

    with pytest.raises(RuntimeError, match="dispatch broke"):
        runner.run_pipeline(_config(tmp_path), work_items=[_work_item()], label="exception")

    assert teardown_spy == [True]
    run_dir = next((tmp_path / "runs").iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["failure_reason"] == "pipeline_exception"
    assert (run_dir / "testing-handoff.json").is_file()


def test_handoff_partial_write_fails_pipeline_and_final_publications(
    tmp_path, monkeypatch, teardown_spy, capsys, caplog
):
    def dispatch(work_items, config, run_dir, **kwargs):
        path = os.path.join(run_dir, "product", "a.py")
        with open(path, "w", encoding="utf-8") as output:
            output.write("value = 1\n")
        return [AgentResult("WI-001", "FeatureAgent", True, ["product/a.py"])]

    updates = []

    def partially_write_handoff(config, run_dir):
        with open(os.path.join(run_dir, "testing-handoff.json"), "w", encoding="utf-8") as output:
            output.write("partial artifact")
        raise RuntimeError("handoff failed")

    monkeypatch.setattr("pipeline.runner.dispatch_work_items", dispatch)
    monkeypatch.setattr(
        "pipeline.runner.write_testing_handoff",
        partially_write_handoff,
    )

    assert not runner.run_pipeline(
        _config(tmp_path), work_items=[_work_item()], label="handoff error", progress=updates.append
    )
    assert teardown_spy == [True]
    run_dir = next(path for path in (tmp_path / "runs").iterdir() if path.is_dir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    index_entry = json.loads((tmp_path / "runs" / "index.jsonl").read_text().splitlines()[-1])
    assert manifest["status"] == "failed"
    assert manifest["failure_reason"] == "testing_handoff_failure"
    assert manifest["testing_handoff_status"] == "failed"
    assert index_entry["status"] == manifest["status"]
    assert index_entry["testing_handoff_status"] == manifest["testing_handoff_status"]
    assert updates[-1]["outcome"] == "failed"
    assert updates[-1]["testing_handoff_status"] == "failed"
    assert not (run_dir / "testing-handoff.json").exists()
    assert not (run_dir / "testing-handoff.md").exists()
    assert "artifacts are incomplete" in capsys.readouterr().err
    assert "handoff failed" in caplog.text


def test_handoff_failure_logs_fallback_manifest_write_failure(
    tmp_path, monkeypatch, teardown_spy, caplog
):
    def dispatch(work_items, config, run_dir, **kwargs):
        path = os.path.join(run_dir, "product", "a.py")
        with open(path, "w", encoding="utf-8") as output:
            output.write("value = 1\n")
        return [AgentResult("WI-001", "FeatureAgent", True, ["product/a.py"])]

    original_write_manifest = runner.write_manifest

    def fail_final_handoff_status(run_dir, manifest):
        if manifest.get("failure_reason") == "testing_handoff_failure":
            raise OSError("manifest fallback failed")
        original_write_manifest(run_dir, manifest)

    monkeypatch.setattr("pipeline.runner.dispatch_work_items", dispatch)
    monkeypatch.setattr(
        "pipeline.runner.write_testing_handoff",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("handoff failed")),
    )
    monkeypatch.setattr("pipeline.runner.write_manifest", fail_final_handoff_status)

    with caplog.at_level(logging.ERROR):
        assert not runner.run_pipeline(
            _config(tmp_path), work_items=[_work_item()], label="manifest fallback"
        )

    assert teardown_spy == [True]
    assert "Testing handoff artifact generation failed" in caplog.text
    assert "Could not write finalized run manifest" in caplog.text
    assert "manifest fallback failed" in caplog.text
    index_entry = json.loads((tmp_path / "runs" / "index.jsonl").read_text().splitlines()[-1])
    assert index_entry["status"] == "failed"
    assert index_entry["testing_handoff_status"] == "failed"


def test_handoff_failure_does_not_mask_pipeline_exception(
    tmp_path, monkeypatch, teardown_spy, caplog
):
    monkeypatch.setattr(
        "pipeline.runner.dispatch_work_items",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("dispatch broke")),
    )
    monkeypatch.setattr(
        "pipeline.runner.write_testing_handoff",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("handoff failed")),
    )

    with pytest.raises(RuntimeError, match="dispatch broke"):
        runner.run_pipeline(_config(tmp_path), work_items=[_work_item()], label="handoff exception")

    assert teardown_spy == [True]
    run_dir = next(path for path in (tmp_path / "runs").iterdir() if path.is_dir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["failure_reason"] == "pipeline_exception"
    assert manifest["testing_handoff_status"] == "failed"
    index_entry = json.loads((tmp_path / "runs" / "index.jsonl").read_text().splitlines()[-1])
    assert index_entry["status"] == manifest["status"]
    assert index_entry["testing_handoff_status"] == manifest["testing_handoff_status"]
    assert "Testing handoff artifact generation failed" in caplog.text


def test_final_manifest_write_failure_publishes_failed_status_and_progress(
    tmp_path, monkeypatch, teardown_spy
):
    def dispatch(work_items, config, run_dir, **kwargs):
        path = os.path.join(run_dir, "product", "a.py")
        with open(path, "w", encoding="utf-8") as output:
            output.write("value = 1\n")
        return [AgentResult("WI-001", "FeatureAgent", True, ["product/a.py"])]

    original_write_manifest = runner.write_manifest

    def fail_final_manifest(run_dir, manifest):
        if manifest.get("testing_handoff_status") == "written":
            raise OSError("final manifest write failed")
        original_write_manifest(run_dir, manifest)

    monkeypatch.setattr(runner, "dispatch_work_items", dispatch)
    monkeypatch.setattr(runner, "write_manifest", fail_final_manifest)
    updates = []

    result = runner.run_pipeline(
        _config(tmp_path),
        work_items=[_work_item()],
        label="final manifest failure",
        progress=updates.append,
    )

    index_entry = json.loads((tmp_path / "runs" / "index.jsonl").read_text().splitlines()[-1])
    assert result is False
    assert index_entry["status"] == "failed"
    assert index_entry["failure_reason"] == "manifest_finalization_failure"
    assert index_entry["validation_status"] == "passed"
    assert updates[-1]["outcome"] == "failed"
    assert updates[-1]["validation_status"] == "passed"
    assert teardown_spy == [True]


def test_final_manifest_failure_does_not_mask_pipeline_exception(
    tmp_path, monkeypatch, teardown_spy
):
    monkeypatch.setattr(
        runner,
        "dispatch_work_items",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("dispatch broke")),
    )
    original_write_manifest = runner.write_manifest

    def fail_final_manifest(run_dir, manifest):
        if manifest.get("testing_handoff_status") == "written":
            raise OSError("final manifest write failed")
        original_write_manifest(run_dir, manifest)

    monkeypatch.setattr(runner, "write_manifest", fail_final_manifest)
    updates = []

    with pytest.raises(RuntimeError, match="dispatch broke"):
        runner.run_pipeline(
            _config(tmp_path),
            work_items=[_work_item()],
            label="exception priority",
            progress=updates.append,
        )

    index_entry = json.loads((tmp_path / "runs" / "index.jsonl").read_text().splitlines()[-1])
    assert index_entry["status"] == "failed"
    assert index_entry["failure_reason"] == "pipeline_exception"
    assert updates[-1]["outcome"] == "failed"
    assert teardown_spy == [True]


def test_index_failure_corrects_manifest_and_reports_failed_progress(
    tmp_path, monkeypatch, teardown_spy
):
    def dispatch(work_items, config, run_dir, **kwargs):
        path = os.path.join(run_dir, "product", "a.py")
        with open(path, "w", encoding="utf-8") as output:
            output.write("value = 1\n")
        return [AgentResult("WI-001", "FeatureAgent", True, ["product/a.py"])]

    monkeypatch.setattr(runner, "dispatch_work_items", dispatch)
    monkeypatch.setattr(
        runner,
        "append_index",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("index write failed")),
    )
    updates = []

    result = runner.run_pipeline(
        _config(tmp_path),
        work_items=[_work_item()],
        label="index failure",
        progress=updates.append,
    )

    run_dir = next(path for path in (tmp_path / "runs").iterdir() if path.is_dir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert result is False
    assert manifest["status"] == "failed"
    assert manifest["failure_reason"] == "run_index_finalization_failure"
    assert manifest["validation_status"] == "passed"
    assert updates[-1]["outcome"] == "failed"
    assert updates[-1]["validation_status"] == "passed"
    assert teardown_spy == [True]


def test_index_and_manifest_correction_failures_log_stale_disk_status(
    tmp_path, monkeypatch, teardown_spy, caplog
):
    def dispatch(work_items, config, run_dir, **kwargs):
        path = os.path.join(run_dir, "product", "a.py")
        with open(path, "w", encoding="utf-8") as output:
            output.write("value = 1\n")
        return [AgentResult("WI-001", "FeatureAgent", True, ["product/a.py"])]

    original_write_manifest = runner.write_manifest

    def fail_manifest_correction(run_dir, manifest):
        if manifest.get("failure_reason") == "run_index_finalization_failure":
            raise OSError("manifest correction failed")
        original_write_manifest(run_dir, manifest)

    monkeypatch.setattr(runner, "dispatch_work_items", dispatch)
    monkeypatch.setattr(runner, "write_manifest", fail_manifest_correction)
    monkeypatch.setattr(
        runner,
        "append_index",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("index write failed")),
    )
    updates = []

    with caplog.at_level(logging.ERROR):
        result = runner.run_pipeline(
            _config(tmp_path),
            work_items=[_work_item()],
            label="publication failures",
            progress=updates.append,
        )

    run_dir = next(path for path in (tmp_path / "runs").iterdir() if path.is_dir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert result is False
    assert manifest["status"] == "validation_passed_external_testing_pending"
    assert updates[-1]["outcome"] == "failed"
    assert updates[-1]["validation_status"] == "passed"
    assert "Could not append finalized run index entry" in caplog.text
    assert "Could not correct run manifest after index failure" in caplog.text
    assert "on-disk manifest may still report a completed workflow" in caplog.text
    assert "manifest correction failed" in caplog.text
    assert teardown_spy == [True]
