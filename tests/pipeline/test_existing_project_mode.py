"""Existing-project context, static validation, and run-artifact isolation."""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline import runner as pipeline_runner
from pipeline.master import MasterAgent
from pipeline.project_context import build_project_context, summarize_project_changes
from pipeline.state import AgentResult, OwnedFile, WorkItem
from pipeline.static_validator import StaticValidator


def _work_item() -> WorkItem:
    return WorkItem(
        id="TASK-existing",
        category="feature",
        language="python",
        module="existing",
        scope="Change existing project",
        title="Update existing project",
        description="Update the application",
        acceptance_criteria=["existing app behavior remains valid"],
        requirement_refs=[],
        owned_files=[OwnedFile("src/app.py", "python")],
    )


def test_master_planning_receives_existing_project_context(monkeypatch):
    captured = []
    plan = {
        "items": [
            {
                "id": "TASK-existing",
                "category": "feature",
                "language": "python",
                "module": "existing",
                "scope": "update existing module",
                "title": "Update existing module",
                "description": "Modify the existing app implementation",
                "acceptance_criteria": ["the app still works"],
                "requirement_refs": [],
                "owned_files": [
                    {"path": "src/app.py", "language": "python", "area": "product"}
                ],
                "depends_on": [],
                "expected_interfaces": [],
            }
        ]
    }

    def fake_plan(messages, model, config, **kwargs):
        captured.extend(messages)
        return json.dumps(plan)

    monkeypatch.setattr("pipeline.master.acp_call_with_retry", fake_plan)
    master = MasterAgent({
        "languages": {"python": {}},
        "models": {"design": {"model": "m", "provider": "acp"}},
        "pipeline": {"output_dir": "product", "tests_dir": "tests"},
    })

    tasks = master.decompose(
        "update the app",
        project_context="Existing project source context:\n--- src/app.py ---\nprint('current')",
    )

    assert tasks[0].owned_files[0].path == "src/app.py"
    assert "update the app" in captured[1]["content"]
    assert "print('current')" in captured[1]["content"]
    assert "existing-project change" in captured[0]["content"]
    assert "relative to the project root" in captured[0]["content"]
    assert "A project-root file is difficulty.js" in captured[0]["content"]
    assert "project/difficulty.js only when the actual owned file" in captured[0]["content"]


def test_project_context_and_change_summary_include_additions_modifications_deletions(
    tmp_path,
):
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "src" / "modified.py").write_text("before\n")
    (project / "src" / "deleted.py").write_text("delete me\n")
    context, before = build_project_context(str(project))
    assert "src/modified.py" in context
    assert "before" in context

    (project / "src" / "modified.py").write_text("after\n")
    (project / "src" / "added.py").write_text("new\n")
    (project / "src" / "deleted.py").unlink()
    _, after = build_project_context(str(project))

    assert summarize_project_changes(before, after) == {
        "added": ["src/added.py"],
        "modified": ["src/modified.py"],
        "deleted": ["src/deleted.py"],
    }


def test_existing_project_static_validation_does_not_require_runtime_tests(tmp_path):
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "src" / "app.py").write_text("def app():\n    return True\n")
    config = {
        "pipeline": {
            "output_dir": "product",
            "tests_dir": "tests",
            "_existing_project_path": str(project),
        },
        "languages": {"python": {}},
    }
    result = StaticValidator().run(config, [_work_item()], str(tmp_path / "run"))

    assert result.passed
    assert result.checked_files == 1
    assert not list(project.glob("**/test_*.py"))


def test_run_pipeline_records_partial_changes_outside_project_without_rollback(
    monkeypatch, tmp_path
):
    project = tmp_path / "target"
    (project / "src").mkdir(parents=True)
    (project / "src" / "edit.py").write_text("before\n")
    (project / "src" / "delete.py").write_text("remove\n")
    runs_root = tmp_path / "server-runs"
    config = {
        "pipeline": {
            "runs_dir": str(runs_root),
            "output_dir": "product",
            "tests_dir": "tests",
            "max_retries": 1,
            "tests_blocking": False,
        },
        "languages": {},
    }
    artifact_dir = runs_root / "0001-edit-project"

    def fake_body(run_config, user_input, work_items, label, progress):
        run_config["pipeline"]["_active_run_dir"] = str(artifact_dir)
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "manifest.json").write_text(json.dumps({"status": "in_progress"}))
        (project / "src" / "edit.py").write_text("after\n")
        (project / "src" / "added.py").write_text("new\n")
        (project / "src" / "delete.py").unlink()
        assert run_config["pipeline"]["tests_blocking"] is False
        assert "src/edit.py" in run_config["pipeline"]["_existing_project_context"]
        return False

    monkeypatch.setattr(pipeline_runner, "_run_pipeline_body", fake_body)
    monkeypatch.setattr(pipeline_runner.acp_client, "teardown", lambda: None)

    result = pipeline_runner.run_pipeline(
        config,
        user_input="edit project",
        existing_project_path=str(project),
    )

    changes = json.loads((artifact_dir / "changes.json").read_text())
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    handoff = json.loads((artifact_dir / "testing-handoff.json").read_text())
    assert result is False
    assert changes["added"] == ["src/added.py"]
    assert changes["modified"] == ["src/edit.py"]
    assert changes["deleted"] == ["src/delete.py"]
    assert changes["partial_edits_possible"] is True
    assert "not rolled back" in changes["warning"]
    assert manifest["changes"] == {
        "added": ["src/added.py"],
        "modified": ["src/edit.py"],
        "deleted": ["src/delete.py"],
    }
    assert handoff["partial_edits_possible"] is True
    assert "not rolled back" in handoff["partial_edit_warning"]
    assert config["pipeline"]["tests_blocking"] is False
    assert not (project / "changes.json").exists()


def test_existing_project_pipeline_passes_static_validation_without_runtime_tests(
    monkeypatch, tmp_path
):
    project = tmp_path / "target"
    (project / "src").mkdir(parents=True)
    (project / "src" / "app.py").write_text("pass\n")
    runs_root = tmp_path / "server-runs"
    config = {
        "pipeline": {
            "runs_dir": str(runs_root),
            "output_dir": "product",
            "tests_dir": "tests",
            "max_retries": 1,
            "tests_blocking": False,
        },
        "languages": {
            "python": {
                "test_file_patterns": ["test_*.py"],
                "test_command": ["pytest", "{tests_dir}"],
                "result_format": "junit",
            }
        },
    }
    item = _work_item()
    monkeypatch.setattr(
        pipeline_runner,
        "dispatch_work_items",
          lambda work_items, config, run_dir, **kwargs: [
              AgentResult(item.id, "FeatureAgent", True, ["src/app.py"])
        ],
    )
    monkeypatch.setattr(pipeline_runner.acp_client, "teardown", lambda: None)

    result = pipeline_runner.run_pipeline(
        config,
        work_items=[item],
        label="existing project without runtime tests",
        existing_project_path=str(project),
    )

    run_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text())
    validation_result = json.loads((run_dirs[0] / "validation/results/attempt-1.json").read_text())
    changes = json.loads((run_dirs[0] / "changes.json").read_text())
    handoff = json.loads((run_dirs[0] / "testing-handoff.json").read_text())
    assert result is True
    assert manifest["status"] == "validation_passed_external_testing_pending"
    assert manifest["external_testing_status"] == "pending"
    assert validation_result["passed"] is True
    assert changes["partial_edits_possible"] is False
    assert handoff["external_testing"]["status"] == "pending"
    assert not (project / "tests").exists()
