"""Existing-project context, static validation, and run-artifact isolation."""

import asyncio
import json
import os
import sys
from pathlib import Path

import acp
import pytest

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


@pytest.mark.parametrize("failure", [None, "static", "handoff", "result", "index", "teardown", "cancel"])
def test_external_code_bridge_receives_final_outcome_and_honest_delivery_evidence(
    monkeypatch, tmp_path, failure
):
    from pipeline import acp_server

    project = tmp_path / "code-worktree"
    (project / "src").mkdir(parents=True)
    (project / "src/app.py").write_text("value = 1\n")
    runs = tmp_path / "provider-runs"
    config = {
        "pipeline": {"runs_dir": str(runs), "output_dir": "product", "tests_dir": "tests", "max_retries": 1},
        "languages": {"python": {}},
    }
    item = _work_item()
    item.requirement_refs = ["AC-1", "REQ-2", "TEST-3"]
    monkeypatch.setattr(acp_server, "load_config", lambda: (config, "/test/config.json"))
    monkeypatch.setattr(MasterAgent, "decompose", lambda self, *args, **kwargs: [item])
    monkeypatch.setattr(pipeline_runner.acp_client, "teardown", lambda: None)
    agent = acp_server.PipelineAgent()

    def dispatch(*args, **kwargs):
        (project / "src/app.py").write_text("return\n" if failure == "static" else "value = 2\n")
        if failure == "cancel":
            next(iter(agent._sessions.values())).cancelled = True
        return [AgentResult(item.id, "FeatureAgent", True, ["src/app.py"])]

    monkeypatch.setattr(pipeline_runner, "dispatch_work_items", dispatch)

    def fail(*args, **kwargs):
        raise OSError("forced finalization failure")

    if failure == "handoff":
        monkeypatch.setattr(pipeline_runner, "write_testing_handoff", fail)
    elif failure == "result":
        monkeypatch.setattr(pipeline_runner, "write_implementation_result", fail)
    elif failure == "index":
        monkeypatch.setattr(pipeline_runner, "append_index", fail)
    elif failure == "teardown":
        monkeypatch.setattr(pipeline_runner.acp_client, "teardown", fail)

    async def host_exchange():
        await agent.initialize(protocol_version=acp.PROTOCOL_VERSION)
        session = await agent.new_session(cwd=str(tmp_path))
        await agent.ext_method("agents-pipeline/session/configure", {
            "sessionId": session.session_id, "mode": "existing_project", "projectPath": str(project),
        })
        response = await agent.prompt(session.session_id, [acp.text_block("Implement approved AC-1 in the Code worktree.")])
        status = await agent.ext_method("agents-pipeline/run/status", {"sessionId": session.session_id})
        return response, status

    response, status = asyncio.run(host_exchange())
    # This is exactly the completion predicate in AI-SDLC's external bridge.
    assert (status["outcome"] == "passed") is (failure is None)
    assert status["schema_version"] == 2
    assert status["outcome_scope"] == "implementation"
    assert status["external_testing_status"] == "pending"
    assert response.stop_reason == ("refusal" if failure == "teardown" else "cancelled" if failure == "cancel" else "end_turn")
    assert status["project_path"] == str(project)
    assert sorted(path.relative_to(project).as_posix() for path in project.rglob("*") if path.is_file()) == ["src/app.py"]

    run_dir = Path(status["run_dir"])
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if failure == "result":
        assert status["implementation_result_path"] is None
        assert manifest["failure_reason"] == "implementation_result_failure"
        return
    evidence = json.loads(Path(status["implementation_result_path"]).read_text())
    assert evidence["mode"] == "existing_project"
    assert evidence["project_path"] == str(project)
    assert evidence["delivery_mode"] == "working_tree"
    assert evidence["unit_tests"] is None
    assert evidence["lint"] == "not_run"
    assert evidence["tasks"] == [{
        "id": "T-001", "work_item_id": item.id, "title": item.title,
        "status": "done", "commits": [], "files": ["src/app.py"],
    }]
    assert evidence["ac_coverage"] == [{"ac_id": "AC-1", "tasks": ["T-001"], "files": ["src/app.py"]}]
    assert evidence["published_files"] == manifest["published_files"]
    if failure is None:
        assert status["status"] == "completed"
        assert status["pipeline_outcome"] == "validation_passed_external_testing_pending"
        assert status["validation_status"] == "passed"
        assert manifest["status"] == "validation_passed_external_testing_pending"
        handoff = json.loads((run_dir / "testing-handoff.json").read_text())
        assert handoff["external_testing"]["status"] == "pending"
    elif failure == "cancel":
        assert status["outcome"] == "cancelled"
