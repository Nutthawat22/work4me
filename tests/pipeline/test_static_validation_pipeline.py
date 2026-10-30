import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline import runner
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
            "max_retries": 2,
            "runs_dir": str(tmp_path / "runs"),
            "output_dir": "product",
            "tests_dir": "tests",
            "tests_blocking": False,
        },
        "languages": {"python": {}},
    }


def _work_item() -> WorkItem:
    return WorkItem(
        id="TASK-app", category="feature", language="python", module="app",
        scope="Implement app", title="App", description="Implement app",
        acceptance_criteria=["app.py is syntactically valid"], requirement_refs=[],
        owned_files=[OwnedFile("app.py", "python")],
    )


def test_static_validation_error_is_sent_to_repair_dispatch_and_clears(tmp_path, monkeypatch):
    dispatch_calls: list[dict] = []

    def dispatch(work_items, config, run_dir, **kwargs):
        dispatch_calls.append(kwargs)
        source = os.path.join(run_dir, "product", "app.py")
        with open(source, "w", encoding="utf-8") as output:
            output.write("def broken(:\n    pass\n" if len(dispatch_calls) == 1 else "def app():\n    return True\n")
        return [AgentResult("TASK-app", "FeatureAgent", True, ["product/app.py"])]

    monkeypatch.setattr(runner, "dispatch_work_items", dispatch)
    monkeypatch.setattr(runner.acp_client, "teardown", lambda: None)

    result = runner.run_pipeline(
        _config(tmp_path), work_items=[_work_item()], label="repair syntax"
    )

    assert result is True
    assert len(dispatch_calls) == 2
    assert dispatch_calls[1]["retry_context"]["TASK-app"]
    run_dir = next(path for path in (tmp_path / "runs").iterdir() if path.is_dir())
    first_attempt = json.loads((run_dir / "validation/results/attempt-1.json").read_text())
    final_attempt = json.loads((run_dir / "validation/results/attempt-2.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert first_attempt["passed"] is False
    assert "Python syntax error" in first_attempt["errors"][0]["error_output"]
    assert final_attempt["passed"] is True
    assert manifest["status"] == "validation_passed_external_testing_pending"
    assert manifest["external_testing_status"] == "pending"


def test_real_dispatch_records_migrated_retry_issues_and_revalidates_repair(tmp_path, monkeypatch):
    specialist_calls = []
    item = _work_item()

    class RepairingFeatureAgent:
        def execute(self, item, config, run_dir, *, retry_failures=None, **kwargs):
            specialist_calls.append(list(retry_failures or []))
            source = os.path.join(run_dir, "product", "app.py")
            with open(source, "w", encoding="utf-8") as output:
                output.write("return\n" if len(specialist_calls) == 1 else "def app():\n    return True\n")
            return AgentResult(item.id, "FeatureAgent", True, ["product/app.py"])

    monkeypatch.setattr("pipeline.dispatch.FeatureAgent", RepairingFeatureAgent)
    monkeypatch.setattr(runner.acp_client, "teardown", lambda: None)

    result = runner.run_pipeline(
        _config(tmp_path), work_items=[item], label="retry migrated issue"
    )

    assert result is True
    assert len(specialist_calls) == 2
    assert specialist_calls[0] == []
    assert specialist_calls[1][0].file_path == "product/app.py"
    assert "Python compile error" in specialist_calls[1][0].error_output
    assert len(item.attempts) == 2
    assert "product/app.py: Python compile error" in item.attempts[1].failure_details
    run_dir = next(path for path in (tmp_path / "runs").iterdir() if path.is_dir())
    first = json.loads((run_dir / "validation/results/attempt-1.json").read_text())
    second = json.loads((run_dir / "validation/results/attempt-2.json").read_text())
    assert first["passed"] is False
    assert second["passed"] is True


@pytest.mark.parametrize("invalid_source", [False, True], ids=["pipeline-success", "pipeline-failure"])
def test_malformed_manifest_shapes_keep_handoff_on_pipeline_success_and_failure(
    tmp_path, monkeypatch, invalid_source
):
    project = tmp_path / "existing-project"
    (project / "src").mkdir(parents=True)
    (project / "src" / "app.py").write_text("value = 1\n")
    (project / "pyproject.toml").write_text("[tool.poetry]\ndependencies = []\n")
    (project / "Pipfile").write_text("packages = []\n")
    (project / "package.json").write_text(json.dumps({"scripts": []}))
    config = _config(tmp_path)
    item = _work_item()
    item.owned_files = [OwnedFile("src/app.py", "python")]

    def dispatch(work_items, run_config, run_dir, **kwargs):
        source = project / "src" / "app.py"
        source.write_text("return\n" if invalid_source else "value = 2\n")
        return [AgentResult(item.id, "FeatureAgent", True, ["src/app.py"])]

    monkeypatch.setattr(runner, "dispatch_work_items", dispatch)
    monkeypatch.setattr(runner.acp_client, "teardown", lambda: None)

    result = runner.run_pipeline(
        config,
        work_items=[item],
        label="malformed existing metadata",
        existing_project_path=str(project),
    )

    run_dir = next(path for path in (tmp_path / "runs").iterdir() if path.is_dir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    handoff = json.loads((run_dir / "testing-handoff.json").read_text())
    assert result is (not invalid_source)
    assert manifest["status"] == ("validation_failed" if invalid_source else "validation_passed_external_testing_pending")
    assert handoff["external_testing"]["status"] == "pending"
    assert any("malformed metadata shapes" in warning for warning in handoff["handoff_warnings"])


@pytest.mark.parametrize("existing_project", [False, True], ids=["new-project", "existing-project"])
def test_pipeline_never_installs_or_executes_generated_project_code(
    tmp_path, monkeypatch, existing_project
):
    sentinel = tmp_path / "generated-code-executed"
    project = tmp_path / "project"
    if existing_project:
        project.mkdir()
    config = _config(tmp_path)
    config["pipeline"]["max_retries"] = 3
    config["languages"]["python"]["test_command"] = ["pytest", "{tests_dir}"]
    app_path = "src/app.py" if existing_project else "app.py"
    test_path = "tests/test_app.py" if existing_project else "test_app.py"
    work_items = [
        WorkItem(
            id="TASK-app", category="feature", language="python", module="app",
            scope="Generate app source", title="App", description="Generate app",
            acceptance_criteria=["app source is syntactically valid"], requirement_refs=[],
            owned_files=[OwnedFile(app_path, "python")],
        ),
        WorkItem(
            id="TASK-tests", category="validation", language="python", module="tests",
            scope="Generate test source", title="Tests", description="Generate test source",
            acceptance_criteria=["test source is syntactically valid"], requirement_refs=[],
            owned_files=[OwnedFile(test_path, "python", "tests")],
        ),
    ]
    dispatch_calls = []

    def dispatch(work_items, run_config, run_dir, **kwargs):
        dispatch_calls.append(1)
        results = []
        for item in work_items:
            written = []
            for owned_file in item.owned_files:
                if existing_project:
                    path = project / owned_file.path
                    relative = owned_file.path
                else:
                    area = "tests" if owned_file.area == "tests" else "product"
                    path = tmp_path / "runs" / os.path.basename(run_dir) / area / owned_file.path
                    relative = f"{area}/{owned_file.path}"
                path.parent.mkdir(parents=True, exist_ok=True)
                if owned_file.area == "tests":
                    path.write_text(f"open({str(sentinel)!r}, 'w').write('executed')\n")
                else:
                    path.write_text("import undeclared_generated_dependency\n")
                written.append(relative)
            results.append(AgentResult(item.id, "FeatureAgent", True, written))
        return results

    def forbidden_process(*args, **kwargs):
        raise AssertionError("generated-project commands must not be started by the pipeline")

    for process_api in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, process_api, forbidden_process)
    monkeypatch.setattr(runner, "dispatch_work_items", dispatch)
    monkeypatch.setattr(runner.acp_client, "teardown", lambda: None)

    result = runner.run_pipeline(
        config,
        work_items=work_items,
        label="static validation only",
        existing_project_path=str(project) if existing_project else None,
    )

    run_dir = next((tmp_path / "runs").iterdir())
    handoff = json.loads((run_dir / "testing-handoff.json").read_text())
    assert result is True
    assert dispatch_calls == [1]
    assert not sentinel.exists()
    assert handoff["external_testing"]["status"] == "pending"
    assert "undeclared_generated_dependency" in handoff["unresolved_or_undeclared_dependencies"]
