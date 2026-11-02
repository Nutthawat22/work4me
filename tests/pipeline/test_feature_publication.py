"""Publication failure is reported and preserves detectable partial edits."""

import json
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline import runner
from pipeline.state import OwnedFile, WorkItem
from specialists import file_delivery
from specialists.llm_client import LLMResponse


def _config(tmp_path) -> dict:
    return {
        "litellm_url": "https://litellm.example.test/v1",
        "litellm_key": "test-secret",
        "models": {
            "design": {"model": "mock", "provider": "acp"},
            "specialist": {"model": "mock", "provider": "acp"},
        },
        "pipeline": {
            "max_retries": 1,
            "runs_dir": str(tmp_path / "runs"),
            "output_dir": "product",
            "tests_dir": "tests",
            "tests_blocking": False,
        },
        "languages": {"python": {"test_command": ["pytest", "{tests_dir}"]}},
    }


def _frame(messages, source: str) -> str:
    markers = re.search(
        r"Framing: first line is (BEGIN-OWNED-FILE:[A-Za-z0-9_-]+); last line is (END-OWNED-FILE:[A-Za-z0-9_-]+)",
        messages[1]["content"],
    )
    assert markers is not None
    return f"{markers.group(1)}\n{source}\n{markers.group(2)}"


def test_partial_publish_rollback_failure_is_reported_in_manifest_and_changes(
    monkeypatch, tmp_path
):
    project = tmp_path / "project"
    project.mkdir()
    (project / "a.py").write_text("old_a = True\n")
    (project / "b.py").write_text("old_b = True\n")
    item = WorkItem(
        id="TASK-feature",
        category="feature",
        language="python",
        module="example",
        scope="Update both related project files",
        title="Example feature",
        description="Update the feature as one owned unit",
        acceptance_criteria=["both files are valid Python"],
        requirement_refs=[],
        owned_files=[OwnedFile("a.py", "python"), OwnedFile("b.py", "python")],
    )
    source_by_file = {"a.py": "new_a = True\n", "b.py": "new_b = True\n"}

    def respond(messages, model, config, **kwargs):
        target = re.search(r"CURRENT TARGET FILE .*?: ([^\n]+)", messages[1]["content"])
        assert target is not None
        name = target.group(1)
        return LLMResponse(_frame(messages, source_by_file[name]), "complete", "end_turn")

    original_replace = os.replace

    def fail_one_publish_and_rollback(source, target):
        source_name = os.path.basename(source)
        target_name = os.path.basename(target)
        if source_name.startswith(".stage-") and target_name == "b.py":
            raise OSError("mock second-file publish failure")
        if source_name.startswith(".rollback-") and target_name == "a.py":
            raise OSError("mock rollback failure")
        return original_replace(source, target)

    monkeypatch.setattr(file_delivery, "call_llm_result", respond)
    monkeypatch.setattr("specialists.base._self_check", lambda *args, **kwargs: None)
    monkeypatch.setattr("pipeline.staging.os.replace", fail_one_publish_and_rollback)
    monkeypatch.setattr(runner.acp_client, "teardown", lambda: None)

    for method in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, method, lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("the pipeline must not execute or install generated code")
        ))

    success = runner.run_pipeline(
        _config(tmp_path),
        work_items=[item],
        label="publication rollback regression",
        existing_project_path=str(project),
    )

    assert success is False
    assert (project / "a.py").read_text() == "new_a = True\n"
    assert (project / "b.py").read_text() == "old_b = True\n"
    run_dir = next(path for path in (tmp_path / "runs").iterdir() if path.is_dir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    changes = json.loads((run_dir / "changes.json").read_text())
    assert manifest["status"] == "failed"
    assert "rollback incomplete for: a.py" in manifest["publication_failures"][0]
    assert changes["modified"] == ["a.py"]
    assert changes["partial_edits_possible"] is True
    assert changes["publication_failures"] == manifest["publication_failures"]
    assert (run_dir / "testing-handoff.json").is_file()


def test_successful_feature_persists_file_progress_but_handoff_excludes_stage(
    monkeypatch, tmp_path
):
    sentinel = tmp_path / "generated-source-executed"
    item = WorkItem(
        id="TASK-feature",
        category="feature",
        language="python",
        module="example",
        scope="Implement a related pair of modules",
        title="Example feature",
        description="Implement a cohesive two-file feature",
        acceptance_criteria=["both source files are valid"],
        requirement_refs=[],
        owned_files=[OwnedFile("pkg/__init__.py", "python"), OwnedFile("pkg/app.py", "python")],
    )
    content = {
        "product/pkg/__init__.py": "",
        "product/pkg/app.py": f"open({str(sentinel)!r}, 'w').write('ran')\nvalue = 1\n",
    }

    def respond(messages, model, config, **kwargs):
        target = re.search(r"CURRENT TARGET FILE .*?: ([^\n]+)", messages[1]["content"])
        assert target is not None
        output_path = target.group(1)
        if output_path.endswith("__init__.py"):
            empty = re.search(r"Empty action .*?: (ACCEPT-EMPTY-FILE:[^\s]+)", messages[1]["content"])
            assert empty is not None
            source = empty.group(1)
        else:
            source = content[output_path]
        return LLMResponse(_frame(messages, source), "complete", "end_turn")

    monkeypatch.setattr(file_delivery, "call_llm_result", respond)
    monkeypatch.setattr("specialists.base._self_check", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner.acp_client, "teardown", lambda: None)
    for method in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, method, lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("the pipeline must not execute or install generated code")
        ))

    success = runner.run_pipeline(
        _config(tmp_path), work_items=[item], label="stage inventory regression"
    )

    assert success is True
    assert not sentinel.exists()
    run_dir = next(path for path in (tmp_path / "runs").iterdir() if path.is_dir())
    plan = json.loads((run_dir / "design/plan.json").read_text())
    task = plan["tasks"][0]
    assert task["status"] == "passed"
    latest_file_status = {file["path"]: file["status"] for file in task["file_progress"]}
    assert set(latest_file_status.values()) == {"published"}
    assert {attempt["status"] for attempt in task["attempts"][0]["file_attempts"]} >= {
        "generating", "staged", "published"
    }
    handoff_text = (run_dir / "testing-handoff.json").read_text()
    assert "staging/" not in handoff_text
    assert "stage.json" not in handoff_text
