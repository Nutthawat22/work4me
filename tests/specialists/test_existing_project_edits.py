"""Existing-project file actions stay path-bound and staged until complete."""

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline.state import OwnedFile, WorkItem
from specialists import file_delivery
from specialists.llm_client import LLMResponse
from specialists.base import run_multifile_specialist


def _task(*paths: str) -> WorkItem:
    return WorkItem(
        id="TASK-edit",
        category="feature",
        language="python",
        module="existing module",
        scope="Update existing implementation",
        title="Update module",
        description="Modify and remove the listed project files",
        acceptance_criteria=["the updated module works"],
        requirement_refs=["REQ-001"],
        owned_files=[OwnedFile(path, "python") for path in paths],
    )


def _config(project_root) -> dict:
    return {
        "models": {"specialist": {"model": "m", "provider": "acp"}},
        "pipeline": {
            "output_dir": "product",
            "tests_dir": "tests",
            "max_retries": 2,
            "_stage_plan_digest": "existing-project-test-plan",
            "_existing_project_path": str(project_root),
            "_existing_project_context": "Existing source context",
        },
    }


def _frame(messages, source: str) -> str:
    markers = re.search(
        r"Framing: first line is (BEGIN-OWNED-FILE:[A-Za-z0-9_-]+); last line is (END-OWNED-FILE:[A-Za-z0-9_-]+)",
        messages[1]["content"],
    )
    assert markers is not None
    return f"{markers.group(1)}\n{source}\n{markers.group(2)}"


def _action(messages, label: str, marker: str) -> str:
    match = re.search(rf"{label} .*?: ({marker}:[^\s]+)", messages[1]["content"])
    assert match is not None
    return match.group(1)


def _install(monkeypatch, responder):
    monkeypatch.setattr(file_delivery, "call_llm_result", responder)
    monkeypatch.setattr("specialists.base._self_check", lambda *args, **kwargs: None)


def test_feature_modifies_adds_and_explicitly_deletes_owned_files(monkeypatch, tmp_path):
    project = tmp_path / "project"
    source_dir = project / "src"
    source_dir.mkdir(parents=True)
    (source_dir / "app.py").write_text("old = True\n")
    (source_dir / "remove.py").write_text("obsolete = True\n")
    (source_dir / "unowned.py").write_text("keep = True\n")
    responses = {
        "src/app.py": "updated = True\n",
        "src/new.py": "created = True\n",
    }
    calls = []

    def respond(messages, model, config, **kwargs):
        calls.append(messages)
        target = re.search(r"CURRENT TARGET FILE .*?: ([^\n]+)", messages[1]["content"])
        assert target is not None
        body = (
            _action(messages, "Deletion action", "INTENTIONAL-DELETE-FILE")
            if target.group(1) == "src/remove.py"
            else responses[target.group(1)]
        )
        return LLMResponse(_frame(messages, body), "complete", "end_turn")

    _install(monkeypatch, respond)
    result = run_multifile_specialist(
        _task("src/app.py", "src/new.py", "src/remove.py"),
        _config(project),
        str(tmp_path / "run"),
        "Implement the task.",
        "FeatureAgent",
    )

    assert result.success
    assert result.files_written == ["src/app.py", "src/new.py"]
    assert result.files_deleted == ["src/remove.py"]
    assert (source_dir / "app.py").read_text() == "updated = True\n"
    assert (source_dir / "new.py").read_text() == "created = True\n"
    assert not (source_dir / "remove.py").exists()
    assert (source_dir / "unowned.py").read_text() == "keep = True\n"
    assert len(calls) == 3
    assert "Existing source context" in calls[0][1]["content"]
    assert "Current project file: src/app.py" in calls[0][1]["content"]
    assert "old = True" in calls[0][1]["content"]
    assert "Accepted staged sibling file: src/app.py" in calls[1][1]["content"]


def test_exact_owned_project_relative_path_is_requested_and_not_model_selected(monkeypatch, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    target = project / "difficulty.py"
    calls = []

    def respond(messages, model, config, **kwargs):
        calls.append(messages)
        return LLMResponse(_frame(messages, "value = 'correct path'\n"), "complete", "end_turn")

    _install(monkeypatch, respond)
    result = run_multifile_specialist(
        _task("difficulty.py"), _config(project), str(tmp_path / "run"), "Implement the task.", "FeatureAgent"
    )

    assert result.success
    assert result.files_written == ["difficulty.py"]
    assert target.read_text() == "value = 'correct path'\n"
    assert len(calls) == 1
    assert "CURRENT TARGET FILE (the only file to return): difficulty.py" in calls[0][1]["content"]
    assert "Do not choose or alter a path" in calls[0][0]["content"]


def test_source_cannot_select_an_unowned_path(monkeypatch, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.py"
    _install(
        monkeypatch,
        lambda messages, *args, **kwargs: LLMResponse(
            _frame(messages, "value = '../outside.py'\n"), "complete", "end_turn"
        ),
    )

    result = run_multifile_specialist(
        _task("owned.py"), _config(project), str(tmp_path / "run"), "Implement the task.", "FeatureAgent"
    )

    assert result.success
    assert result.files_written == ["owned.py"]
    assert not outside.exists()
    assert (project / "owned.py").is_file()


def test_specialist_rejects_project_path_traversal_before_llm_call(monkeypatch, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    called = []
    monkeypatch.setattr(file_delivery, "call_llm_result", lambda *args, **kwargs: called.append(True))

    result = run_multifile_specialist(
        _task("../outside.py"), _config(project), str(tmp_path / "run"), "Implement the task.", "FeatureAgent"
    )

    assert not result.success
    assert "unsafe component" in result.notes
    assert called == []


def test_specialist_rejects_symlink_escape_before_llm_call(monkeypatch, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / "escape").symlink_to(outside, target_is_directory=True)
    called = []
    monkeypatch.setattr(file_delivery, "call_llm_result", lambda *args, **kwargs: called.append(True))

    result = run_multifile_specialist(
        _task("escape/owned.py"), _config(project), str(tmp_path / "run"), "Implement the task.", "FeatureAgent"
    )

    assert not result.success
    assert "escapes or aliases the project" in result.notes
    assert not (outside / "owned.py").exists()
    assert called == []


def test_specialist_cannot_write_into_pipeline_run_artifacts(monkeypatch, tmp_path):
    project = tmp_path / "project"
    protected_runs = project / "runs"
    protected_runs.mkdir(parents=True)
    config = _config(project)
    config["pipeline"]["_protected_run_root"] = str(protected_runs)
    called = []
    monkeypatch.setattr(file_delivery, "call_llm_result", lambda *args, **kwargs: called.append(True))

    result = run_multifile_specialist(
        _task("runs/unrelated.py"), config, str(tmp_path / "run"), "Implement the task.", "FeatureAgent"
    )

    assert not result.success
    assert "reserved for pipeline run artifacts" in result.notes
    assert not (protected_runs / "unrelated.py").exists()
    assert called == []


def test_specialist_rejects_generated_and_vendor_directories(monkeypatch, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    called = []
    monkeypatch.setattr(file_delivery, "call_llm_result", lambda *args, **kwargs: called.append(True))

    result = run_multifile_specialist(
        _task("node_modules/pkg/file.py"), _config(project), str(tmp_path / "run"), "Implement the task.", "FeatureAgent"
    )

    assert not result.success
    assert "excluded generated/vendor directory" in result.notes
    assert called == []


def test_owned_path_under_a_directory_named_project_is_preserved(monkeypatch, tmp_path):
    project = tmp_path / "project"
    owned_dir = project / "project"
    owned_dir.mkdir(parents=True)
    target = owned_dir / "difficulty.py"
    _install(
        monkeypatch,
        lambda messages, *args, **kwargs: LLMResponse(
            _frame(messages, "legitimate = True\n"), "complete", "end_turn"
        ),
    )

    result = run_multifile_specialist(
        _task("project/difficulty.py"), _config(project), str(tmp_path / "run"), "Implement the task.", "FeatureAgent"
    )

    assert result.success
    assert result.files_written == ["project/difficulty.py"]
    assert target.read_text() == "legitimate = True\n"
