"""Existing-project specialist writes are restricted to owned project paths."""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline.state import OwnedFile, WorkItem
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
            "_existing_project_path": str(project_root),
            "_existing_project_context": "Existing source context: src/app.py",
        },
    }


def test_specialist_modifies_adds_and_deletes_owned_files_in_place(
    monkeypatch, tmp_path
):
    project = tmp_path / "project"
    source_dir = project / "src"
    source_dir.mkdir(parents=True)
    (source_dir / "app.py").write_text("old implementation\n")
    (source_dir / "remove.py").write_text("obsolete\n")
    (source_dir / "unowned.py").write_text("keep me\n")
    calls = []
    responses = [
        json.dumps({
            "files": {
                "src/app.py": "updated implementation\n",
                "src/new.py": "new implementation\n",
            },
            "deleted_files": ["src/remove.py"],
        }),
        json.dumps({"accepted": True, "issues": [], "reasoning": "matches task"}),
    ]

    def fake_call_llm(messages, *args, **kwargs):
        calls.append(messages)
        return responses.pop(0)

    monkeypatch.setattr("specialists.base.call_llm", fake_call_llm)
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
    assert (source_dir / "app.py").read_text() == "updated implementation\n"
    assert (source_dir / "new.py").read_text() == "new implementation\n"
    assert not (source_dir / "remove.py").exists()
    assert (source_dir / "unowned.py").read_text() == "keep me\n"
    generation_prompt = calls[0][1]["content"]
    assert "Existing source context" in generation_prompt
    assert "Current project file: src/app.py" in generation_prompt
    assert "old implementation" in generation_prompt
    assert "src/remove.py" in calls[1][1]["content"]


def test_prefixed_output_path_gets_one_correction_before_any_write(monkeypatch, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    target = project / "difficulty.js"
    calls = []
    responses = [
        json.dumps({"files": {"project/difficulty.js": "wrong path"}}),
        json.dumps({"files": {"difficulty.js": "corrected content"}}),
        json.dumps({"accepted": True, "issues": [], "reasoning": "matches task"}),
    ]

    def fake_call_llm(messages, *args, **kwargs):
        calls.append(messages)
        if len(calls) == 2:
            assert not target.exists()
            feedback = messages[1]["content"]
            assert "project/difficulty.js" in feedback
            assert "difficulty.js" in feedback
            assert "Do not prepend `project/`" in feedback
        return responses.pop(0)

    monkeypatch.setattr("specialists.base.call_llm", fake_call_llm)
    result = run_multifile_specialist(
        _task("difficulty.js"),
        _config(project),
        str(tmp_path / "run"),
        "Implement the task.",
        "FeatureAgent",
    )

    assert result.success
    assert result.files_written == ["difficulty.js"]
    assert target.read_text() == "corrected content"
    assert len(calls) == 3
    assert "- difficulty.js (python)" in calls[0][1]["content"]
    assert "- project/difficulty.js (python)" not in calls[0][1]["content"]


def test_persistent_unowned_path_fails_after_one_correction_without_writes(
    monkeypatch, tmp_path
):
    project = tmp_path / "project"
    project.mkdir()
    responses = [
        json.dumps({"files": {"project/difficulty.js": "first"}}),
        json.dumps({"files": {"project/difficulty.js": "still wrong"}}),
    ]
    calls = []

    def fake_call_llm(messages, *args, **kwargs):
        calls.append(messages)
        return responses.pop(0)

    monkeypatch.setattr("specialists.base.call_llm", fake_call_llm)
    result = run_multifile_specialist(
        _task("difficulty.js"),
        _config(project),
        str(tmp_path / "run"),
        "Implement the task.",
        "FeatureAgent",
    )

    assert not result.success
    assert result.files_written == []
    assert "supplied paths=['project/difficulty.js']" in result.notes
    assert "exact owned project-root-relative paths=['difficulty.js']" in result.notes
    assert len(calls) == 2
    assert not (project / "difficulty.js").exists()
    assert not (project / "project" / "difficulty.js").exists()


def test_unsafe_path_discards_whole_candidate_before_corrected_retry(monkeypatch, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    target = project / "difficulty.js"
    outside = tmp_path / "outside.js"
    calls = []
    responses = [
        json.dumps({
            "files": {
                "difficulty.js": "must not be written",
                "../outside.js": "unsafe",
            }
        }),
        json.dumps({"files": {"difficulty.js": "corrected content"}}),
        json.dumps({"accepted": True, "issues": [], "reasoning": "matches task"}),
    ]

    def fake_call_llm(messages, *args, **kwargs):
        calls.append(messages)
        if len(calls) == 2:
            assert not target.exists()
            assert "../outside.js" in messages[1]["content"]
            assert "exact owned project-root-relative paths are: ['difficulty.js']" in messages[1]["content"]
        return responses.pop(0)

    monkeypatch.setattr("specialists.base.call_llm", fake_call_llm)
    result = run_multifile_specialist(
        _task("difficulty.js"),
        _config(project),
        str(tmp_path / "run"),
        "Implement the task.",
        "FeatureAgent",
    )

    assert result.success
    assert len(calls) == 3
    assert result.files_written == ["difficulty.js"]
    assert target.read_text() == "corrected content"
    assert not outside.exists()


def test_persistent_unsafe_path_fails_without_writing_valid_subset(monkeypatch, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    target = project / "difficulty.js"
    calls = []
    invalid = json.dumps({
        "files": {
            "difficulty.js": "must never be written",
            "../outside.js": "unsafe",
        }
    })

    def fake_call_llm(messages, *args, **kwargs):
        calls.append(messages)
        if len(calls) == 2:
            assert not target.exists()
        return invalid

    monkeypatch.setattr("specialists.base.call_llm", fake_call_llm)
    result = run_multifile_specialist(
        _task("difficulty.js"),
        _config(project),
        str(tmp_path / "run"),
        "Implement the task.",
        "FeatureAgent",
    )

    assert not result.success
    assert result.files_written == []
    assert "Unsafe output path '../outside.js'" in result.notes
    assert "exact owned project-root-relative paths=['difficulty.js']" in result.notes
    assert len(calls) == 2
    assert not target.exists()
    assert not (tmp_path / "outside.js").exists()


def test_owned_path_inside_project_directory_is_not_stripped(monkeypatch, tmp_path):
    project = tmp_path / "project"
    owned_dir = project / "project"
    owned_dir.mkdir(parents=True)
    target = owned_dir / "difficulty.js"
    calls = []
    responses = [
        json.dumps({"files": {"project/difficulty.js": "legitimate content"}}),
        json.dumps({"accepted": True, "issues": [], "reasoning": "matches task"}),
    ]

    def fake_call_llm(messages, *args, **kwargs):
        calls.append(messages)
        return responses.pop(0)

    monkeypatch.setattr("specialists.base.call_llm", fake_call_llm)
    result = run_multifile_specialist(
        _task("project/difficulty.js"),
        _config(project),
        str(tmp_path / "run"),
        "Implement the task.",
        "FeatureAgent",
    )

    assert result.success
    assert result.files_written == ["project/difficulty.js"]
    assert target.read_text() == "legitimate content"
    assert "- project/difficulty.js (python)" in calls[0][1]["content"]


def test_specialist_rejects_project_path_traversal_before_llm_call(monkeypatch, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.py"
    called = []
    monkeypatch.setattr("specialists.base.call_llm", lambda *args, **kwargs: called.append(True))

    result = run_multifile_specialist(
        _task("../outside.py"),
        _config(project),
        str(tmp_path / "run"),
        "Implement the task.",
        "FeatureAgent",
    )

    assert not result.success
    assert "unsafe component" in result.notes
    assert not outside.exists()
    assert called == []


def test_specialist_rejects_symlink_escape_before_write(monkeypatch, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / "escape").symlink_to(outside, target_is_directory=True)
    called = []
    monkeypatch.setattr("specialists.base.call_llm", lambda *args, **kwargs: called.append(True))

    result = run_multifile_specialist(
        _task("escape/owned.py"),
        _config(project),
        str(tmp_path / "run"),
        "Implement the task.",
        "FeatureAgent",
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
    monkeypatch.setattr("specialists.base.call_llm", lambda *args, **kwargs: called.append(True))

    result = run_multifile_specialist(
        _task("runs/unrelated.py"),
        config,
        str(tmp_path / "run"),
        "Implement the task.",
        "FeatureAgent",
    )

    assert not result.success
    assert "reserved for pipeline run artifacts" in result.notes
    assert not (protected_runs / "unrelated.py").exists()
    assert called == []


def test_specialist_rejects_generated_and_vendor_directories(monkeypatch, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    called = []
    monkeypatch.setattr("specialists.base.call_llm", lambda *args, **kwargs: called.append(True))

    result = run_multifile_specialist(
        _task("node_modules/pkg/file.py"),
        _config(project),
        str(tmp_path / "run"),
        "Implement the task.",
        "FeatureAgent",
    )

    assert not result.success
    assert "excluded generated/vendor directory" in result.notes
    assert called == []
