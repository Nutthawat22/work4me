"""A task fails before writing unless it returns exactly its safe owned files."""

import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline.state import OwnedFile, WorkItem
from specialists.base import run_multifile_specialist


def _task(*paths: str) -> WorkItem:
    return WorkItem(
        id="TASK-feature",
        category="feature",
        language="python",
        module="example",
        scope="Implement one cohesive example feature",
        title="Example feature",
        description="Implement the module files listed below",
        acceptance_criteria=["the feature works"],
        requirement_refs=["REQ-001"],
        owned_files=[OwnedFile(path, "python") for path in paths],
    )


def _run(monkeypatch, tmp_path, item: WorkItem, response: str):
    monkeypatch.setattr("specialists.base.call_llm", lambda *args, **kwargs: response)
    config = {
        "models": {"specialist": {"model": "m", "provider": "acp"}},
        "pipeline": {"output_dir": "product", "tests_dir": "tests"},
    }
    return run_multifile_specialist(item, config, str(tmp_path), "Feature", "FeatureAgent")


@pytest.mark.parametrize(
    "files",
    [
        {},
        {"one.py": "content"},
        {"one.py": "content", "two.py": "content", "extra.py": "content"},
        {"one.py": "   ", "two.py": "content"},
        {"../outside.py": "content", "two.py": "content"},
        {"/tmp/outside.py": "content", "two.py": "content"},
    ],
)
def test_malformed_missing_extra_empty_or_unsafe_output_fails_without_writes(
    monkeypatch, tmp_path, files
):
    result = _run(
        monkeypatch,
        tmp_path,
        _task("one.py", "two.py"),
        json.dumps({"files": files}),
    )

    assert not result.success
    assert result.files_written == []
    if (tmp_path / "product").exists():
        assert not list((tmp_path / "product").rglob("*.py"))
    assert "set mismatch" in result.notes or "Unsafe output path" in result.notes or "empty file" in result.notes


def test_symlink_escape_is_rejected_before_write(monkeypatch, tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "product").mkdir()
    (tmp_path / "product" / "escape").symlink_to(outside, target_is_directory=True)

    result = _run(
        monkeypatch,
        tmp_path,
        _task("escape/owned.py"),
        json.dumps({"files": {"escape/owned.py": "content"}}),
    )

    assert not result.success
    assert "escapes or aliases its configured area" in result.notes
    assert not (outside / "owned.py").exists()


def test_internal_symlink_cannot_alias_another_owned_path(monkeypatch, tmp_path):
    product = tmp_path / "product"
    (product / "actual").mkdir(parents=True)
    (product / "alias").symlink_to(product / "actual", target_is_directory=True)
    (product / "actual" / "owned.py").write_text("original")

    result = _run(
        monkeypatch,
        tmp_path,
        _task("alias/owned.py"),
        json.dumps({"files": {"alias/owned.py": "replacement"}}),
    )

    assert not result.success
    assert not result.files_written
    assert (product / "actual" / "owned.py").read_text() == "original"
