"""Path-bound file framing and publish-time symlink checks."""

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline.state import OwnedFile, WorkItem
from specialists import file_delivery
from specialists.llm_client import LLMResponse


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


def _config() -> dict:
    return {
        "models": {"specialist": {"model": "m", "provider": "acp"}},
        "pipeline": {
            "output_dir": "product",
            "tests_dir": "tests",
            "max_retries": 1,
            "_stage_plan_digest": "test-plan",
        },
    }


def _frame(messages, source: str) -> str:
    markers = re.search(
        r"Framing: first line is (BEGIN-OWNED-FILE:[A-Za-z0-9_-]+); last line is (END-OWNED-FILE:[A-Za-z0-9_-]+)",
        messages[1]["content"],
    )
    assert markers is not None
    return f"{markers.group(1)}\n{source}\n{markers.group(2)}"


def _run(monkeypatch, tmp_path, item: WorkItem, responder):
    monkeypatch.setattr(file_delivery, "call_llm_result", responder)
    monkeypatch.setattr("specialists.base._self_check", lambda *args, **kwargs: None)
    config = _config()
    return file_delivery.run_file_delivery(item, config, str(tmp_path), "Feature", "FeatureAgent")


def test_stale_or_malformed_framing_fails_without_product_writes(monkeypatch, tmp_path):
    result = _run(
        monkeypatch,
        tmp_path,
        _task("one.py", "two.py"),
        lambda *args, **kwargs: LLMResponse("BEGIN-OWNED-FILE:previous\nvalue = 1\nEND-OWNED-FILE:previous", "complete", "end_turn"),
    )

    assert not result.success
    assert "framing" in result.notes.casefold()
    assert not (tmp_path / "product").exists()


def test_symlink_escape_is_rejected_before_publication(monkeypatch, tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    product = tmp_path / "product"
    product.mkdir()
    (product / "escape").symlink_to(outside, target_is_directory=True)

    result = _run(
        monkeypatch,
        tmp_path,
        _task("escape/owned.py"),
        lambda messages, *args, **kwargs: LLMResponse(
            _frame(messages, "value = 1\n"), "complete", "end_turn"
        ),
    )

    assert not result.success
    assert "symbolic link" in result.notes
    assert not (outside / "owned.py").exists()


def test_internal_symlink_cannot_alias_another_owned_path(monkeypatch, tmp_path):
    product = tmp_path / "product"
    (product / "actual").mkdir(parents=True)
    (product / "alias").symlink_to(product / "actual", target_is_directory=True)
    (product / "actual/owned.py").write_text("original = True\n")

    result = _run(
        monkeypatch,
        tmp_path,
        _task("alias/owned.py"),
        lambda messages, *args, **kwargs: LLMResponse(
            _frame(messages, "replacement = True\n"), "complete", "end_turn"
        ),
    )

    assert not result.success
    assert (product / "actual/owned.py").read_text() == "original = True\n"
