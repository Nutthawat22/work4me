"""Feature self-checks review staged files before publication."""

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline.state import OwnedFile, ReviewResult, WorkItem
from specialists import file_delivery
from specialists.llm_client import LLMResponse


def _config() -> dict:
    return {
        "models": {"specialist": {"model": "m", "provider": "acp"}},
        "pipeline": {
            "output_dir": "output",
            "tests_dir": "tests",
            "max_retries": 2,
            "_stage_plan_digest": "self-check-plan",
        },
    }


def _work_item() -> WorkItem:
    return WorkItem(
        id="WI-001",
        category="feature",
        language="python",
        module="calculator",
        scope="Implement addition",
        title="Add two numbers",
        description="Implement add(a, b)",
        acceptance_criteria=["add(2, 3) == 5"],
        requirement_refs=["REQ-001"],
        owned_files=[OwnedFile("add.py", "python")],
        expected_interfaces=["exports add(a, b)"],
    )


def _frame(messages, content: str) -> str:
    markers = re.search(
        r"Framing: first line is (BEGIN-OWNED-FILE:[A-Za-z0-9_-]+); last line is (END-OWNED-FILE:[A-Za-z0-9_-]+)",
        messages[1]["content"],
    )
    assert markers is not None
    return f"{markers.group(1)}\n{content}\n{markers.group(2)}"


def _install_generator(monkeypatch, content: str, calls: list[list[dict]]) -> None:
    def respond(messages, model, config, **kwargs):
        calls.append(messages)
        return LLMResponse(_frame(messages, content), "complete", "end_turn")

    monkeypatch.setattr(file_delivery, "call_llm_result", respond)


def test_review_consumes_staged_source_before_any_target_is_published(monkeypatch, tmp_path):
    generation_calls: list[list[dict]] = []
    seen: list[dict[str, str]] = []

    def self_check(item, content, dependencies, config, run_dir, **kwargs):
        seen.append(dict(content))
        assert not (tmp_path / "output/add.py").exists()
        assert content["output/add.py"] == "def add(a, b):\n    return a + b\n"
        return ReviewResult(True, [], "looks good")

    _install_generator(monkeypatch, "def add(a, b):\n    return a + b\n", generation_calls)
    monkeypatch.setattr("specialists.base._self_check", self_check)
    result = file_delivery.run_file_delivery(
        _work_item(), _config(), str(tmp_path), "Feature", "FeatureAgent"
    )

    assert result.success
    assert len(generation_calls) == 1
    assert seen == [{"output/add.py": "def add(a, b):\n    return a + b\n"}]
    assert (tmp_path / "output/add.py").read_text().endswith("return a + b\n")


def test_review_rejection_invalidates_feature_stage_and_retries_with_feedback(monkeypatch, tmp_path):
    generation_calls: list[list[dict]] = []
    reviews = iter(
        [
            ReviewResult(False, ["wrong arithmetic"], "subtraction was generated"),
            ReviewResult(True, [], "fixed"),
        ]
    )
    _install_generator(monkeypatch, "def add(a, b):\n    return a + b\n", generation_calls)
    monkeypatch.setattr("specialists.base._self_check", lambda *args, **kwargs: next(reviews))
    item = _work_item()
    first = file_delivery.run_file_delivery(item, _config(), str(tmp_path), "Feature", "FeatureAgent")
    assert not first.success
    assert not (tmp_path / "output/add.py").exists()

    second = file_delivery.run_file_delivery(item, _config(), str(tmp_path), "Feature", "FeatureAgent")

    assert second.success
    assert len(generation_calls) == 2
    assert "wrong arithmetic" in generation_calls[1][1]["content"]


def test_broken_self_check_remains_fail_open(monkeypatch, tmp_path):
    calls: list[list[dict]] = []
    _install_generator(monkeypatch, "def add(a, b):\n    return a + b\n", calls)
    monkeypatch.setattr("specialists.base._self_check", lambda *args, **kwargs: None)

    result = file_delivery.run_file_delivery(
        _work_item(), _config(), str(tmp_path), "Feature", "FeatureAgent"
    )

    assert result.success
    assert len(calls) == 1
