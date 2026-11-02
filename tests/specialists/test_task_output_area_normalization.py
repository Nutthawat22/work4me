"""Configured output prefixes are applied once to each per-file request."""

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline.state import OwnedFile, WorkItem
from specialists import file_delivery
from specialists.llm_client import LLMResponse


def _config() -> dict:
    return {
        "models": {"specialist": {"model": "m", "provider": "acp"}},
        "pipeline": {
            "output_dir": "product",
            "tests_dir": "tests",
            "max_retries": 1,
            "_stage_plan_digest": "normalization-plan",
        },
    }


def _frame(messages, source: str) -> str:
    markers = re.search(
        r"Framing: first line is (BEGIN-OWNED-FILE:[A-Za-z0-9_-]+); last line is (END-OWNED-FILE:[A-Za-z0-9_-]+)",
        messages[1]["content"],
    )
    assert markers is not None
    return f"{markers.group(1)}\n{source}\n{markers.group(2)}"


def _install(monkeypatch, contents: dict[str, str]) -> None:
    def respond(messages, model, config, **kwargs):
        target = re.search(r"CURRENT TARGET FILE .*?: ([^\n]+)", messages[1]["content"])
        assert target is not None
        return LLMResponse(_frame(messages, contents[target.group(1)]), "complete", "end_turn")

    monkeypatch.setattr(file_delivery, "call_llm_result", respond)
    monkeypatch.setattr("specialists.base._self_check", lambda *args, **kwargs: None)


def test_tests_dir_prefix_is_written_only_once(monkeypatch, tmp_path):
    content = "describe('movement', () => {});"
    _install(monkeypatch, {"tests/movement.test.js": content})
    item = WorkItem(
        id="TASK-tests", category="validation", language="javascript", module="movement",
        scope="Implement movement validation", title="Movement validation",
        description="Validate movement behavior", acceptance_criteria=["movement works"],
        requirement_refs=["TEST-001"],
        owned_files=[OwnedFile("tests/movement.test.js", "javascript", "tests")],
    )

    result = file_delivery.run_file_delivery(
        item, _config(), str(tmp_path), "Validation for {language}", "ValidationAgent"
    )

    assert result.success
    assert result.files_written == ["tests/movement.test.js"]
    assert (tmp_path / "tests/movement.test.js").read_text() == content
    assert not (tmp_path / "tests/tests/movement.test.js").exists()


def test_product_dir_prefix_is_normalized_for_every_owned_file(monkeypatch, tmp_path):
    contents = {
        "product/src/one.js": "export const one = 1;",
        "product/src/two.js": "export const two = 2;",
    }
    _install(monkeypatch, contents)
    item = WorkItem(
        id="TASK-feature", category="feature", language="javascript", module="example",
        scope="Add two modules", title="Example", description="Two related modules",
        acceptance_criteria=["both modules are present"], requirement_refs=[],
        owned_files=[
            OwnedFile("product/src/one.js", "javascript"),
            OwnedFile("src/two.js", "javascript"),
        ],
    )

    result = file_delivery.run_file_delivery(item, _config(), str(tmp_path), "Feature", "FeatureAgent")

    assert result.success
    assert set(result.files_written) == set(contents)
    assert all((tmp_path / path).read_text() == content for path, content in contents.items())
