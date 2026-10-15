"""Multi-file task paths never repeat configured output-area prefixes."""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline.state import OwnedFile, WorkItem
from specialists.base import run_multifile_specialist


def _config() -> dict:
    return {
        "models": {"specialist": {"model": "m", "provider": "acp"}},
        "pipeline": {"output_dir": "product", "tests_dir": "tests"},
    }


def _accept_json() -> str:
    return json.dumps({"accepted": True, "issues": [], "reasoning": "ok"})


def _queue_call_llm(monkeypatch, responses: list[str]):
    remaining = list(responses)

    def fake_call_llm(messages, model, config, **kwargs):
        return remaining.pop(0)

    monkeypatch.setattr("specialists.base.call_llm", fake_call_llm)


def _task(file_path: str, area: str) -> WorkItem:
    return WorkItem(
        id="TASK-tests",
        category="validation" if area == "tests" else "feature",
        language="javascript",
        module="movement",
        scope="Implement movement validation",
        title="Movement validation",
        description="Validate movement behavior",
        acceptance_criteria=["movement works"],
        requirement_refs=["TEST-001"],
        owned_files=[OwnedFile(file_path, "javascript", area)],
    )


def test_tests_dir_prefix_is_written_only_once(monkeypatch, tmp_path):
    content = "describe('movement', () => {});"
    _queue_call_llm(monkeypatch, [
        json.dumps({"files": {"tests/movement.test.js": content}}),
        _accept_json(),
    ])

    result = run_multifile_specialist(
        _task("tests/movement.test.js", "tests"), _config(), str(tmp_path),
        "Validation for {language}", "ValidationAgent",
    )

    assert result.success
    assert result.files_written == ["tests/movement.test.js"]
    assert (tmp_path / "tests" / "movement.test.js").read_text() == content
    assert not (tmp_path / "tests" / "tests" / "movement.test.js").exists()


def test_product_dir_prefix_is_normalized_and_multi_file_set_is_written(monkeypatch, tmp_path):
    _queue_call_llm(monkeypatch, [
        json.dumps({"files": {
            "product/src/one.js": "export const one = 1;",
            "src/two.js": "export const two = 2;",
        }}),
        _accept_json(),
    ])
    item = WorkItem(
        id="TASK-feature", category="feature", language="javascript", module="example",
        scope="Add two modules", title="Example", description="Two related modules",
        acceptance_criteria=["both modules are present"], requirement_refs=[],
        owned_files=[OwnedFile("product/src/one.js", "javascript"), OwnedFile("src/two.js", "javascript")],
    )

    result = run_multifile_specialist(item, _config(), str(tmp_path), "Feature", "FeatureAgent")

    assert result.success
    assert set(result.files_written) == {"product/src/one.js", "product/src/two.js"}


if __name__ == "__main__":
    pass
