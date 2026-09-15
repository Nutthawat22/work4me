"""
tests/specialists/test_test_agent_output_path_nesting.py

Regression coverage for the test-type WorkItem output_path double-nesting
bug: TestAgent-authored WorkItems (type="test") set output_path like
"tests/movementCollision.test.js" (see the real WI-005 in
runs/0013-handoff-snake-game-handoff/design/plan.json), matching the
convention every other WorkItem type uses relative to run_dir. But
run_specialist's subdir_key="tests_dir" call ALSO joins output_path under
config["pipeline"]["tests_dir"] (itself literally "tests"), producing an
actual double-nested "tests/tests/movementCollision.test.js" path on
disk.

Fix (in specialists/base.py's run_specialist): when subdir_key ==
"tests_dir", strip one leading "{tests_dir}/" prefix from output_path
before joining, so a test WorkItem's output_path is always treated as
relative to tests_dir directly — the same convention output_dir-relative
WorkItems (ui/logic/config/scaffold/integrate) already use.

No real LLM calls: specialists.base.call_llm is monkeypatched, mirroring
tests/specialists/test_base_self_check.py's pattern.
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from specialists.base import run_specialist
from pipeline.state import WorkItem


def _config() -> dict:
    return {
        "models": {"specialist": {"model": "m", "provider": "acp"}},
        "pipeline": {"output_dir": "product", "tests_dir": "tests"},
    }


def _accept_json() -> str:
    return json.dumps({"accepted": True, "issues": [], "reasoning": "ok"})


def _queue_call_llm(monkeypatch, responses: list[str]):
    remaining = list(responses)

    def fake_call_llm(messages, model, config, provider="acp", timeout=60,
                       response_schema=None, session_scope=None):
        return remaining.pop(0)

    monkeypatch.setattr("specialists.base.call_llm", fake_call_llm)


class TestOutputPathWithTestsDirPrefixIsNotDoubled:
    def test_wi005_style_output_path_writes_without_double_nesting(
        self, monkeypatch, tmp_path
    ):
        """
        Exact WI-005-style output_path from the snake game plan.json:
        "tests/movementCollision.test.js". Must resolve to
        run_dir/tests/movementCollision.test.js, NOT
        run_dir/tests/tests/movementCollision.test.js.
        """
        _queue_call_llm(monkeypatch, [
            "describe('x', () => { it('y', () => {}); });\n",
            _accept_json(),
        ])

        work_item = WorkItem(
            id="WI-005",
            type="test",
            language="javascript",
            title="[WP-002] Movement/collision tests",
            description="Test movement and collision mechanics.",
            acceptance_criteria=["snake grows by one segment per food eaten"],
            output_path="tests/movementCollision.test.js",
            depends_on=["WI-002"],
        )

        result = run_specialist(
            work_item, _config(), str(tmp_path),
            "You are an expert {language} engineer.", "TestAgent", "tests_dir",
        )

        assert result.success is True
        assert result.files_written == ["tests/movementCollision.test.js"]

        expected_path = tmp_path / "tests" / "movementCollision.test.js"
        assert expected_path.is_file()

        double_nested_path = tmp_path / "tests" / "tests" / "movementCollision.test.js"
        assert not double_nested_path.exists()

    def test_output_path_without_tests_dir_prefix_still_works(
        self, monkeypatch, tmp_path
    ):
        """Non-prefixed output_path (e.g. a bare filename or nested
        subdir with no leading "tests/") must be unaffected by the
        prefix-stripping fix."""
        _queue_call_llm(monkeypatch, [
            "describe('x', () => { it('y', () => {}); });\n",
            _accept_json(),
        ])

        work_item = WorkItem(
            id="WI-006",
            type="test",
            language="javascript",
            title="Some other test",
            description="Test something.",
            acceptance_criteria=["works"],
            output_path="unit/foo.test.js",
            depends_on=[],
        )

        result = run_specialist(
            work_item, _config(), str(tmp_path),
            "You are an expert {language} engineer.", "TestAgent", "tests_dir",
        )

        assert result.success is True
        assert result.files_written == ["tests/unit/foo.test.js"]
        assert (tmp_path / "tests" / "unit" / "foo.test.js").is_file()

    def test_output_dir_subdir_key_is_never_prefix_stripped(
        self, monkeypatch, tmp_path
    ):
        """The prefix-stripping logic only applies to subdir_key ==
        "tests_dir" — a non-test WorkItem (subdir_key="output_dir") whose
        output_path happens to start with "product/" (the configured
        output_dir name) must NOT have that prefix stripped, since that
        convention was never broken for output_dir-relative WorkItems."""
        _queue_call_llm(monkeypatch, [
            "console.log('hi');\n",
            _accept_json(),
        ])

        work_item = WorkItem(
            id="WI-007",
            type="logic",
            language="javascript",
            title="Some logic file literally under a 'product' subdir",
            description="Implement something.",
            acceptance_criteria=["works"],
            output_path="product/nested/thing.js",
            depends_on=[],
        )

        result = run_specialist(
            work_item, _config(), str(tmp_path),
            "You are an expert {language} engineer.", "LogicAgent", "output_dir",
        )

        assert result.success is True
        assert result.files_written == ["product/product/nested/thing.js"]
        assert (tmp_path / "product" / "product" / "nested" / "thing.js").is_file()
