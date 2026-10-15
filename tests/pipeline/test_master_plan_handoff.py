"""Mocked planning tests for design handoffs and the unified task schema."""

import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline.master import MasterAgent, MasterPlanError, TASK_PLANNING_PROMPT_TEMPLATE


def _config() -> dict:
    return {
        "languages": {"python": {}, "typescript": {}},
        "models": {"design": {"model": "m", "provider": "acp"}},
    }


def _task(task_id: str, path: str, category: str = "feature", depends_on=None) -> dict:
    area = "tests" if category == "validation" else "product"
    language = "typescript" if path.endswith((".ts", ".tsx")) else "python"
    return {
        "id": task_id,
        "category": category,
        "language": language,
        "module": task_id.lower(),
        "scope": f"Implement {task_id}",
        "title": task_id,
        "description": f"Implement the cohesive {task_id} module",
        "acceptance_criteria": [f"{task_id} works"],
        "requirement_refs": ["REQ-001"],
        "owned_files": [{"path": path, "language": language, "area": area}],
        "depends_on": list(depends_on or []),
        "expected_interfaces": [f"{task_id} contract"],
    }


def _plan_json(count: int) -> str:
    tasks = [_task(f"TASK-{index:03}", f"src/module_{index}.py") for index in range(count)]
    if count > 1:
        tasks[1]["depends_on"] = [tasks[0]["id"]]
    return json.dumps({"items": tasks})


def test_handoff_plan_uses_unified_multi_file_task_contract(monkeypatch):
    plan = {
        "items": [
            {
                **_task("TASK-auth", "src/auth.py"),
                "owned_files": [
                    {"path": "src/auth.py", "language": "python", "area": "product"},
                    {"path": "src/login.tsx", "language": "typescript", "area": "product"},
                ],
                "language": "python",
            },
            _task("TASK-auth-validation", "test_auth.py", "validation", ["TASK-auth"]),
        ]
    }
    monkeypatch.setattr("pipeline.master.acp_call_with_retry", lambda *a, **k: json.dumps(plan))

    tasks = MasterAgent(_config()).plan_handoff("approved design")

    assert [task.category for task in tasks] == ["feature", "validation"]
    assert [file.language for file in tasks[0].owned_files] == ["python", "typescript"]
    assert tasks[0].expected_interfaces == ["TASK-auth contract"]
    assert tasks[0].status == "ready"
    assert tasks[1].status == "blocked"


def test_handoff_task_count_tracks_design_complexity(monkeypatch):
    monkeypatch.setattr("pipeline.master.acp_call_with_retry", lambda *a, **k: _plan_json(2))
    small = MasterAgent(_config()).plan_handoff("small design")
    monkeypatch.setattr("pipeline.master.acp_call_with_retry", lambda *a, **k: _plan_json(7))
    large = MasterAgent(_config()).plan_handoff("large design")
    assert len(small) == 2
    assert len(large) == 7


@pytest.mark.parametrize(
    ("tasks", "message"),
    [
        ([_task("A", "a.py", depends_on=["missing"])], "unknown WorkItem id"),
        (
            [_task("A", "a.py"), _task("B", "a.py")],
            "multiple task owners",
        ),
        ([_task("A", "../outside.py")], "unsafe path"),
        (
            [_task("A", "a.py", depends_on=["B"]), _task("B", "b.py", depends_on=["A"])],
            "Cycle detected",
        ),
    ],
)
def test_handoff_rejects_invalid_task_contracts(monkeypatch, tasks, message):
    monkeypatch.setattr(
        "pipeline.master.acp_call_with_retry",
        lambda *a, **k: json.dumps({"items": tasks}),
    )
    with pytest.raises(MasterPlanError, match=message):
        MasterAgent(_config()).plan_handoff("design")


def test_requirements_text_is_appended_to_handoff_prompt(monkeypatch):
    captured = {}

    def fake_call(messages, model, config, **kwargs):
        captured["messages"] = messages
        return _plan_json(1)

    monkeypatch.setattr("pipeline.master.acp_call_with_retry", fake_call)
    MasterAgent(_config()).plan_handoff("design body", requirements_text="requirements body")
    assert "design body" in captured["messages"][1]["content"]
    assert "requirements body" in captured["messages"][1]["content"]


def test_planning_prompt_lists_categories_and_forbids_single_file_types():
    prompt = TASK_PLANNING_PROMPT_TEMPLATE.format(valid_languages="python, typescript")
    assert '"category":"feature"' in prompt
    assert '"owned_files"' in prompt
    assert "Do not use legacy logic/ui/config/test/scaffold/integrate types." in prompt
