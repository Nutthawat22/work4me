"""Ownership-aware dependency context and retry selection regressions."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline.dispatch import build_dependency_context, dispatch_work_items
from pipeline.master import MasterAgent
from pipeline.state import (
    AgentResult,
    OwnedFile,
    ValidationIssue,
    ValidationResult,
    WorkItem,
)
from specialists.base import _build_dependency_section


def _task(task_id, category, files, depends_on=None):
    area = "tests" if category == "validation" else "product"
    return WorkItem(
        id=task_id,
        category=category,
        language="python",
        module=task_id,
        scope=f"Implement {task_id}",
        title=task_id,
        description=f"Implement {task_id}",
        acceptance_criteria=[f"{task_id} works"],
        requirement_refs=[],
        owned_files=[OwnedFile(path, "python", area) for path in files],
        depends_on=list(depends_on or []),
    )


def test_dependency_context_includes_every_dependency_output_and_marks_omissions(
    tmp_path, monkeypatch
):
    (tmp_path / "product").mkdir()
    (tmp_path / "product" / "a.py").write_text("export_a = True")
    (tmp_path / "product" / "b.py").write_text("export_b = True")
    dependency = AgentResult(
        work_item_id="TASK-dep", agent_name="FeatureAgent", success=True,
        files_written=["product/a.py", "product/b.py"],
    )
    consumer = _task("TASK-consumer", "feature", ["consumer.py"], ["TASK-dep"])

    context = build_dependency_context(
        consumer, {"TASK-dep": dependency}, str(tmp_path)
    )
    assert set(context) == {"product/a.py", "product/b.py"}

    monkeypatch.setattr("pipeline.dispatch.MAX_DEPENDENCY_FILES_SHOWN", 1)
    limited = build_dependency_context(
        consumer, {"TASK-dep": dependency}, str(tmp_path)
    )
    assert "(dependency files omitted by dispatch budget)" in limited
    omitted_note = limited["(dependency files omitted by dispatch budget)"]
    assert "product/a.py" in omitted_note or "product/b.py" in omitted_note


def test_dependency_context_and_worker_prompt_respect_character_budgets(
    tmp_path, monkeypatch
):
    product = tmp_path / "product"
    product.mkdir()
    (product / "a.py").write_text("a" * 100)
    (product / "b.py").write_text("b" * 100)
    dependency = AgentResult(
        work_item_id="TASK-dep", agent_name="FeatureAgent", success=True,
        files_written=["product/a.py", "product/b.py"],
    )
    consumer = _task("TASK-consumer", "feature", ["consumer.py"], ["TASK-dep"])

    monkeypatch.setattr("pipeline.dispatch.MAX_DEPENDENCY_TOTAL_CHARS", 35)
    context = build_dependency_context(
        consumer, {"TASK-dep": dependency}, str(tmp_path)
    )
    shown_files = [
        content for path, content in context.items()
        if path != "(dependency files omitted by dispatch budget)"
    ]
    assert sum(map(len, shown_files)) <= 35

    monkeypatch.setattr("specialists.base.MAX_DEPENDENCY_FILES_SHOWN", 2)
    monkeypatch.setattr("specialists.base.MAX_DEPENDENCY_CHARS_IN_PROMPT", 40)
    monkeypatch.setattr("specialists.base.MAX_DEPENDENCY_TOTAL_CHARS_IN_PROMPT", 55)
    section = _build_dependency_section(
        {"product/a.py": "a" * 100, "product/b.py": "b" * 100}
    )
    first_content = section.split("--- product/a.py ---\n", 1)[1].split(
        "\n\n--- product/b.py ---", 1
    )[0]
    second_content = section.split("--- product/b.py ---\n", 1)[1].split(
        "\n\n", 1
    )[0]
    shown_contents = [first_content, second_content]
    assert [len(content) for content in shown_contents] == [40, 15]
    assert sum(map(len, shown_contents)) <= 55


def test_dispatch_blocks_consumers_when_a_dependency_fails(monkeypatch, tmp_path):
    foundation = _task("TASK-foundation", "foundation", ["foundation.py"])
    feature = _task("TASK-feature", "feature", ["feature.py"], [foundation.id])
    called = []

    class FailedFoundation:
        def execute(self, work_item, *args, **kwargs):
            return AgentResult(work_item.id, "FoundationAgent", False, [], "generation failed")

    class MustNotRun:
        def execute(self, work_item, *args, **kwargs):
            called.append(work_item.id)
            return AgentResult(work_item.id, "FeatureAgent", True, [], "")

    monkeypatch.setattr("pipeline.dispatch.FoundationAgent", FailedFoundation)
    monkeypatch.setattr("pipeline.dispatch.FeatureAgent", MustNotRun)

    results = dispatch_work_items([foundation, feature], {}, str(tmp_path))

    assert [result.success for result in results] == [False, False]
    assert foundation.status == "failed"
    assert len(foundation.attempts) == 1
    assert feature.status == "blocked"
    assert feature.attempts == []
    assert called == []


def test_retry_subset_uses_known_successful_dependency_outputs(monkeypatch, tmp_path):
    (tmp_path / "product").mkdir()
    (tmp_path / "product" / "one.py").write_text("one = 1")
    (tmp_path / "product" / "two.py").write_text("two = 2")
    dependency = _task("TASK-feature", "feature", ["one.py", "two.py"])
    integration = _task("TASK-integration", "integration", ["main.py"], [dependency.id])
    seen = {}

    class CapturingIntegration:
        def execute(self, work_item, config, run_dir, **kwargs):
            seen.update(kwargs["dependency_context"])
            return AgentResult(work_item.id, "IntegrationAgent", True, ["product/main.py"])

    monkeypatch.setattr("pipeline.dispatch.IntegrationAgent", CapturingIntegration)
    results = dispatch_work_items(
        [integration],
        {},
        str(tmp_path),
        strict=False,
        known_results=[
            AgentResult(
                dependency.id,
                "FeatureAgent",
                True,
                ["product/one.py", "product/two.py"],
            )
        ],
    )

    assert results[0].success
    assert seen == {"product/one.py": "one = 1", "product/two.py": "two = 2"}


def test_failed_task_retries_its_dependents_but_not_unrelated_tasks():
    foundation = _task("TASK-foundation", "foundation", ["foundation.py"])
    feature = _task("TASK-feature", "feature", ["feature.py"], [foundation.id])
    integration = _task("TASK-integration", "integration", ["main.py"], [feature.id])
    unrelated = _task("TASK-unrelated", "feature", ["other.py"])
    feature.status = "failed"
    results = [
        AgentResult(foundation.id, "FoundationAgent", True, ["product/foundation.py"]),
        AgentResult(feature.id, "FeatureAgent", False, [], "error"),
        AgentResult(integration.id, "IntegrationAgent", False, [], "blocked"),
        AgentResult(unrelated.id, "FeatureAgent", True, ["product/other.py"]),
    ]
    clean_validation = ValidationResult(passed=True, checked_files=0)

    retry = MasterAgent({"languages": {"python": {}}}).map_validation_issues_to_work_items(
        clean_validation, results, [foundation, feature, integration, unrelated]
    )

    assert set(retry) == {feature.id, integration.id}
    assert retry[feature.id] == []
    assert retry[integration.id] == []


def test_validation_failure_maps_to_declared_implementation_dependency_and_dependents():
    feature = _task("TASK-feature", "feature", ["feature.py"])
    integration = _task("TASK-integration", "integration", ["main.py"], [feature.id])
    validation = _task("TASK-validation", "validation", ["feature_test.py"], [feature.id])
    results = [
        AgentResult(feature.id, "FeatureAgent", True, ["product/feature.py"]),
        AgentResult(integration.id, "IntegrationAgent", True, ["product/main.py"]),
        AgentResult(validation.id, "ValidationAgent", True, ["tests/feature_test.py"]),
    ]
    failed_validation = ValidationResult(
        passed=False,
        checked_files=1,
        errors=[ValidationIssue("tests/feature_test.py", validation.id, "syntax error")],
    )

    retry = MasterAgent({"languages": {"python": {}}}).map_validation_issues_to_work_items(
        failed_validation, results, [feature, integration, validation]
    )

    assert set(retry) == {feature.id, integration.id, validation.id}
    assert len(retry[validation.id]) == 1
    assert retry[feature.id] == []
    assert retry[integration.id] == []
