"""Ownership-aware dependency context and retry selection regressions."""

import dataclasses
import hashlib
import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline.dispatch import build_dependency_context, dispatch_work_items
from pipeline.master import MasterAgent, MasterPlanError
from pipeline.resume import (
    ResumeError,
    compatibility_digest,
    import_resume_sources,
    load_resume_request,
    resume_source_lock,
)
from pipeline import runner
from pipeline.runner import _stage_plan_identity
from pipeline.staging import open_stage, persist_stage, publish_stage, read_stage, write_stage_content
from pipeline.state import (
    AgentResult,
    OwnedFile,
    TaskAttempt,
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


def _resume_fixture(tmp_path):
    runs = tmp_path / "runs"
    source = runs / "0001-repair-source"
    (source / "design").mkdir(parents=True)
    (source / "validation/results").mkdir(parents=True)
    (source / "product").mkdir()
    (source / "staging").mkdir()
    config = {
        "models": {
            "design": {"model": "design-model", "provider": "acp"},
            "specialist": {"model": "specialist-model", "provider": "acp"},
        },
        "pipeline": {
            "runs_dir": str(runs),
            "output_dir": "product",
            "tests_dir": "tests",
            "max_retries": 3,
        },
        "languages": {"python": {}},
    }
    foundation = _task("TASK-foundation", "foundation", ["shared.py"])
    foundation.status = "passed"
    foundation.attempts = [TaskAttempt(
        number=1,
        status="passed",
        files_written=["product/shared.py"],
    )]
    feature = _task("TASK-feature", "feature", ["feature.py"], [foundation.id])
    feature.status = "failed"
    feature.attempts = [TaskAttempt(number=1, status="failed", failure_details="repair required")]
    tasks = [foundation, feature]
    plan_digest = _stage_plan_identity(tasks)
    published_source = b"published = 1\n"
    (source / "product/shared.py").write_bytes(published_source)
    foundation_key = hashlib.sha256(foundation.id.encode()).hexdigest()[:24]
    foundation_stage = source / "staging" / foundation_key
    foundation_stage.mkdir()
    (foundation_stage / "files/product").mkdir(parents=True)
    (foundation_stage / "files/product/shared.py").write_bytes(published_source)
    (foundation_stage / "stage.json").write_text(json.dumps({
        "schema_version": 1,
        "identity": "foundation-stage",
        "plan_digest": plan_digest,
        "task_id": foundation.id,
        "status": "published",
        "files": {
            "product/shared.py": {
                "status": "staged",
                "action": "write",
                "digest": hashlib.sha256(published_source).hexdigest(),
                "attempts": [{"number": 1, "status": "staged", "digest": hashlib.sha256(published_source).hexdigest()}],
            }
        },
    }))
    feature_key = hashlib.sha256(feature.id.encode()).hexdigest()[:24]
    feature_stage = source / "staging" / feature_key
    feature_stage.mkdir()
    (feature_stage / "files/product").mkdir(parents=True)
    (feature_stage / "stage.json").write_text(json.dumps({
        "schema_version": 1,
        "identity": "feature-stage",
        "plan_digest": plan_digest,
        "task_id": feature.id,
        "status": "incomplete",
        "files": {
            "product/feature.py": {
                "status": "rejected",
                "action": None,
                "digest": "",
                "failure_details": "incomplete framing",
                "attempts": [{"number": 1, "status": "rejected", "digest": "", "failure_details": "incomplete framing"}],
            }
        },
    }))
    (source / "design/plan.json").write_text(json.dumps({
        "format_version": 2,
        "tasks": [dataclasses.asdict(item) for item in tasks],
    }))
    (source / "validation/results/attempt-1.json").write_text(json.dumps({
        "passed": False,
        "errors": [{
            "file_path": "product/feature.py",
            "work_item_id": feature.id,
            "error_output": "Expected owned file is missing",
            "severity": "error",
        }],
    }))
    (source / "manifest.json").write_text(json.dumps({
        "run_id": source.name,
        "status": "failed",
        "pipeline_schema_version": 3,
        "pipeline_config_digest": compatibility_digest(config),
        "stage_plan_digest": plan_digest,
        "published_files": {
            "product/shared.py": hashlib.sha256(published_source).hexdigest(),
        },
    }))
    return config, source, tasks


@pytest.mark.parametrize(
    ("language", "source_path", "compiler_config"),
    [
        ("typescript", "src/api.ts", "tsconfig.json"),
        ("javascript", "src/api.js", "jsconfig.json"),
    ],
)
def test_new_node_projects_require_owned_root_tooling(language, source_path, compiler_config):
    config = {
        "languages": {"python": {}, "javascript": {}, "typescript": {}},
        "pipeline": {"output_dir": "product", "tests_dir": "tests"},
    }
    master = MasterAgent(config)

    def plan(paths):
        task = _task("TASK-node", "feature", paths)
        task.language = language
        task.owned_files = [OwnedFile(path, language, "product") for path in paths]
        return [task]

    valid_plan = plan([source_path, "package.json", compiler_config])
    master._validate_task_plan(valid_plan)
    assert any("runtime dependencies" in item for item in valid_plan[0].acceptance_criteria)
    assert any(compiler_config in item for item in valid_plan[0].acceptance_criteria)

    with pytest.raises(MasterPlanError, match="exact-owned root package.json"):
        master._validate_task_plan(plan([source_path, compiler_config]))
    with pytest.raises(MasterPlanError, match=f"exact-owned root {compiler_config}"):
        master._validate_task_plan(plan([source_path, "package.json"]))


def test_resume_preflight_selects_exact_task_without_planning_or_regenerating_dependencies(tmp_path):
    config, source, tasks = _resume_fixture(tmp_path)

    request = load_resume_request(config, source.name, ["TASK-feature"])

    assert request.requested_task_ids == ["TASK-feature"]
    assert request.selected_task_ids == ["TASK-feature"]
    assert [result.work_item_id for result in request.successful_results] == ["TASK-foundation"]
    assert request.retry_context["TASK-feature"][0].file_path == "product/feature.py"


def test_resume_child_imports_verified_sources_and_retains_selected_stage_history(tmp_path):
    config, source, tasks = _resume_fixture(tmp_path)
    request = load_resume_request(config, source.name, ["TASK-feature"])
    child = tmp_path / "runs/0002-resume-child"
    (child / "product").mkdir(parents=True)
    (child / "tests").mkdir()
    config["pipeline"]["_stage_plan_digest"] = _stage_plan_identity(request.work_items)

    import_resume_sources(request, config, str(child))

    assert (child / "product/shared.py").read_text() == "published = 1\n"
    feature_key = hashlib.sha256(b"TASK-feature").hexdigest()[:24]
    stage_manifest = child / "staging" / feature_key / "stage.json"
    data = json.loads(stage_manifest.read_text())
    assert data["status"] == "in_progress"
    assert data["files"]["product/feature.py"]["attempts"][0]["number"] == 1


def test_produced_run_can_resume_twice_with_passed_stage_provenance_and_immutable_parent(
    tmp_path, monkeypatch
):
    config = {
        "models": {
            "design": {"model": "design-model", "provider": "acp"},
            "specialist": {"model": "specialist-model", "provider": "acp"},
        },
        "pipeline": {
            "runs_dir": str(tmp_path / "runs"),
            "output_dir": "product",
            "tests_dir": "tests",
            "max_retries": 3,
        },
        "languages": {"python": {}},
    }
    items = [
        _task("TASK-alpha", "feature", ["alpha.py"]),
        _task("TASK-beta", "feature", ["beta.py"]),
        _task("TASK-gamma", "feature", ["gamma.py"]),
    ]
    calls = 0
    dispatched_ids = []

    def publish_task(item, run_config, run_dir):
        relative = f"product/{item.owned_files[0].path}"
        content = f"value_{item.id.rsplit('-', 1)[-1]} = True\n"
        layout = open_stage(item, run_config, run_dir)
        stage = read_stage(layout)
        digest = write_stage_content(layout, relative, content)
        stage["files"][relative] = {
            "status": "staged",
            "action": "write",
            "digest": digest,
            "attempts": [{"number": 1, "status": "staged", "digest": digest}],
        }
        persist_stage(layout, stage)
        written, deleted, error = publish_stage(layout, run_config, run_dir, [("write", relative)])
        assert not error and not deleted
        stage["files"][relative]["status"] = "published"
        stage["status"] = "published"
        persist_stage(layout, stage)
        return AgentResult(item.id, f"{item.category.title()}Agent", True, written)

    def dispatch(work_items, run_config, run_dir, **kwargs):
        nonlocal calls
        calls += 1
        results = []
        for item in work_items:
            dispatched_ids.append(item.id)
            if calls == 1 and item.id != "TASK-alpha":
                results.append(AgentResult(item.id, "FeatureAgent", False, [], "generation failed"))
            else:
                results.append(publish_task(item, run_config, run_dir))
        return results

    monkeypatch.setattr(runner, "dispatch_work_items", dispatch)
    monkeypatch.setattr(runner.acp_client, "teardown", lambda: None)
    monkeypatch.setattr(MasterAgent, "map_validation_issues_to_work_items", lambda self, *args: {})

    assert runner.run_pipeline(config, work_items=items, label="producer source") is False
    parent = next(
        path for path in (tmp_path / "runs").iterdir()
        if path.is_dir() and (path / "manifest.json").is_file()
    )
    parent_manifest_path = parent / "manifest.json"
    parent_plan_path = parent / "design/plan.json"
    parent_stage_path = parent / "staging" / hashlib.sha256(b"TASK-alpha").hexdigest()[:24] / "stage.json"
    parent_snapshot = (
        parent_manifest_path.read_bytes(),
        parent_plan_path.read_bytes(),
        parent_stage_path.read_bytes(),
        (parent / "product/alpha.py").read_bytes(),
    )
    manifest = json.loads(parent_snapshot[0])
    assert json.loads(parent_snapshot[1])["format_version"] == 2
    assert manifest["stage_plan_digest"] == _stage_plan_identity(items)
    assert _stage_plan_identity(items) == manifest["stage_plan_digest"]
    before_status_update = _stage_plan_identity(items)
    items[0].status = "failed"
    items[0].attempts.append(TaskAttempt(number=9, status="failed"))
    assert _stage_plan_identity(items) == before_status_update
    items[0].status = "passed"
    items[0].attempts.pop()

    assert runner.resume_pipeline(config, parent.name, ["TASK-beta"]) is False
    first_child = max(
        (path for path in (tmp_path / "runs").iterdir()
         if path.is_dir() and (path / "manifest.json").is_file() and path != parent),
        key=lambda path: path.name,
    )
    child_manifest = json.loads((first_child / "manifest.json").read_text())
    assert child_manifest["resume"]["source_run_id"] == parent.name
    assert (first_child / "staging" / hashlib.sha256(b"TASK-alpha").hexdigest()[:24] / "stage.json").is_file()

    published_alpha = (first_child / "product/alpha.py").read_bytes()
    (first_child / "product/alpha.py").write_text("drift = True\n")
    with pytest.raises(ResumeError, match="changed since"):
        load_resume_request(config, first_child.name, ["TASK-gamma"])
    (first_child / "product/alpha.py").write_bytes(published_alpha)

    next_request = load_resume_request(config, first_child.name, ["TASK-gamma"])
    assert {result.work_item_id for result in next_request.successful_results} == {
        "TASK-alpha", "TASK-beta"
    }
    assert runner.resume_pipeline(config, first_child.name, ["TASK-gamma"]) is True
    assert dispatched_ids == [
        "TASK-alpha", "TASK-beta", "TASK-gamma", "TASK-beta", "TASK-gamma"
    ]
    assert parent_snapshot == (
        parent_manifest_path.read_bytes(),
        parent_plan_path.read_bytes(),
        parent_stage_path.read_bytes(),
        (parent / "product/alpha.py").read_bytes(),
    )


def test_plan_ownership_collision_checks_each_normalized_owned_path_before_dispatch(
    tmp_path, monkeypatch
):
    config = {
        "models": {
            "design": {"model": "design-model", "provider": "acp"},
            "specialist": {"model": "specialist-model", "provider": "acp"},
        },
        "languages": {"python": {}},
        "pipeline": {
            "runs_dir": str(tmp_path / "runs"),
            "output_dir": "product",
            "tests_dir": "tests",
            "max_retries": 1,
        },
    }
    first = _task("TASK-first", "feature", ["shared.py", "first.py"])
    second = _task("TASK-second", "feature", ["product/shared.py", "second.py"])
    dispatched = False

    def dispatch(*args, **kwargs):
        nonlocal dispatched
        dispatched = True
        raise AssertionError("invalid ownership plan reached dispatch")

    monkeypatch.setattr(runner, "dispatch_work_items", dispatch)
    monkeypatch.setattr(runner.acp_client, "teardown", lambda: None)

    assert runner.run_pipeline(
        config, work_items=[first, second], label="duplicate ownership"
    ) is False
    assert not dispatched


def test_resume_rejects_stale_source_config_schema_and_unknown_task_ids(tmp_path):
    config, source, _tasks = _resume_fixture(tmp_path)
    (source / "product/shared.py").write_text("changed outside the pipeline\n")
    with pytest.raises(ResumeError, match="changed since"):
        load_resume_request(config, source.name, ["TASK-feature"])

    config, source, _tasks = _resume_fixture(tmp_path / "fresh")
    config["pipeline"]["max_retries"] = 4
    with pytest.raises(ResumeError, match="configuration is incompatible"):
        load_resume_request(config, source.name, ["TASK-feature"])

    config, source, _tasks = _resume_fixture(tmp_path / "legacy")
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["pipeline_schema_version"] = 2
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ResumeError, match="authorized plan amendment"):
        load_resume_request(config, source.name, ["TASK-feature"])

    config, source, _tasks = _resume_fixture(tmp_path / "unknown")
    with pytest.raises(ResumeError, match="Unknown task id"):
        load_resume_request(config, source.name, ["TASK-typo"])

    config, source, _tasks = _resume_fixture(tmp_path / "snapshot-error")
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["published_files_error"] = "OSError"
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ResumeError, match="trustworthy published-file snapshot"):
        load_resume_request(config, source.name, ["TASK-feature"])


def test_resume_source_lock_rejects_concurrent_repair_for_same_parent(tmp_path):
    config, source, _tasks = _resume_fixture(tmp_path)

    with resume_source_lock(config, str(source)):
        with pytest.raises(ResumeError, match="already active"):
            with resume_source_lock(config, str(source)):
                pass


def test_resume_pipeline_skips_planning_and_dispatches_only_selected_task(tmp_path, monkeypatch):
    config, source, _tasks = _resume_fixture(tmp_path)
    dispatched = []

    def dispatch(work_items, configured, run_dir, **kwargs):
        dispatched.extend(item.id for item in work_items)
        assert {result.work_item_id for result in kwargs["known_results"]} == {"TASK-foundation"}
        assert (tmp_path / "runs" / os.path.basename(run_dir) / "product/shared.py").read_text() == "published = 1\n"
        output = os.path.join(run_dir, "product", "feature.py")
        with open(output, "w", encoding="utf-8") as target:
            target.write("feature = True\n")
        return [AgentResult("TASK-feature", "FeatureAgent", True, ["product/feature.py"])]

    monkeypatch.setattr(runner, "dispatch_work_items", dispatch)
    monkeypatch.setattr(runner.acp_client, "teardown", lambda: None)
    monkeypatch.setattr(
        "pipeline.master.MasterAgent.decompose",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("resume must not replan")),
    )

    result = runner.resume_pipeline(config, source.name, ["TASK-feature"])

    assert result is True
    assert dispatched == ["TASK-feature"]
    child_runs = [
        path for path in (tmp_path / "runs").iterdir()
        if path.is_dir() and path.name not in {source.name, ".resume-locks"}
    ]
    assert len(child_runs) == 1
    child = child_runs[0]
    child_manifest = json.loads((child / "manifest.json").read_text())
    assert child_manifest["resume"]["source_run_id"] == source.name
    assert child_manifest["resume"]["selected_task_ids"] == ["TASK-feature"]
    assert json.loads((child / "resume-history.json").read_text())["selected_tasks"][0]["prior_attempt_count"] == 1
    assert json.loads((source / "manifest.json").read_text())["status"] == "failed"


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
        AgentResult(integration.id, "IntegrationAgent", False, [], "Task blocked by dependencies: TASK-feature"),
        AgentResult(unrelated.id, "FeatureAgent", True, ["product/other.py"]),
    ]
    clean_validation = ValidationResult(passed=True, checked_files=0)

    retry = MasterAgent({"languages": {"python": {}}}).map_validation_issues_to_work_items(
        clean_validation, results, [foundation, feature, integration, unrelated]
    )

    assert set(retry) == {feature.id, integration.id}
    assert retry[feature.id] == []
    assert retry[integration.id] == []


def test_validation_failure_retries_only_actual_failing_test_task():
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

    assert set(retry) == {validation.id}
    assert len(retry[validation.id]) == 1


def test_blocked_missing_test_files_do_not_retry_passed_implementation_dependencies():
    feature = _task("TASK-feature", "feature", ["feature.py"])
    validation = _task("TASK-validation", "validation", ["feature_test.py"], [feature.id])
    results = [
        AgentResult(feature.id, "FeatureAgent", True, ["product/feature.py"]),
        AgentResult(validation.id, "ValidationAgent", False, [], "Task blocked by dependencies: TASK-other"),
    ]
    missing_blocked_test = ValidationResult(
        passed=False,
        checked_files=0,
        errors=[ValidationIssue("tests/feature_test.py", validation.id, "Expected owned file is missing")],
    )

    retry = MasterAgent({"languages": {"python": {}}}).map_validation_issues_to_work_items(
        missing_blocked_test, results, [feature, validation]
    )

    assert retry == {}
