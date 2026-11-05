"""Mocked regressions for staged, per-owned-file delivery."""

from __future__ import annotations

import json
import hashlib
import os
import re
import sys

import pytest
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline.state import OwnedFile, ReviewResult, ValidationIssue, WorkItem
from specialists import file_delivery
from specialists.llm_client import LLMResponse


def _task(*paths: str) -> WorkItem:
    return WorkItem(
        id="TASK-owned-feature",
        category="feature",
        language="python",
        module="example",
        scope="Implement one cohesive feature across the owned files",
        title="Example feature",
        description="Implement the complete feature described by this task",
        acceptance_criteria=["the feature works"],
        requirement_refs=["REQ-001"],
        owned_files=[OwnedFile(path, "python") for path in paths],
    )


def _config(max_retries: int = 3) -> dict:
    return {
        "litellm_key": "test-secret",
        "models": {"specialist": {"model": "mock-model", "provider": "acp"}},
        "pipeline": {
            "output_dir": "product",
            "tests_dir": "tests",
            "max_retries": max_retries,
            "_stage_plan_digest": "approved-plan-digest",
        },
    }


def _frame(messages: list[dict], body: str) -> str:
    text = messages[1]["content"]
    framing = re.search(
        r"Framing: first line is (BEGIN-OWNED-FILE:[A-Za-z0-9_-]+); last line is (END-OWNED-FILE:[A-Za-z0-9_-]+)",
        text,
    )
    assert framing is not None
    return f"{framing.group(1)}\n{body}\n{framing.group(2)}"


def _empty_action(messages: list[dict]) -> str:
    token = re.search(r"Empty action .*?: (ACCEPT-EMPTY-FILE:[^\s]+)", messages[1]["content"])
    assert token is not None
    return token.group(1)


def _delete_action(messages: list[dict]) -> str:
    token = re.search(r"Deletion action .*?: (INTENTIONAL-DELETE-FILE:[^\s]+)", messages[1]["content"])
    assert token is not None
    return token.group(1)


def _install(monkeypatch, responder):
    monkeypatch.setattr(file_delivery, "call_llm_result", responder)
    monkeypatch.setattr("specialists.base._self_check", lambda *args, **kwargs: None)


def _deliver(item, config, run_dir, **kwargs):
    return file_delivery.run_file_delivery(
        item, config, str(run_dir), "Feature specialist", "FeatureAgent", **kwargs
    )


def test_each_owned_file_is_delivered_as_plain_framed_source_with_quotes_and_newlines(
    monkeypatch, tmp_path
):
    contents = {
        "product/pkg/__init__.py": "",
        "product/pkg/a.py": 'VALUE = "quoted \\\"text\\\""\n',
        "product/pkg/b.py": 'from .a import VALUE\nTEXT = """two\nlines"""\n',
    }
    calls: list[tuple[str, str]] = []

    def respond(messages, model, config, **kwargs):
        request = messages[1]["content"]
        target = re.search(r"CURRENT TARGET FILE .*?: ([^\n]+)", request)
        assert target is not None
        output_path = target.group(1)
        calls.append((output_path, messages[1]["content"]))
        body = _empty_action(messages) if output_path.endswith("__init__.py") else contents[output_path]
        return LLMResponse(_frame(messages, body), "complete", "end_turn")

    _install(monkeypatch, respond)
    item = _task("pkg/__init__.py", "pkg/a.py", "pkg/b.py")
    result = _deliver(item, _config(), tmp_path)

    assert result.success
    assert len(calls) == 3
    assert all("\"files\"" not in request for _, request in calls)
    assert (tmp_path / "product/pkg/__init__.py").read_bytes() == b""
    assert (tmp_path / "product/pkg/a.py").read_text() == contents["product/pkg/a.py"]
    assert (tmp_path / "product/pkg/b.py").read_text() == contents["product/pkg/b.py"]
    assert (tmp_path / "staging").is_dir()
    stage_manifest = next((tmp_path / "staging").rglob("stage.json"))
    data = json.loads(stage_manifest.read_text())
    assert data["schema_version"] == 1
    assert all(record["status"] == "staged" for record in data["files"].values())


def test_crlf_framing_preserves_source_line_endings(monkeypatch, tmp_path):
    source = "FIRST = 1\r\nSECOND = 2\r\n"

    def respond(messages, model, config, **kwargs):
        request = messages[1]["content"]
        markers = re.search(
            r"Framing: first line is (BEGIN-OWNED-FILE:[A-Za-z0-9_-]+); last line is (END-OWNED-FILE:[A-Za-z0-9_-]+)",
            request,
        )
        assert markers is not None
        return LLMResponse(
            f"{markers.group(1)}\r\n{source}\r\n{markers.group(2)}",
            "complete",
            "end_turn",
        )

    _install(monkeypatch, respond)
    result = _deliver(_task("value.py"), _config(), tmp_path)

    assert result.success
    assert (tmp_path / "product/value.py").read_bytes() == source.encode()


def test_unterminated_outer_fence_is_rejected_and_retained(monkeypatch, tmp_path):
    def respond(messages, model, config, **kwargs):
        return LLMResponse(f"```python\n{_frame(messages, 'VALUE = 1')}\n", "complete", "end_turn")

    _install(monkeypatch, respond)
    result = _deliver(_task("value.py"), _config(max_retries=1), tmp_path)

    assert not result.success
    candidate = next((tmp_path / "diagnostics/rejected").rglob("*.source"))
    retained = candidate.read_text()
    assert retained.startswith("```python\nBEGIN-OWNED-FILE:")
    assert "VALUE = 1" in retained
    assert retained.endswith("\n")


def test_rejected_candidate_source_and_nonce_metrics_are_protected_and_retained(monkeypatch, tmp_path):
    def respond(messages, model, config, **kwargs):
        request = messages[1]["content"]
        markers = re.search(
            r"Framing: first line is (BEGIN-OWNED-FILE:[A-Za-z0-9_-]+); last line is (END-OWNED-FILE:[A-Za-z0-9_-]+)",
            request,
        )
        assert markers is not None
        return LLMResponse(
            "BEGIN-OWNED-FILE:stale-nonce\nTOKEN_VALUE = 'candidate'\nEND-OWNED-FILE:stale-nonce",
            "complete",
            "end_turn",
        )

    _install(monkeypatch, respond)
    result = _deliver(_task("value.py"), _config(max_retries=1), tmp_path)

    assert not result.success
    task_dir = next((tmp_path / "diagnostics/rejected").iterdir())
    source = next(task_dir.glob("*.source"))
    metadata_path = next(task_dir.glob("*.json"))
    metadata = json.loads(metadata_path.read_text())
    assert source.read_text() == "TOKEN_VALUE = 'candidate'"
    assert source.stat().st_mode & 0o777 == 0o600
    assert metadata_path.stat().st_mode & 0o777 == 0o600
    assert metadata["stale_nonce"] is True
    assert metadata["current_nonce"] is False
    assert metadata["source_length"] == len("TOKEN_VALUE = 'candidate'")
    assert metadata["completion_status"] == "complete"
    assert "test-secret" not in metadata_path.read_text()


def test_scoped_review_rejects_one_file_and_preserves_staged_siblings(monkeypatch, tmp_path):
    calls: list[str] = []
    reviews = iter([
        ReviewResult(
            False,
            ["a.py has a mismatch"],
            "Only a.py needs a repair",
            affected_paths=["a.py"],
            scope_reliable=True,
        ),
        ReviewResult(True, [], "fixed"),
    ])

    def respond(messages, model, config, **kwargs):
        target = re.search(r"CURRENT TARGET FILE .*?: ([^\n]+)", messages[1]["content"])
        assert target is not None
        path = target.group(1)
        calls.append(path)
        return LLMResponse(_frame(messages, f"VALUE_{path[-4]} = 1\n"), "complete", "end_turn")

    _install(monkeypatch, respond)
    monkeypatch.setattr("specialists.base._self_check", lambda *args, **kwargs: next(reviews))
    item = _task("a.py", "b.py", "c.py")
    first = _deliver(item, _config(), tmp_path)

    assert not first.success
    stage_manifest = next((tmp_path / "staging").rglob("stage.json"))
    stage = json.loads(stage_manifest.read_text())
    assert stage["repair_scope"] == {
        "kind": "review_scoped",
        "paths": ["product/a.py"],
        "work_item_ids": [],
    }
    assert stage["files"]["product/a.py"]["status"] == "rejected"
    assert stage["files"]["product/b.py"]["status"] == "staged"
    assert stage["files"]["product/c.py"]["status"] == "staged"
    assert (tmp_path / "staging" / stage_manifest.parent.name / "files/product/b.py").is_file()

    second = _deliver(item, _config(), tmp_path)

    assert second.success
    assert calls == ["product/a.py", "product/b.py", "product/c.py", "product/a.py"]


def test_retry_retains_accepted_file_and_regenerates_only_rejected_file(monkeypatch, tmp_path):
    calls: list[str] = []

    def respond(messages, model, config, **kwargs):
        request = messages[1]["content"]
        target = re.search(r"CURRENT TARGET FILE .*?: ([^\n]+)", request)
        assert target is not None
        path = target.group(1)
        calls.append(path)
        if path.endswith("/b.py") and len(calls) == 2:
            return LLMResponse("truncated content without current framing", "complete", "end_turn")
        return LLMResponse(_frame(messages, f"VALUE_{path[-4]} = 1\n"), "complete", "end_turn")

    _install(monkeypatch, respond)
    item = _task("a.py", "b.py")
    first = _deliver(item, _config(), tmp_path)
    assert not first.success
    assert not (tmp_path / "product/a.py").exists()
    assert (tmp_path / "staging").is_dir()

    failures = [
        ValidationIssue("product/a.py", item.id, "Expected owned file is missing"),
        ValidationIssue("product/b.py", item.id, "Expected owned file is missing"),
    ]
    second = _deliver(item, _config(), tmp_path, retry_failures=failures)

    assert second.success
    assert calls == ["product/a.py", "product/b.py", "product/b.py"]
    assert (tmp_path / "product/a.py").read_text() == "VALUE_a = 1\n"
    assert (tmp_path / "product/b.py").read_text() == "VALUE_b = 1\n"
    assert len([
        attempt for attempt in item.file_progress
        if attempt.path == "product/a.py" and attempt.status == "staged"
    ]) == 1


def test_cross_file_consistency_feedback_invalidates_the_owned_feature_set(monkeypatch, tmp_path):
    calls: list[str] = []
    prompts: list[str] = []

    def respond(messages, model, config, **kwargs):
        request = messages[1]["content"]
        target = re.search(r"CURRENT TARGET FILE .*?: ([^\n]+)", request)
        assert target is not None
        calls.append(target.group(1))
        prompts.append(request)
        return LLMResponse(_frame(messages, f"VALUE_{target.group(1)[-4]} = 1\n"), "complete", "end_turn")

    _install(monkeypatch, respond)
    item = _task("a.py", "b.py")
    first = _deliver(item, _config(), tmp_path)
    assert first.success

    second = _deliver(
        item,
        _config(),
        tmp_path,
        retry_failures=[ValidationIssue("<feature>", item.id, "Cross-file interface mismatch between a.py and b.py")],
    )

    assert second.success
    assert calls == ["product/a.py", "product/b.py", "product/a.py", "product/b.py"]
    assert all("Cross-file consistency repair feedback" not in prompt for prompt in prompts[:2])
    assert all("Cross-file consistency repair feedback" in prompt for prompt in prompts[2:])


def test_per_file_generation_budget_is_bounded_by_configured_attempts(monkeypatch, tmp_path):
    calls = 0

    def respond(messages, model, config, **kwargs):
        nonlocal calls
        calls += 1
        return LLMResponse("malformed response", "complete", "end_turn")

    _install(monkeypatch, respond)
    item = _task("only.py")
    first = _deliver(item, _config(max_retries=1), tmp_path)
    assert not first.success

    second = _deliver(
        item,
        _config(max_retries=1),
        tmp_path,
        retry_failures=[ValidationIssue("product/only.py", item.id, "Expected owned file is missing")],
    )

    assert not second.success
    assert calls == 1


def test_empty_response_is_failure_but_explicit_empty_action_is_accepted(monkeypatch, tmp_path):
    mode = "accepted"

    def respond(messages, model, config, **kwargs):
        body = _empty_action(messages) if mode == "accepted" else ""
        return LLMResponse(_frame(messages, body), "complete", "end_turn")

    _install(monkeypatch, respond)
    accepted = _deliver(_task("empty.py"), _config(), tmp_path / "accepted")
    assert accepted.success
    assert (tmp_path / "accepted/product/empty.py").read_bytes() == b""

    mode = "failed"
    rejected = _deliver(_task("empty.py"), _config(), tmp_path / "failed")
    assert not rejected.success
    assert "explicit empty-file action" in rejected.notes
    assert not (tmp_path / "failed/product/empty.py").exists()


@pytest.mark.parametrize(
    ("completion_status", "stop_reason"),
    [("refused", "refusal"), ("incomplete", "max_tokens"), ("unsupported", None)],
)
def test_refused_interrupted_and_unsupported_completion_metadata_never_publish(
    monkeypatch, tmp_path, completion_status, stop_reason
):
    def respond(messages, model, config, **kwargs):
        return LLMResponse(
            _frame(messages, "VALUE = 1\n"),
            completion_status,
            stop_reason,
            "refusal" if completion_status == "refused" else None,
        )

    _install(monkeypatch, respond)
    item = _task("value.py")
    result = _deliver(item, _config(), tmp_path)

    assert not result.success
    assert not (tmp_path / "product/value.py").exists()
    assert any(attempt.completion_status == completion_status for attempt in item.file_progress)


def test_existing_project_deletion_is_explicit_and_incomplete_feature_leaves_originals_untouched(
    monkeypatch, tmp_path
):
    project = tmp_path / "project"
    project.mkdir()
    (project / "obsolete.py").write_text("old = True\n")
    (project / "new.py").write_text("old = False\n")
    config = _config()
    config["pipeline"].update(
        _existing_project_path=str(project),
        _existing_project_context="Project context",
        _existing_project_baseline={"obsolete.py": "old", "new.py": "old"},
    )

    def respond(messages, model, configured, **kwargs):
        target = re.search(r"CURRENT TARGET FILE .*?: ([^\n]+)", messages[1]["content"])
        assert target is not None
        if target.group(1) == "obsolete.py":
            body = _delete_action(messages)
        else:
            return LLMResponse("partial output", "incomplete", "max_tokens", "interrupted")
        return LLMResponse(_frame(messages, body), "complete", "end_turn")

    _install(monkeypatch, respond)
    result = _deliver(_task("obsolete.py", "new.py"), config, tmp_path / "run")

    assert not result.success
    assert (project / "obsolete.py").read_text() == "old = True\n"
    assert (project / "new.py").read_text() == "old = False\n"
    assert not result.files_deleted


def test_explicit_existing_project_delete_publishes_after_feature_validation(monkeypatch, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "obsolete.py").write_text("old = True\n")
    config = _config(max_retries=1)
    config["pipeline"].update(
        _existing_project_path=str(project),
        _existing_project_context="Project context",
        _existing_project_baseline={
            "obsolete.py": hashlib.sha256(b"old = True\n").hexdigest(),
        },
    )

    def respond(messages, model, configured, **kwargs):
        return LLMResponse(_frame(messages, _delete_action(messages)), "complete", "end_turn")

    _install(monkeypatch, respond)
    result = _deliver(_task("obsolete.py"), config, tmp_path / "run")

    assert result.success
    assert result.files_deleted == ["obsolete.py"]
    assert not (project / "obsolete.py").exists()


def test_candidate_validation_resolves_staged_relative_imports_without_execution(monkeypatch, tmp_path):
    sentinel = tmp_path / "executed"
    dependency_root = tmp_path / "product/pkg"
    dependency_root.mkdir(parents=True)
    (dependency_root / "helper.py").write_text("VALUE = 2\n")
    config = _config()
    config["pipeline"]["_stage_plan_digest"] = "with-dependency"
    contents = {
        "__init__.py": "",
        "current.py": f"from .helper import VALUE\nopen({str(sentinel)!r}, 'w').write('ran')\n",
    }

    def respond(messages, model, configured, **kwargs):
        target = re.search(r"CURRENT TARGET FILE .*?: ([^\n]+)", messages[1]["content"])
        assert target is not None
        name = target.group(1).rsplit("/", 1)[-1]
        body = _empty_action(messages) if name == "__init__.py" else contents[name]
        return LLMResponse(_frame(messages, body), "complete", "end_turn")

    _install(monkeypatch, respond)
    result = _deliver(_task("pkg/__init__.py", "pkg/current.py"), config, tmp_path)

    assert result.success
    assert not sentinel.exists()


def test_supported_python_parser_rejects_unterminated_structure_before_publish(
    monkeypatch, tmp_path
):
    _install(
        monkeypatch,
        lambda messages, *args, **kwargs: LLMResponse(
            _frame(messages, "value = 'unterminated\n"), "complete", "end_turn"
        ),
    )

    result = _deliver(_task("broken.py"), _config(), tmp_path)

    assert not result.success
    assert "unterminated string literal" in result.notes
    assert not (tmp_path / "product/broken.py").exists()


def test_stale_stage_is_reset_when_plan_identity_changes(monkeypatch, tmp_path):
    calls: list[str] = []

    def respond(messages, model, config, **kwargs):
        target = re.search(r"CURRENT TARGET FILE .*?: ([^\n]+)", messages[1]["content"])
        assert target is not None
        path = target.group(1)
        calls.append(path)
        if path.endswith("/b.py") and sum(item.endswith("/b.py") for item in calls) == 1:
            return LLMResponse("bad frame", "complete", "end_turn")
        return LLMResponse(_frame(messages, f"VALUE_{path[-4]} = 1\n"), "complete", "end_turn")

    _install(monkeypatch, respond)
    item = _task("a.py", "b.py")
    first = _deliver(item, _config(), tmp_path)
    assert not first.success
    stage_manifest = next((tmp_path / "staging").rglob("stage.json"))
    old_identity = json.loads(stage_manifest.read_text())["identity"]

    item.description = "A changed feature plan must not reuse prior stage content"
    second = _deliver(item, _config(), tmp_path)

    assert second.success
    assert calls == ["product/a.py", "product/b.py", "product/a.py", "product/b.py"]
    assert json.loads(stage_manifest.read_text())["identity"] != old_identity


def test_existing_source_change_resets_unpublished_stage(monkeypatch, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "a.py").write_text("old_a = True\n")
    (project / "b.py").write_text("old_b = True\n")
    config = _config()
    config["pipeline"].update(
        _existing_project_path=str(project),
        _existing_project_context="Project context",
        _existing_project_baseline={
            "a.py": hashlib.sha256(b"old_a = True\n").hexdigest(),
            "b.py": hashlib.sha256(b"old_b = True\n").hexdigest(),
        },
    )
    calls: list[str] = []

    def respond(messages, model, configured, **kwargs):
        target = re.search(r"CURRENT TARGET FILE .*?: ([^\n]+)", messages[1]["content"])
        assert target is not None
        path = target.group(1)
        calls.append(path)
        if path == "b.py" and calls.count("b.py") == 1:
            return LLMResponse("interrupted", "incomplete", "max_tokens", "interrupted")
        return LLMResponse(_frame(messages, f"VALUE_{path[0]} = 1\n"), "complete", "end_turn")

    _install(monkeypatch, respond)
    item = _task("a.py", "b.py")
    first = _deliver(item, config, tmp_path / "run")
    assert not first.success
    stage_manifest = next((tmp_path / "run/staging").rglob("stage.json"))
    old_identity = json.loads(stage_manifest.read_text())["identity"]

    (project / "a.py").write_text("external_edit = True\n")
    second = _deliver(item, config, tmp_path / "run")

    assert not second.success
    assert calls == ["a.py", "b.py", "a.py", "b.py"]
    assert (project / "a.py").read_text() == "external_edit = True\n"
    assert (project / "b.py").read_text() == "old_b = True\n"
    assert json.loads(stage_manifest.read_text())["identity"] != old_identity
