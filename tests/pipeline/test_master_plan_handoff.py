"""
tests/pipeline/test_master_plan_handoff.py

Tests for MasterAgent.plan_handoff() (Phase 2 of the ACP-native
rearchitecture — see
dev-plans/agents/features/2026-09-14-acp-native-master-architecture.md).
Renamed from decompose_handoff to reflect that the system prompt no
longer instructs a fixed, template-shaped translation of an authored
Work Package list — the LLM decides the actual item count/shape.

No real LLM/subprocess calls: every test monkeypatches
pipeline.master.acp_call_with_retry directly with a canned JSON response
(or an error string), exercising plan_handoff's parse/validate/
dependency-graph logic in isolation.
"""

import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline.master import MasterAgent, MasterPlanError


def _config() -> dict:
    return {
        "languages": {"python": {}, "typescript": {}},
        "models": {"design": {"model": "m", "provider": "acp"}},
    }


def _small_plan_json() -> str:
    """A small (2-item) plan: scaffold + integrate only, no middle items —
    the minimum valid shape under the structural invariants."""
    return json.dumps({
        "items": [
            {
                "id": "WI-000",
                "type": "scaffold",
                "language": "python",
                "title": "[WP-001] Project scaffold",
                "description": "Source IDs: REQ-001 | Completion criteria: package installs",
                "acceptance_criteria": ["package installs cleanly"],
                "output_path": "pyproject.toml",
                "depends_on": [],
            },
            {
                "id": "WI-999",
                "type": "integrate",
                "language": "python",
                "title": "[WP-001] Wire everything",
                "description": "Source IDs: REQ-001 | Completion criteria: app boots",
                "acceptance_criteria": ["app boots end-to-end"],
                "output_path": "src/main.py",
                "depends_on": ["WI-000"],
            },
        ]
    })


def _large_plan_json(n_middle: int) -> str:
    """A larger plan with n_middle implementation items between the
    single scaffold and single integrate item — used to confirm the
    code path handles varying item counts without assuming a fixed
    shape."""
    items = [
        {
            "id": "WI-000",
            "type": "scaffold",
            "language": "typescript",
            "title": "[WP-001] Project scaffold",
            "description": "Source IDs: REQ-001 | Completion criteria: builds cleanly",
            "acceptance_criteria": ["project builds"],
            "output_path": "package.json",
            "depends_on": [],
        },
    ]
    middle_ids = []
    for i in range(n_middle):
        item_id = f"WI-{i + 1:03d}"
        middle_ids.append(item_id)
        items.append({
            "id": item_id,
            "type": "logic",
            "language": "typescript",
            "title": f"[WP-{i + 2:03d}] Module {i}",
            "description": f"Source IDs: REQ-{i + 2:03d} | Completion criteria: module {i} works",
            "acceptance_criteria": [f"module {i} passes its acceptance criteria"],
            "output_path": f"src/module_{i}.ts",
            "depends_on": ["WI-000"],
        })
    items.append({
        "id": "WI-999",
        "type": "integrate",
        "language": "typescript",
        "title": "[WP-999] Wire everything",
        "description": "Source IDs: REQ-999 | Completion criteria: app boots",
        "acceptance_criteria": ["app boots end-to-end"],
        "output_path": "src/index.ts",
        "depends_on": middle_ids,
    })
    return json.dumps({"items": items})


# ── Dynamic shape: item count is not hardcoded ──────────────────────────────

class TestDynamicItemCount:
    def test_small_task_produces_small_plan(self, monkeypatch):
        monkeypatch.setattr(
            "pipeline.master.acp_call_with_retry", lambda *a, **k: _small_plan_json()
        )
        work_items = MasterAgent(_config()).plan_handoff("small design doc")
        assert len(work_items) == 2
        assert {wi.type for wi in work_items} == {"scaffold", "integrate"}

    def test_large_task_produces_large_plan(self, monkeypatch):
        monkeypatch.setattr(
            "pipeline.master.acp_call_with_retry", lambda *a, **k: _large_plan_json(12)
        )
        work_items = MasterAgent(_config()).plan_handoff("large design doc")
        assert len(work_items) == 14  # 1 scaffold + 12 middle + 1 integrate

    def test_different_inputs_yield_different_counts(self, monkeypatch):
        """
        Confirms plan_handoff itself imposes no fixed-count assumption:
        the same code path accepts a 2-item response for one input and a
        7-item response for another, with no special-casing.
        """
        config = _config()

        monkeypatch.setattr(
            "pipeline.master.acp_call_with_retry", lambda *a, **k: _small_plan_json()
        )
        small = MasterAgent(config).plan_handoff("tiny task")

        monkeypatch.setattr(
            "pipeline.master.acp_call_with_retry", lambda *a, **k: _large_plan_json(5)
        )
        medium = MasterAgent(config).plan_handoff("bigger task")

        assert len(small) != len(medium)
        assert len(small) == 2
        assert len(medium) == 7


# ── Structural invariants still enforced ────────────────────────────────────

class TestStructuralInvariants:
    def test_valid_plan_has_scaffold_first_and_integrate_last(self, monkeypatch):
        monkeypatch.setattr(
            "pipeline.master.acp_call_with_retry", lambda *a, **k: _large_plan_json(3)
        )
        work_items = MasterAgent(_config()).plan_handoff("design doc")

        scaffold_items = [wi for wi in work_items if wi.type == "scaffold"]
        integrate_items = [wi for wi in work_items if wi.type == "integrate"]

        assert len(scaffold_items) == 1
        assert scaffold_items[0].depends_on == []

        assert len(integrate_items) == 1
        # Rule 6b: integrate depends on every other IMPLEMENTATION item
        # (not necessarily the scaffold item directly — scaffold is
        # reached transitively via the middle items' own depends_on).
        implementation_ids = {
            wi.id for wi in work_items if wi.type not in ("integrate", "scaffold")
        }
        assert set(integrate_items[0].depends_on) == implementation_ids

    def test_cycle_raises_master_plan_error(self, monkeypatch):
        cyclic_json = json.dumps({
            "items": [
                {
                    "id": "WI-000",
                    "type": "scaffold",
                    "language": "python",
                    "title": "scaffold",
                    "description": "d",
                    "acceptance_criteria": [],
                    "output_path": "pyproject.toml",
                    "depends_on": [],
                },
                {
                    "id": "WI-001",
                    "type": "logic",
                    "language": "python",
                    "title": "a",
                    "description": "d",
                    "acceptance_criteria": [],
                    "output_path": "a.py",
                    "depends_on": ["WI-002"],
                },
                {
                    "id": "WI-002",
                    "type": "logic",
                    "language": "python",
                    "title": "b",
                    "description": "d",
                    "acceptance_criteria": [],
                    "output_path": "b.py",
                    "depends_on": ["WI-001"],
                },
                {
                    "id": "WI-999",
                    "type": "integrate",
                    "language": "python",
                    "title": "integrate",
                    "description": "d",
                    "acceptance_criteria": [],
                    "output_path": "main.py",
                    "depends_on": ["WI-001", "WI-002"],
                },
            ]
        })
        monkeypatch.setattr(
            "pipeline.master.acp_call_with_retry", lambda *a, **k: cyclic_json
        )
        with pytest.raises(MasterPlanError, match="invalid dependency graph"):
            MasterAgent(_config()).plan_handoff("design doc with a cycle")

    def test_dangling_dependency_raises_master_plan_error(self, monkeypatch):
        dangling_json = json.dumps({
            "items": [
                {
                    "id": "WI-000",
                    "type": "scaffold",
                    "language": "python",
                    "title": "scaffold",
                    "description": "d",
                    "acceptance_criteria": [],
                    "output_path": "pyproject.toml",
                    "depends_on": [],
                },
                {
                    "id": "WI-999",
                    "type": "integrate",
                    "language": "python",
                    "title": "integrate",
                    "description": "d",
                    "acceptance_criteria": [],
                    "output_path": "main.py",
                    "depends_on": ["WI-000", "WI-DOES-NOT-EXIST"],
                },
            ]
        })
        monkeypatch.setattr(
            "pipeline.master.acp_call_with_retry", lambda *a, **k: dangling_json
        )
        with pytest.raises(MasterPlanError, match="invalid dependency graph"):
            MasterAgent(_config()).plan_handoff("design doc with a dangling ref")


# ── Parse/validation error paths ────────────────────────────────────────────

class TestParseAndValidationErrors:
    def test_retry_exhaustion_error_string_raises_clean_master_plan_error(self, monkeypatch):
        """Bug A: plan_handoff must detect acp_call_with_retry's
        error-string-on-exhaustion return BEFORE parsing it as JSON."""
        error_string = (
            "[m] Error: acp_call_with_retry failed after 3 attempts, "
            "last error: [m] Error: ACP prompt timed out after 240s."
        )
        monkeypatch.setattr(
            "pipeline.master.acp_call_with_retry", lambda *a, **k: error_string
        )
        with pytest.raises(MasterPlanError) as exc_info:
            MasterAgent(_config()).plan_handoff("design doc")

        assert str(exc_info.value) == error_string
        assert "Failed to parse" not in str(exc_info.value)

    def test_invalid_json_raises_master_plan_error(self, monkeypatch):
        monkeypatch.setattr(
            "pipeline.master.acp_call_with_retry", lambda *a, **k: "not json"
        )
        with pytest.raises(MasterPlanError):
            MasterAgent(_config()).plan_handoff("design doc")

    def test_missing_required_field_raises_master_plan_error(self, monkeypatch):
        bad_json = json.dumps({
            "items": [
                {
                    "id": "WI-000",
                    "type": "scaffold",
                    "language": "python",
                    "title": "scaffold",
                    # missing "description", "acceptance_criteria", etc.
                    "output_path": "pyproject.toml",
                    "depends_on": [],
                },
            ]
        })
        monkeypatch.setattr(
            "pipeline.master.acp_call_with_retry", lambda *a, **k: bad_json
        )
        with pytest.raises(MasterPlanError, match="missing required fields"):
            MasterAgent(_config()).plan_handoff("design doc")

    def test_invalid_type_raises_master_plan_error(self, monkeypatch):
        bad_json = json.dumps({
            "items": [
                {
                    "id": "WI-000",
                    "type": "migration",  # not a valid WorkItemType
                    "language": "python",
                    "title": "scaffold",
                    "description": "d",
                    "acceptance_criteria": [],
                    "output_path": "pyproject.toml",
                    "depends_on": [],
                },
            ]
        })
        monkeypatch.setattr(
            "pipeline.master.acp_call_with_retry", lambda *a, **k: bad_json
        )
        with pytest.raises(MasterPlanError, match="invalid type"):
            MasterAgent(_config()).plan_handoff("design doc")


# ── requirements_text plumbing ───────────────────────────────────────────────

class TestRequirementsTextPlumbing:
    def test_requirements_text_appended_to_user_content(self, monkeypatch):
        captured = {}

        def _fake_acp_call_with_retry(messages, model, config, **kwargs):
            captured["messages"] = messages
            return _small_plan_json()

        monkeypatch.setattr(
            "pipeline.master.acp_call_with_retry", _fake_acp_call_with_retry
        )
        MasterAgent(_config()).plan_handoff(
            "the design doc body", requirements_text="the requirements body"
        )

        user_message = captured["messages"][1]["content"]
        assert "the design doc body" in user_message
        assert "the requirements body" in user_message

    def test_no_requirements_text_omits_section(self, monkeypatch):
        captured = {}

        def _fake_acp_call_with_retry(messages, model, config, **kwargs):
            captured["messages"] = messages
            return _small_plan_json()

        monkeypatch.setattr(
            "pipeline.master.acp_call_with_retry", _fake_acp_call_with_retry
        )
        MasterAgent(_config()).plan_handoff("the design doc body")

        user_message = captured["messages"][1]["content"]
        assert user_message == "the design doc body"
