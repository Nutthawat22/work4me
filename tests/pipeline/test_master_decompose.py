"""
tests/pipeline/test_master_decompose.py

Tests for MasterAgent.decompose() (free-text prompt -> flat WorkItem
array), moved from the former pipeline/design.py::DesignAgent.decompose
in Phase 2 of the ACP-native rearchitecture. No real LLM/subprocess
calls: monkeypatches pipeline.master.acp_call_with_retry directly.
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
        "languages": {"python": {}},
        "models": {"design": {"model": "m", "provider": "acp"}},
    }


def _work_items_json() -> str:
    return json.dumps({
        "items": [
            {
                "id": "WI-001",
                "type": "logic",
                "language": "python",
                "title": "Add two numbers",
                "description": "Implement add(a, b)",
                "acceptance_criteria": ["add(2, 3) == 5"],
                "output_path": "add.py",
                "depends_on": [],
            },
        ]
    })


class TestDecomposeHappyPath:
    def test_decompose_returns_work_items(self, monkeypatch):
        monkeypatch.setattr(
            "pipeline.master.acp_call_with_retry", lambda *a, **k: _work_items_json()
        )
        work_items = MasterAgent(_config()).decompose("build an adder")
        assert len(work_items) == 1
        assert work_items[0].id == "WI-001"
        assert work_items[0].type == "logic"

    def test_decompose_accepts_bare_array(self, monkeypatch):
        bare_array = json.dumps([
            {
                "id": "WI-001",
                "type": "logic",
                "language": "python",
                "title": "t",
                "description": "d",
                "acceptance_criteria": [],
                "output_path": "a.py",
                "depends_on": [],
            },
        ])
        monkeypatch.setattr(
            "pipeline.master.acp_call_with_retry", lambda *a, **k: bare_array
        )
        work_items = MasterAgent(_config()).decompose("build something")
        assert len(work_items) == 1


class TestDecomposeErrors:
    def test_retry_exhaustion_error_string_raises_clean_master_plan_error(self, monkeypatch):
        """
        Bug A: acp_call_with_retry never raises -- on total exhaustion it
        returns an f"[{model}] Error: ..." string. decompose() must detect
        that BEFORE calling _parse_json, raising a clean MasterPlanError
        whose message IS that error string, not a nested "failed to parse
        as JSON" wrapper around it.
        """
        error_string = (
            "[m] Error: acp_call_with_retry failed after 3 attempts, "
            "last error: [m] Error: ACP prompt timed out after 120s."
        )
        monkeypatch.setattr(
            "pipeline.master.acp_call_with_retry", lambda *a, **k: error_string
        )
        with pytest.raises(MasterPlanError) as exc_info:
            MasterAgent(_config()).decompose("build something")

        assert str(exc_info.value) == error_string
        assert "Failed to parse" not in str(exc_info.value)

    def test_invalid_json_raises_master_plan_error(self, monkeypatch):
        monkeypatch.setattr(
            "pipeline.master.acp_call_with_retry", lambda *a, **k: "not json"
        )
        with pytest.raises(MasterPlanError):
            MasterAgent(_config()).decompose("build something")

    def test_concatenated_json_documents_raises_specific_master_plan_error(self):
        """
        Bug B: two complete JSON documents concatenated with no separator
        must raise a MasterPlanError that specifically calls out "more
        than one JSON document" / extra data, not a generic
        json.JSONDecodeError-derived message.
        """
        concatenated = '{"items": []}{"items": []}'
        with pytest.raises(MasterPlanError, match="more than one JSON document"):
            MasterAgent._parse_json(concatenated)

    def test_invalid_language_raises_master_plan_error(self, monkeypatch):
        bad_json = json.dumps({
            "items": [
                {
                    "id": "WI-001",
                    "type": "logic",
                    "language": "rust",  # not in config["languages"]
                    "title": "t",
                    "description": "d",
                    "acceptance_criteria": [],
                    "output_path": "a.rs",
                    "depends_on": [],
                },
            ]
        })
        monkeypatch.setattr(
            "pipeline.master.acp_call_with_retry", lambda *a, **k: bad_json
        )
        with pytest.raises(MasterPlanError, match="invalid language"):
            MasterAgent(_config()).decompose("build something")
