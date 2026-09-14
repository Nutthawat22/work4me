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
    def test_invalid_json_raises_master_plan_error(self, monkeypatch):
        monkeypatch.setattr(
            "pipeline.master.acp_call_with_retry", lambda *a, **k: "not json"
        )
        with pytest.raises(MasterPlanError):
            MasterAgent(_config()).decompose("build something")

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
