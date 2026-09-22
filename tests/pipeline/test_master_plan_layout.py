"""
tests/pipeline/test_master_plan_layout.py

Tests for MasterAgent.plan_layout() (Phase 4 of the ACP-native
rearchitecture — see
dev-plans/agents/features/2026-09-14-acp-native-master-architecture.md,
"Phase 4 — file_manifest.py as a Selectable Strategy").

No real LLM/subprocess calls: the "llm_decided" strategy monkeypatches
pipeline.file_manifest.acp_call_with_retry directly with a canned JSON
response; the "external" strategy never calls into an LLM at all, so
those tests confirm no LLM call is attempted rather than monkeypatching
one.
"""

import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline.master import MasterAgent, MasterPlanError
from pipeline.state import Feature


def _config() -> dict:
    return {
        "languages": {"typescript": {}},
        "models": {"design": {"model": "m", "provider": "acp"}},
    }


def _features() -> list[Feature]:
    return [
        Feature(
            id="FEAT-auth",
            title="Auth",
            description="Authentication",
            acceptance_criteria=["users can log in"],
            language="typescript",
        ),
        Feature(
            id="FEAT-requests",
            title="Requests",
            description="Requests feature",
            acceptance_criteria=["users can submit requests"],
            language="typescript",
        ),
    ]


def _manifest_json() -> str:
    return json.dumps({
        "files": [
            {"path": "package.json", "role": "scaffold", "language": "typescript",
             "contributing_features": [], "requirements": ["deps"],
             "depends_on_files": []},
            {"path": "src/server/routes/auth.ts", "role": "feature", "language": "typescript",
             "contributing_features": ["FEAT-auth"], "requirements": ["login route"],
             "depends_on_files": ["package.json"]},
            {"path": "src/server/routes/requests.ts", "role": "feature", "language": "typescript",
             "contributing_features": ["FEAT-requests"], "requirements": ["requests route"],
             "depends_on_files": ["package.json"]},
        ]
    })


def _external_layout_dict() -> dict:
    return {
        "files": [
            {"path": "package.json", "role": "scaffold", "language": "typescript",
             "contributing_features": [], "requirements": ["deps"],
             "depends_on_files": []},
            {"path": "src/server/routes/auth.ts", "role": "feature", "language": "typescript",
             "contributing_features": ["FEAT-auth"], "requirements": ["login route"],
             "depends_on_files": ["package.json"]},
            {"path": "src/server/routes/requests.ts", "role": "feature", "language": "typescript",
             "contributing_features": ["FEAT-requests"], "requirements": ["requests route"],
             "depends_on_files": ["package.json"]},
        ]
    }


# ── llm_decided strategy ─────────────────────────────────────────────────────

class TestLlmDecidedStrategy:
    def test_happy_path_returns_work_items(self, monkeypatch):
        monkeypatch.setattr(
            "pipeline.file_manifest.acp_call_with_retry", lambda *a, **k: _manifest_json()
        )
        work_items = MasterAgent(_config()).plan_layout(_features(), strategy="llm_decided")

        assert len(work_items) == 3  # scaffold + 2 feature groups
        by_id = {wi.id: wi for wi in work_items}
        assert by_id["GRP-scaffold"].type == "scaffold"
        assert by_id["GRP-feature-FEAT-auth"].type == "feature"
        assert by_id["GRP-feature-FEAT-requests"].type == "feature"

    def test_default_strategy_is_llm_decided(self, monkeypatch):
        monkeypatch.setattr(
            "pipeline.file_manifest.acp_call_with_retry", lambda *a, **k: _manifest_json()
        )
        work_items = MasterAgent(_config()).plan_layout(_features())
        assert len(work_items) == 3

    def test_file_manifest_parse_error_wrapped_as_master_plan_error(self, monkeypatch):
        monkeypatch.setattr(
            "pipeline.file_manifest.acp_call_with_retry", lambda *a, **k: "not json"
        )
        with pytest.raises(MasterPlanError):
            MasterAgent(_config()).plan_layout(_features(), strategy="llm_decided")


# ── external strategy ─────────────────────────────────────────────────────────

class TestExternalStrategy:
    def test_happy_path_returns_work_items_no_llm_call(self, monkeypatch):
        def _fail_if_called(*args, **kwargs):
            raise AssertionError("acp_call_with_retry must not be called for strategy='external'")

        monkeypatch.setattr("pipeline.file_manifest.acp_call_with_retry", _fail_if_called)

        work_items = MasterAgent(_config()).plan_layout(
            _features(), strategy="external", external_layout=_external_layout_dict()
        )

        assert len(work_items) == 3
        by_id = {wi.id: wi for wi in work_items}
        assert by_id["GRP-scaffold"].type == "scaffold"
        assert by_id["GRP-feature-FEAT-auth"].type == "feature"
        assert by_id["GRP-feature-FEAT-requests"].type == "feature"

    def test_none_external_layout_raises(self):
        with pytest.raises(MasterPlanError):
            MasterAgent(_config()).plan_layout(
                _features(), strategy="external", external_layout=None
            )

    def test_malformed_file_spec_raises(self):
        malformed = {
            "files": [
                {
                    "path": "package.json",
                    "role": "scaffold",
                    # missing "language", "contributing_features",
                    # "requirements", "depends_on_files"
                },
            ]
        }
        with pytest.raises(MasterPlanError):
            MasterAgent(_config()).plan_layout(
                _features(), strategy="external", external_layout=malformed
            )

    def test_validation_failure_raises_with_details(self):
        """Dropped feature (FEAT-requests contributes to no file) must
        surface as a MasterPlanError whose message includes the
        validate_file_manifest error detail."""
        layout = {
            "files": [
                {"path": "package.json", "role": "scaffold", "language": "typescript",
                 "contributing_features": [], "requirements": [], "depends_on_files": []},
                {"path": "src/server/routes/auth.ts", "role": "feature", "language": "typescript",
                 "contributing_features": ["FEAT-auth"], "requirements": [], "depends_on_files": []},
                # FEAT-requests never contributes to any file.
            ]
        }
        with pytest.raises(MasterPlanError, match="dropped feature") as exc_info:
            MasterAgent(_config()).plan_layout(
                _features(), strategy="external", external_layout=layout
            )
        assert "FEAT-requests" in str(exc_info.value)

    def test_dangling_dependency_raises_with_details(self):
        layout = {
            "files": [
                {"path": "package.json", "role": "scaffold", "language": "typescript",
                 "contributing_features": [], "requirements": [], "depends_on_files": []},
                {"path": "src/server/routes/auth.ts", "role": "feature", "language": "typescript",
                 "contributing_features": ["FEAT-auth"], "requirements": [],
                 "depends_on_files": ["src/does/not/exist.ts"]},
                {"path": "src/server/routes/requests.ts", "role": "feature", "language": "typescript",
                 "contributing_features": ["FEAT-requests"], "requirements": [], "depends_on_files": []},
            ]
        }
        with pytest.raises(MasterPlanError, match="dangling depends_on_files"):
            MasterAgent(_config()).plan_layout(
                _features(), strategy="external", external_layout=layout
            )


# ── unknown strategy ──────────────────────────────────────────────────────────

class TestUnknownStrategy:
    def test_unknown_strategy_raises_with_message(self):
        with pytest.raises(MasterPlanError, match="Unknown layout strategy"):
            MasterAgent(_config()).plan_layout(_features(), strategy="bogus")
