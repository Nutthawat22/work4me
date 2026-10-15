"""
tests/pipeline/test_manifest_wiring.py

Tests for wiring the file-manifest planning stage into the live pipeline:
the file_groups_to_work_items adapter and MasterAgent.decompose_features.
No network / no real LLM calls.
"""

import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline.state import Feature, FileManifest, FileSpec
from pipeline.file_manifest import (
    file_groups_to_work_items,
    group_into_work_items,
)
from pipeline.master import MasterAgent, MasterPlanError


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _manifest() -> FileManifest:
    return FileManifest(files=[
        FileSpec(
            path="package.json",
            role="scaffold",
            language="typescript",
            contributing_features=[],
            requirements=["deps"],
        ),
        FileSpec(
            path="src/server/routes/auth.ts",
            role="feature",
            language="typescript",
            contributing_features=["FEAT-auth"],
            requirements=["login route"],
        ),
        FileSpec(
            path="src/server/auth/service.ts",
            role="feature",
            language="typescript",
            contributing_features=["FEAT-auth"],
            requirements=["auth service"],
        ),
        FileSpec(
            path="src/server/routes/requests.ts",
            role="feature",
            language="typescript",
            contributing_features=["FEAT-requests"],
            requirements=["requests route"],
        ),
        FileSpec(
            path="src/i18n/en.json",
            role="shared",
            language="typescript",
            contributing_features=["FEAT-auth", "FEAT-requests"],
            requirements=["auth strings", "requests strings"],
        ),
        FileSpec(
            path="src/index.ts",
            role="entrypoint",
            language="typescript",
            contributing_features=["FEAT-auth", "FEAT-requests"],
            requirements=["wire routes"],
            depends_on_files=[
                "src/server/routes/auth.ts",
                "src/server/routes/requests.ts",
            ],
        ),
    ])


def _config() -> dict:
    return {
        "languages": {"typescript": {}},
        "models": {"design": {"model": "m", "provider": "acp"}},
    }


def _features() -> list[Feature]:
    return [
        Feature("FEAT-auth", "Auth", "Authentication", ["users can log in"], "typescript", ["REQ-001"]),
        Feature("FEAT-requests", "Requests", "Requesting", ["users can submit"], "typescript", ["REQ-002"]),
    ]


# ── file_groups_to_work_items ────────────────────────────────────────────────

def test_file_groups_to_work_items_roles_and_mapping():
    manifest = _manifest()
    groups = group_into_work_items(manifest)
    work_items = file_groups_to_work_items(groups, manifest, _features())

    by_id = {wi.id: wi for wi in work_items}

    # scaffold
    assert by_id["TASK-foundation"].category == "foundation"

    # feature groups
    assert by_id["TASK-feature-FEAT-auth"].category == "feature"
    assert by_id["TASK-feature-FEAT-requests"].category == "feature"

    # shared group (en.json contributed by both)
    shared = by_id["TASK-shared-feat-auth-feat-requests"]
    assert shared.category == "shared"

    # entrypoint -> integrate
    entry = by_id["TASK-integration"]
    assert entry.category == "integration"

    # depends_on preserved (group ids)
    assert set(entry.depends_on) == {"TASK-feature-FEAT-auth", "TASK-feature-FEAT-requests"}


def test_file_groups_to_work_items_owned_files_and_description():
    manifest = _manifest()
    groups = group_into_work_items(manifest)
    work_items = file_groups_to_work_items(groups, manifest, _features())
    by_id = {wi.id: wi for wi in work_items}

    auth = by_id["TASK-feature-FEAT-auth"]
    group = next(g for g in groups if g.id == "TASK-feature-FEAT-auth")

    assert {owned.path for owned in auth.owned_files} == set(group.files)

    # description embeds each file path.
    for path in group.files:
        assert path in auth.description

    # per-file requirements are embedded.
    assert "login route" in auth.description
    assert "auth service" in auth.description


def test_file_groups_to_work_items_order_preserved():
    manifest = _manifest()
    groups = group_into_work_items(manifest)
    work_items = file_groups_to_work_items(groups, manifest, _features())
    assert [g.id for g in groups] == [wi.id for wi in work_items]


# ── decompose_features ───────────────────────────────────────────────────────

def test_decompose_features_happy_path(monkeypatch):
    features_json = json.dumps({
        "features": [
            {
                "id": "FEAT-auth",
                "title": "Authentication",
                "description": "Users can log in.",
                "acceptance_criteria": ["users can log in"],
                "language": "typescript",
                "source_ids": ["REQ-001"],
                "depends_on": [],
            },
            {
                "id": "FEAT-requests",
                "title": "Request authoring",
                "description": "Users can submit requests.",
                "acceptance_criteria": ["users can submit requests"],
                "language": "typescript",
                "source_ids": ["REQ-002"],
                "depends_on": ["FEAT-auth"],
            },
        ]
    })
    monkeypatch.setattr("pipeline.master.acp_call_with_retry", lambda *a, **k: features_json)

    features = MasterAgent(_config()).decompose_features("design text")
    assert [f.id for f in features] == ["FEAT-auth", "FEAT-requests"]
    assert all(isinstance(f, Feature) for f in features)
    assert features[1].depends_on == ["FEAT-auth"]


def test_decompose_features_invalid_json_raises(monkeypatch):
    monkeypatch.setattr("pipeline.master.acp_call_with_retry", lambda *a, **k: "not json")
    with pytest.raises(MasterPlanError):
        MasterAgent(_config()).decompose_features("design text")


def test_decompose_features_retry_exhaustion_error_string_raises_clean_master_plan_error(monkeypatch):
    """Bug A: decompose_features must detect acp_call_with_retry's
    error-string-on-exhaustion return BEFORE parsing it as JSON."""
    error_string = (
        "[m] Error: acp_call_with_retry failed after 3 attempts, "
        "last error: [m] Error: ACP prompt timed out after 240s."
    )
    monkeypatch.setattr("pipeline.master.acp_call_with_retry", lambda *a, **k: error_string)

    with pytest.raises(MasterPlanError) as exc_info:
        MasterAgent(_config()).decompose_features("design text")

    assert str(exc_info.value) == error_string
    assert "Failed to parse" not in str(exc_info.value)


def test_parse_features_json_concatenated_documents_raises_specific_error():
    """Bug B: same "extra data" detection applies to
    _parse_features_json as _parse_json."""
    concatenated = '{"features": []}{"features": []}'
    with pytest.raises(MasterPlanError, match="more than one JSON document"):
        MasterAgent._parse_features_json(concatenated)
