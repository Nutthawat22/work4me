"""
tests/pipeline/test_manifest_wiring.py

Tests for wiring the file-manifest planning stage into the live pipeline:
the file_groups_to_work_items adapter and DesignAgent.decompose_features.
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
from pipeline.design import DesignAgent, DesignParseError


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
        "models": {"design": {"model": "m", "provider": "responses"}},
    }


# ── file_groups_to_work_items ────────────────────────────────────────────────

def test_file_groups_to_work_items_roles_and_mapping():
    manifest = _manifest()
    groups = group_into_work_items(manifest)
    work_items = file_groups_to_work_items(groups, manifest)

    by_id = {wi.id: wi for wi in work_items}

    # scaffold
    assert by_id["GRP-scaffold"].type == "scaffold"

    # feature groups
    assert by_id["GRP-feature-FEAT-auth"].type == "feature"
    assert by_id["GRP-feature-FEAT-requests"].type == "feature"

    # shared group (en.json contributed by both)
    shared = by_id["GRP-shared-src-i18n-en-json"]
    assert shared.type == "shared"

    # entrypoint -> integrate
    entry = by_id["GRP-entrypoint-src-index-ts"]
    assert entry.type == "integrate"

    # depends_on preserved (group ids)
    assert set(entry.depends_on) == {"GRP-feature-FEAT-auth", "GRP-feature-FEAT-requests"}


def test_file_groups_to_work_items_output_path_and_description():
    manifest = _manifest()
    groups = group_into_work_items(manifest)
    work_items = file_groups_to_work_items(groups, manifest)
    by_id = {wi.id: wi for wi in work_items}

    auth = by_id["GRP-feature-FEAT-auth"]
    group = next(g for g in groups if g.id == "GRP-feature-FEAT-auth")

    # output_path is one of the group's files.
    assert auth.output_path in group.files

    # description embeds each file path.
    for path in group.files:
        assert path in auth.description

    # per-file requirements are embedded.
    assert "login route" in auth.description
    assert "auth service" in auth.description


def test_file_groups_to_work_items_order_preserved():
    manifest = _manifest()
    groups = group_into_work_items(manifest)
    work_items = file_groups_to_work_items(groups, manifest)
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
    monkeypatch.setattr("pipeline.design.call_llm", lambda *a, **k: features_json)

    features = DesignAgent(_config()).decompose_features("design text")
    assert [f.id for f in features] == ["FEAT-auth", "FEAT-requests"]
    assert all(isinstance(f, Feature) for f in features)
    assert features[1].depends_on == ["FEAT-auth"]


def test_decompose_features_invalid_json_raises(monkeypatch):
    monkeypatch.setattr("pipeline.design.call_llm", lambda *a, **k: "not json")
    with pytest.raises(DesignParseError):
        DesignAgent(_config()).decompose_features("design text")
