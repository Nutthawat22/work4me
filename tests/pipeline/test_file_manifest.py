"""
tests/pipeline/test_file_manifest.py

Tests for the additive, not-yet-wired file-indexed planning scaffold
(pipeline/file_manifest.py). No LLM calls and no network: exercises
group_into_work_items() and validate_file_manifest() with hand-made
FileManifests, and confirms plan_file_manifest() is still a stub.
"""

import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline.state import Feature, FileGroup, FileManifest, FileSpec
from pipeline.file_manifest import (
    FileManifestParseError,
    group_into_work_items,
    plan_file_manifest,
    validate_file_manifest,
)


# ── Fixtures / builders ──────────────────────────────────────────────────────

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


def _manifest() -> FileManifest:
    return FileManifest(files=[
        # (i) scaffold
        FileSpec(
            path="package.json",
            role="scaffold",
            language="typescript",
            contributing_features=[],
            requirements=["deps"],
        ),
        # (ii) two feature files for FEAT-auth
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
        # ...one for FEAT-requests
        FileSpec(
            path="src/server/routes/requests.ts",
            role="feature",
            language="typescript",
            contributing_features=["FEAT-requests"],
            requirements=["requests route"],
        ),
        # (iii) a shared en.json contributed by both
        FileSpec(
            path="src/i18n/en.json",
            role="shared",
            language="typescript",
            contributing_features=["FEAT-auth", "FEAT-requests"],
            requirements=["auth strings", "requests strings"],
        ),
        # (iv) entrypoint depending on auth+requests route files
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


def _by_id(groups: list[FileGroup]) -> dict[str, FileGroup]:
    return {g.id: g for g in groups}


# ── group_into_work_items ────────────────────────────────────────────────────

def test_group_scaffold_single_group():
    groups = _by_id(group_into_work_items(_manifest()))
    assert "GRP-scaffold" in groups
    assert groups["GRP-scaffold"].role == "scaffold"
    assert groups["GRP-scaffold"].files == ["package.json"]


def test_group_feature_auth_has_both_files():
    groups = _by_id(group_into_work_items(_manifest()))
    assert "GRP-feature-FEAT-auth" in groups
    g = groups["GRP-feature-FEAT-auth"]
    assert g.role == "feature"
    assert g.files == ["src/server/auth/service.ts", "src/server/routes/auth.ts"]
    assert g.contributing_features == ["FEAT-auth"]


def test_group_feature_requests():
    groups = _by_id(group_into_work_items(_manifest()))
    assert "GRP-feature-FEAT-requests" in groups
    assert groups["GRP-feature-FEAT-requests"].files == ["src/server/routes/requests.ts"]


def test_group_shared_en_json_own_group():
    groups = _by_id(group_into_work_items(_manifest()))
    gid = "GRP-shared-src-i18n-en-json"
    assert gid in groups
    g = groups[gid]
    assert g.role == "shared"
    assert g.files == ["src/i18n/en.json"]
    assert g.contributing_features == ["FEAT-auth", "FEAT-requests"]


def test_group_entrypoint_own_group_and_depends_on_feature_groups():
    groups = _by_id(group_into_work_items(_manifest()))
    gid = "GRP-entrypoint-src-index-ts"
    assert gid in groups
    g = groups[gid]
    assert g.role == "entrypoint"
    assert g.files == ["src/index.ts"]
    # index.ts depends_on_files the auth + requests route files, which live in
    # their respective feature groups.
    assert set(g.depends_on) == {"GRP-feature-FEAT-auth", "GRP-feature-FEAT-requests"}


def test_group_ordering_scaffold_first_entrypoint_last():
    groups = group_into_work_items(_manifest())
    roles = [g.role for g in groups]
    assert roles[0] == "scaffold"
    assert roles[-1] == "entrypoint"


def test_group_convergent_feature_file_treated_as_shared():
    manifest = FileManifest(files=[
        FileSpec(
            path="src/shared/util.ts",
            role="feature",
            language="typescript",
            contributing_features=["FEAT-auth", "FEAT-requests"],
        ),
    ])
    groups = _by_id(group_into_work_items(manifest))
    gid = "GRP-shared-src-shared-util-ts"
    assert gid in groups
    assert groups[gid].role == "shared"


# ── validate_file_manifest ───────────────────────────────────────────────────

def test_validate_wellformed_returns_empty():
    assert validate_file_manifest(_manifest(), _features()) == []


def test_validate_duplicate_path():
    manifest = _manifest()
    manifest.files.append(FileSpec(
        path="src/server/routes/auth.ts",
        role="feature",
        language="typescript",
        contributing_features=["FEAT-auth"],
    ))
    errors = validate_file_manifest(manifest, _features())
    assert any("duplicate file path" in e for e in errors)


def test_validate_dangling_dependency():
    manifest = _manifest()
    manifest.files[-1].depends_on_files.append("src/does/not/exist.ts")
    errors = validate_file_manifest(manifest, _features())
    assert any("dangling depends_on_files" in e for e in errors)


def test_validate_self_dependency():
    manifest = _manifest()
    manifest.files[1].depends_on_files.append(manifest.files[1].path)
    errors = validate_file_manifest(manifest, _features())
    assert any("depends on itself" in e for e in errors)


def test_validate_dropped_feature():
    features = _features()
    features.append(Feature(
        id="FEAT-orphan",
        title="Orphan",
        description="nobody builds this",
        acceptance_criteria=[],
        language="typescript",
    ))
    errors = validate_file_manifest(_manifest(), features)
    assert any("dropped feature" in e and "FEAT-orphan" in e for e in errors)


def test_validate_unknown_feature_id():
    manifest = _manifest()
    manifest.files[1].contributing_features.append("FEAT-ghost")
    errors = validate_file_manifest(manifest, _features())
    assert any("unknown feature id" in e and "FEAT-ghost" in e for e in errors)


def test_validate_non_scaffold_without_contributing_feature():
    manifest = _manifest()
    manifest.files.append(FileSpec(
        path="src/lonely.ts",
        role="feature",
        language="typescript",
        contributing_features=[],
    ))
    errors = validate_file_manifest(manifest, _features())
    assert any("no contributing features" in e for e in errors)


def test_validate_cycle_detection():
    manifest = FileManifest(files=[
        FileSpec(path="a.ts", role="feature", language="typescript",
                 contributing_features=["FEAT-auth"], depends_on_files=["b.ts"]),
        FileSpec(path="b.ts", role="feature", language="typescript",
                 contributing_features=["FEAT-auth"], depends_on_files=["a.ts"]),
    ])
    features = [Feature(id="FEAT-auth", title="a", description="a",
                        acceptance_criteria=[], language="typescript")]
    errors = validate_file_manifest(manifest, features)
    assert any("dependency cycle among files" in e for e in errors)


# ── plan_file_manifest (LLM call, monkeypatched) ─────────────────────────────

def _config() -> dict:
    return {
        "languages": {"typescript": {}},
        "models": {"design": {"model": "m", "provider": "responses"}},
    }


def test_plan_file_manifest_happy_path(monkeypatch):
    manifest_json = json.dumps({
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
            {"path": "src/i18n/en.json", "role": "shared", "language": "typescript",
             "contributing_features": ["FEAT-auth", "FEAT-requests"],
             "requirements": ["auth strings", "requests strings"],
             "depends_on_files": []},
            {"path": "src/index.ts", "role": "entrypoint", "language": "typescript",
             "contributing_features": ["FEAT-auth", "FEAT-requests"],
             "requirements": ["wire routes"],
             "depends_on_files": ["src/server/routes/auth.ts", "src/server/routes/requests.ts"]},
        ]
    })

    def fake(*args, **kwargs):
        return manifest_json

    monkeypatch.setattr("pipeline.file_manifest.call_llm", fake)

    manifest = plan_file_manifest(_features(), _config())
    assert len(manifest.files) == 5
    shared = next(f for f in manifest.files if f.path == "src/i18n/en.json")
    assert set(shared.contributing_features) == {"FEAT-auth", "FEAT-requests"}


def test_plan_file_manifest_invalid_json_raises(monkeypatch):
    monkeypatch.setattr("pipeline.file_manifest.call_llm", lambda *a, **k: "not json")
    with pytest.raises(FileManifestParseError):
        plan_file_manifest(_features(), _config())


def test_plan_file_manifest_validation_failure_raises(monkeypatch):
    bad_json = json.dumps({
        "files": [
            {"path": "package.json", "role": "scaffold", "language": "typescript",
             "contributing_features": [], "requirements": [], "depends_on_files": []},
            {"path": "src/dup.ts", "role": "feature", "language": "typescript",
             "contributing_features": ["FEAT-auth"], "requirements": [], "depends_on_files": []},
            {"path": "src/dup.ts", "role": "feature", "language": "typescript",
             "contributing_features": ["FEAT-requests"], "requirements": [], "depends_on_files": []},
        ]
    })
    monkeypatch.setattr("pipeline.file_manifest.call_llm", lambda *a, **k: bad_json)
    with pytest.raises(FileManifestParseError):
        plan_file_manifest(_features(), _config())


def test_plan_file_manifest_llm_error_sentinel(monkeypatch):
    monkeypatch.setattr("pipeline.file_manifest.call_llm", lambda *a, **k: "[m] Error: boom")
    with pytest.raises(FileManifestParseError):
        plan_file_manifest(_features(), _config())
