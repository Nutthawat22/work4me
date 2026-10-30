import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline.project_context import snapshot_project_files
from pipeline.state import AgentResult, OwnedFile, WorkItem
from pipeline.static_validator import StaticValidator
from pipeline.testing_handoff import build_testing_handoff, write_testing_handoff


def _work_item(path: str, language: str = "python", area: str = "product") -> WorkItem:
    return WorkItem(
        id="TASK-one",
        category="feature",
        language=language,
        module="one",
        scope="Implement one feature",
        title="One feature",
        description="Implement one feature",
        acceptance_criteria=["The feature has an owned source file."],
        requirement_refs=[],
        owned_files=[OwnedFile(path, language, area)],
    )


def _work_item_for_paths(paths: list[str]) -> WorkItem:
    item = _work_item(paths[0])
    item.owned_files = [OwnedFile(path, "python") for path in paths]
    return item


def _config() -> dict:
    return {
        "pipeline": {"output_dir": "product", "tests_dir": "tests"},
        "languages": {
      "python": {"test_command": ["pytest", "--api-key", "API-SUPER-SECRET", "--header", "Authorization: Bearer HEADER-SECRET", "{tests_dir}"]},
            "typescript": {"test_command": ["bunx", "vitest", "run", "{tests_dir}"]},
        },
    }


def test_python_validation_parses_without_importing_or_executing_source(tmp_path):
    run_dir = tmp_path / "run"
    product = run_dir / "product"
    product.mkdir(parents=True)
    sentinel = tmp_path / "executed"
    (product / "app.py").write_text(f"open({str(sentinel)!r}, 'w').write('bad')\n")

    result = StaticValidator().run(
        _config(), [_work_item("app.py")], str(run_dir),
        [AgentResult("TASK-one", "FeatureAgent", True, ["product/app.py"])],
    )

    assert result.passed
    assert result.checked_files == 1
    assert not sentinel.exists()


def test_python_syntax_and_relative_import_errors_are_definite(tmp_path):
    run_dir = tmp_path / "run"
    product = run_dir / "product"
    product.mkdir(parents=True)
    source = product / "app.py"
    source.write_text("from .missing import value\ndef broken(:\n    pass\n")

    result = StaticValidator().run(_config(), [_work_item("app.py")], str(run_dir))

    assert not result.passed
    assert any("Python syntax error" in issue.error_output for issue in result.errors)

    source.write_text("from .missing import value\n")
    result = StaticValidator().run(_config(), [_work_item("app.py")], str(run_dir))
    assert any("Relative Python import target not found" in issue.error_output for issue in result.errors)


def test_unused_private_python_declaration_is_warning_only(tmp_path):
    run_dir = tmp_path / "run"
    product = run_dir / "product"
    product.mkdir(parents=True)
    (product / "app.py").write_text("def _unused_helper():\n    return True\n")

    result = StaticValidator().run(_config(), [_work_item("app.py")], str(run_dir))

    assert result.passed
    assert len(result.warnings) == 1
    assert "_unused_helper" in result.warnings[0].error_output


def test_python_callback_and_dynamic_exports_are_not_reported_as_unused(tmp_path):
    run_dir = tmp_path / "run"
    product = run_dir / "product"
    product.mkdir(parents=True)
    (product / "app.py").write_text(
        "__all__ = ['_exported']\n"
        "def _exported():\n    return True\n"
        "def _on_click():\n    return True\n"
        "def _registered_handler():\n    return False\n"
        "def _framework_registered():\n    return None\n"
        "registry = {'callback': '_framework_registered'}\n"
        "def _unused_helper():\n    return True\n"
    )

    result = StaticValidator().run(_config(), [_work_item("app.py")], str(run_dir))

    assert result.passed
    assert len(result.warnings) == 1
    assert "_unused_helper" in result.warnings[0].error_output


def test_typescript_syntax_is_reported_unsupported_not_regex_validated(tmp_path):
    run_dir = tmp_path / "run"
    product = run_dir / "product"
    product.mkdir(parents=True)
    (product / "app.ts").write_text("const broken: string = ;\n")

    result = StaticValidator().run(_config(), [_work_item("app.ts", "typescript")], str(run_dir))

    assert result.passed
    assert any("syntax, type correctness" in limitation for limitation in result.limitations)
    assert any("local import targets are not parsed" in limitation for limitation in result.limitations)
    assert any("unused-symbol analysis is unsupported" in limitation for limitation in result.limitations)


def test_literal_javascript_local_import_and_manifest_errors_are_reported(tmp_path):
    run_dir = tmp_path / "run"
    product = run_dir / "product"
    product.mkdir(parents=True)
    (product / "app.js").write_text('import missing from "./missing.js";\n')
    (product / "package.json").write_text("{broken json")
    item = WorkItem(
        id="TASK-one", category="feature", language="javascript", module="one",
        scope="Implement app", title="App", description="Implement app",
        acceptance_criteria=["App source and manifest exist"], requirement_refs=[],
        owned_files=[OwnedFile("app.js", "javascript"), OwnedFile("package.json", "javascript")],
    )

    result = StaticValidator().run(_config(), [item], str(run_dir))

    assert not result.passed
    assert any("Invalid package.json manifest" in issue.error_output for issue in result.errors)
    assert any("local import targets are not parsed" in limitation for limitation in result.limitations)


def test_python_relative_import_target_is_checked_with_ast(tmp_path):
    run_dir = tmp_path / "run"
    product = run_dir / "product"
    product.mkdir(parents=True)
    source = product / "app.py"
    source.write_text("from . import missing\n")

    result = StaticValidator().run(_config(), [_work_item("app.py")], str(run_dir))

    assert not result.passed
    assert any("Relative Python import target not found" in issue.error_output for issue in result.errors)


@pytest.mark.parametrize("existing_project", [False, True], ids=["new-project", "existing-project"])
def test_python_relative_imports_resolve_siblings_and_parent_packages(tmp_path, existing_project):
    root = tmp_path / "project" if existing_project else tmp_path / "run"
    project_base = root if existing_project else root / "product"
    source_dir = project_base / "pkg" / "sub"
    source_dir.mkdir(parents=True)
    (source_dir / "app.py").write_text("from .helpers import sibling\nfrom ..shared import parent\n")
    (source_dir / "helpers.py").write_text("sibling = True\n")
    (project_base / "pkg" / "shared.py").write_text("parent = True\n")
    paths = ["pkg/sub/app.py", "pkg/sub/helpers.py", "pkg/shared.py"]
    config = _config()
    if existing_project:
        config["pipeline"]["_existing_project_path"] = str(root)

    result = StaticValidator().run(config, [_work_item_for_paths(paths)], str(tmp_path / "run"))

    assert result.passed
    (project_base / "pkg" / "shared.py").unlink()
    missing = StaticValidator().run(config, [_work_item_for_paths(paths)], str(tmp_path / "run"))
    assert any("Relative Python import target not found" in issue.error_output for issue in missing.errors)


@pytest.mark.parametrize("source", ["return\n", "break\n"])
def test_module_scope_control_flow_compile_errors_are_reported(tmp_path, source):
    product = tmp_path / "run" / "product"
    product.mkdir(parents=True)
    (product / "app.py").write_text(source)

    result = StaticValidator().run(_config(), [_work_item("app.py")], str(tmp_path / "run"))

    assert not result.passed
    assert any("Python compile error" in issue.error_output for issue in result.errors)


def test_poetry_metadata_shape_errors_are_reported(tmp_path):
    product = tmp_path / "run" / "product"
    product.mkdir(parents=True)
    (product / "pyproject.toml").write_text("[tool.poetry]\ndependencies = []\n")

    result = StaticValidator().run(_config(), [_work_item("pyproject.toml")], str(tmp_path / "run"))

    assert not result.passed
    assert any("tool.poetry.dependencies must be a table" in issue.error_output for issue in result.errors)


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("Pipfile", "packages = []\n"),
        ("Cargo.toml", "dependencies = []\n"),
        ("package.json", '{"scripts": []}\n'),
    ],
)
def test_manifest_nested_shape_errors_are_reported(tmp_path, filename, content):
    product = tmp_path / "run" / "product"
    product.mkdir(parents=True)
    (product / filename).write_text(content)

    result = StaticValidator().run(_config(), [_work_item(filename)], str(tmp_path / "run"))

    assert not result.passed
    assert any(
        (filename in issue.file_path and "metadata shape" in issue.error_output)
        or (filename == "package.json" and "Invalid package.json manifest" in issue.error_output)
        for issue in result.errors
    )


def test_valid_multiple_requirement_specifiers_are_not_blocked(tmp_path):
    product = tmp_path / "run" / "product"
    product.mkdir(parents=True)
    (product / "requirements.txt").write_text(
        "requests>=2.31,<3\nurllib3>=1.26,!=2.0.0\n"
    )

    result = StaticValidator().run(
        _config(), [_work_item("requirements.txt")], str(tmp_path / "run")
    )

    assert result.passed
    assert not any("Invalid requirements.txt" in issue.error_output for issue in result.errors)


def test_invalid_toml_and_requirements_manifests_are_reported(tmp_path):
    run_dir = tmp_path / "run"
    product = run_dir / "product"
    product.mkdir(parents=True)
    (product / "pyproject.toml").write_text("[project\nname = 'bad'\n")
    (product / "requirements.txt").write_text("invalid requirement syntax here\n")
    item = WorkItem(
        id="TASK-one", category="feature", language="python", module="one",
        scope="Implement app", title="App", description="Implement app",
        acceptance_criteria=["App manifests are valid"], requirement_refs=[],
        owned_files=[
            OwnedFile("pyproject.toml", "python"),
            OwnedFile("requirements.txt", "python"),
        ],
    )

    result = StaticValidator().run(_config(), [item], str(run_dir))

    assert not result.passed
    assert any("Invalid TOML manifest" in issue.error_output for issue in result.errors)
    assert any("Invalid requirements.txt entry syntax" in issue.error_output for issue in result.errors)


def test_unowned_generated_files_and_missing_expected_files_block(tmp_path):
    run_dir = tmp_path / "run"
    product = run_dir / "product"
    product.mkdir(parents=True)
    (product / "unexpected.py").write_text("value = 1\n")

    result = StaticValidator().run(_config(), [_work_item("missing.py")], str(run_dir))

    assert not result.passed
    assert any("Expected owned file is missing" in issue.error_output for issue in result.errors)
    assert any("not owned by a planned task" in issue.error_output for issue in result.errors)


def test_existing_project_allows_owned_deletion_and_rejects_unowned_edits(tmp_path):
    project = tmp_path / "existing"
    source = project / "src" / "app.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n")
    baseline = snapshot_project_files(str(project))
    source.unlink()
    config = {
        "pipeline": {
            "output_dir": "product",
            "tests_dir": "tests",
            "_existing_project_path": str(project),
            "_existing_project_baseline": baseline,
        },
        "languages": {},
    }

    deleted = StaticValidator().run(
        config,
        [_work_item("src/app.py")],
        str(tmp_path / "run"),
        [AgentResult("TASK-one", "FeatureAgent", True, [], files_deleted=["src/app.py"])],
    )
    assert deleted.passed

    source.write_text("value = 1\n")
    clean_baseline = snapshot_project_files(str(project))
    (project / "unowned.py").write_text("value = 2\n")
    config["pipeline"]["_existing_project_baseline"] = clean_baseline
    changed = StaticValidator().run(
        config,
        [_work_item("src/app.py")],
        str(tmp_path / "run"),
        [AgentResult("TASK-one", "FeatureAgent", True, ["src/app.py"])],
    )
    assert not changed.passed
    assert any("outside planned file ownership" in issue.error_output for issue in changed.errors)


def test_testing_handoff_lists_declared_metadata_without_secrets(tmp_path):
    run_dir = tmp_path / "runs" / "0001-demo"
    product = run_dir / "product"
    product.mkdir(parents=True)
    (product / "package.json").write_text(json.dumps({
        "name": "demo",
        "scripts": {"test": "vitest run", "build": "tsc -b"},
        "dependencies": {"left-pad": "^1.3.0"},
    }))
    (product / "package-lock.json").write_text('{"lockfileVersion":3}')
    (product / "Cargo.toml").write_text(
        '[package]\nname = "demo"\nversion = "0.1.0"\nrust-version = "1.81"\n'
    )
    (product / ".nvmrc").write_text("22.4.0\n")
    (product / ".env.example").write_text("API_KEY=secret-value-must-not-leak\n")
    (product / "src.ts").write_text('import helper from "external-lib";\nconsole.log(process.env.API_KEY, helper);\n')
    (product / "compose.yaml").write_text("services:\n  database:\n    image: postgres\n")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": "0001-demo", "status": "validation_passed_external_testing_pending"}))

    metadata = build_testing_handoff(_config(), str(run_dir))
    write_testing_handoff(_config(), str(run_dir))
    encoded = (run_dir / "testing-handoff.json").read_text() + (run_dir / "testing-handoff.md").read_text()

    assert metadata["run_id"] == "0001-demo"
    assert metadata["external_testing"]["status"] == "pending"
    assert "package-lock.json" in metadata["lockfiles"][0]
    assert {"runtime": "Rust", "version": "1.81", "source": str(product / "Cargo.toml")} in metadata["runtime_version_requirements"]
    assert any(
        item["name"] == "left-pad" and item["version_specification"] == "^1.3.0"
        for item in metadata["declared_dependencies"]
    )
    assert "npm run test" in [entry["command"] for entry in metadata["test_and_build_commands"]]
    assert metadata["required_environment_variable_names"] == ["API_KEY"]
    assert metadata["declared_services"] == ["database"]
    assert "external-lib" in metadata["unresolved_or_undeclared_dependencies"]
    assert "secret-value-must-not-leak" not in encoded
    assert "API-SUPER-SECRET" not in encoded
    assert "HEADER-SECRET" not in encoded
    assert "isolated copy" in (run_dir / "testing-handoff.md").read_text()


def test_testing_handoff_omits_unsafe_and_unrecognized_runtime_version_contents(tmp_path):
    run_dir = tmp_path / "run"
    product = run_dir / "product"
    product.mkdir(parents=True)
    secret = tmp_path / "external-secret-version"
    secret.write_text("external-version-secret-value\n")
    (product / ".nvmrc").symlink_to(secret)
    invalid_value = "runtime-secret-value-should-not-appear"
    (product / ".python-version").write_text(invalid_value)
    (product / ".node-version").write_text("9" * 300)

    metadata = build_testing_handoff(_config(), str(run_dir))
    write_testing_handoff(_config(), str(run_dir))
    encoded = (run_dir / "testing-handoff.json").read_text() + (run_dir / "testing-handoff.md").read_text()

    assert metadata["runtime_version_requirements"] == []
    assert any("path is unsafe" in warning for warning in metadata["handoff_warnings"])
    assert any("recognized runtime version" in warning for warning in metadata["handoff_warnings"])
    assert any("size limit" in warning for warning in metadata["handoff_warnings"])
    assert "external-version-secret-value" not in encoded
    assert invalid_value not in encoded
    assert "9" * 80 not in encoded


def test_testing_handoff_scans_runs_storage_symlink_without_following_project_links(tmp_path):
    real_runs = tmp_path / "canonical-storage" / "runs"
    run_dir = real_runs / "0001-demo"
    product = run_dir / "product"
    tests = run_dir / "tests"
    product.mkdir(parents=True)
    tests.mkdir()
    (product / "package.json").write_text(json.dumps({
        "dependencies": {"declared-package": "^1.2.0"},
    }))
    (product / "app.py").write_text("import declared_package\n")
    (product / ".nvmrc").write_text("22.4.0\n")
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "0001-demo", "status": "validation_passed_external_testing_pending"})
    )

    external = tmp_path / "external-secret-project"
    external.mkdir()
    (external / "package.json").write_text(json.dumps({
        "dependencies": {"leaked-package": "RAW-SUPER-SECRET-VALUE"},
    }))
    (external / "secret.py").write_text('API_KEY="RAW-SOURCE-SECRET"\n')
    (product / "external-link").symlink_to(external, target_is_directory=True)

    storage_alias = tmp_path / "storage-alias"
    storage_alias.symlink_to(real_runs.parent, target_is_directory=True)
    alias_run_dir = storage_alias / "runs" / "0001-demo"
    config = _config()
    config["pipeline"]["runs_dir"] = str(storage_alias / "runs")

    metadata = build_testing_handoff(config, str(alias_run_dir))
    write_testing_handoff(config, str(alias_run_dir))
    encoded = (
        (run_dir / "testing-handoff.json").read_text()
        + (run_dir / "testing-handoff.md").read_text()
    )

    assert any(
        dependency["name"] == "declared-package"
        for dependency in metadata["declared_dependencies"]
    )
    assert "product/package.json" in metadata["changed_files"]["added"]
    assert "product/app.py" in metadata["changed_files"]["added"]
    assert any(item["runtime"] == "Node.js" and item["version"] == "22.4.0"
               for item in metadata["runtime_version_requirements"])
    assert any("Symbolic-link entries were skipped" in warning
               for warning in metadata["handoff_warnings"])
    assert "leaked-package" not in encoded
    assert "RAW-SUPER-SECRET-VALUE" not in encoded
    assert "RAW-SOURCE-SECRET" not in encoded


def test_testing_handoff_rejects_project_root_symlink_and_warns(tmp_path):
    run_dir = tmp_path / "runs" / "0001-demo"
    run_dir.mkdir(parents=True)
    (run_dir / "tests").mkdir()
    external = tmp_path / "external-project"
    external.mkdir()
    (external / "package.json").write_text(json.dumps({
        "dependencies": {"leaked-package": "ROOT-SECRET-VALUE"},
    }))
    (run_dir / "product").symlink_to(external, target_is_directory=True)
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": "0001-demo"}))
    config = _config()
    config["pipeline"]["runs_dir"] = str(tmp_path / "runs")

    metadata = build_testing_handoff(config, str(run_dir))
    write_testing_handoff(config, str(run_dir))
    encoded = (
        (run_dir / "testing-handoff.json").read_text()
        + (run_dir / "testing-handoff.md").read_text()
    )

    assert metadata["declared_dependencies"] == []
    assert any("output path is unavailable or unsafe" in warning
               for warning in metadata["handoff_warnings"])
    assert "ROOT-SECRET-VALUE" not in encoded
