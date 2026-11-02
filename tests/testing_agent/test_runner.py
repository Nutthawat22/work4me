import hashlib
import json
import sys
from pathlib import Path
from threading import Event

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from testing_agent.contract import HandoffError, validate_handoff
from testing_agent.runner import AgentSettings, DockerCli, DockerFailure, run_external_tests


@pytest.fixture(autouse=True)
def _make_readonly_test_artifacts_removable(tmp_path):
    yield
    for path in sorted(tmp_path.rglob("*"), key=lambda item: len(item.parts)):
        if path.is_dir():
            path.chmod(0o700)
        elif path.exists():
            path.chmod(0o600)


class FakeDocker:
    def __init__(
        self,
        *,
        outcomes=None,
        test_output="1 passed in 0.01s",
        cancel_event=None,
        npm_version="10.9.2",
        cleanup_error=None,
    ):
        self.outcomes = outcomes or {}
        self.test_output = test_output
        self.cancel_event = cancel_event
        self.npm_version = npm_version
        self.cleanup_error = cleanup_error
        self.commands = []
        self.started_images = []
        self.removed = []
        self.disconnected = []
        self.copied = []

    def check(self, timeout, cancel_event):
        if self.outcomes.get("docker") == "unavailable":
            raise DockerFailure("Docker daemon unavailable")
        return "29.6.2"

    def start(self, image, settings, cancel_event):
        self.started_images.append(image)
        return "container-test-id"

    def copy_into(self, container_id, source, cancel_event):
        self.copied.append((container_id, source))

    def disconnect_network(self, container_id, cancel_event):
        self.disconnected.append(container_id)

    def execute(self, container_id, argv, workdir, timeout, cancel_event):
        self.commands.append((argv, workdir, timeout))
        key = "setup" if argv == ("npm", "--version") or argv[:2] == ("npm", "ci") else "test" if "pytest" in argv or "unittest" in argv or "npm" in argv else "setup"
        status = self.outcomes.get(key, "passed")
        output = self.npm_version if argv == ("npm", "--version") else self.test_output if key == "test" else "dependency installation complete"
        if status == "failed" and key == "test" and self.test_output == "1 passed in 0.01s":
            output = "1 failed in 0.01s"
        if status == "timed_out":
            return status, None, 4.0, "command timed out"
        if status == "cancelled":
            if self.cancel_event is not None:
                self.cancel_event.set()
            return status, None, 0.1, "cancelled"
        return status, 0 if status == "passed" else 1, 0.1, output

    def remove(self, container_id):
        self.removed.append(container_id)
        return self.cleanup_error


def _new_handoff(tmp_path, *, mode="new_project", node=False):
    run_dir = tmp_path / "runs" / "0001-demo"
    product = run_dir / "product"
    tests = run_dir / "tests"
    product.mkdir(parents=True)
    tests.mkdir()
    if node:
        (product / ".nvmrc").write_text("22.16.0\n")
        (product / "package.json").write_text(json.dumps({
            "name": "demo",
            "packageManager": "npm@10.9.2",
            "scripts": {"test": "vitest run", "build": "vite build"},
            "devDependencies": {"vitest": "2.1.8"},
        }))
        (product / "package-lock.json").write_text(json.dumps({"name": "demo", "lockfileVersion": 3, "packages": {}}))
        (tests / "app.test.ts").write_text("import { test } from 'vitest'; test('works', () => {});\n")
        dependencies = [{"name": "vitest", "version_specification": "2.1.8", "group": "devDependencies", "source": "package.json"}]
    else:
        (product / ".python-version").write_text("3.12.11\n")
        (product / "requirements.txt").write_text("pytest==9.1.1\n")
        (product / "app.py").write_text("def value():\n    return 42\n")
        (tests / "test_app.py").write_text("def test_value():\n    assert True\n")
        dependencies = [{"name": "pytest", "version_specification": "==9.1.1", "group": "requirements.txt", "source": "requirements.txt"}]
    project_root = product
    if mode == "existing_project":
        project = tmp_path / "existing-project"
        project.mkdir()
        (project / "requirements.txt").write_text("pytest==9.1.1\n")
        (project / "app.py").write_text("def value():\n    return 42\n")
        (project / "test_app.py").write_text("def test_value():\n    assert True\n")
        changes = {"project_path": str(project), "added": [], "modified": ["app.py"], "deleted": [], "partial_edits_possible": False}
        (run_dir / "changes.json").write_text(json.dumps(changes))
        project_root = project
        product_path = tests_path = project
    else:
        product_path = product
        tests_path = tests
    run_id = "0001-demo"
    manifest = {"run_id": run_id, "status": "validation_passed_external_testing_pending"}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    handoff = {
        "schema_version": 1,
        "run_id": run_id,
        "pipeline_status": manifest["status"],
        "project_mode": mode,
        "paths": {
            "run_dir": str(run_dir),
            "project_root": str(project_root),
            "product_dir": str(product_path),
            "tests_dir": str(tests_path),
            "changes_json": str(run_dir / "changes.json") if mode == "existing_project" else None,
        },
        "declared_dependencies": dependencies,
        "test_and_build_commands": [{"command": "sh -c hostile-host-command", "kind": "pipeline-configured-test"}],
        "required_environment_variable_names": ["API_TOKEN"],
        "external_testing": {"status": "pending", "contract_version": 1},
    }
    handoff_path = run_dir / "testing-handoff.json"
    handoff_path.write_text(json.dumps(handoff))
    return run_dir, product_path, tests_path, handoff_path, project_root, manifest


def _settings():
    return AgentSettings()


def test_python_handoff_runs_only_in_mocked_container_and_preserves_original(tmp_path):
    run_dir, product, tests, handoff, _, manifest_before = _new_handoff(tmp_path)
    original = {
        path.relative_to(run_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (product / "requirements.txt", product / "app.py", tests / "test_app.py", handoff, run_dir / "manifest.json")
    }
    fake = FakeDocker()

    result = run_external_tests(handoff, settings=_settings(), docker=fake)

    assert result["outcome"] == "passed"
    assert result["run_id"] == "0001-demo"
    assert result["test_evidence"]["test_count"] == 1
    assert result["test_evidence"]["status"] == "verified"
    assert fake.started_images == [_settings().python_image]
    assert fake.disconnected == ["container-test-id"]
    assert fake.removed == ["container-test-id"]
    assert all("sh -c hostile-host-command" not in " ".join(argv) for argv, _, _ in fake.commands)
    for relative, digest in original.items():
        assert hashlib.sha256((run_dir / relative).read_bytes()).hexdigest() == digest
    assert json.loads((run_dir / "manifest.json").read_text()) == manifest_before
    result_path = Path(result["result_path"])
    saved = json.loads(result_path.read_text())
    assert saved["source_snapshot"]["sha256"] == result["source_snapshot"]["sha256"]
    assert (result_path.parent / "report.md").is_file()
    assert list((result_path.parent / "logs").glob("*.log"))


@pytest.mark.parametrize(
    ("docker_outcomes", "test_output", "expected"),
    [
        ({"test": "failed"}, "1 failed in 0.01s", "tests_failed"),
        ({"setup": "failed"}, "", "setup_failed"),
        ({"test": "timed_out"}, "", "tests_failed"),
    ],
)
def test_command_failures_and_timeouts_are_evidence_not_passes(
    tmp_path, docker_outcomes, test_output, expected
):
    _, _, _, handoff, _, _ = _new_handoff(tmp_path)
    fake = FakeDocker(outcomes=docker_outcomes, test_output=test_output)

    result = run_external_tests(handoff, settings=_settings(), docker=fake)

    assert result["outcome"] == expected
    assert result["commands"]
    assert fake.removed == ["container-test-id"]


def test_cancel_stops_test_and_cleans_container(tmp_path):
    _, _, _, handoff, _, _ = _new_handoff(tmp_path)
    cancelled = Event()
    fake = FakeDocker(outcomes={"test": "cancelled"}, cancel_event=cancelled)

    result = run_external_tests(handoff, settings=_settings(), docker=fake, cancel_event=cancelled)

    assert result["outcome"] == "cancelled"
    assert fake.removed == ["container-test-id"]


def test_cleanup_failure_overrides_pass_and_is_reported(tmp_path):
    _, _, _, handoff, _, _ = _new_handoff(tmp_path)
    fake = FakeDocker(cleanup_error="Docker daemon did not confirm removal")

    result = run_external_tests(handoff, settings=_settings(), docker=fake)

    assert result["outcome"] == "setup_failed"
    assert result["container_cleanup"]["status"] == "failed"
    assert result["test_evidence"]["status"] == "verified"


def test_no_tests_does_not_start_docker_and_keeps_static_state_pending(tmp_path):
    run_dir, product, _, handoff, _, manifest_before = _new_handoff(tmp_path)
    (product / "requirements.txt").unlink()
    (product / "app.py").write_text("value = 1\n")
    (run_dir / "tests" / "test_app.py").unlink()
    fake = FakeDocker()

    result = run_external_tests(handoff, settings=_settings(), docker=fake)

    assert result["outcome"] == "no_tests"
    assert fake.started_images == []
    assert result["environment_setup"]["status"] == "not_attempted"
    assert json.loads((run_dir / "manifest.json").read_text()) == manifest_before


def test_node_tests_are_discovered_in_separate_handoff_tests_root(tmp_path):
    _, _, _, handoff, _, _ = _new_handoff(tmp_path, node=True)
    fake = FakeDocker(test_output="Tests  2 passed (2)")

    result = run_external_tests(handoff, settings=_settings(), docker=fake)

    assert result["outcome"] == "passed"
    assert result["runtime"] == "node"
    assert result["test_evidence"]["test_count"] == 2
    assert fake.commands[-1][0] == ("npm", "exec", "--no", "--", "vitest", "run")


def test_declared_npm_version_must_match_the_configured_image(tmp_path):
    _, _, _, handoff, _, _ = _new_handoff(tmp_path, node=True)
    fake = FakeDocker(npm_version="9.0.0")

    result = run_external_tests(handoff, settings=_settings(), docker=fake)

    assert result["outcome"] == "setup_failed"
    assert "Expected npm 10.9.2" in result["environment_setup"]["details"]
    assert fake.commands[-1][0] == ("npm", "--version")


def test_node_test_script_shell_composition_is_rejected_before_docker(tmp_path):
    _, product, _, handoff, _, _ = _new_handoff(tmp_path, node=True)
    package = json.loads((product / "package.json").read_text())
    package["scripts"]["test"] = "vitest run && echo 'Tests 999 passed'"
    (product / "package.json").write_text(json.dumps(package))
    fake = FakeDocker()

    result = run_external_tests(handoff, settings=_settings(), docker=fake)

    assert result["outcome"] == "setup_failed"
    assert "shell wrappers are not executed" in result["environment_setup"]["details"]
    assert fake.started_images == []


def test_incompatible_declared_runtime_is_a_setup_failure(tmp_path):
    _, product, _, handoff, _, _ = _new_handoff(tmp_path)
    (product / ".python-version").write_text("3.11.9\n")
    fake = FakeDocker()

    result = run_external_tests(handoff, settings=_settings(), docker=fake)

    assert result["outcome"] == "setup_failed"
    assert "does not match configured image" in result["environment_setup"]["details"]
    assert fake.started_images == []


def test_missing_test_runner_is_reported_as_setup_failure(tmp_path):
    _, _, _, handoff, _, _ = _new_handoff(tmp_path)
    fake = FakeDocker(test_output="/usr/local/bin/python: No module named pytest")
    fake.outcomes["test"] = "failed"

    result = run_external_tests(handoff, settings=_settings(), docker=fake)

    assert result["outcome"] == "setup_failed"
    assert result["test_evidence"]["status"] == "runner_unavailable"


def test_docker_unavailable_is_actionable_setup_failure(tmp_path):
    _, _, _, handoff, _, _ = _new_handoff(tmp_path)
    fake = FakeDocker(outcomes={"docker": "unavailable"})

    result = run_external_tests(handoff, settings=_settings(), docker=fake)

    assert result["outcome"] == "setup_failed"
    assert "Docker daemon unavailable" in result["environment_setup"]["details"]
    assert fake.started_images == []


def test_existing_project_requires_explicit_matching_authorization_and_does_not_edit_source(tmp_path):
    _, _, _, handoff, project, _ = _new_handoff(tmp_path, mode="existing_project")
    original = {path.name: path.read_bytes() for path in project.iterdir()}
    with pytest.raises(HandoffError, match="explicit"):
        validate_handoff(handoff)

    fake = FakeDocker()
    result = run_external_tests(handoff, authorized_project_root=project, docker=fake)

    assert result["outcome"] == "passed"
    assert {path.name: path.read_bytes() for path in project.iterdir()} == original


def test_handoff_rejects_outside_paths_and_stale_manifest(tmp_path):
    run_dir, product, _, handoff, _, _ = _new_handoff(tmp_path)
    data = json.loads(handoff.read_text())
    outside = tmp_path / "outside"
    outside.mkdir()
    data["paths"]["product_dir"] = str(outside)
    data["paths"]["project_root"] = str(outside)
    handoff.write_text(json.dumps(data))
    with pytest.raises(HandoffError, match="outside"):
        validate_handoff(handoff)

    _, _, _, stale_handoff, _, _ = _new_handoff(tmp_path / "other")
    stale = json.loads(stale_handoff.read_text())
    stale["pipeline_status"] = "validation_failed"
    stale_handoff.write_text(json.dumps(stale))
    with pytest.raises(HandoffError, match="static validation"):
        validate_handoff(stale_handoff)


def test_handoff_rejects_symlinked_project_root(tmp_path):
    run_dir, product, _, handoff, _, _ = _new_handoff(tmp_path)
    real_product = run_dir / "real-product"
    product.rename(real_product)
    product.symlink_to(real_product, target_is_directory=True)

    with pytest.raises(HandoffError, match="non-symlink directory"):
        validate_handoff(handoff)


def test_snapshot_omits_symlinks_and_sensitive_files_and_redacts_logs(tmp_path):
    _, product, _, handoff, _, _ = _new_handoff(tmp_path)
    (product / ".env").write_text("API_TOKEN=must-not-copy\n")
    (product / ".npmrc").write_text("//registry/:_authToken=must-not-copy\n")
    external_secret = tmp_path / "linked-secret"
    external_secret.write_text("linked-secret")
    (product / "linked.txt").symlink_to(external_secret)
    fake = FakeDocker(test_output="token=must-not-leak\n1 passed in 0.01s")

    result = run_external_tests(handoff, settings=_settings(), docker=fake)

    snapshot = Path(result["result_path"]).parent / "source-snapshot" / "product"
    assert not (snapshot / ".env").exists()
    assert not (snapshot / ".npmrc").exists()
    assert not (snapshot / "linked.txt").exists()
    assert "must-not-leak" not in "".join(path.read_text() for path in (Path(result["result_path"]).parent / "logs").glob("*.log"))
    assert result["source_snapshot"]["omitted_entries"] >= 3


def test_container_configuration_has_no_host_mounts_privilege_or_host_network(monkeypatch):
    docker = DockerCli(_settings())
    captured = []

    def checked(args, cancel_event, timeout=None):
        captured.extend(args)
        return "0123456789abcdef\n"

    monkeypatch.setattr(docker, "_checked", checked)
    docker.start(_settings().python_image, _settings(), Event())

    assert "--cap-drop=ALL" in captured
    assert "--network" in captured
    assert "host" not in captured
    assert "--privileged" not in captured
    assert "--mount" not in captured
    assert not any("docker.sock" in arg for arg in captured)
