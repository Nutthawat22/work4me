"""Opt-in Docker integration fixtures; run with AGENTS_DOCKER_SMOKE=1."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from testing_agent.runner import run_external_tests


pytestmark = pytest.mark.skipif(
    os.environ.get("AGENTS_DOCKER_SMOKE") != "1",
    reason="set AGENTS_DOCKER_SMOKE=1 to run disposable Docker integration fixtures",
)


def _docker_available():
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            check=False,
            timeout=10,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.mark.parametrize("runtime", ["python", "node"])
@pytest.mark.parametrize("should_pass", [True, False], ids=["pass", "fail"])
def test_real_docker_runtime_smoke_preserves_input_source(tmp_path, runtime, should_pass):
    if not _docker_available():
        pytest.skip("Docker CLI or daemon is unavailable")
    run_dir = tmp_path / "runs" / f"0001-{runtime}-{'pass' if should_pass else 'fail'}"
    product = run_dir / "product"
    tests = run_dir / "tests"
    product.mkdir(parents=True)
    tests.mkdir()
    if runtime == "python":
        (product / ".python-version").write_text("3.12.11\n")
        (tests / "test_smoke.py").write_text(
            "import unittest\n"
            "class SmokeTests(unittest.TestCase):\n"
            f"    def test_result(self): self.assertEqual(1, {1 if should_pass else 2})\n"
        )
        dependencies = []
    else:
        (product / ".nvmrc").write_text("22.16.0\n")
        (product / "package.json").write_text(json.dumps({
            "name": "docker-smoke",
            "version": "0.1.0",
            "scripts": {"test": "node --test"},
        }))
        (product / "package-lock.json").write_text(json.dumps({
            "name": "docker-smoke",
            "version": "0.1.0",
            "lockfileVersion": 3,
            "requires": True,
            "packages": {"": {"name": "docker-smoke", "version": "0.1.0"}},
        }))
        (tests / "smoke.test.js").write_text(
            "const test = require('node:test');\n"
            "const assert = require('node:assert/strict');\n"
            f"test('smoke', () => assert.equal(1, {1 if should_pass else 2}));\n"
        )
        dependencies = []
    source_hashes = {
        path.relative_to(run_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (*product.rglob("*"), *tests.rglob("*"))
        if path.is_file()
    }
    run_id = run_dir.name
    manifest = {"run_id": run_id, "status": "validation_passed_external_testing_pending"}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    handoff = {
        "schema_version": 1,
        "run_id": run_id,
        "pipeline_status": manifest["status"],
        "project_mode": "new_project",
        "paths": {
            "run_dir": str(run_dir),
            "project_root": str(product),
            "product_dir": str(product),
            "tests_dir": str(tests),
        },
        "declared_dependencies": dependencies,
        "test_and_build_commands": [],
        "external_testing": {"status": "pending", "contract_version": 1},
    }
    handoff_path = run_dir / "testing-handoff.json"
    handoff_path.write_text(json.dumps(handoff))
    source_hashes["manifest.json"] = hashlib.sha256((run_dir / "manifest.json").read_bytes()).hexdigest()
    source_hashes["testing-handoff.json"] = hashlib.sha256(handoff_path.read_bytes()).hexdigest()

    result = run_external_tests(handoff_path)

    assert result["outcome"] == ("passed" if should_pass else "tests_failed")
    assert result["run_id"] == run_id
    assert result["source_snapshot"]["sha256"]
    assert result["environment_setup"]["network_policy"].endswith("disconnected before build/test commands")
    assert result["container_cleanup"]["status"] == "removed"
    for relative, digest in source_hashes.items():
        assert hashlib.sha256((run_dir / relative).read_bytes()).hexdigest() == digest
