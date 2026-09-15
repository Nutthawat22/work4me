"""
tests/pipeline/test_test_runner_cwd_install.py

Coverage for two test_runner.py fixes:

1. jest/vitest test subprocess must run with cwd=output_dir (the run's
   product dir where package.json/node_modules live), not PROJECT_ROOT
   (the repo root, which has neither). pytest/junit keeps cwd=PROJECT_ROOT
   unchanged.
2. Before running jest/vitest, an install step (npm install / bun
   install, mirroring pipeline/instructions.py's uses_bun heuristic) must
   run in output_dir if package.json exists but node_modules doesn't —
   and any install failure must degrade to a TestResult failure, not an
   exception.

No live npm/bun/jest processes are invoked — subprocess.run is
monkeypatched throughout, following the pattern in
test_test_runner_vitest.py's TestGracefulDegradationMissingBinary.
"""

import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline.test_runner import PROJECT_ROOT, TestRunner


JEST_LANG_CONFIG = {
    "test_command": ["npx", "jest", "--ci", "--json", "--outputFile", "{result_path}"],
    "result_format": "jest_json",
}

VITEST_LANG_CONFIG = {
    "test_command": ["bunx", "vitest", "run", "--reporter=json", "--outputFile={result_path}"],
    "result_format": "vitest_json",
}

JEST_JSON_EMPTY_PASS = {
    "numTotalTests": 0,
    "numFailedTests": 0,
    "testResults": [],
}


@pytest.fixture
def runner():
    return TestRunner()


def _fake_run_recorder(calls: list, write_result_for: list[str] | None = None):
    """
    Build a fake subprocess.run replacement that records every call as
    (command, cwd) and, for any command whose first token is in
    write_result_for (e.g. "npx"/"bunx" — the actual test-runner
    invocation, not the install step), writes a minimal passing
    jest/vitest-shaped JSON result to the outputFile path so
    _run_language_suite's parse step succeeds cleanly.
    """
    write_result_for = write_result_for or []

    def fake_run(command, cwd=None, env=None, capture_output=None, text=None, timeout=None):
        calls.append((list(command), cwd))
        if command and command[0] in write_result_for:
            result_path = command[-1]
            if result_path.startswith("--outputFile="):
                result_path = result_path.split("=", 1)[1]
            with open(result_path, "w") as f:
                json.dump(JEST_JSON_EMPTY_PASS, f)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    return fake_run


class TestJsTsSubprocessCwd:
    def test_jest_runs_with_cwd_output_dir_not_project_root(
        self, runner, tmp_path, monkeypatch
    ):
        output_dir = tmp_path / "product"
        output_dir.mkdir()
        (output_dir / "package.json").write_text("{}")
        (output_dir / "node_modules").mkdir()

        calls: list = []
        monkeypatch.setattr(
            subprocess, "run", _fake_run_recorder(calls, write_result_for=["npx"])
        )

        runner._run_language_suite(
            "javascript", JEST_LANG_CONFIG, str(tmp_path / "tests"), str(output_dir)
        )

        test_calls = [c for c in calls if c[0][0] == "npx"]
        assert len(test_calls) == 1
        _, cwd = test_calls[0]
        assert cwd == str(output_dir)
        assert cwd != PROJECT_ROOT

    def test_vitest_runs_with_cwd_output_dir_not_project_root(
        self, runner, tmp_path, monkeypatch
    ):
        output_dir = tmp_path / "product"
        output_dir.mkdir()
        (output_dir / "package.json").write_text("{}")
        (output_dir / "node_modules").mkdir()

        calls: list = []
        monkeypatch.setattr(
            subprocess, "run", _fake_run_recorder(calls, write_result_for=["bunx"])
        )

        runner._run_language_suite(
            "typescript", VITEST_LANG_CONFIG, str(tmp_path / "tests"), str(output_dir)
        )

        test_calls = [c for c in calls if c[0][0] == "bunx"]
        assert len(test_calls) == 1
        _, cwd = test_calls[0]
        assert cwd == str(output_dir)
        assert cwd != PROJECT_ROOT


class TestInstallStepTriggered:
    def test_install_runs_when_node_modules_missing(self, runner, tmp_path, monkeypatch):
        output_dir = tmp_path / "product"
        output_dir.mkdir()
        (output_dir / "package.json").write_text("{}")
        # no node_modules dir created

        calls: list = []
        monkeypatch.setattr(
            subprocess, "run", _fake_run_recorder(calls, write_result_for=["npx"])
        )

        runner._run_language_suite(
            "javascript", JEST_LANG_CONFIG, str(tmp_path / "tests"), str(output_dir)
        )

        install_calls = [c for c in calls if c[0][:2] == ["npm", "install"]]
        assert len(install_calls) == 1
        _, cwd = install_calls[0]
        assert cwd == str(output_dir)

        # install call must happen before the test-run call
        commands = [c[0] for c in calls]
        assert commands.index(["npm", "install"]) < next(
            i for i, c in enumerate(commands) if c[0] == "npx"
        )

    def test_install_uses_bun_when_test_command_uses_bun(
        self, runner, tmp_path, monkeypatch
    ):
        output_dir = tmp_path / "product"
        output_dir.mkdir()
        (output_dir / "package.json").write_text("{}")

        calls: list = []
        monkeypatch.setattr(
            subprocess, "run", _fake_run_recorder(calls, write_result_for=["bunx"])
        )

        runner._run_language_suite(
            "typescript", VITEST_LANG_CONFIG, str(tmp_path / "tests"), str(output_dir)
        )

        install_calls = [c for c in calls if c[0][:2] == ["bun", "install"]]
        assert len(install_calls) == 1


class TestInstallStepSkipped:
    def test_install_skipped_when_node_modules_present(
        self, runner, tmp_path, monkeypatch
    ):
        output_dir = tmp_path / "product"
        output_dir.mkdir()
        (output_dir / "package.json").write_text("{}")
        (output_dir / "node_modules").mkdir()

        calls: list = []
        monkeypatch.setattr(
            subprocess, "run", _fake_run_recorder(calls, write_result_for=["npx"])
        )

        runner._run_language_suite(
            "javascript", JEST_LANG_CONFIG, str(tmp_path / "tests"), str(output_dir)
        )

        install_calls = [
            c for c in calls if c[0][:2] in (["npm", "install"], ["bun", "install"])
        ]
        assert install_calls == []

    def test_install_skipped_when_no_package_json(self, runner, tmp_path, monkeypatch):
        output_dir = tmp_path / "product"
        output_dir.mkdir()
        # no package.json at all

        calls: list = []
        monkeypatch.setattr(
            subprocess, "run", _fake_run_recorder(calls, write_result_for=["npx"])
        )

        runner._run_language_suite(
            "javascript", JEST_LANG_CONFIG, str(tmp_path / "tests"), str(output_dir)
        )

        install_calls = [
            c for c in calls if c[0][:2] in (["npm", "install"], ["bun", "install"])
        ]
        assert install_calls == []


class TestInstallFailureIsGraceful:
    def test_install_nonzero_exit_returns_failure_not_exception(
        self, runner, tmp_path, monkeypatch
    ):
        output_dir = tmp_path / "product"
        output_dir.mkdir()
        (output_dir / "package.json").write_text("{}")

        def fake_run(command, cwd=None, env=None, capture_output=None, text=None, timeout=None):
            if command[:2] == ["npm", "install"]:
                return subprocess.CompletedProcess(
                    command, 1, stdout="", stderr="npm ERR! something broke"
                )
            raise AssertionError("test command should not run after install failure")

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = runner._run_language_suite(
            "javascript", JEST_LANG_CONFIG, str(tmp_path / "tests"), str(output_dir)
        )

        assert result.passed is False
        assert len(result.failures) == 1
        assert "install failed" in result.failures[0].test_name
        assert "npm ERR!" in result.failures[0].error_output

    def test_install_missing_binary_returns_failure_not_exception(
        self, runner, tmp_path, monkeypatch
    ):
        output_dir = tmp_path / "product"
        output_dir.mkdir()
        (output_dir / "package.json").write_text("{}")

        def fake_run(command, cwd=None, env=None, capture_output=None, text=None, timeout=None):
            if command[:2] == ["npm", "install"]:
                raise FileNotFoundError("npm: command not found")
            raise AssertionError("test command should not run after install failure")

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = runner._run_language_suite(
            "javascript", JEST_LANG_CONFIG, str(tmp_path / "tests"), str(output_dir)
        )

        assert result.passed is False
        assert len(result.failures) == 1
        assert "install failed" in result.failures[0].test_name

    def test_install_timeout_returns_failure_not_exception(
        self, runner, tmp_path, monkeypatch
    ):
        output_dir = tmp_path / "product"
        output_dir.mkdir()
        (output_dir / "package.json").write_text("{}")

        def fake_run(command, cwd=None, env=None, capture_output=None, text=None, timeout=None):
            if command[:2] == ["npm", "install"]:
                raise subprocess.TimeoutExpired(command, timeout)
            raise AssertionError("test command should not run after install failure")

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = runner._run_language_suite(
            "javascript", JEST_LANG_CONFIG, str(tmp_path / "tests"), str(output_dir)
        )

        assert result.passed is False
        assert len(result.failures) == 1
        assert "install failed" in result.failures[0].test_name
