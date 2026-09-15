"""
tests/pipeline/test_test_runner_tests_dir_arg.py

Coverage for the jest/vitest "missing tests_dir path argument" bug fix:

Jest and vitest's test_command (unlike pytest's, which explicitly passes
"{tests_dir}" as a positional arg) previously had no way to point at
tests_dir. Combined with the earlier cwd=output_dir fix
(test_test_runner_cwd_install.py), running jest/vitest from output_dir
with no tests_dir argument meant they defaulted to searching cwd (the
product dir) and never found any tests, since real test files live in
the sibling tests_dir.

Fix: config.example.json / config.json's jest test_command now includes
"--roots" "{tests_dir}", and vitest's test_command includes a positional
"{tests_dir}" filter argument appended after "run". This module verifies:

1. The real config.example.json template actually has these args (so
   the fix is committed to the template, not just the user's live
   config).
2. test_runner.py's format_context already exposes "tests_dir" to the
   jest_json/vitest_json branches (it does, unconditionally, before the
   result_format if/elif — this is a non-regression check, not a new
   requirement).
3. The formatted command (test_command tokens with format_context
   substituted) actually contains the resolved tests_dir path, using the
   same subprocess.run-monkeypatch style as test_test_runner_cwd_install.py.
"""

import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline.test_runner import TestRunner


def load_example_config() -> dict:
    config_path = os.path.join(REPO_ROOT, "config.example.json")
    with open(config_path, "r") as f:
        return json.load(f)


JEST_JSON_EMPTY_PASS = {
    "numTotalTests": 0,
    "numFailedTests": 0,
    "testResults": [],
}


@pytest.fixture
def runner():
    return TestRunner()


def _fake_run_recorder(calls: list, write_result_for: list[str] | None = None):
    write_result_for = write_result_for or []

    def fake_run(command, cwd=None, env=None, capture_output=None, text=None, timeout=None):
        calls.append((list(command), cwd))
        if command and command[0] in write_result_for:
            result_path = None
            for i, token in enumerate(command):
                if token == "--outputFile" and i + 1 < len(command):
                    result_path = command[i + 1]
                    break
                if token.startswith("--outputFile="):
                    result_path = token.split("=", 1)[1]
                    break
            if result_path is not None:
                with open(result_path, "w") as f:
                    json.dump(JEST_JSON_EMPTY_PASS, f)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    return fake_run


class TestConfigExampleHasTestsDirArg:
    def test_jest_test_command_includes_roots_tests_dir(self):
        config = load_example_config()
        jest_command = config["languages"]["javascript"]["test_command"]
        assert "--roots" in jest_command
        idx = jest_command.index("--roots")
        assert jest_command[idx + 1] == "{tests_dir}"

    def test_vitest_test_command_includes_positional_tests_dir(self):
        config = load_example_config()
        vitest_command = config["languages"]["typescript"]["test_command"]
        assert "{tests_dir}" in vitest_command
        # Must come after "run" (the subcommand) as a filter arg.
        assert vitest_command.index("{tests_dir}") > vitest_command.index("run")


class TestFormatContextIncludesTestsDirForJsTs:
    def test_jest_command_resolves_with_real_tests_dir_path(
        self, runner, tmp_path, monkeypatch
    ):
        config = load_example_config()
        jest_lang_config = config["languages"]["javascript"]

        output_dir = tmp_path / "product"
        output_dir.mkdir()
        (output_dir / "package.json").write_text("{}")
        (output_dir / "node_modules").mkdir()
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        calls: list = []
        monkeypatch.setattr(
            subprocess, "run", _fake_run_recorder(calls, write_result_for=["npx"])
        )

        runner._run_language_suite(
            "javascript", jest_lang_config, str(tests_dir), str(output_dir)
        )

        test_calls = [c for c in calls if c[0][0] == "npx"]
        assert len(test_calls) == 1
        command, _ = test_calls[0]
        assert "--roots" in command
        assert command[command.index("--roots") + 1] == str(tests_dir)

    def test_vitest_command_resolves_with_real_tests_dir_path(
        self, runner, tmp_path, monkeypatch
    ):
        config = load_example_config()
        vitest_lang_config = config["languages"]["typescript"]

        output_dir = tmp_path / "product"
        output_dir.mkdir()
        (output_dir / "package.json").write_text("{}")
        (output_dir / "node_modules").mkdir()
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        calls: list = []
        monkeypatch.setattr(
            subprocess, "run", _fake_run_recorder(calls, write_result_for=["bunx"])
        )

        runner._run_language_suite(
            "typescript", vitest_lang_config, str(tests_dir), str(output_dir)
        )

        test_calls = [c for c in calls if c[0][0] == "bunx"]
        assert len(test_calls) == 1
        command, _ = test_calls[0]
        assert str(tests_dir) in command
