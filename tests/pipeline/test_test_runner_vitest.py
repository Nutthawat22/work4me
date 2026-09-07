"""
tests/pipeline/test_test_runner_vitest.py

Fixture-based tests for TestRunner._parse_vitest_json (HO-4: TypeScript/
Vitest test support) plus graceful-degradation coverage when the
bunx/vitest binary is missing. No live pipeline run, no LLM calls.
"""

import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline.test_runner import TestRunner


VITEST_JSON_ALL_PASSING = {
    "numTotalTestSuites": 1,
    "numPassedTestSuites": 1,
    "numFailedTestSuites": 0,
    "numTotalTests": 2,
    "numPassedTests": 2,
    "numFailedTests": 0,
    "success": True,
    "testResults": [
        {
            "name": "tests/example.test.ts",
            "status": "passed",
            "assertionResults": [
                {
                    "ancestorTitles": ["example suite"],
                    "fullName": "example suite adds numbers",
                    "title": "adds numbers",
                    "status": "passed",
                    "failureMessages": [],
                },
                {
                    "ancestorTitles": ["example suite"],
                    "fullName": "example suite subtracts numbers",
                    "title": "subtracts numbers",
                    "status": "passed",
                    "failureMessages": [],
                },
            ],
        }
    ],
}

VITEST_JSON_WITH_FAILURE = {
    "numTotalTestSuites": 1,
    "numPassedTestSuites": 0,
    "numFailedTestSuites": 1,
    "numTotalTests": 2,
    "numPassedTests": 1,
    "numFailedTests": 1,
    "success": False,
    "testResults": [
        {
            "name": "tests/example.test.ts",
            "status": "failed",
            "assertionResults": [
                {
                    "ancestorTitles": ["example suite"],
                    "fullName": "example suite adds numbers",
                    "title": "adds numbers",
                    "status": "passed",
                    "failureMessages": [],
                },
                {
                    "ancestorTitles": ["example suite"],
                    "fullName": "example suite subtracts numbers",
                    "title": "subtracts numbers",
                    "status": "failed",
                    "failureMessages": [
                        "AssertionError: expected 1 to be 2 // Object.is equality"
                    ],
                },
            ],
        }
    ],
}


@pytest.fixture
def runner():
    return TestRunner()


class TestParseVitestJsonPassing:
    def test_all_passing_reports_passed_true(self, runner, tmp_path):
        result_path = tmp_path / "vitest_result.json"
        result_path.write_text(json.dumps(VITEST_JSON_ALL_PASSING))

        result = runner._parse_vitest_json(str(result_path))

        assert result.passed is True
        assert result.total == 2
        assert result.failed == 0
        assert result.failures == []


class TestParseVitestJsonFailing:
    def test_failing_case_reports_passed_false(self, runner, tmp_path):
        result_path = tmp_path / "vitest_result.json"
        result_path.write_text(json.dumps(VITEST_JSON_WITH_FAILURE))

        result = runner._parse_vitest_json(str(result_path))

        assert result.passed is False
        assert result.total == 2
        assert result.failed == 1
        assert len(result.failures) == 1

    def test_failure_test_name_follows_file_colon_colon_convention(
        self, runner, tmp_path
    ):
        result_path = tmp_path / "vitest_result.json"
        result_path.write_text(json.dumps(VITEST_JSON_WITH_FAILURE))

        result = runner._parse_vitest_json(str(result_path))

        failure = result.failures[0]
        # master.py's map_failures_to_work_items splits on "::" and takes
        # the FIRST segment as the file path.
        file_path = failure.test_name.split("::", 1)[0]
        assert file_path == "tests/example.test.ts"
        assert "subtracts numbers" in failure.test_name

    def test_failure_error_output_captured(self, runner, tmp_path):
        result_path = tmp_path / "vitest_result.json"
        result_path.write_text(json.dumps(VITEST_JSON_WITH_FAILURE))

        result = runner._parse_vitest_json(str(result_path))

        assert "expected 1 to be 2" in result.failures[0].error_output


class TestParseVitestJsonMissingOrMalformed:
    def test_missing_file_returns_failure_placeholder(self, runner, tmp_path):
        missing_path = str(tmp_path / "does_not_exist.json")

        result = runner._parse_vitest_json(missing_path)

        assert result.passed is False
        assert result.total == 0
        assert len(result.failures) == 1
        assert "not found" in result.failures[0].test_name

    def test_malformed_json_returns_failure_placeholder(self, runner, tmp_path):
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("{not valid json")

        result = runner._parse_vitest_json(str(bad_path))

        assert result.passed is False
        assert len(result.failures) == 1
        assert "parse error" in result.failures[0].test_name


class TestGracefulDegradationMissingBinary:
    def test_run_language_suite_handles_missing_binary(self, runner, tmp_path, monkeypatch):
        """
        Simulate bunx/vitest not being installed: subprocess.run raises
        FileNotFoundError, and _run_language_suite must degrade
        gracefully (matching the existing pytest/jest fallback path)
        rather than raising.
        """

        def fake_run(*args, **kwargs):
            raise FileNotFoundError("bunx: command not found")

        monkeypatch.setattr(subprocess, "run", fake_run)

        lang_config = {
            "test_command": ["bunx", "vitest", "run", "--reporter=json", "--outputFile={result_path}"],
            "result_format": "vitest_json",
        }

        result = runner._run_language_suite(
            "typescript", lang_config, str(tmp_path / "tests"), str(tmp_path / "product")
        )

        assert result.passed is False
        assert len(result.failures) == 1
        assert "bunx not found" in result.failures[0].test_name
        assert "typescript" in result.failures[0].error_output
