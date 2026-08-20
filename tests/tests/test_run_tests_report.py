import subprocess
import sys
import os
import textwrap
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER_SCRIPT = os.path.join(REPO_ROOT, "scripts", "run_tests_report.py")


@pytest.fixture
def mixed_fixture_module(tmp_path):
    """
    Creates a fixture test module containing at least one passing
    and one intentionally failing test.
    """
    test_file = tmp_path / "test_fixture_mixed.py"
    test_file.write_text(
        textwrap.dedent(
            """
            def test_passing_case():
                assert 1 + 1 == 2

            def test_failing_case():
                assert 1 + 1 == 3, "This test is intentionally failing"
            """
        )
    )
    return test_file


@pytest.fixture
def all_passing_fixture_module(tmp_path):
    """
    Creates a fixture test module containing only passing tests.
    """
    test_file = tmp_path / "test_fixture_all_passing.py"
    test_file.write_text(
        textwrap.dedent(
            """
            def test_passing_case_one():
                assert 1 + 1 == 2

            def test_passing_case_two():
                assert "abc".upper() == "ABC"
            """
        )
    )
    return test_file


def run_runner_script(*args):
    """
    Helper to invoke the test runner script as a subprocess and
    capture its output and exit code.
    """
    assert os.path.exists(RUNNER_SCRIPT), (
        f"Expected runner script to exist at {RUNNER_SCRIPT}"
    )
    cmd = [sys.executable, RUNNER_SCRIPT, *args]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return result


class TestRunTestsReportScriptExists:
    def test_runner_script_exists(self):
        assert os.path.isfile(RUNNER_SCRIPT), (
            "scripts/run_tests_report.py must exist for these tests to run"
        )


class TestRunTestsReportFailureDetection:
    def test_runner_lists_failing_test_by_name(self, mixed_fixture_module):
        result = run_runner_script(str(mixed_fixture_module))

        combined_output = result.stdout + result.stderr

        assert "test_failing_case" in combined_output, (
            "Runner script output should mention the name of the failing test.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_runner_does_not_report_failure_for_passing_test_as_failed(
        self, mixed_fixture_module
    ):
        result = run_runner_script(str(mixed_fixture_module))

        combined_output = result.stdout + result.stderr

        # Ensure the passing test isn't incorrectly reported as a failure.
        # We check that if "test_passing_case" appears alongside failure
        # indicators, it isn't marked FAILED.
        passing_failed_pattern = "test_passing_case FAILED"
        assert passing_failed_pattern not in combined_output, (
            "Passing test should not be reported as failed.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_runner_exit_code_nonzero_on_failures(self, mixed_fixture_module):
        result = run_runner_script(str(mixed_fixture_module))

        assert result.returncode != 0, (
            "Runner script should exit with a non-zero exit code when "
            "there are failing tests.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


class TestRunTestsReportAllPassing:
    def test_runner_exit_code_zero_when_all_tests_pass(
        self, all_passing_fixture_module
    ):
        result = run_runner_script(str(all_passing_fixture_module))

        assert result.returncode == 0, (
            "Runner script should exit with a zero exit code when all "
            "fixture tests pass.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_runner_reports_no_failures_when_all_tests_pass(
        self, all_passing_fixture_module
    ):
        result = run_runner_script(str(all_passing_fixture_module))

        combined_output = result.stdout + result.stderr

        assert "FAILED" not in combined_output, (
            "Runner script output should not report any failures when "
            "all fixture tests pass.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )