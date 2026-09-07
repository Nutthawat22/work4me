"""
pipeline/test_runner.py

TestRunner: dispatches per-language test suites (via subprocess) based on
the languages actually present in a run's WorkItems, parses each
language's native result format, and merges everything into a single
TestResult. Decoupled from AgentResult — the mapping of failures back to
WorkItem ids happens in MasterAgent.map_failures_to_work_items(), not
here.

Convention: TestFailure has no dedicated file-path field (see
pipeline/state.py), so the resolved test file path is embedded as a
prefix in `test_name`, joined with "::":

    test_name = f"{file_path}::{classname}::{name}"   # junit (pytest)
    test_name = f"{file_path}::{full_name}"            # jest_json
    test_name = f"{file_path}::{full_name}"            # vitest_json

Callers (MasterAgent) split on "::" and take the FIRST segment to
recover the file path for matching against AgentResult.files_written.
Any new result-format parser added here must keep the file path as the
first "::"-delimited segment so master.py's matching logic keeps working
unchanged. This convention must stay consistent between this file and
pipeline/master.py.

SCOPE NOTE — Playwright (e2e) is intentionally NOT implemented here.
Supertest needs no separate runner: it's a request-assertion library used
*inside* Vitest test files, so it's exercised by the same
`bunx vitest run` command as any other TS unit/integration test — no
wiring change needed as long as Supertest-based specs match the
`typescript` language's test_file_patterns (*.test.ts / *.spec.ts).

Playwright IS a separate e2e test runner (its own CLI `playwright test`
and its own JSON reporter shape, distinct from Vitest/Jest's
testResults/assertionResults schema) and is out of scope for this
handoff. To add it later:
  1. Add a `playwright` entry to config's "languages" map (or a
     dedicated non-language "e2e" bucket if WorkItem.language shouldn't
     conflate unit-test language with e2e suite) with its own
     test_command (e.g. ["bunx", "playwright", "test",
     "--reporter=json"]) and result_format (e.g. "playwright_json").
  2. Implement `_parse_playwright_json` here, following Playwright's own
     JSON reporter schema (suites[].specs[].tests[].results[]), still
     preserving the `file::name` test_name convention required by
     master.py's map_failures_to_work_items.
  3. Wire the new result_format into the dispatch in
     `_run_language_suite` and into config_validation.py's
     VALID_RESULT_FORMATS.
  4. Decide whether e2e specs share the `typescript` language's
     test_file_patterns or need a distinct pattern (e.g. *.e2e.ts) to
     avoid Vitest also picking them up and failing (Playwright's test()/
     expect() API is incompatible with Vitest's runtime).
"""

import fnmatch
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from pipeline.state import TestFailure, TestResult, WorkItem

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAX_ERROR_OUTPUT_LEN = 2000


class TestRunner:
    def run(self, config: dict, work_items: list[WorkItem], run_dir: str) -> TestResult:
        """
        Run each language's test suite that's actually represented in
        work_items, then merge results.

        For each unique WorkItem.language, look up config["languages"][language],
        check whether any files under run_dir/config["pipeline"]["tests_dir"]
        match that language's test_file_patterns, and if so run that
        language's test_command. Languages with no matching test files
        contribute nothing (not a failure). If no language across the
        whole run has any test files, behavior matches the old no-tests
        case: passed=True, total=0, failures=[].

        Args:
            run_dir: Path to this pipeline run's output directory (see
                pipeline/run_paths.py). config["pipeline"]["tests_dir"]
                and config["pipeline"]["output_dir"] are resolved relative
                to run_dir, not PROJECT_ROOT.
        """
        tests_dir = os.path.join(run_dir, config["pipeline"]["tests_dir"])
        output_dir = os.path.join(run_dir, config["pipeline"]["output_dir"])
        languages_config = config.get("languages", {})

        seen_languages: list[str] = []
        for item in work_items:
            if item.language not in seen_languages:
                seen_languages.append(item.language)

        ran_results: list[TestResult] = []

        for language in seen_languages:
            lang_config = languages_config.get(language)
            if lang_config is None:
                continue

            patterns = lang_config.get("test_file_patterns", [])
            if not self._find_test_files(tests_dir, patterns):
                continue

            ran_results.append(
                self._run_language_suite(language, lang_config, tests_dir, output_dir)
            )

        if not ran_results:
            return TestResult(passed=True, total=0, failed=0, failures=[])

        total = sum(r.total for r in ran_results)
        failed = sum(r.failed for r in ran_results)
        failures = [f for r in ran_results for f in r.failures]
        passed = all(r.passed for r in ran_results)

        return TestResult(passed=passed, total=total, failed=failed, failures=failures)

    def _run_language_suite(
        self, language: str, lang_config: dict, tests_dir: str, output_dir: str
    ) -> TestResult:
        result_format = lang_config.get("result_format")
        test_command = lang_config["test_command"]

        format_context = {"tests_dir": tests_dir}
        env = os.environ.copy()

        if result_format == "junit":
            fd, result_path = tempfile.mkstemp(suffix=".xml", prefix="pytest_junit_")
            os.close(fd)
            format_context["junit_path"] = result_path

            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                output_dir
                if not existing_pythonpath
                else output_dir + os.pathsep + existing_pythonpath
            )
        elif result_format == "jest_json":
            fd, result_path = tempfile.mkstemp(suffix=".json", prefix="jest_result_")
            os.close(fd)
            format_context["result_path"] = result_path
        elif result_format == "vitest_json":
            fd, result_path = tempfile.mkstemp(suffix=".json", prefix="vitest_result_")
            os.close(fd)
            format_context["result_path"] = result_path
        else:
            return TestResult(
                passed=False,
                total=0,
                failed=0,
                failures=[
                    TestFailure(
                        test_name=f"<unsupported result_format: {result_format}>",
                        work_item_id="",
                        error_output=(
                            f"Language {language!r} has unrecognized "
                            f"result_format {result_format!r}"
                        ),
                    )
                ],
            )

        command = [token.format(**format_context) for token in test_command]

        try:
            try:
                subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except FileNotFoundError:
                return TestResult(
                    passed=False,
                    total=0,
                    failed=0,
                    failures=[
                        TestFailure(
                            test_name=f"<{command[0]} not found>",
                            work_item_id="",
                            error_output=(
                                f"{command[0]} executable not found — is it "
                                f"installed? (language: {language})"
                            ),
                        )
                    ],
                )
            except subprocess.TimeoutExpired:
                return TestResult(
                    passed=False,
                    total=0,
                    failed=0,
                    failures=[
                        TestFailure(
                            test_name=f"<{language} test run timeout>",
                            work_item_id="",
                            error_output=f"{language} test run exceeded 120s timeout",
                        )
                    ],
                )

            if result_format == "junit":
                return self._parse_junit_xml(result_path)
            elif result_format == "vitest_json":
                return self._parse_vitest_json(result_path)
            else:
                return self._parse_jest_json(result_path)
        finally:
            if os.path.exists(result_path):
                os.remove(result_path)

    @staticmethod
    def _find_test_files(tests_dir: str, patterns: list[str]) -> bool:
        if not patterns:
            return False
        for root, _dirs, files in os.walk(tests_dir):
            for name in files:
                for pattern in patterns:
                    if fnmatch.fnmatch(name, pattern):
                        return True
        return False

    def _parse_junit_xml(self, junit_path: str) -> TestResult:
        try:
            tree = ET.parse(junit_path)
        except ET.ParseError:
            return TestResult(
                passed=False,
                total=0,
                failed=0,
                failures=[
                    TestFailure(
                        test_name="<junit xml parse error>",
                        work_item_id="",
                        error_output=f"Could not parse JUnit XML at {junit_path}",
                    )
                ],
            )

        root = tree.getroot()
        testcases = (
            root.findall(".//testcase") if root.tag != "testcase" else [root]
        )

        total = len(testcases)
        failures: list[TestFailure] = []

        for tc in testcases:
            failure_el = tc.find("failure")
            error_el = tc.find("error")
            fault_el = failure_el if failure_el is not None else error_el
            if fault_el is None:
                continue

            classname = tc.get("classname", "")
            name = tc.get("name", "")
            file_path = tc.get("file", classname.replace(".", "/") + ".py")

            error_text = (fault_el.text or fault_el.get("message", "")).strip()
            if len(error_text) > MAX_ERROR_OUTPUT_LEN:
                error_text = error_text[:MAX_ERROR_OUTPUT_LEN] + "...[truncated]"

            failures.append(
                TestFailure(
                    test_name=f"{file_path}::{classname}::{name}",
                    work_item_id="",
                    error_output=error_text,
                )
            )

        failed = len(failures)
        passed = failed == 0 and total > 0

        return TestResult(passed=passed, total=total, failed=failed, failures=failures)

    def _parse_jest_json(self, result_path: str) -> TestResult:
        """
        Parse jest's `--json --outputFile <path>` output.

        Schema (relevant fields):
          { "numTotalTests": int, "numFailedTests": int,
            "testResults": [
              { "name": "<file path>",
                "assertionResults": [
                  { "status": "passed"|"failed"|..., "fullName": str,
                    "title": str, "failureMessages": [str, ...] },
                  ...
                ]
              },
              ...
            ]
          }
        """
        try:
            with open(result_path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            return TestResult(
                passed=False,
                total=0,
                failed=0,
                failures=[
                    TestFailure(
                        test_name="<jest result file not found>",
                        work_item_id="",
                        error_output=f"Could not find jest result file at {result_path}",
                    )
                ],
            )
        except json.JSONDecodeError:
            return TestResult(
                passed=False,
                total=0,
                failed=0,
                failures=[
                    TestFailure(
                        test_name="<jest json parse error>",
                        work_item_id="",
                        error_output=f"Could not parse jest JSON at {result_path}",
                    )
                ],
            )

        test_file_results = data.get("testResults", [])

        failures: list[TestFailure] = []
        for tr in test_file_results:
            file_path = tr.get("name", "<unknown file>")
            for ar in tr.get("assertionResults", []):
                if ar.get("status") != "failed":
                    continue

                test_title = ar.get("fullName") or ar.get("title") or "<unnamed test>"
                messages = ar.get("failureMessages") or []
                error_text = "\n".join(messages).strip()
                if len(error_text) > MAX_ERROR_OUTPUT_LEN:
                    error_text = error_text[:MAX_ERROR_OUTPUT_LEN] + "...[truncated]"

                failures.append(
                    TestFailure(
                        test_name=f"{file_path}::{test_title}",
                        work_item_id="",
                        error_output=error_text,
                    )
                )

        total = data.get("numTotalTests")
        if total is None:
            total = sum(len(tr.get("assertionResults", [])) for tr in test_file_results)

        failed = len(failures)
        passed = failed == 0 and total > 0

        return TestResult(passed=passed, total=total, failed=failed, failures=failures)

    def _parse_vitest_json(self, result_path: str) -> TestResult:
        """
        Parse vitest's `run --reporter=json --outputFile=<path>` output.

        Vitest's JSON reporter is documented as producing a report "in a
        JSON format compatible with Jest's --json option"
        (https://vitest.dev/guide/reporters.html#json-reporter), so the
        schema is identical in the fields we care about:

          { "numTotalTests": int, "numFailedTests": int,
            "testResults": [
              { "name": "<file path>",
                "assertionResults": [
                  { "status": "passed"|"failed"|..., "fullName": str,
                    "title": str, "failureMessages": [str, ...] },
                  ...
                ]
              },
              ...
            ]
          }

        Kept as a separate method (rather than aliasing _parse_jest_json)
        so vitest-specific error messages are clear and the two formats
        can diverge independently if a future vitest version changes its
        JSON shape.
        """
        try:
            with open(result_path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            return TestResult(
                passed=False,
                total=0,
                failed=0,
                failures=[
                    TestFailure(
                        test_name="<vitest result file not found>",
                        work_item_id="",
                        error_output=f"Could not find vitest result file at {result_path}",
                    )
                ],
            )
        except json.JSONDecodeError:
            return TestResult(
                passed=False,
                total=0,
                failed=0,
                failures=[
                    TestFailure(
                        test_name="<vitest json parse error>",
                        work_item_id="",
                        error_output=f"Could not parse vitest JSON at {result_path}",
                    )
                ],
            )

        test_file_results = data.get("testResults", [])

        failures: list[TestFailure] = []
        for tr in test_file_results:
            file_path = tr.get("name", "<unknown file>")
            for ar in tr.get("assertionResults", []):
                if ar.get("status") != "failed":
                    continue

                test_title = ar.get("fullName") or ar.get("title") or "<unnamed test>"
                messages = ar.get("failureMessages") or []
                error_text = "\n".join(messages).strip()
                if len(error_text) > MAX_ERROR_OUTPUT_LEN:
                    error_text = error_text[:MAX_ERROR_OUTPUT_LEN] + "...[truncated]"

                failures.append(
                    TestFailure(
                        test_name=f"{file_path}::{test_title}",
                        work_item_id="",
                        error_output=error_text,
                    )
                )

        total = data.get("numTotalTests")
        if total is None:
            total = sum(len(tr.get("assertionResults", [])) for tr in test_file_results)

        failed = len(failures)
        passed = failed == 0 and total > 0

        return TestResult(passed=passed, total=total, failed=failed, failures=failures)
