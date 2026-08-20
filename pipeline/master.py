"""
pipeline/master.py

MasterAgent: holds the user's intent throughout a pipeline run, and
(Phase 3) evaluates test failures to decide what to re-dispatch and
builds the final FailReport when retries are exhausted.
"""

import os

from pipeline.state import AgentResult, FailReport, TestFailure, TestResult, WorkItem


class MasterAgent:
    def __init__(self, config: dict, user_prompt: str = ""):
        self.config = config
        self.intent: str = user_prompt

    def set_intent(self, user_prompt: str) -> None:
        """Update the held intent for a new pipeline run."""
        self.intent = user_prompt

    def map_failures_to_work_items(
        self,
        test_result: TestResult,
        agent_results: list[AgentResult],
        work_items: list[WorkItem],
    ) -> dict[str, list[TestFailure]]:
        """
        Given a failed TestResult, resolve which WorkItem ids should be
        retried: the WorkItem that produced the failing test file, plus
        everything in that WorkItem's depends_on (either could be at
        fault).

        Convention (must match pipeline/test_runner.py): TestFailure.test_name
        is f"{file_path}::{classname}::{name}" — the file path is the first
        "::"-delimited segment.

        Returns:
            dict mapping work_item_id -> list of TestFailure that directly
            implicated it. WorkItem ids added to the retry set only via
            depends_on propagation (i.e. they didn't directly produce a
            failing file) map to an empty list — the caller can tell
            "retried because a dependent failed" from an empty list.
        """
        by_id = {item.id: item for item in work_items}

        # file_path (normalized) -> work_item_id, from every AgentResult's
        # files_written (paths are relative to project_root).
        reverse_map: dict[str, str] = {}
        for result in agent_results:
            for path in result.files_written:
                reverse_map[os.path.normpath(path)] = result.work_item_id

        retry_failures: dict[str, list[TestFailure]] = {}

        for failure in test_result.failures:
            file_path = failure.test_name.split("::", 1)[0]
            normalized = os.path.normpath(file_path)

            work_item_id = reverse_map.get(normalized)

            if work_item_id is None:
                # Fallback: endswith matching, handles relative vs
                # project-root-relative path discrepancies between how
                # pytest reports the file and how it's stored in the map.
                for mapped_path, mapped_id in reverse_map.items():
                    if normalized.endswith(mapped_path) or mapped_path.endswith(
                        normalized
                    ):
                        work_item_id = mapped_id
                        break

            if work_item_id is None:
                # Couldn't resolve — skip retry-targeting for this failure,
                # but the raw failure still surfaces in the FailReport via
                # test_result.failures regardless.
                continue

            retry_failures.setdefault(work_item_id, []).append(failure)
            item = by_id.get(work_item_id)
            if item is not None:
                for dep_id in item.depends_on:
                    retry_failures.setdefault(dep_id, [])

        return retry_failures

    def build_fail_report(
        self,
        run_number: int,
        test_result: TestResult,
        unresolved_work_item_ids: list[str],
    ) -> FailReport:
        """
        Build the final FailReport after retries are exhausted.

        Note: the design doc's example shows an LLM-generated narrative
        summary. For this Phase 3 baseline, the summary is a deterministic
        templated string (no LLM round-trip) — see final report deviation
        notes.
        """
        summary = (
            f"{test_result.failed} of {test_result.total} tests failed "
            f"across {len(unresolved_work_item_ids)} unresolved work "
            f"item(s) after exhausting retries."
        )

        return FailReport(
            run_number=run_number,
            failures=test_result.failures,
            unresolved_items=unresolved_work_item_ids,
            summary=summary,
        )
