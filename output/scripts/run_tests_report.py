#!/usr/bin/env python3
"""
scripts/run_tests_report.py

A test runner script that programmatically invokes pytest, captures the
results, and reports a clear, structured list of failing tests -- including
a summary of the failure reason/traceback for each.

Usage:
    python scripts/run_tests_report.py [pytest args...]
    python scripts/run_tests_report.py --format json --output report.json
    python scripts/run_tests_report.py tests/ -k "not slow"

Exit status:
    0  -- all tests passed (or no tests failed)
    !=0 -- one or more tests failed, or pytest encountered an error

Any arguments not recognized by this script are forwarded directly to
pytest, so you can pass normal pytest options/paths as usual.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import pytest
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "Error: pytest is not installed. Install it with `pip install pytest`.\n"
    )
    sys.exit(2)


@dataclass
class FailedTest:
    """Represents a single failed test and its failure details."""

    nodeid: str
    outcome: str  # "failed" or "error"
    when: str  # "setup", "call", or "teardown"
    duration: float
    message: str
    traceback_summary: str

    def to_dict(self) -> dict:
        return {
            "nodeid": self.nodeid,
            "outcome": self.outcome,
            "when": self.when,
            "duration": round(self.duration, 4),
            "message": self.message,
            "traceback_summary": self.traceback_summary,
        }


@dataclass
class ResultCollector:
    """Pytest plugin that collects pass/fail results for every test."""

    total: int = 0
    passed: int = 0
    failed_tests: List[FailedTest] = field(default_factory=list)

    def pytest_runtest_logreport(self, report):
        # Only count each test once toward the total (on the "call" phase),
        # but capture failures that occur during setup/call/teardown.
        if report.when == "call":
            self.total += 1
            if report.outcome == "passed":
                self.passed += 1

        if report.failed:
            outcome = "error" if report.when != "call" else "failed"
            message = self._extract_message(report)
            traceback_summary = self._extract_traceback_summary(report)
            self.failed_tests.append(
                FailedTest(
                    nodeid=report.nodeid,
                    outcome=outcome,
                    when=report.when,
                    duration=getattr(report, "duration", 0.0),
                    message=message,
                    traceback_summary=traceback_summary,
                )
            )

    @staticmethod
    def _extract_message(report) -> str:
        """Return a short, one-line failure message."""
        longrepr = getattr(report, "longrepr", None)
        if longrepr is None:
            return "Unknown failure (no details available)"

        # longrepr can be a tuple, a string, or an object with reprcrash
        reprcrash = getattr(longrepr, "reprcrash", None)
        if reprcrash is not None:
            return str(reprcrash.message)

        text = str(longrepr)
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        return first_line or "Unknown failure"

    @staticmethod
    def _extract_traceback_summary(report) -> str:
        """Return a compact traceback/summary string for the failure."""
        longrepr = getattr(report, "longrepr", None)
        if longrepr is None:
            return ""

        text = str(longrepr)
        lines = text.strip().splitlines()

        # Keep it reasonably short: last N lines usually contain the
        # most relevant assertion/exception info.
        max_lines = 15
        if len(lines) > max_lines:
            lines = ["... (truncated) ..."] + lines[-max_lines:]

        return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the test suite via pytest and report a structured list of "
            "failing tests with failure details."
        ),
        add_help=True,
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format for the failure report (default: text).",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Write the report to FILE instead of stdout.",
    )
    parser.add_argument(
        "--quiet-pytest",
        action="store_true",
        help="Suppress pytest's own console output (only show this report).",
    )
    return parser


def format_text_report(collector: ResultCollector) -> str:
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("TEST RUN SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Total tests run : {collector.total}")
    lines.append(f"Passed          : {collector.passed}")
    lines.append(f"Failed          : {len(collector.failed_tests)}")
    lines.append("")

    if not collector.failed_tests:
        lines.append("All tests passed. 🎉")
        return "\n".join(lines)

    lines.append("FAILED TESTS")
    lines.append("-" * 70)
    for i, ft in enumerate(collector.failed_tests, start=1):
        lines.append(f"{i}. {ft.nodeid}  [{ft.outcome} @ {ft.when}]")
        lines.append(f"   Duration: {ft.duration:.4f}s")
        lines.append(f"   Message : {ft.message}")
        lines.append("   Traceback summary:")
        for tb_line in ft.traceback_summary.splitlines():
            lines.append(f"     {tb_line}")
        lines.append("-" * 70)

    return "\n".join(lines)


def format_json_report(collector: ResultCollector) -> str:
    report = {
        "total": collector.total,
        "passed": collector.passed,
        "failed_count": len(collector.failed_tests),
        "failed_tests": [ft.to_dict() for ft in collector.failed_tests],
    }
    return json.dumps(report, indent=2)


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    arg_parser = build_arg_parser()
    known_args, pytest_args = arg_parser.parse_known_args(argv)

    collector = ResultCollector()

    pytest_run_args = list(pytest_args)
    if known_args.quiet_pytest:
        pytest_run_args.append("-q")
        pytest_run_args.append("--no-header")

    exit_code = pytest.main(pytest_run_args, plugins=[collector])

    if known_args.format == "json":
        report_text = format_json_report(collector)
    else:
        report_text = format_text_report(collector)

    if known_args.output:
        with open(known_args.output, "w", encoding="utf-8") as f:
            f.write(report_text + "\n")
        print(f"Report written to {known_args.output}")
    else:
        print(report_text)

    if collector.failed_tests and exit_code == 0:
        # Safety net: ensure non-zero exit if failures were captured
        # but pytest's own exit code didn't reflect it for some reason.
        exit_code = 1

    return int(exit_code)


if __name__ == "__main__":
    sys.exit(main())