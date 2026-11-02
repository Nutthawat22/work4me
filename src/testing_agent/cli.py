"""Command-line entry point for a manual external-testing delegation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_SRC = str(Path(__file__).resolve().parents[1])
if REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)

from testing_agent.config import load_settings
from testing_agent.runner import run_external_tests


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a pipeline testing handoff inside Docker isolation.")
    parser.add_argument("--handoff", required=True, help="absolute path to testing-handoff.json")
    parser.add_argument("--project-root", help="explicitly authorize the exact existing-project root in the handoff")
    parser.add_argument("--output-dir", help="new absolute directory for result, report, snapshot, and logs")
    args = parser.parse_args()
    try:
        result = run_external_tests(
            args.handoff,
            authorized_project_root=args.project_root,
            output_dir=args.output_dir,
            settings=load_settings(),
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
    except Exception as error:
        print(f"External testing could not start: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return {
        "passed": 0,
        "tests_failed": 1,
        "setup_failed": 2,
        "no_tests": 3,
        "cancelled": 130,
    }.get(str(result.get("outcome")), 2)


if __name__ == "__main__":
    raise SystemExit(main())
