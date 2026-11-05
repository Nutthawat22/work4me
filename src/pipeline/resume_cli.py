"""Explicit CLI for resuming a finalized pipeline run without replanning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from pipeline.config_validation import validate_config
from pipeline.resume import ResumeError
from pipeline.runner import load_config, resume_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Source run id or absolute path under configured runs storage")
    parser.add_argument("--task", required=True, action="append", dest="task_ids", help="Exact task id to repair; repeat as needed")
    parser.add_argument("--project-root", help="Exact existing-project root for existing-project resumes")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config, config_path = load_config()
        config_errors = validate_config(config)
        if config_errors:
            print(f"Invalid pipeline config ({config_path}):", file=sys.stderr)
            for error in config_errors:
                print(f"- {error}", file=sys.stderr)
            return 2
        success = resume_pipeline(
            config,
            args.run,
            args.task_ids,
            existing_project_path=args.project_root,
        )
        return 0 if success else 1
    except ResumeError as error:
        print(f"Resume refused: {error}", file=sys.stderr)
        return 2
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Resume failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
