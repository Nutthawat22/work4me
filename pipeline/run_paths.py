"""
pipeline/run_paths.py

Helpers for creating a per-run output directory under config's
"runs_dir": runs/{NNNN-slug}/{design,product,tests/results}. One run dir
is created per user prompt handled by the runner.py REPL loop.
"""

import json
import os
import re


def slugify(prompt: str, max_len: int = 40) -> str:
    """Lowercase, collapse non-alphanumeric runs to single hyphens, strip
    edge hyphens, truncate to max_len (breaking on a hyphen boundary when
    possible). Falls back to "run" if the result would be empty."""
    lowered = prompt.lower()
    collapsed = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")

    if len(collapsed) > max_len:
        truncated = collapsed[:max_len]
        last_hyphen = truncated.rfind("-")
        if last_hyphen > 0:
            truncated = truncated[:last_hyphen]
        collapsed = truncated.strip("-")

    return collapsed or "run"


def next_run_seq(runs_dir: str) -> int:
    """Scan immediate subdirectories of runs_dir for a leading NNNN- prefix
    and return max+1 (1 if none exist or runs_dir doesn't exist yet)."""
    if not os.path.isdir(runs_dir):
        return 1

    max_seq = 0
    pattern = re.compile(r"^(\d{4})-")

    for entry in os.listdir(runs_dir):
        if not os.path.isdir(os.path.join(runs_dir, entry)):
            continue
        match = pattern.match(entry)
        if not match:
            continue
        max_seq = max(max_seq, int(match.group(1)))

    return max_seq + 1


def create_run_dir(runs_dir: str, prompt: str) -> str:
    """Create runs_dir/{NNNN-slug}/{design,product,tests/results} and
    return the full run dir path."""
    seq = next_run_seq(runs_dir)
    run_id = f"{seq:04d}-{slugify(prompt)}"
    run_dir = os.path.join(runs_dir, run_id)

    os.makedirs(os.path.join(run_dir, "design"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "product"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "tests", "results"), exist_ok=True)

    return run_dir


def write_manifest(run_dir: str, manifest: dict) -> None:
    """Overwrite run_dir/manifest.json with the given manifest dict."""
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def append_index(runs_dir: str, entry: dict) -> None:
    """Append one compact JSON line to runs_dir/index.jsonl. Creates
    runs_dir if it doesn't exist yet (mirrors create_run_dir's behavior)."""
    os.makedirs(runs_dir, exist_ok=True)
    with open(os.path.join(runs_dir, "index.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")
