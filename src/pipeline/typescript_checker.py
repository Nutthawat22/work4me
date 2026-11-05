"""Run the pinned, non-executing TypeScript compiler API checker."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

MAX_INPUT_BYTES = 4 * 1024 * 1024
MAX_SOURCE_FILES = 2000
CHECK_TIMEOUT_SECONDS = 30
CHECKER_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "typescript-checker" / "check.mjs"


def check_typescript(
    files: dict[str, str], contracts: list[dict[str, str]], *,
    require_all_contracts: bool = True,
    required_contract_names: set[str] | None = None,
) -> dict[str, Any]:
    if not files:
        return {"status": "not_applicable", "diagnostics": [], "imports": [], "contract_findings": [], "limitations": []}
    if len(files) > MAX_SOURCE_FILES:
        return _unsupported("TypeScript checking skipped: source count exceeds the 2,000-file safety budget.")

    payload = json.dumps(
        {"files": [{"path": path, "content": content} for path, content in files.items()],
         "contracts": contracts[:20],
         "requireAllContracts": require_all_contracts,
         "requiredContractNames": sorted(required_contract_names or set())},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if len(payload.encode("utf-8")) > MAX_INPUT_BYTES:
        return _unsupported("TypeScript checking skipped: authorized source snapshot exceeds the 4 MiB safety budget.")
    if not CHECKER_SCRIPT.is_file():
        return _unsupported("TypeScript checking is unsupported: trusted checker script is unavailable.")

    try:
        completed = subprocess.run(
            ["node", "--max-old-space-size=512", str(CHECKER_SCRIPT)],
            input=payload,
            text=True,
            capture_output=True,
            timeout=CHECK_TIMEOUT_SECONDS,
            check=False,
            cwd=str(CHECKER_SCRIPT.parent),
            env={"PATH": os.environ.get("PATH", ""), "NODE_OPTIONS": ""},
        )
    except (OSError, subprocess.TimeoutExpired):
        return _unsupported("TypeScript checking is unsupported: pinned Node/TypeScript tooling is missing or exceeded its 30-second limit.")
    if len(completed.stdout) > 1024 * 1024:
        return _unsupported("TypeScript checking is unsupported: checker output exceeded the 1 MiB result budget.")
    if completed.returncode != 0:
        return _unsupported("TypeScript checking is unsupported: the trusted checker process exited abnormally.")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return _unsupported("TypeScript checking is unsupported: the trusted checker returned invalid output.")
    if not isinstance(result, dict) or result.get("status") not in {"checked", "unsupported"}:
        return _unsupported("TypeScript checking is unsupported: the trusted checker returned an invalid result shape.")
    return result


def _unsupported(message: str) -> dict[str, Any]:
    return {
        "status": "unsupported",
        "diagnostics": [],
        "imports": [],
        "contract_findings": [],
        "limitations": [message],
    }
