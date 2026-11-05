"""Implementation evidence consumed by AI-SDLC's external Code-stage session."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import defaultdict

from pipeline.task_validation import normalize_relative_path

RESULT_FILENAME = "implementation-result.json"


def write_implementation_result(config: dict, run_dir: str, manifest: dict) -> str:
    """Export task delivery and declared AC coverage without inventing test results.

    The host owns tasks.md, code_manifest.json, its index, and Code acceptance.
    WorkItem ids remain in plan.json; deterministic T- ids are the host's view.
    ACP's terminal outcome remains the authority for whether the run completed.
    """
    plan_path = os.path.join(run_dir, "design", "plan.json")
    try:
        with open(plan_path, encoding="utf-8") as source:
            plan = json.load(source)
    except FileNotFoundError:
        if manifest.get("validation_status") == "passed":
            raise ValueError("A validated implementation requires a task plan")
        plan = {"tasks": []}
    if not isinstance(plan, dict) or not isinstance(plan.get("tasks"), list):
        raise ValueError("Implementation task plan is malformed")

    published = manifest.get("published_files", {})
    if not isinstance(published, dict):
        raise ValueError("Published source snapshot is malformed")
    tasks = []
    coverage: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"tasks": [], "files": []})
    project_path = config["pipeline"].get("_existing_project_path")
    for index, item in enumerate(plan["tasks"], start=1):
        task_id = f"T-{index:03d}"
        status = {"passed": "done", "running": "in_progress"}.get(item["status"], "todo")
        files = []
        for owned in item["owned_files"]:
            path = normalize_relative_path(owned["path"])
            if not project_path:
                area = "tests_dir" if owned["area"] == "tests" else "output_dir"
                root = normalize_relative_path(config["pipeline"][area])
                if not path.startswith(root + "/"):
                    path = f"{root}/{path}"
            if path in published:
                files.append(path)
        tasks.append({
            "id": task_id,
            "work_item_id": item["id"],
            "title": item["title"],
            "status": status,
            "commits": [],
            "files": files,
        })
        if status != "done":
            continue
        for reference in dict.fromkeys(item["requirement_refs"]):
            if re.fullmatch(r"AC-\d+", reference) is None:
                continue
            claim = coverage[reference]
            claim["tasks"].append(task_id)
            claim["files"].extend(path for path in files if path not in claim["files"])

    result = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "mode": "existing_project" if project_path else "new_project",
        "project_path": project_path,
        "delivery_mode": "working_tree",
        "tasks": tasks,
        "ac_coverage": [{"ac_id": ac_id, **claim} for ac_id, claim in coverage.items()],
        "lint": "not_run",
        "unit_tests": None,
        "validation_status": manifest.get("validation_status", "not_completed"),
        "external_testing_status": manifest.get("external_testing_status", "pending"),
        "published_files": published,
    }
    destination = os.path.join(run_dir, RESULT_FILENAME)
    descriptor, temporary = tempfile.mkstemp(prefix=".implementation-result-", dir=run_dir)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination
