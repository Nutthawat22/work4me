"""
src/pipeline/dispatch.py

Dependency ordering (topological sort over WorkItem.depends_on) and
sequential dispatch to the matching specialist agent.
"""

import os
from collections import deque
from typing import Callable

from pipeline.state import AgentResult, TaskAttempt, TestFailure, WorkItem
from pipeline.task_validation import normalize_relative_path, truncate_with_marker
from specialists.feature import FeatureAgent
from specialists.integrate import IntegrationAgent
from specialists.foundation import FoundationAgent
from specialists.shared import SharedAgent
from specialists.test_writer import ValidationAgent

# Called as progress(item, index, total) immediately BEFORE that item is
# dispatched to its specialist (index is 1-based). Dispatch is sequential
# and each specialist call is a blocking LLM request that can take a
# while, so this is the caller's only hook to show "still working, on
# item N of M" rather than appearing to hang for the whole batch.
DispatchProgressCallback = Callable[[WorkItem, int, int], None]
TaskUpdateCallback = Callable[[WorkItem], None]

# Limits how much dependency content is loaded before the worker prompt applies
# its own smaller budget. Omitted paths are reported to the worker.
MAX_DEPENDENCY_FILES_SHOWN = 40
MAX_DEPENDENCY_FILE_CHARS = 12000
MAX_DEPENDENCY_TOTAL_CHARS = 80000


class CycleError(Exception):
    """Raised when WorkItems cannot be topologically ordered — either a
    dependency cycle exists, or a WorkItem depends on an id not present
    in the given list."""


def topological_sort(work_items: list[WorkItem], strict: bool = True) -> list[WorkItem]:
    """
    Order work_items so that every item comes after all of its
    depends_on ids, using Kahn's algorithm.

    Args:
        work_items: The (possibly partial) batch of WorkItems to order.
        strict: When True (default — full initial dispatch), a depends_on
            id that isn't present in work_items is treated as a missing
            dependency and raises CycleError. When False (retry subsets),
            such ids are treated as already-satisfied (out-of-batch) and
            simply don't count toward in-degree.

    Raises:
        CycleError: if a cycle is detected among ids present in
            work_items, or (strict=True only) an item's depends_on
            references an id not present in work_items.
    """
    by_id = {item.id: item for item in work_items}

    if strict:
        for item in work_items:
            for dep_id in item.depends_on:
                if dep_id not in by_id:
                    raise CycleError(
                        f"WorkItem {item.id!r} depends on unknown WorkItem id {dep_id!r}"
                    )

    # Only count depends_on edges where the dependency is present in this
    # batch — out-of-batch ids (non-strict mode) are pre-satisfied.
    in_degree = {
        item.id: sum(1 for dep_id in item.depends_on if dep_id in by_id)
        for item in work_items
    }
    dependents: dict[str, list[str]] = {item.id: [] for item in work_items}
    for item in work_items:
        for dep_id in item.depends_on:
            if dep_id in by_id:
                dependents[dep_id].append(item.id)

    queue = deque(sorted(item_id for item_id, deg in in_degree.items() if deg == 0))
    ordered: list[WorkItem] = []

    while queue:
        item_id = queue.popleft()
        ordered.append(by_id[item_id])
        for dependent_id in dependents[item_id]:
            in_degree[dependent_id] -= 1
            if in_degree[dependent_id] == 0:
                queue.append(dependent_id)

    if len(ordered) != len(work_items):
        remaining = sorted(set(by_id.keys()) - {item.id for item in ordered})
        raise CycleError(
            f"Cycle detected among WorkItems (or unresolved dependencies): {remaining}"
        )

    return ordered


def _read_dependency_content(
    result: AgentResult,
    run_dir: str,
    max_chars: int = MAX_DEPENDENCY_FILE_CHARS,
) -> dict[str, str] | None:
    """Read (and truncate) every file a successful dependency task wrote.

    Returns None if the dependency's result was unsuccessful or its file
    can't be read for any reason — callers should note this rather than
    crash.
    """
    if not result.success or not result.files_written:
        return None
    contents: dict[str, str] = {}
    run_root = os.path.realpath(run_dir)
    for relative_path in result.files_written:
        try:
            normalized = normalize_relative_path(relative_path)
            full_path = os.path.realpath(os.path.join(run_root, normalized))
            if os.path.commonpath((run_root, full_path)) != run_root:
                continue
            with open(full_path, "r") as source:
                content = source.read()
        except (OSError, ValueError):
            continue
        content = truncate_with_marker(content, max_chars, "\n... (truncated)")
        contents[normalized] = content
    return contents or None


def build_dependency_context(
    item: WorkItem,
    results_by_id: dict[str, AgentResult],
    run_dir: str,
) -> dict[str, str]:
    """Build the {run-relative output path: file_content} map for dependencies.
    dependencies, resolving from already-dispatched AgentResults.

    Missing/failed/unreadable dependencies are noted inline rather than
    omitted silently. File and total-character caps are explicit, with
    omitted dependency paths reported in a sentinel context entry.
    """
    context: dict[str, str] = {}
    omitted: list[str] = []
    total_chars = 0
    shown_files = 0
    for dep_id in item.depends_on:
        result = results_by_id.get(dep_id)
        if result is None:
            context[dep_id] = f"(dependency {dep_id} unavailable — not yet dispatched)"
            continue
        files = _read_dependency_content(result, run_dir)
        if files is None:
            reason = "specialist failed" if not result.success else "file unreadable"
            context[dep_id] = f"(dependency {dep_id} unavailable — {reason})"
            continue
        for output_path, content in files.items():
            if shown_files >= MAX_DEPENDENCY_FILES_SHOWN or total_chars >= MAX_DEPENDENCY_TOTAL_CHARS:
                omitted.append(output_path)
                continue
            remaining = MAX_DEPENDENCY_TOTAL_CHARS - total_chars
            content = truncate_with_marker(
                content, remaining, "\n... (truncated by dispatch context budget)"
            )
            context[output_path] = content
            total_chars += len(content)
            shown_files += 1
    if omitted:
        context["(dependency files omitted by dispatch budget)"] = ", ".join(omitted)
    return context


def dispatch_work_items(
    work_items: list[WorkItem],
    config: dict,
    run_dir: str,
    strict: bool = True,
    retry_context: dict[str, list[TestFailure]] | None = None,
    known_results: list[AgentResult] | None = None,
    progress: DispatchProgressCallback | None = None,
    task_update: TaskUpdateCallback | None = None,
) -> list[AgentResult]:
    """
    Topologically sort work_items by depends_on, then dispatch each to
    its matching specialist agent sequentially.

    Args:
        work_items: The (possibly partial) batch of WorkItems to dispatch.
        config: Loaded config dict.
        run_dir: Path to this pipeline run's output directory (see
            src/pipeline/run_paths.py), passed through to each specialist.
        strict: Passed through to topological_sort — True (default) for a
            full/self-contained batch, False for a retry subset where
            depends_on may reference ids outside this batch.
        retry_context: Optional map of work_item_id -> TestFailures that
            implicated it, from MasterAgent.map_failures_to_work_items().
            None (default) for the initial/strict dispatch, which has no
            failure history. Passed through to each specialist's
            execute() so it can inform the LLM prompt on retries.
        known_results: Optional list of AgentResults from prior dispatch
            rounds (e.g. src/pipeline/runner.py's all_agent_results
            accumulator). Needed during retry-subset dispatch (strict=
            False) so dependency file lookups can resolve WorkItems that
            already succeeded in an earlier round and therefore aren't
            present in this call's own work_items/results. Merged with
            results produced in this call (this call's own results take
            precedence for ids in both).
        progress: Optional callback invoked as progress(item, index,
            total) immediately before each item (1-based index) is
            dispatched. Dispatch is strictly sequential and each
            specialist call blocks on an LLM request that can take
            anywhere from seconds to minutes — without this, a caller
            driving a large batch (e.g. 20 WorkItems) has no visibility
            into progress until the whole batch finishes. None (default)
            disables progress reporting entirely.

    Raises:
        CycleError: propagated from topological_sort.
    """
    ordered = topological_sort(work_items, strict=strict)

    registry = {
        "foundation": FoundationAgent(),
        "feature": FeatureAgent(),
        "shared": SharedAgent(),
        "integration": IntegrationAgent(),
        "validation": ValidationAgent(),
    }

    results_by_id: dict[str, AgentResult] = {
        r.work_item_id: r for r in (known_results or [])
    }

    total = len(ordered)
    results: list[AgentResult] = []
    dependency_root = config.get("pipeline", {}).get("_existing_project_path", run_dir)
    for index, item in enumerate(ordered, start=1):
        failed_dependencies = [
            dep_id for dep_id in item.depends_on
            if dep_id in results_by_id and not results_by_id[dep_id].success
        ]
        unavailable_dependencies = [dep_id for dep_id in item.depends_on if dep_id not in results_by_id]
        if failed_dependencies or unavailable_dependencies:
            item.status = "blocked"
            result = AgentResult(
                work_item_id=item.id,
                agent_name=f"{item.category.title()}Agent",
                success=False,
                files_written=[],
                notes=("Task blocked by dependencies: " + ", ".join(
                    failed_dependencies + unavailable_dependencies
                )),
            )
            results.append(result)
            results_by_id[item.id] = result
            if task_update is not None:
                task_update(item)
            continue

        item.status = "ready"
        if task_update is not None:
            task_update(item)
        item.status = "running"
        if task_update is not None:
            task_update(item)
        if progress is not None:
            progress(item, index, total)
        agent = registry[item.category]
        retry_failures = (retry_context or {}).get(item.id, [])
        dependency_context = build_dependency_context(item, results_by_id, dependency_root)
        result = agent.execute(
            item,
            config,
            run_dir,
            retry_failures=retry_failures,
            dependency_context=dependency_context,
        )
        results.append(result)
        results_by_id[item.id] = result
        item.status = "passed" if result.success else "failed"
        item.attempts.append(
            TaskAttempt(
                number=len(item.attempts) + 1,
                status="passed" if result.success else "failed",
                files_written=list(result.files_written),
                result_summary=(
                    f"Wrote {len(result.files_written)} file(s); deleted {len(result.files_deleted)} file(s)"
                    if result.success
                    else ""
                ),
                files_deleted=list(result.files_deleted),
                failure_details=(
                    result.notes
                    if not result.success
                    else "\n".join(
                        f"{failure.test_name}: {failure.error_output[:500]}"
                        for failure in retry_failures[:5]
                    )
                ),
            )
        )
        if task_update is not None:
            task_update(item)

    return results
