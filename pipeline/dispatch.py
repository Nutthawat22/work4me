"""
pipeline/dispatch.py

Dependency ordering (topological sort over WorkItem.depends_on) and
sequential dispatch to the matching specialist agent.
"""

import os
from collections import deque
from typing import Callable

from pipeline.state import AgentResult, TestFailure, WorkItem
from specialists.config import ConfigAgent
from specialists.logic import LogicAgent
from specialists.test_writer import TestAgent
from specialists.ui import UIAgent

# Called as progress(item, index, total) immediately BEFORE that item is
# dispatched to its specialist (index is 1-based). Dispatch is sequential
# and each specialist call is a blocking LLM request that can take a
# while, so this is the caller's only hook to show "still working, on
# item N of M" rather than appearing to hang for the whole batch.
DispatchProgressCallback = Callable[[WorkItem, int, int], None]

# Cap how much dependency content gets injected into a dependent
# WorkItem's prompt — avoids blowing up context on deeply-chained items.
MAX_DEPENDENCY_FILES_SHOWN = 5
MAX_DEPENDENCY_FILE_CHARS = 3000


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
    dep_id: str,
    result: AgentResult,
    run_dir: str,
) -> str | None:
    """Read (and truncate) the file content a dependency WorkItem wrote.

    Returns None if the dependency's result was unsuccessful or its file
    can't be read for any reason — callers should note this rather than
    crash.
    """
    if not result.success or not result.files_written:
        return None
    try:
        full_path = os.path.join(run_dir, result.files_written[0])
        with open(full_path, "r") as f:
            content = f.read()
    except OSError:
        return None
    if len(content) > MAX_DEPENDENCY_FILE_CHARS:
        content = content[:MAX_DEPENDENCY_FILE_CHARS] + "\n... (truncated)"
    return content


def build_dependency_context(
    item: WorkItem,
    results_by_id: dict[str, AgentResult],
    run_dir: str,
) -> dict[str, str]:
    """Build the {output_path: file_content} map for a WorkItem's
    dependencies, resolving from already-dispatched AgentResults.

    Missing/failed/unreadable dependencies are noted inline rather than
    omitted silently, and the total number of dependency files shown is
    capped at MAX_DEPENDENCY_FILES_SHOWN.
    """
    context: dict[str, str] = {}
    for dep_id in item.depends_on[:MAX_DEPENDENCY_FILES_SHOWN]:
        result = results_by_id.get(dep_id)
        if result is None:
            context[dep_id] = f"(dependency {dep_id} unavailable — not yet dispatched)"
            continue
        content = _read_dependency_content(dep_id, result, run_dir)
        if content is None:
            reason = "specialist failed" if not result.success else "file unreadable"
            context[dep_id] = f"(dependency {dep_id} unavailable — {reason})"
            continue
        output_path = result.files_written[0]
        context[output_path] = content
    return context


def dispatch_work_items(
    work_items: list[WorkItem],
    config: dict,
    run_dir: str,
    strict: bool = True,
    retry_context: dict[str, list[TestFailure]] | None = None,
    known_results: list[AgentResult] | None = None,
    progress: DispatchProgressCallback | None = None,
) -> list[AgentResult]:
    """
    Topologically sort work_items by depends_on, then dispatch each to
    its matching specialist agent sequentially.

    Args:
        work_items: The (possibly partial) batch of WorkItems to dispatch.
        config: Loaded config dict.
        run_dir: Path to this pipeline run's output directory (see
            pipeline/run_paths.py), passed through to each specialist.
        strict: Passed through to topological_sort — True (default) for a
            full/self-contained batch, False for a retry subset where
            depends_on may reference ids outside this batch.
        retry_context: Optional map of work_item_id -> TestFailures that
            implicated it, from MasterAgent.map_failures_to_work_items().
            None (default) for the initial/strict dispatch, which has no
            failure history. Passed through to each specialist's
            execute() so it can inform the LLM prompt on retries.
        known_results: Optional list of AgentResults from prior dispatch
            rounds (e.g. pipeline/runner.py's all_agent_results
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
        "logic": LogicAgent(),
        "ui": UIAgent(),
        "config": ConfigAgent(),
        "test": TestAgent(),
    }

    results_by_id: dict[str, AgentResult] = {
        r.work_item_id: r for r in (known_results or [])
    }

    total = len(ordered)
    results: list[AgentResult] = []
    for index, item in enumerate(ordered, start=1):
        if progress is not None:
            progress(item, index, total)
        agent = registry[item.type]
        retry_failures = (retry_context or {}).get(item.id, [])
        dependency_context = build_dependency_context(item, results_by_id, run_dir)
        result = agent.execute(
            item,
            config,
            run_dir,
            retry_failures=retry_failures,
            dependency_context=dependency_context,
        )
        results.append(result)
        results_by_id[item.id] = result

    return results
