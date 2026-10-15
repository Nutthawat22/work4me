"""
specialists/shared.py

SharedAgent: implements shared modules and contracts with one task owner.
"""

from pipeline.state import AgentResult, TestFailure, WorkItem
from specialists.base import run_multifile_specialist

SYSTEM_PROMPT = (
    "You implement a shared module or contract used by multiple features. "
    "Satisfy the combined requirements and preserve all stated interfaces. "
    "Use the complete dependency context and do not take ownership of files "
    "outside this task. Output ONLY a JSON object "
    '{"files": {"<relative path>": "<full file content>"}} — no fences, no '
    "explanation."
)


class SharedAgent:
    def execute(
        self,
        work_item: WorkItem,
        config: dict,
        run_dir: str,
        retry_failures: list[TestFailure] | None = None,
        dependency_context: dict[str, str] | None = None,
    ) -> AgentResult:
        return run_multifile_specialist(
            work_item, config, run_dir, SYSTEM_PROMPT, "SharedAgent",
            retry_failures=retry_failures,
            dependency_context=dependency_context,
        )
