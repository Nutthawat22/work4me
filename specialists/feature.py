"""
specialists/feature.py

FeatureAgent: implements one cohesive capability over its explicitly owned
set of related files.
"""

from pipeline.state import AgentResult, TestFailure, WorkItem
from specialists.base import run_multifile_specialist

SYSTEM_PROMPT = (
    "You are implementing one cohesive end-user capability, including related "
    "frontend and backend modules when the task owns them. Use each owned file's "
    "declared language, respect dependency contracts, and do not split work by "
    "file or invent additional files. Output ONLY a JSON object "
    '{"files": {"<relative path>": "<full file content>", ...}} — no '
    "markdown fences, no explanation."
)


class FeatureAgent:
    def execute(
        self,
        work_item: WorkItem,
        config: dict,
        run_dir: str,
        retry_failures: list[TestFailure] | None = None,
        dependency_context: dict[str, str] | None = None,
    ) -> AgentResult:
        return run_multifile_specialist(
            work_item, config, run_dir, SYSTEM_PROMPT, "FeatureAgent",
            retry_failures=retry_failures,
            dependency_context=dependency_context,
        )
