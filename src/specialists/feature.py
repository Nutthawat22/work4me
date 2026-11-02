"""
src/specialists/feature.py

FeatureAgent: implements one cohesive capability over its explicitly owned
set of related files.
"""

from pipeline.state import AgentResult, ValidationIssue, WorkItem
from specialists.base import run_multifile_specialist

SYSTEM_PROMPT = (
    "You are implementing one cohesive end-user capability, including related "
    "frontend and backend modules when the task owns them. Use each owned file's "
    "declared language, respect dependency contracts, and do not split feature "
    "ownership or invent additional files. Implement only the explicitly requested "
    "owned file per response; the complete feature context is supplied each time."
)


class FeatureAgent:
    def execute(
        self,
        work_item: WorkItem,
        config: dict,
        run_dir: str,
        retry_failures: list[ValidationIssue] | None = None,
        dependency_context: dict[str, str] | None = None,
    ) -> AgentResult:
        return run_multifile_specialist(
            work_item, config, run_dir, SYSTEM_PROMPT, "FeatureAgent",
            retry_failures=retry_failures,
            dependency_context=dependency_context,
        )
