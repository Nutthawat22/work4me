"""
src/specialists/foundation.py

FoundationAgent: implements project setup and primitives shared across modules.
"""

from pipeline.state import AgentResult, ValidationIssue, WorkItem
from specialists.base import run_multifile_specialist

SYSTEM_PROMPT = (
    "You implement the foundation category: project setup and shared primitives "
    "that multiple modules depend on. Follow the owned-file languages, implement "
    "only this task's scope, and state the real interfaces described by the task. "
    'Output ONLY a JSON object of the exact shape {"files": {"<relative '
    'path>": "<full file content>", ...}} — no markdown fences, no '
    "explanation. Every value is the complete raw file content."
)


class FoundationAgent:
    def execute(
        self,
        work_item: WorkItem,
        config: dict,
        run_dir: str,
        retry_failures: list[ValidationIssue] | None = None,
        dependency_context: dict[str, str] | None = None,
    ) -> AgentResult:
        return run_multifile_specialist(
            work_item, config, run_dir, SYSTEM_PROMPT, "FoundationAgent",
            retry_failures=retry_failures,
            dependency_context=dependency_context,
        )
