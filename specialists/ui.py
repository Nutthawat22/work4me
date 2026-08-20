"""
specialists/ui.py

UIAgent: implements frontend / interface WorkItems.
"""

from pipeline.state import AgentResult, TestFailure, WorkItem
from specialists.base import run_specialist

SYSTEM_PROMPT = (
    "You are an expert {language} frontend engineer. Write clean, correct, "
    "production-quality {language} UI code implementing the following work "
    "item. Output ONLY the raw code for the file — no markdown fences, no "
    "explanation."
)


class UIAgent:
    def execute(
        self,
        work_item: WorkItem,
        config: dict,
        run_dir: str,
        retry_failures: list[TestFailure] | None = None,
        dependency_context: dict[str, str] | None = None,
    ) -> AgentResult:
        return run_specialist(
            work_item, config, run_dir, SYSTEM_PROMPT, "UIAgent", "output_dir",
            retry_failures=retry_failures,
            dependency_context=dependency_context,
        )
