"""
specialists/logic.py

LogicAgent: implements business logic / core functionality WorkItems.
"""

from pipeline.state import AgentResult, TestFailure, WorkItem
from specialists.base import run_specialist

SYSTEM_PROMPT = (
    "You are an expert {language} software engineer. Write clean, correct, "
    "production-quality {language} code implementing the following work "
    "item. Output ONLY the raw code for the file — no markdown fences, no "
    "explanation."
)


class LogicAgent:
    def execute(
        self,
        work_item: WorkItem,
        config: dict,
        run_dir: str,
        retry_failures: list[TestFailure] | None = None,
        dependency_context: dict[str, str] | None = None,
    ) -> AgentResult:
        return run_specialist(
            work_item, config, run_dir, SYSTEM_PROMPT, "LogicAgent", "output_dir",
            retry_failures=retry_failures,
            dependency_context=dependency_context,
        )
