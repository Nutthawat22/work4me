"""
specialists/config.py

ConfigAgent: implements configuration / env / infra WorkItems.
"""

from pipeline.state import AgentResult, TestFailure, WorkItem
from specialists.base import run_specialist

SYSTEM_PROMPT = (
    "You are an expert in {language} configuration, environment, and "
    "infrastructure files. Write a correct, production-quality config/env/"
    "infra file implementing the following work item. Output ONLY the raw "
    "file content — no markdown fences, no explanation."
)


class ConfigAgent:
    def execute(
        self,
        work_item: WorkItem,
        config: dict,
        run_dir: str,
        retry_failures: list[TestFailure] | None = None,
        dependency_context: dict[str, str] | None = None,
    ) -> AgentResult:
        return run_specialist(
            work_item, config, run_dir, SYSTEM_PROMPT, "ConfigAgent", "output_dir",
            retry_failures=retry_failures,
            dependency_context=dependency_context,
        )
