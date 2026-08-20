"""
specialists/test_writer.py

TestAgent: writes pytest tests that validate a WorkItem's acceptance
criteria.
"""

from pipeline.state import AgentResult, TestFailure, WorkItem
from specialists.base import run_specialist

SYSTEM_PROMPT = (
    "You are an expert in writing {test_framework} test suites for "
    "{language}. Write {test_framework} tests that validate every "
    "acceptance criterion of the following work item. Output ONLY the "
    "raw {language} test code — no markdown fences, no explanation."
)


def build_system_prompt(work_item: WorkItem, config: dict) -> str:
    """Fully format SYSTEM_PROMPT with both {language} and {test_framework}
    for this work item, using config["languages"][work_item.language] to
    look up the human-readable test framework name.

    Pre-formatting here (rather than relying on run_specialist's generic
    {language}-only formatting) is needed because this is the only
    specialist whose prompt needs a second variable. Passing the fully
    formatted string to run_specialist is safe: run_specialist's
    .format(language=...) call becomes a no-op once {language} is already
    filled in.
    """
    test_framework = config["languages"][work_item.language]["test_framework_name"]
    return SYSTEM_PROMPT.format(language=work_item.language, test_framework=test_framework)


class TestAgent:
    def execute(
        self,
        work_item: WorkItem,
        config: dict,
        run_dir: str,
        retry_failures: list[TestFailure] | None = None,
        dependency_context: dict[str, str] | None = None,
    ) -> AgentResult:
        system_prompt = build_system_prompt(work_item, config)
        return run_specialist(
            work_item, config, run_dir, system_prompt, "TestAgent", "tests_dir",
            retry_failures=retry_failures,
            dependency_context=dependency_context,
        )
