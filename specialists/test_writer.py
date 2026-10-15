"""ValidationAgent: writes task-owned automated validation assets."""

from pipeline.state import AgentResult, TestFailure, WorkItem
from specialists.base import run_multifile_specialist

SYSTEM_PROMPT = (
    "You implement the validation category for the configured test framework "
    "and each owned file's language. Write tests for the stated acceptance "
    "criteria and dependency contracts. Create exactly the owned test files. "
    'Output ONLY {"files": {"relative/path": "complete test content"}}.'
)


def build_system_prompt(work_item: WorkItem, config: dict) -> str:
    frameworks = sorted({
        config["languages"][owned_file.language]["test_framework_name"]
        for owned_file in work_item.owned_files
        if owned_file.language in config["languages"]
    })
    return SYSTEM_PROMPT.replace("configured test framework", ", ".join(frameworks) or "configured test framework")


class ValidationAgent:
    def execute(
        self,
        work_item: WorkItem,
        config: dict,
        run_dir: str,
        retry_failures: list[TestFailure] | None = None,
        dependency_context: dict[str, str] | None = None,
    ) -> AgentResult:
        system_prompt = build_system_prompt(work_item, config)
        return run_multifile_specialist(
            work_item, config, run_dir, system_prompt, "ValidationAgent",
            retry_failures=retry_failures,
            dependency_context=dependency_context,
        )
