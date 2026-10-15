"""
specialists/integrate.py

IntegrationAgent: implements composition and wiring assets for a runnable
application, using declared dependencies and owned files.
"""

from pipeline.state import AgentResult, TestFailure, WorkItem
from specialists.base import run_multifile_specialist

SYSTEM_PROMPT = (
    "You implement application composition and integration for this task. "
    "Use actual exports and contracts from all dependency files shown, wire "
    "only the owned entry points/aggregation modules, and do not reimplement "
    "feature logic or create unowned files. Output ONLY a JSON object of the exact "
    'shape {"files": {"<relative path>": "<full file content>", ...}} — no '
    "markdown fences, no explanation. Every value is the complete raw file "
    "content."
)


class IntegrationAgent:
    def execute(
        self,
        work_item: WorkItem,
        config: dict,
        run_dir: str,
        retry_failures: list[TestFailure] | None = None,
        dependency_context: dict[str, str] | None = None,
    ) -> AgentResult:
        return run_multifile_specialist(
            work_item, config, run_dir, SYSTEM_PROMPT, "IntegrationAgent",
            retry_failures=retry_failures,
            dependency_context=dependency_context,
        )
