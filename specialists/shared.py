"""
specialists/shared.py

SharedAgent: implements a SHARED file that multiple features contribute to
(e.g. a router, an i18n bundle, a shared schema). The exact file(s) and the
merged requirements from every contributing feature are listed in the work
item description. Emits MULTIPLE files via run_multifile_specialist.
"""

from pipeline.state import AgentResult, TestFailure, WorkItem
from specialists.base import run_multifile_specialist

SYSTEM_PROMPT = (
    "You are implementing a SHARED file that multiple features contribute to "
    "(e.g. a router, an i18n bundle, a shared schema). The work item "
    "description lists the exact file(s) to create and the merged "
    "requirements from every contributing feature. Satisfy ALL of them in "
    "the single file. Output ONLY a JSON object "
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
            work_item, config, run_dir, SYSTEM_PROMPT, "SharedAgent", "output_dir",
            retry_failures=retry_failures,
            dependency_context=dependency_context,
        )
