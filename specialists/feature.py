"""
specialists/feature.py

FeatureAgent: implements ONE cohesive feature as a small set of related
files. The exact file paths and per-file requirements are listed in the
work item description (built by file_groups_to_work_items). Emits MULTIPLE
files via run_multifile_specialist.
"""

from pipeline.state import AgentResult, TestFailure, WorkItem
from specialists.base import run_multifile_specialist

SYSTEM_PROMPT = (
    "You are an expert {language} engineer implementing ONE cohesive feature "
    "as a small set of related files. The exact file paths you must create "
    "and each file's requirements are listed in the work item description. "
    "Create EXACTLY those files — no more, no fewer — using the shown "
    "dependency files' real exports. Output ONLY a JSON object "
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
            work_item, config, run_dir, SYSTEM_PROMPT, "FeatureAgent", "output_dir",
            retry_failures=retry_failures,
            dependency_context=dependency_context,
        )
