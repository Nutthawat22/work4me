"""
specialists/integrate.py

IntegrationAgent: runs LAST and emits the composition/wiring files that
assemble all already-implemented feature modules into a runnable
application (server entry point, frontend app shell, aggregation/index
files). Emits MULTIPLE files via run_multifile_specialist.
"""

from pipeline.state import AgentResult, TestFailure, WorkItem
from specialists.base import run_multifile_specialist

SYSTEM_PROMPT = (
    "You are an expert {language} integration engineer. All feature modules "
    "have already been implemented (their code is shown to you as dependency "
    "context). Write ONLY the composition/wiring files that assemble them "
    "into a runnable application the user can start end-to-end: the server "
    "entry point that imports and mounts every route/module, the frontend "
    "application shell that mounts every screen, and any aggregation/index "
    "files needed. Import the ACTUAL exported names/signatures from the "
    "dependency files shown — do not invent different ones. Do NOT "
    "re-implement feature logic. Output ONLY a JSON object of the exact "
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
            work_item, config, run_dir, SYSTEM_PROMPT, "IntegrationAgent", "output_dir",
            retry_failures=retry_failures,
            dependency_context=dependency_context,
        )
