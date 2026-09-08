"""
specialists/scaffold.py

ScaffoldAgent: runs FIRST and emits the project skeleton + shared
foundation (dependency manifest, build/compiler config, .gitignore, and
shared foundation modules other feature code imports). Emits MULTIPLE
files via run_multifile_specialist.
"""

from pipeline.state import AgentResult, TestFailure, WorkItem
from specialists.base import run_multifile_specialist

SYSTEM_PROMPT = (
    "You are an expert {language} build/tooling engineer. Produce the "
    "project scaffolding and SHARED FOUNDATION for a runnable application: "
    "dependency manifest (e.g. package.json), compiler/build config (e.g. "
    "tsconfig.json, vite config), .gitignore, and shared foundation modules "
    "the feature code will import (e.g. database setup, migrations, shared "
    "config loader, shared types). Do NOT implement feature/business logic "
    "— only the skeleton and shared primitives other modules depend on. "
    'Output ONLY a JSON object of the exact shape {"files": {"<relative '
    'path>": "<full file content>", ...}} — no markdown fences, no '
    "explanation. Every value is the complete raw file content."
)


class ScaffoldAgent:
    def execute(
        self,
        work_item: WorkItem,
        config: dict,
        run_dir: str,
        retry_failures: list[TestFailure] | None = None,
        dependency_context: dict[str, str] | None = None,
    ) -> AgentResult:
        return run_multifile_specialist(
            work_item, config, run_dir, SYSTEM_PROMPT, "ScaffoldAgent", "output_dir",
            retry_failures=retry_failures,
            dependency_context=dependency_context,
        )
