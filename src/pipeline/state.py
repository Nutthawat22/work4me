"""
src/pipeline/state.py

Data structures shared across the agentic pipeline: WorkItem, AgentResult,
TestResult, TestFailure, ReviewResult, FailReport, PipelineState.

See docs/task-workflow.md for the active task model and run artifact contract.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


# ── Work Item ────────────────────────────────────────────────────────────────

TaskCategory = Literal["foundation", "feature", "shared", "integration", "validation"]
TaskStatus = Literal["blocked", "ready", "running", "passed", "failed"]


@dataclass
class OwnedFile:
    path: str
    language: str
    area: Literal["product", "tests"] = "product"


@dataclass
class TaskAttempt:
    number: int
    status: Literal["passed", "failed"]
    files_written: list[str] = field(default_factory=list)
    result_summary: str = ""
    failure_details: str = ""
    files_deleted: list[str] = field(default_factory=list)


@dataclass
class WorkItem:
    id: str
    category: TaskCategory
    language: str
    module: str
    scope: str
    title: str
    description: str
    acceptance_criteria: list[str]
    requirement_refs: list[str]
    owned_files: list[OwnedFile]
    depends_on: list[str] = field(default_factory=list)
    expected_interfaces: list[str] = field(default_factory=list)
    status: TaskStatus = "blocked"
    attempts: list[TaskAttempt] = field(default_factory=list)


# ── Agent Result ─────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    work_item_id: str
    agent_name: str
    success: bool
    files_written: list[str]         # relative paths
    notes: str = ""                  # agent commentary, warnings
    files_deleted: list[str] = field(default_factory=list)


# ── Test Result ──────────────────────────────────────────────────────────────

@dataclass
class TestFailure:
    test_name: str
    work_item_id: str                # which acceptance criterion failed
    error_output: str                # raw pytest stderr/stdout for this test


@dataclass
class TestResult:
    passed: bool
    total: int
    failed: int
    failures: list[TestFailure]


# ── Review Result ────────────────────────────────────────────────────────────

@dataclass
class ReviewResult:
    """Accept/reject verdict shape used by specialists/base.py's internal
    self-check loop (see run_multifile_specialist): after
    writing its own output, a specialist verifies it against the
    WorkItem's acceptance_criteria/dependency_context via a separate
    self-check LLM call, and this is that call's parsed response. Not a
    separate Master-owned review step — the whole self-check-and-retry
    loop lives inside one specialist's execute() call."""
    accepted: bool
    issues: list[str]
    reasoning: str


# ── Fail Report ──────────────────────────────────────────────────────────────

@dataclass
class FailReport:
    run_number: int
    failures: list[TestFailure]
    unresolved_items: list[str]      # WorkItem ids that still fail after max retries
    summary: str                     # human-readable produced by MasterAgent


# ── Feature ──────────────────────────────────────────────────────────────────

@dataclass
class Feature:
    """Feature intent used as input to the optional file-manifest strategy.
    Unlike a dispatchable WorkItem, a Feature does not name output files."""
    id: str                          # e.g. "FEAT-auth"
    title: str
    description: str
    acceptance_criteria: list[str]
    language: str
    source_ids: list[str] = field(default_factory=list)   # REQ-*/AC-*/TEST-* traceability
    depends_on: list[str] = field(default_factory=list)    # other Feature ids


# ── File Spec ────────────────────────────────────────────────────────────────

FileRole = Literal["scaffold", "feature", "shared", "entrypoint", "validation"]


@dataclass
class FileSpec:
    """One file in the file-indexed plan. Each physical path appears in a
    FileManifest exactly once, so every file has exactly one author — this
    is the structural fix for feature-vs-feature write collisions on shared
    files (routers, i18n bundles, schema)."""
    path: str                              # e.g. "src/server/routes/auth.ts"
    role: FileRole
    language: str
    contributing_features: list[str]       # Feature ids that need something in this file
    requirements: list[str] = field(default_factory=list)     # merged, file-scoped requirement bullets
    depends_on_files: list[str] = field(default_factory=list)  # other FileSpec.path values this imports
    interfaces: list[str] = field(default_factory=list)        # contracts this file exposes or consumes


# ── File Manifest ────────────────────────────────────────────────────────────

@dataclass
class FileManifest:
    """The file-indexed plan produced by plan_file_manifest(): the complete
    set of files the app needs, each with the merged requirements it owes
    across all contributing features."""
    files: list[FileSpec] = field(default_factory=list)


# ── File Group ───────────────────────────────────────────────────────────────

@dataclass
class FileGroup:
    """A cohesion cluster of FileSpecs that should be authored together by
    one multi-file specialist task. Produced by group_into_work_items() and
    adapted into the shared WorkItem model by file_groups_to_work_items()."""
    id: str                          # stable architecture/task id
    role: FileRole
    language: str
    module: str
    title: str
    files: list[str]                 # FileSpec.path values in this group
    contributing_features: list[str]
    requirements: list[str] = field(default_factory=list)
    interfaces: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)   # other FileGroup ids


# ── Pipeline State ───────────────────────────────────────────────────────────

@dataclass
class PipelineState:
    user_prompt: str
    intent: str                      # MasterAgent's parsed intent
    work_items: list[WorkItem] = field(default_factory=list)
    agent_results: list[AgentResult] = field(default_factory=list)
    test_results: list[TestResult] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    status: Literal["running", "passed", "failed"] = "running"
