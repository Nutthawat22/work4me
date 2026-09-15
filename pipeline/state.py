"""
pipeline/state.py

Data structures shared across the agentic pipeline: WorkItem, AgentResult,
TestResult, TestFailure, ReviewResult, FailReport, PipelineState.

See ~/dev-plans/agents/features/2026-08-10-agentic-pipeline.md for the
full design spec these are derived from.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


# ── Work Item ────────────────────────────────────────────────────────────────

WorkItemType = Literal["logic", "ui", "config", "test", "scaffold", "integrate", "feature", "shared"]


@dataclass
class WorkItem:
    id: str                          # e.g. "WI-001"
    type: WorkItemType               # routes to the correct specialist
    language: str                    # e.g. "python" — must match a key in config's "languages"
    title: str                       # one-line description
    description: str                 # full task spec
    acceptance_criteria: list[str]   # bullet list — used by TestAgent & TestRunner
    output_path: str                 # relative path where agent writes its file(s)
    depends_on: list[str] = field(default_factory=list)  # other WorkItem ids


# ── Agent Result ─────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    work_item_id: str
    agent_name: str
    success: bool
    files_written: list[str]         # relative paths
    notes: str = ""                  # agent commentary, warnings


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
    self-check loop (see run_specialist/run_multifile_specialist): after
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
    """A feature-level unit of work (the WHAT), emitted by a future
    feature-oriented decomposition stage. Unlike WorkItem, a Feature does
    NOT name files — it describes intent only. The file layout is decided
    later by plan_file_manifest()."""
    id: str                          # e.g. "FEAT-auth"
    title: str
    description: str
    acceptance_criteria: list[str]
    language: str
    source_ids: list[str] = field(default_factory=list)   # REQ-*/AC-*/TEST-* traceability
    depends_on: list[str] = field(default_factory=list)    # other Feature ids


# ── File Spec ────────────────────────────────────────────────────────────────

FileRole = Literal["scaffold", "feature", "shared", "entrypoint"]


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
    one multi-file specialist call. Produced by group_into_work_items().
    This is a planning artifact only — it is intentionally NOT the live
    WorkItem type (that stays untouched); a future wiring step will adapt
    FileGroups into dispatchable work."""
    id: str                          # e.g. "GRP-feature-auth"
    role: FileRole
    language: str
    title: str
    files: list[str]                 # FileSpec.path values in this group
    contributing_features: list[str]
    requirements: list[str] = field(default_factory=list)
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
