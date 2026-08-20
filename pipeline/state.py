"""
pipeline/state.py

Data structures shared across the agentic pipeline: WorkItem, AgentResult,
TestResult, TestFailure, FailReport, PipelineState.

See ~/dev-plans/agents/features/2026-08-10-agentic-pipeline.md for the
full design spec these are derived from.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


# ── Work Item ────────────────────────────────────────────────────────────────

WorkItemType = Literal["logic", "ui", "config", "test"]


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


# ── Fail Report ──────────────────────────────────────────────────────────────

@dataclass
class FailReport:
    run_number: int
    failures: list[TestFailure]
    unresolved_items: list[str]      # WorkItem ids that still fail after max retries
    summary: str                     # human-readable produced by MasterAgent


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
