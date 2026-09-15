"""
tests/specialists/test_base_self_check.py

Tests for specialists/base.py's self-check loop (run_specialist /
run_multifile_specialist verifying their OWN output against the
WorkItem's acceptance_criteria/dependency_context via a separate
self-check LLM call, regenerating on rejection up to
SELF_CHECK_MAX_ATTEMPTS extra times). No real LLM/subprocess calls:
monkeypatches specialists.base.call_llm directly with a canned sequence
of responses (generation, self-check, generation, self-check, ...),
mirroring tests/pipeline/test_master_decompose.py's
acp_call_with_retry-monkeypatching style.
"""

import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from specialists.base import (
    SELF_CHECK_MAX_ATTEMPTS,
    run_multifile_specialist,
    run_specialist,
)
from pipeline.state import WorkItem


def _config() -> dict:
    return {
        "models": {"specialist": {"model": "m", "provider": "acp"}},
        "pipeline": {"output_dir": "output", "tests_dir": "tests"},
    }


def _work_item() -> WorkItem:
    return WorkItem(
        id="WI-001",
        type="logic",
        language="python",
        title="Add two numbers",
        description="Implement add(a, b)",
        acceptance_criteria=["add(2, 3) == 5"],
        output_path="add.py",
        depends_on=[],
    )


def _accept_json(reasoning="looks good") -> str:
    return json.dumps({"accepted": True, "issues": [], "reasoning": reasoning})


def _reject_json(issue="missing something") -> str:
    return json.dumps({"accepted": False, "issues": [issue], "reasoning": issue})


def _queue_call_llm(monkeypatch, responses: list[str]):
    """Monkeypatch specialists.base.call_llm to return each of `responses`
    in order across successive calls (generation call, self-check call,
    generation call, self-check call, ...). Records every call's
    messages list for later inspection."""
    calls = []
    remaining = list(responses)

    def fake_call_llm(messages, model, config, provider="acp", timeout=60,
                       response_schema=None, session_scope=None):
        calls.append({"messages": messages, "response_schema": response_schema})
        return remaining.pop(0)

    monkeypatch.setattr("specialists.base.call_llm", fake_call_llm)
    return calls


class TestSelfCheckAcceptsFirstTry:
    def test_accepted_on_first_attempt_does_not_regenerate(self, monkeypatch, tmp_path):
        calls = _queue_call_llm(monkeypatch, [
            "def add(a, b):\n    return a + b\n",
            _accept_json(),
        ])

        result = run_specialist(
            _work_item(), _config(), str(tmp_path),
            "You are an expert {language} engineer.", "LogicAgent", "output_dir",
        )

        assert len(calls) == 2  # one generation, one self-check — no regeneration
        assert result.success is True
        assert result.files_written == ["output/add.py"]
        assert result.notes == ""

        written = (tmp_path / "output" / "add.py").read_text()
        assert "def add" in written

    def test_multifile_accepted_on_first_attempt_does_not_regenerate(self, monkeypatch, tmp_path):
        calls = _queue_call_llm(monkeypatch, [
            json.dumps({"files": {"add.py": "def add(a, b):\n    return a + b\n"}}),
            _accept_json(),
        ])

        result = run_multifile_specialist(
            _work_item(), _config(), str(tmp_path),
            "You are an expert {language} engineer.", "FeatureAgent", "output_dir",
        )

        assert len(calls) == 2
        assert result.success is True
        assert result.files_written == ["output/add.py"]


class TestSelfCheckRejectsThenAccepts:
    def test_rejected_then_accepted_regenerates_once(self, monkeypatch, tmp_path):
        calls = _queue_call_llm(monkeypatch, [
            "def add(a, b):\n    return a - b\n",   # v1: wrong implementation
            _reject_json("subtracts instead of adding"),
            "def add(a, b):\n    return a + b\n",   # v2: fixed
            _accept_json(),
        ])

        result = run_specialist(
            _work_item(), _config(), str(tmp_path),
            "You are an expert {language} engineer.", "LogicAgent", "output_dir",
        )

        assert len(calls) == 4
        assert result.success is True

        written = (tmp_path / "output" / "add.py").read_text()
        assert "a + b" in written

        # The second generation call's user prompt should carry the
        # self-check's rejection feedback.
        second_gen_user_content = calls[2]["messages"][1]["content"]
        assert "subtracts instead of adding" in second_gen_user_content


class TestSelfCheckExhaustsAttempts:
    def test_always_rejected_exhausts_attempts_and_fails(self, monkeypatch, tmp_path):
        # SELF_CHECK_MAX_ATTEMPTS extra attempts beyond the first -> that
        # many + 1 total generation+self-check pairs, all rejecting.
        total_attempts = SELF_CHECK_MAX_ATTEMPTS + 1
        responses = []
        for i in range(total_attempts):
            responses.append(f"def add(a, b):\n    return {i}\n")
            responses.append(_reject_json(f"issue {i}"))

        calls = _queue_call_llm(monkeypatch, responses)

        result = run_specialist(
            _work_item(), _config(), str(tmp_path),
            "You are an expert {language} engineer.", "LogicAgent", "output_dir",
        )

        assert len(calls) == total_attempts * 2
        assert result.success is False
        assert result.files_written == []
        assert f"issue {total_attempts - 1}" in result.notes
        assert "Self-check rejected" in result.notes


class TestSelfCheckFailsOpenOnBrokenCheckCall:
    def test_self_check_llm_error_does_not_block_success(self, monkeypatch, tmp_path):
        calls = _queue_call_llm(monkeypatch, [
            "def add(a, b):\n    return a + b\n",
            "[m] Error: ACP prompt timed out after 60s.",
        ])

        result = run_specialist(
            _work_item(), _config(), str(tmp_path),
            "You are an expert {language} engineer.", "LogicAgent", "output_dir",
        )

        assert len(calls) == 2
        assert result.success is True  # fail-open: broken self-check doesn't block
