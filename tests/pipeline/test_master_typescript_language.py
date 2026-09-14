"""
tests/pipeline/test_master_typescript_language.py

HO-4 known-issue verification (no live LLM calls): confirms that once
"typescript" is registered in config["languages"] (as this handoff adds
to config.example.json), MasterAgent._validate_and_build resolves
WorkItems with language="typescript" successfully, and that
MasterAgent.plan_handoff's system prompt correctly lists "typescript" as
a valid language once it's present in the config.

Prior to this handoff, MasterAgent.plan_handoff() (then
DesignAgent.decompose_handoff()) would raise MasterPlanError (then
DesignParseError) for language="typescript" because config.example.json
had no such entry (see pipeline/master.py module docstring / HO-2's
documented known issue). This test hand-crafts an LLM JSON response and
feeds it directly to _validate_and_build — it never calls the real LLM.
"""

import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline.master import MasterAgent, MasterPlanError, PLAN_HANDOFF_SYSTEM_PROMPT_TEMPLATE


def load_real_example_config() -> dict:
    config_path = os.path.join(REPO_ROOT, "config.example.json")
    with open(config_path, "r") as f:
        return json.load(f)


MOCK_HANDOFF_LLM_RESPONSE = [
    {
        "id": "WI-001",
        "type": "logic",
        "language": "typescript",
        "title": "[WP-002] Request draft API",
        "description": (
            "Source IDs: REQ-003 | Test IDs: TEST-001 | "
            "Completion criteria: endpoint returns 201 on valid draft request"
        ),
        "acceptance_criteria": ["POST /drafts returns 201 with a draft id"],
        "output_path": "src/api/drafts.ts",
        "depends_on": [],
    },
    {
        "id": "WI-002",
        "type": "test",
        "language": "typescript",
        "title": "[WP-002] Request draft API tests",
        "description": (
            "Source IDs: REQ-003 | Test IDs: TEST-001 | "
            "Completion criteria: endpoint returns 201 on valid draft request"
        ),
        "acceptance_criteria": ["POST /drafts returns 201 with a draft id"],
        "output_path": "tests/api/drafts.test.ts",
        "depends_on": ["WI-001"],
    },
]


class TestTypescriptResolvesAgainstUpdatedConfig:
    def test_config_example_now_has_typescript_language_entry(self):
        config = load_real_example_config()
        assert "typescript" in config["languages"], (
            "config.example.json must register a 'typescript' languages "
            "entry for this handoff (HO-4) to close the known issue from "
            "HO-2/HO-3."
        )

    def test_validate_and_build_accepts_language_typescript(self):
        config = load_real_example_config()
        valid_languages = set(config["languages"].keys())

        work_items = MasterAgent._validate_and_build(
            MOCK_HANDOFF_LLM_RESPONSE, valid_languages
        )

        assert len(work_items) == 2
        assert all(item.language == "typescript" for item in work_items)

    def test_validate_and_build_rejects_typescript_without_config_entry(self):
        """
        Regression guard for the exact HO-2/HO-3 known issue: without a
        'typescript' entry in valid_languages, language='typescript' must
        still hard-fail (not silently downgrade to 'javascript' or
        anything else) — that silent-downgrade behavior was observed in
        the live LLM, never in this validation function itself, but this
        confirms _validate_and_build's enforcement is real, not a no-op.
        """
        valid_languages = {"python", "javascript"}  # pre-HO-4 config shape

        with pytest.raises(MasterPlanError, match="invalid language 'typescript'"):
            MasterAgent._validate_and_build(MOCK_HANDOFF_LLM_RESPONSE, valid_languages)


class TestHandoffSystemPromptListsTypescript:
    def test_system_prompt_lists_typescript_when_present_in_config(self):
        config = load_real_example_config()
        valid_languages = set(config["languages"].keys())

        system_prompt = PLAN_HANDOFF_SYSTEM_PROMPT_TEMPLATE.format(
            valid_languages=", ".join(sorted(valid_languages))
        )

        assert "typescript" in system_prompt
        # Distinctness check for the known js/ts LLM-name-confusion issue:
        # both language names must appear as their own tokens in the
        # rendered valid-language list (not just as a substring of one
        # another via naive matching).
        rendered_list = ", ".join(sorted(valid_languages))
        assert "javascript" in rendered_list.split(", ")
        assert "typescript" in rendered_list.split(", ")
