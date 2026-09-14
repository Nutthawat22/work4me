"""
tests/specialists/providers/test_providers_registry.py

Tests for specialists/providers/__init__.py's PROVIDERS registry and its
dispatch from specialists/llm_client.py::call_llm. No real subprocess is
spawned — the "acp" entry is verified via a stubbed acp_call (patched
onto the PROVIDERS dict itself), not a live opencode acp process.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO_ROOT)

from specialists.providers import PROVIDERS
from specialists.providers.acp_client import acp_call
from specialists.llm_client import call_llm


def test_providers_registry_has_acp_entry_pointing_at_acp_call():
    assert PROVIDERS["acp"] is acp_call


def test_call_llm_dispatches_acp_provider_to_acp_call(monkeypatch):
    calls = []

    def fake_acp_call(messages, model, config, timeout=60, response_schema=None, session_scope=None):
        calls.append((messages, model, config, timeout, response_schema, session_scope))
        return "stubbed acp response"

    monkeypatch.setitem(PROVIDERS, "acp", fake_acp_call)

    result = call_llm(
        [{"role": "user", "content": "hi"}],
        "claude-sonnet-5",
        {"litellm_url": "https://litellm.example.test/v1", "litellm_key": "sk-test"},
        provider="acp",
        timeout=5,
    )

    assert result == "stubbed acp response"
    assert len(calls) == 1
    assert calls[0][1] == "claude-sonnet-5"
