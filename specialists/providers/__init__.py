"""
specialists/providers/__init__.py

Registry of LLM provider adapters, keyed by the string used in
config["models"][role]["provider"]. All adapters share the call_llm
contract: (messages, model, config, timeout) -> str.

ACP is the sole provider (see the ACP-native rearchitecture design doc,
Phase 1) — chat_completions and responses adapters have been removed.
"""

from specialists.providers.acp_client import acp_call

PROVIDERS = {
    "acp": acp_call,
}
