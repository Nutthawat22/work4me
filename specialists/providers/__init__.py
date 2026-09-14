"""
specialists/providers/__init__.py

Registry of LLM provider adapters, keyed by the string used in
config["models"][role]["provider"]. All adapters share the call_llm
contract: (messages, model, config, timeout) -> str.
"""

from specialists.providers.acp_client import acp_call
from specialists.providers.chat_completions import chat_completions_call
from specialists.providers.responses_api import responses_call

PROVIDERS = {
    "chat_completions": chat_completions_call,
    "responses": responses_call,
    "acp": acp_call,
}
