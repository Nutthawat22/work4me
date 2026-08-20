"""
specialists/llm_client.py

Thin dispatcher used by all agents (Master, Design, and specialists) to
call an LLM. Resolves config["models"][role]["provider"] against the
specialists.providers.PROVIDERS registry and delegates the actual
request/response handling to the resolved adapter.
"""

from typing import Any

from specialists.providers import PROVIDERS


def call_llm(
    messages: list[dict],
    model: str,
    config: dict[str, Any],
    provider: str = "chat_completions",
    timeout: int = 60,
) -> str:
    """
    Dispatch an LLM call to the adapter registered for `provider`.

    Args:
        messages: Full list of {"role": ..., "content": ...} messages to send.
        model: Model name string (e.g. config["models"]["design"]["model"]).
        config: Loaded config dict, must contain "litellm_url" and "litellm_key".
        provider: Key into specialists.providers.PROVIDERS selecting which
            adapter/API shape to use.
        timeout: Request timeout in seconds.

    Returns:
        The assistant's response content string, or a formatted error string
        on connection/timeout/other failure (see individual adapters).

    Raises:
        ValueError: if `provider` is not a registered provider.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider!r}. Valid providers: {sorted(PROVIDERS.keys())}")

    return PROVIDERS[provider](messages, model, config, timeout)
