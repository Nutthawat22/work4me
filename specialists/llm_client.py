"""
specialists/llm_client.py

Thin dispatcher used by all agents (Master, Design, and specialists) to
call an LLM. Resolves config["models"][role]["provider"] against the
specialists.providers.PROVIDERS registry and delegates the actual
request/response handling to the resolved adapter.
"""

from typing import Any, Optional

from specialists.providers import PROVIDERS


def call_llm(
    messages: list[dict],
    model: str,
    config: dict[str, Any],
    provider: str = "acp",
    timeout: int = 60,
    response_schema: Optional[dict[str, Any]] = None,
    session_scope: Optional[dict] = None,
) -> str:
    """
    Dispatch an LLM call to the adapter registered for `provider`.

    Args:
        messages: Full list of {"role": ..., "content": ...} messages to send.
        model: Model name string (e.g. config["models"]["design"]["model"]).
        config: Loaded config dict, must contain "litellm_url" and "litellm_key".
        provider: Key into specialists.providers.PROVIDERS selecting which
            adapter/API shape to use. "acp" is currently the only
            registered provider (see specialists/providers/__init__.py).
        timeout: Request timeout in seconds.
        response_schema: Optional dict with keys "name" and "schema" (a
            JSON Schema object) forwarded to the resolved adapter. ACP
            has no token-level schema enforcement -- this is only
            prepended as a best-effort prompt instruction (see
            acp_client.py's _build_schema_instruction). None (default)
            sends no schema instruction at all.
        session_scope: Optional dict of shape {"project": str, "role": str}
            identifying which pipeline run ("project") and config role
            this call belongs to. Forwarded to acp_call, which uses it
            to select the pooled ACP session (see acp_client.py's
            _role_key_for). None (default) falls back to acp_call's own
            (model, litellm_url) pooling key.

    Returns:
        The assistant's response content string, or a formatted error string
        on connection/timeout/other failure (see individual adapters).

    Raises:
        ValueError: if `provider` is not a registered provider.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider!r}. Valid providers: {sorted(PROVIDERS.keys())}")

    return PROVIDERS[provider](
        messages, model, config, timeout,
        response_schema=response_schema, session_scope=session_scope,
    )
