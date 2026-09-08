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
    provider: str = "chat_completions",
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
            adapter/API shape to use.
        timeout: Request timeout in seconds.
        response_schema: Optional dict with keys "name" and "schema" (a
            JSON Schema object) forwarded to the resolved adapter to
            request structured-output enforcement (the model's token
            generation is constrained to match the schema, eliminating
            malformed/missing-field JSON as a failure mode rather than
            just discouraging it via prompt text). None (default) — no
            adapters currently in PROVIDERS require this, and passing
            None preserves prior behavior exactly.
        session_scope: Optional dict of shape {"project": str, "role": str}
            identifying which pipeline run ("project") and config role
            this call belongs to. Only meaningful to session-based
            adapters (currently just specialists/providers/acp_client.py's
            acp_call, not yet registered in PROVIDERS — see the ACP
            migration design doc); stateless HTTP adapters
            (chat_completions_call, responses_call) accept and ignore it.
            None (default) — callers that don't have a project identifier
            available simply omit it, and session-based adapters fall
            back to their pre-existing pooling behavior.

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
