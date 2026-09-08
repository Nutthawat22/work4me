"""
specialists/providers/chat_completions.py

Chat Completions API adapter. Standardized on the /chat/completions shape
since it's the more common LiteLLM proxy interface.
"""

import requests
from typing import Any, Optional


def chat_completions_call(
    messages: list[dict],
    model: str,
    config: dict[str, Any],
    timeout: int = 60,
    response_schema: Optional[dict[str, Any]] = None,
    session_scope: Optional[dict] = None,
) -> str:
    """
    Send a chat-completions request to the configured LiteLLM proxy.

    Args:
        messages: Full list of {"role": ..., "content": ...} messages to send.
        model: Model name string (e.g. config["models"]["design"]["model"]).
        config: Loaded config dict, must contain "litellm_url" and "litellm_key".
        timeout: Request timeout in seconds.
        response_schema: Optional dict with keys "name" and "schema" (a JSON
            Schema object). When given, sent as the Chat Completions API's
            `response_format` with `type: "json_schema"` and `strict: true`
            (OpenAI-compatible shape) — same structural-enforcement intent
            as specialists.providers.responses_api.responses_call's
            response_schema, kept symmetric across both adapters even
            though today's config only routes through the "responses"
            provider. None (default) sends no format constraint.
        session_scope: Accepted-but-ignored. This is a stateless HTTP call
            with no session/conversation concept — the kwarg exists purely
            so llm_client.py can call every registered provider with the
            same signature (see specialists/providers/acp_client.py, the
            only adapter that actually uses it).

    Returns:
        The assistant's response content string, or a formatted error string
        on connection/timeout/other failure.
    """
    url = f"{config['litellm_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config['litellm_key']}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if response_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_schema["name"],
                "strict": True,
                "schema": response_schema["schema"],
            },
        }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        return f"[{model}] Error: could not connect to LiteLLM proxy at {config['litellm_url']}."
    except requests.exceptions.Timeout:
        return f"[{model}] Error: request timed out."
    except Exception as e:
        return f"[{model}] Error: {e}"
