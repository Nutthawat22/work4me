"""
specialists/providers/responses_api.py

OpenAI Responses API adapter. Used by models served through the
Responses API shape (e.g. GPT 5.6 Luna via OpenCode Go).
"""

import requests
from typing import Any, Optional


def responses_call(
    messages: list[dict],
    model: str,
    config: dict[str, Any],
    timeout: int = 60,
    response_schema: Optional[dict[str, Any]] = None,
) -> str:
    """
    Send a Responses API request to the configured LiteLLM proxy.

    Args:
        messages: Full list of {"role": ..., "content": ...} messages to send.
            Passed through as the Responses API "input" array (EasyInputMessage
            shape: role + plain string content).
        model: Model name string (e.g. config["models"]["design"]["model"]).
        config: Loaded config dict, must contain "litellm_url" and "litellm_key".
        timeout: Request timeout in seconds.
        response_schema: Optional dict with keys "name" and "schema" (a JSON
            Schema object). When given, sent as the Responses API's
            `text.format` with `type: "json_schema"` and `strict: true`,
            which constrains the model's actual token generation to match
            the schema — this is a hard structural guarantee, not a prompt
            hint, and is used by callers (e.g. DesignAgent) to eliminate
            the "LLM drops a required field somewhere in a large JSON
            array" failure mode entirely rather than just reducing its
            likelihood. None (default) sends no format constraint, same
            as before this parameter existed.

    Returns:
        The assistant's response content string, or a formatted error string
        on connection/timeout/other failure.
    """
    url = f"{config['litellm_url']}/responses"
    headers = {
        "Authorization": f"Bearer {config['litellm_key']}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {"model": model, "input": messages}
    if response_schema is not None:
        payload["text"] = {
            "format": {
                "type": "json_schema",
                "name": response_schema["name"],
                "strict": True,
                "schema": response_schema["schema"],
            }
        }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        text_parts = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                for content_item in item.get("content", []):
                    if content_item.get("type") == "output_text":
                        text_parts.append(content_item.get("text", ""))
        if not text_parts:
            return f"[{model}] Error: no output_text found in response"
        return "".join(text_parts)
    except requests.exceptions.ConnectionError:
        return f"[{model}] Error: could not connect to LiteLLM proxy at {config['litellm_url']}."
    except requests.exceptions.Timeout:
        return f"[{model}] Error: request timed out."
    except Exception as e:
        return f"[{model}] Error: {e}"
