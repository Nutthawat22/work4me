"""
specialists/llm_client.py

Shared LiteLLM proxy client used by all agents (Master, Design, and
future Phase 2 specialists). Standardized on the /chat/completions
shape since it's the more common LiteLLM proxy interface.
"""

import requests
from typing import Any


def call_llm(messages: list[dict], model: str, config: dict[str, Any], timeout: int = 60) -> str:
    """
    Send a chat-completions request to the configured LiteLLM proxy.

    Args:
        messages: Full list of {"role": ..., "content": ...} messages to send.
        model: Model name string (e.g. config["models"]["design"]).
        config: Loaded config dict, must contain "litellm_url" and "litellm_key".
        timeout: Request timeout in seconds.

    Returns:
        The assistant's response content string, or a formatted error string
        on connection/timeout/other failure.
    """
    url = f"{config['litellm_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config['litellm_key']}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(url, headers=headers, json={"model": model, "messages": messages}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        return f"[{model}] Error: could not connect to LiteLLM proxy at {config['litellm_url']}."
    except requests.exceptions.Timeout:
        return f"[{model}] Error: request timed out."
    except Exception as e:
        return f"[{model}] Error: {e}"
