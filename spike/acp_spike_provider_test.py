"""
Step 0 research script (Phase 2 prep, NOT Phase 1 spike): verify the exact
opencode.json provider/model config shape that routes `opencode acp`
through OUR LiteLLM proxy (config["litellm_url"]/config["litellm_key"]),
using a real config.json-shaped test config, and confirm it does NOT
silently fall back to a free/default "zen" model.

Standalone, throwaway. Does NOT modify acp_spike.py or
acp_spike_minimal.py. Reuses the "winning combo" config shape discovered
in acp_spike_minimal.py's Trial 6 (HOME isolation + OPENCODE_CONFIG with
an inline `provider.<id>.options.{apiKey,baseURL}` block using the
`@ai-sdk/openai-compatible` npm provider), but this time:
  1. Loads real litellm_url/litellm_key from an actual config.json on
     disk (config.example.json-shaped), not a hardcoded default.
  2. Asks a question a "free zen model" would almost certainly answer
     differently/generically, to distinguish "routed through our proxy"
     from "silently fell back to some other model" -- specifically we
     ask the model to self-report its own model name/identity string,
     and cross-check token usage against a direct (non-ACP) HTTP call to
     the same proxy/model for a sanity baseline.

Run:
    cd spike && source venv/bin/activate && python3 acp_spike_provider_test.py
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import acp

# Real config.json (config.example.json-shaped) with actual litellm_url/key.
# Not committed to the repo; lives outside it. Falls back to env vars if
# not found, matching acp_spike_minimal.py's existing fallback convention.
REAL_CONFIG_PATH = os.path.expanduser("~/.config/Agents/config.json")

IDENTITY_PROMPT = (
    "What is your exact model name/identifier string, as you would "
    "report it in an API response's `model` field? Answer with ONLY "
    "the model identifier, no other text."
)


def _load_real_config() -> dict:
    if os.path.isfile(REAL_CONFIG_PATH):
        with open(REAL_CONFIG_PATH) as f:
            return json.load(f)
    return {
        "litellm_url": os.environ.get("LITELLM_BASE_URL", "https://api.rctz.online/v1"),
        "litellm_key": os.environ.get("LITELLM_API_KEY", "sk-REPLACE-ME"),
    }


@dataclass
class SpikeClient:
    chunks: list[str] = field(default_factory=list)
    raw_updates: list[dict] = field(default_factory=list)

    async def session_update(self, session_id: str, update, **kwargs) -> None:
        self.raw_updates.append({"session_id": session_id, "update": update.model_dump(by_alias=True)})
        if update.sessionUpdate == "agent_message_chunk":
            text = getattr(update.content, "text", None)
            if text:
                self.chunks.append(text)

    async def request_permission(self, session_id, tool_call, options, **kwargs):
        raise acp.RequestError.method_not_found("request_permission not supported")

    async def write_text_file(self, session_id, path, content, **kwargs):
        raise acp.RequestError.method_not_found("write_text_file not supported")

    async def read_text_file(self, session_id, path, line=None, limit=None, **kwargs):
        raise acp.RequestError.method_not_found("read_text_file not supported")

    async def create_terminal(self, session_id, command, args=None, env=None, cwd=None, output_byte_limit=None, **kwargs):
        raise acp.RequestError.method_not_found("create_terminal not supported")

    async def terminal_output(self, session_id, terminal_id, **kwargs):
        raise acp.RequestError.method_not_found("terminal_output not supported")

    async def release_terminal(self, session_id, terminal_id, **kwargs):
        raise acp.RequestError.method_not_found("release_terminal not supported")

    async def wait_for_terminal_exit(self, session_id, terminal_id, **kwargs):
        raise acp.RequestError.method_not_found("wait_for_terminal_exit not supported")

    async def kill_terminal(self, session_id, terminal_id, **kwargs):
        raise acp.RequestError.method_not_found("kill_terminal not supported")

    async def create_elicitation(self, message, mode, **kwargs):
        raise acp.RequestError.method_not_found("create_elicitation not supported")

    async def complete_elicitation(self, elicitation_id, **kwargs):
        return None

    async def ext_method(self, method, params):
        raise acp.RequestError.method_not_found(f"ext_method {method} not supported")

    async def ext_notification(self, method, params):
        return None

    def on_connect(self, conn) -> None:
        return None


def build_provider_config(real_config: dict, model: str) -> dict:
    """
    The candidate opencode.json shape for Step 0: a custom
    `@ai-sdk/openai-compatible` provider entry pointing baseURL/apiKey
    at our LiteLLM proxy, with a minimal deny-all agent as default.
    """
    return {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "litellm": {
                "name": "LiteLLM Gateway",
                "npm": "@ai-sdk/openai-compatible",
                "options": {
                    "apiKey": real_config["litellm_key"],
                    "baseURL": real_config["litellm_url"],
                },
                "models": {model: {"name": model}},
            },
        },
        "model": f"litellm/{model}",
        "agent": {
            "minimal": {
                "mode": "primary",
                "description": "Minimal ACP provider-routing test agent",
                "prompt": "Respond directly. No commentary.",
                "tools": {"*": False},
                "permission": {
                    "edit": "deny", "bash": "deny", "read": "deny", "glob": "deny",
                    "grep": "deny", "list": "deny", "task": "deny", "webfetch": "deny",
                    "websearch": "deny", "skill": "deny", "external_directory": "deny",
                },
            },
        },
        "default_agent": "minimal",
    }


async def run_acp_prompt(config_dict: dict, prompt_text: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="fake-home-") as fake_home, \
         tempfile.TemporaryDirectory(prefix="acp-provider-test-cfg-") as config_dir, \
         tempfile.TemporaryDirectory(prefix="acp-provider-test-cwd-") as cwd:

        config_path = Path(config_dir) / "opencode.json"
        config_path.write_text(json.dumps(config_dict))

        env = {
            **os.environ,
            "HOME": fake_home,
            "OPENCODE_CONFIG": str(config_path),
        }

        client = SpikeClient()
        async with acp.spawn_agent_process(client, "opencode", "acp", cwd=cwd, env=env) as (conn, process):
            await conn.initialize(
                protocol_version=acp.PROTOCOL_VERSION,
                client_capabilities=acp.schema.ClientCapabilities(
                    fs=acp.schema.FileSystemCapabilities(read_text_file=False, write_text_file=False),
                    terminal=False,
                ),
            )
            session_resp = await conn.new_session(cwd=cwd, mcp_servers=[])
            session_id = session_resp.session_id

            prompt_resp = await conn.prompt(
                session_id=session_id,
                prompt=[acp.text_block(prompt_text)],
            )

            usage = prompt_resp.model_dump(by_alias=True).get("usage") or {}
            accumulated = "".join(client.chunks)

    return {
        "accumulated_text": accumulated,
        "usage": usage,
        "stop_reason": prompt_resp.stopReason,
    }


def run_direct_http_baseline(real_config: dict, model: str, prompt_text: str) -> dict:
    """Direct (non-ACP) HTTP call to the same proxy/model, for cross-check."""
    import urllib.request

    url = f"{real_config['litellm_url']}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
    }).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {real_config['litellm_key']}",
            "Content-Type": "application/json",
            "User-Agent": "acp-spike-provider-test/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return {
        "reported_model": data.get("model"),
        "content": data["choices"][0]["message"]["content"],
        "usage": data.get("usage"),
    }


async def run_bad_key_negative_control(real_config: dict, model: str) -> dict:
    """
    Negative control: spawn the SAME config shape but with a deliberately
    wrong apiKey. If the ACP session were silently falling back to some
    free/default "zen" model instead of actually routing through our
    openai-compatible provider block, this call would still succeed
    (the fallback path wouldn't care about our bad key). If it instead
    fails with OUR proxy's own authentication error, that's definitive
    proof the ACP session is really calling our LiteLLM proxy with our
    credentials -- not falling back to anything else.
    """
    bad_cfg = build_provider_config(
        {**real_config, "litellm_key": "sk-DEFINITELY-WRONG-KEY-00000"}, model
    )
    try:
        result = await run_acp_prompt(bad_cfg, "say hello in exactly 3 words")
        return {"raised": False, "result": result}
    except Exception as e:
        return {"raised": True, "error_type": type(e).__name__, "error_message": str(e)}


async def main() -> None:
    real_config = _load_real_config()
    model = "claude-sonnet-5"  # matches a real model entry in real_config's models block

    print("=== Direct HTTP baseline (no ACP) -- sanity check the proxy/key work at all ===")
    baseline = run_direct_http_baseline(real_config, model, IDENTITY_PROMPT)
    print(json.dumps(baseline, indent=2))

    print()
    print("=== ACP round-trip via generated openai-compatible provider config (correct key) ===")
    provider_cfg = build_provider_config(real_config, model)
    print("Generated opencode.json:")
    print(json.dumps(provider_cfg, indent=2))
    print()
    result = await run_acp_prompt(provider_cfg, IDENTITY_PROMPT)
    print(json.dumps(result, indent=2))

    print()
    print("=== NEGATIVE CONTROL: same config shape, deliberately WRONG apiKey ===")
    print("(If this call still silently succeeds, the config isn't actually routing through our proxy.)")
    negative = await run_bad_key_negative_control(real_config, model)
    print(json.dumps(negative, indent=2))

    print()
    print("=== VERDICT ===")
    acp_text = result["accumulated_text"].strip()
    print(f"ACP-routed response (correct key): {acp_text!r}")
    print(f"stopReason: {result['stop_reason']!r}, usage: {result['usage']}")
    print()
    if negative["raised"] and "litellm" in negative["error_message"].lower() or "proxy" in negative.get("error_message", "").lower() or "token" in negative.get("error_message", "").lower():
        print("CONFIRMED: bad-key call failed with OUR proxy's own auth error, not a silent fallback.")
        print(f"  error: {negative.get('error_message')}")
        print("This proves the correct-key call above is genuinely routed through config['litellm_url'] with config['litellm_key'], not a free zen-model fallback.")
    elif negative["raised"]:
        print("Bad-key call raised an exception (good sign -- no silent success), but error text doesn't obviously reference our proxy. Inspect manually:")
        print(f"  {negative.get('error_type')}: {negative.get('error_message')}")
    else:
        print("WARNING: bad-key call did NOT fail -- this config shape may be silently falling back to a different model/provider. Needs further investigation.")


if __name__ == "__main__":
    asyncio.run(main())
