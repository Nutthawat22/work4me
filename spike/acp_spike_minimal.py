"""
Phase 1 spike (follow-up): test whether ANY flag/config reduces per-call
token overhead of `opencode acp` sessions.

Does NOT modify acp_spike.py. Standalone experiment script.

Tests attempted, in order:
  1. Baseline (repeat of acp_spike.py numbers, for a same-run comparison).
  2. --agent <custom-minimal-agent> with all tools/permissions denied,
     custom short system prompt, via a project-local opencode.json
     (OPENCODE_CONFIG env var pointing at a config with an empty
     mcp/plugin/instructions and a `default_agent` set to the minimal agent).
  3. Same minimal config + --pure flag together.
  4. Minimal config + client_capabilities all False (already done in
     acp_spike.py, repeated here for direct comparison in the same run).

Run:
    cd spike && source venv/bin/activate && python3 acp_spike_minimal.py
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import acp

PROMPT_TEXT = "say hello in exactly 3 words"

MINIMAL_CONFIG = {
    "$schema": "https://opencode.ai/config.json",
    "agent": {
        "minimal": {
            "mode": "primary",
            "description": "Minimal test agent",
            "prompt": "Respond directly. No commentary.",
            "permission": {
                "edit": "deny",
                "bash": "deny",
                "read": "deny",
                "glob": "deny",
                "grep": "deny",
                "list": "deny",
                "task": "deny",
                "webfetch": "deny",
                "websearch": "deny",
                "skill": "deny",
                "external_directory": "deny",
            },
            "tools": {"*": False},
        }
    },
    "mcp": {},
    "plugin": [],
    "instructions": [],
    "default_agent": "minimal",
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
        raise acp.RequestError.method_not_found("request_permission not supported in spike")

    async def write_text_file(self, session_id, path, content, **kwargs):
        raise acp.RequestError.method_not_found("write_text_file not supported in spike")

    async def read_text_file(self, session_id, path, line=None, limit=None, **kwargs):
        raise acp.RequestError.method_not_found("read_text_file not supported in spike")

    async def create_terminal(self, session_id, command, args=None, env=None, cwd=None, output_byte_limit=None, **kwargs):
        raise acp.RequestError.method_not_found("create_terminal not supported in spike")

    async def terminal_output(self, session_id, terminal_id, **kwargs):
        raise acp.RequestError.method_not_found("terminal_output not supported in spike")

    async def release_terminal(self, session_id, terminal_id, **kwargs):
        raise acp.RequestError.method_not_found("release_terminal not supported in spike")

    async def wait_for_terminal_exit(self, session_id, terminal_id, **kwargs):
        raise acp.RequestError.method_not_found("wait_for_terminal_exit not supported in spike")

    async def kill_terminal(self, session_id, terminal_id, **kwargs):
        raise acp.RequestError.method_not_found("kill_terminal not supported in spike")

    async def create_elicitation(self, message, mode, **kwargs):
        raise acp.RequestError.method_not_found("create_elicitation not supported in spike")

    async def complete_elicitation(self, elicitation_id, **kwargs):
        return None

    async def ext_method(self, method, params):
        raise acp.RequestError.method_not_found(f"ext_method {method} not supported in spike")

    async def ext_notification(self, method, params):
        return None

    def on_connect(self, conn) -> None:
        return None


async def run_trial(
    label: str,
    *,
    extra_args: list[str] | None = None,
    env_overrides: dict[str, str] | None = None,
    agent: str | None = None,
) -> dict:
    """Spawn opencode acp fresh, do initialize+session/new+prompt, return usage info."""
    extra_args = extra_args or []
    env = {**os.environ, **(env_overrides or {})}

    with tempfile.TemporaryDirectory(prefix="acp-spike-min-") as cwd:
        client = SpikeClient()
        args = list(extra_args)

        async with acp.spawn_agent_process(client, "opencode", "acp", *args, cwd=cwd, env=env) as (conn, process):
            await conn.initialize(
                protocol_version=acp.PROTOCOL_VERSION,
                client_capabilities=acp.schema.ClientCapabilities(
                    fs=acp.schema.FileSystemCapabilities(read_text_file=False, write_text_file=False),
                    terminal=False,
                ),
            )
            session_resp = await conn.new_session(cwd=cwd, mcp_servers=[])
            session_id = session_resp.session_id

            if agent is not None:
                try:
                    await conn.set_config_option(session_id=session_id, config_id="mode", value=agent)
                except Exception as e:
                    print(f"  [warn] could not switch agent/mode to {agent!r}: {e}")

            prompt_resp = await conn.prompt(
                session_id=session_id,
                prompt=[acp.text_block(PROMPT_TEXT)],
            )

            usage = prompt_resp.model_dump(by_alias=True).get("usage") or {}
            accumulated = "".join(client.chunks)
            n_updates = len(client.raw_updates)
            has_available_commands = any(
                u["update"].get("sessionUpdate") == "available_commands_update" for u in client.raw_updates
            )

    result = {
        "label": label,
        "input_tokens": usage.get("inputTokens"),
        "output_tokens": usage.get("outputTokens"),
        "total_tokens": usage.get("totalTokens"),
        "accumulated_text": accumulated,
        "n_session_updates": n_updates,
        "has_available_commands_update": has_available_commands,
    }
    return result


async def main() -> None:
    results = []

    print("=== Trial 1: baseline (no flags, no custom config) ===")
    r1 = await run_trial("baseline")
    print(json.dumps(r1, indent=2))
    results.append(r1)

    print()
    print("=== Trial 2: --pure flag (external plugins disabled) ===")
    r2 = await run_trial("pure_flag", extra_args=["--pure"])
    print(json.dumps(r2, indent=2))
    results.append(r2)

    print()
    print("=== Trial 3: minimal opencode.json via OPENCODE_CONFIG env var ===")
    with tempfile.TemporaryDirectory(prefix="acp-spike-config-") as config_dir:
        config_path = Path(config_dir) / "opencode.json"
        config_path.write_text(json.dumps(MINIMAL_CONFIG))
        r3 = await run_trial(
            "minimal_config_env",
            env_overrides={"OPENCODE_CONFIG": str(config_path)},
        )
    print(json.dumps(r3, indent=2))
    results.append(r3)

    print()
    print("=== Trial 4: minimal opencode.json + --pure ===")
    with tempfile.TemporaryDirectory(prefix="acp-spike-config-") as config_dir:
        config_path = Path(config_dir) / "opencode.json"
        config_path.write_text(json.dumps(MINIMAL_CONFIG))
        r4 = await run_trial(
            "minimal_config_env_pure",
            extra_args=["--pure"],
            env_overrides={"OPENCODE_CONFIG": str(config_path)},
        )
    print(json.dumps(r4, indent=2))
    results.append(r4)

    print()
    print("=== Trial 5: minimal opencode.json + --agent explore (built-in smaller agent) ===")
    with tempfile.TemporaryDirectory(prefix="acp-spike-config-") as config_dir:
        config_path = Path(config_dir) / "opencode.json"
        config_path.write_text(json.dumps(MINIMAL_CONFIG))
        r5 = await run_trial(
            "minimal_config_agent_explore",
            env_overrides={"OPENCODE_CONFIG": str(config_path)},
            agent="explore",
        )
    print(json.dumps(r5, indent=2))
    results.append(r5)

    print()
    print("=== Trial 6: WINNING COMBO -- fake HOME (isolate ~/.config/opencode) ===")
    print("===           + OPENCODE_CONFIG w/ explicit provider creds + minimal agent (tools off) ===")
    winning_cfg = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "litellm": {
                "name": "LiteLLM Gateway",
                "npm": "@ai-sdk/openai-compatible",
                "options": {
                    "apiKey": os.environ.get("LITELLM_API_KEY", "sk-REPLACE-ME"),
                    "baseURL": os.environ.get("LITELLM_BASE_URL", "https://api.rctz.online/v1"),
                },
                "models": {"claude-sonnet-5": {"name": "claude-sonnet-5"}},
            },
        },
        "model": "litellm/claude-sonnet-5",
        "agent": {
            "minimal": {
                "mode": "primary",
                "description": "Minimal test agent",
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
    with tempfile.TemporaryDirectory(prefix="fake-home-") as fake_home, \
         tempfile.TemporaryDirectory(prefix="acp-spike-config-") as config_dir:
        config_path = Path(config_dir) / "opencode.json"
        config_path.write_text(json.dumps(winning_cfg))
        r6 = await run_trial(
            "winning_combo_fake_home_plus_config",
            env_overrides={"HOME": fake_home, "OPENCODE_CONFIG": str(config_path)},
        )
    print(json.dumps(r6, indent=2))
    results.append(r6)

    print()
    print("=== SUMMARY ===")
    for r in results:
        print(
            f"{r['label']:30s}  input_tokens={r['input_tokens']!s:>8}  "
            f"total_tokens={r['total_tokens']!s:>8}  "
            f"n_updates={r['n_session_updates']:>3}  "
            f"has_avail_cmds={r['has_available_commands_update']}"
        )


if __name__ == "__main__":
    asyncio.run(main())
