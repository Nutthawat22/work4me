"""
Throwaway ACP CLIENT script for manually exercising pipeline/acp_server.py
(the ACP AGENT/server built for Phase 5 of the ACP-native rearchitecture).

Mirrors spike/acp_spike.py's SpikeClient pattern, but instead of spawning
`opencode acp`, it spawns THIS PROJECT'S OWN pipeline/acp_server.py as the
agent subprocess, using the project's own venv python
(venv/bin/python -- this repo's actual convention, see spike/acp_spike.py's
own venv/bin/activate usage note).

Run (from the project root):

    venv/bin/python spike/acp_server_test_client.py

Or, to try a different prompt:

    venv/bin/python spike/acp_server_test_client.py "build a cli that reverses a string"

This will:
  1. Spawn `venv/bin/python pipeline/acp_server.py` as a subprocess.
  2. initialize() -> new_session() -> prompt() with the given free-text task.
  3. Print every session_update notification as it arrives (this is the
     pipeline's own print() output, streamed back live).
  4. Print the final PromptResponse (stop_reason).

NOTE: this actually runs the real pipeline end-to-end (real LLM calls via
whatever config ~/.config/Agents/config.json or the repo's config.json
points at) -- it is NOT a mocked/unit-test path. Expect it to take
anywhere from tens of seconds to several minutes depending on the prompt.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from dataclasses import dataclass, field

import acp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT_PYTHON = os.path.join(PROJECT_ROOT, "venv", "bin", "python")
SERVER_MODULE_PATH = os.path.join(PROJECT_ROOT, "pipeline", "acp_server.py")

DEFAULT_PROMPT = "create a simple hello world html page"


@dataclass
class TestClient:
    """Minimal ACP Client implementation -- same shape as
    spike/acp_spike.py's SpikeClient. Accumulates agent_message_chunk
    text AND prints every session_update notification as it arrives
    (this script's whole point is to demonstrate live progress
    streaming from the server, not just the final accumulated text)."""

    chunks: list[str] = field(default_factory=list)

    async def session_update(self, session_id: str, update, **kwargs) -> None:
        if getattr(update, "sessionUpdate", None) == "agent_message_chunk":
            content = update.content
            text = getattr(content, "text", None)
            if text:
                self.chunks.append(text)
                print(text, end="", flush=True)

    async def request_permission(self, session_id, tool_call, options, **kwargs):
        raise acp.RequestError.method_not_found("request_permission not supported in this test client")

    async def write_text_file(self, session_id, path, content, **kwargs):
        raise acp.RequestError.method_not_found("write_text_file not supported in this test client")

    async def read_text_file(self, session_id, path, line=None, limit=None, **kwargs):
        raise acp.RequestError.method_not_found("read_text_file not supported in this test client")

    async def create_terminal(self, session_id, command, args=None, env=None, cwd=None, output_byte_limit=None, **kwargs):
        raise acp.RequestError.method_not_found("create_terminal not supported in this test client")

    async def terminal_output(self, session_id, terminal_id, **kwargs):
        raise acp.RequestError.method_not_found("terminal_output not supported in this test client")

    async def release_terminal(self, session_id, terminal_id, **kwargs):
        raise acp.RequestError.method_not_found("release_terminal not supported in this test client")

    async def wait_for_terminal_exit(self, session_id, terminal_id, **kwargs):
        raise acp.RequestError.method_not_found("wait_for_terminal_exit not supported in this test client")

    async def kill_terminal(self, session_id, terminal_id, **kwargs):
        raise acp.RequestError.method_not_found("kill_terminal not supported in this test client")

    async def create_elicitation(self, message, mode, **kwargs):
        raise acp.RequestError.method_not_found("create_elicitation not supported in this test client")

    async def complete_elicitation(self, elicitation_id, **kwargs):
        return None

    async def ext_method(self, method, params):
        raise acp.RequestError.method_not_found(f"ext_method {method} not supported in this test client")

    async def ext_notification(self, method, params):
        return None

    def on_connect(self, conn) -> None:
        return None


async def main() -> None:
    prompt_text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROMPT

    print(f"=== Spawning pipeline/acp_server.py, prompting: {prompt_text!r} ===\n")

    with tempfile.TemporaryDirectory(prefix="acp-server-test-client-") as cwd:
        client = TestClient()

        async with acp.spawn_agent_process(
            client, PROJECT_ROOT_PYTHON, SERVER_MODULE_PATH, cwd=cwd
        ) as (conn, process):
            init_resp = await conn.initialize(
                protocol_version=acp.PROTOCOL_VERSION,
                client_capabilities=acp.schema.ClientCapabilities(
                    fs=acp.schema.FileSystemCapabilities(read_text_file=False, write_text_file=False),
                    terminal=False,
                ),
            )
            print(f"[initialize] response: {init_resp.model_dump(by_alias=True)}\n")

            session_resp = await conn.new_session(cwd=cwd, mcp_servers=[])
            session_id = session_resp.session_id
            print(f"[session/new] sessionId={session_id}\n")
            print("=== Streamed pipeline output ===\n")

            prompt_resp = await conn.prompt(
                session_id=session_id,
                prompt=[acp.text_block(prompt_text)],
            )

    print()
    print("=== Final PromptResponse ===")
    print(prompt_resp.model_dump(by_alias=True))


if __name__ == "__main__":
    asyncio.run(main())
