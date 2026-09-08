"""
Phase 1 spike: validate raw ACP round-trip against `opencode acp` subprocess.

Throwaway script -- not integrated into the pipeline. Uses the
`agent-client-protocol` PyPI package (imports as `acp`), which provides
Pydantic models + an async JSON-RPC client/agent connection over stdio.

Run:
    cd spike && source venv/bin/activate && python3 acp_spike.py
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from dataclasses import dataclass, field

import acp


@dataclass
class SpikeClient:
    """Minimal ACP Client implementation.

    We only care about `session_update` notifications (to accumulate
    agent_message_chunk text). Every other Client method is a no-op /
    permission-denial since this spike never expects the agent to need
    file access, terminals, or permission prompts for a trivial prompt.
    """

    chunks: list[str] = field(default_factory=list)
    raw_updates: list[dict] = field(default_factory=list)

    async def session_update(self, session_id: str, update, **kwargs) -> None:
        self.raw_updates.append({"session_id": session_id, "update": update.model_dump(by_alias=True)})
        if update.sessionUpdate == "agent_message_chunk":
            content = update.content
            text = getattr(content, "text", None)
            if text:
                self.chunks.append(text)

    async def request_permission(self, session_id, tool_call, options, **kwargs):
        # Deny everything -- this spike expects a pure text completion,
        # no tool calls should be needed for "say hello in exactly 3 words".
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


async def main() -> None:
    prompt_text = "say hello in exactly 3 words"
    timings: dict[str, float] = {}

    t_wall_start = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="acp-spike-") as spike_cwd:
        client = SpikeClient()

        t_spawn_start = time.perf_counter()
        async with acp.spawn_agent_process(client, "opencode", "acp", cwd=spike_cwd) as (conn, process):
            timings["spawn_s"] = time.perf_counter() - t_spawn_start

            t_handshake_start = time.perf_counter()

            init_resp = await conn.initialize(
                protocol_version=acp.PROTOCOL_VERSION,
                client_capabilities=acp.schema.ClientCapabilities(
                    fs=acp.schema.FileSystemCapabilities(read_text_file=False, write_text_file=False),
                    terminal=False,
                ),
            )
            print(f"[initialize] response: {init_resp.model_dump(by_alias=True)}")

            session_resp = await conn.new_session(cwd=spike_cwd, mcp_servers=[])
            session_id = session_resp.sessionId
            print(f"[session/new] sessionId={session_id}")

            timings["handshake_s"] = time.perf_counter() - t_handshake_start

            t_prompt_start = time.perf_counter()

            prompt_resp = await conn.prompt(
                session_id=session_id,
                prompt=[acp.text_block(prompt_text)],
            )

            timings["prompt_round_trip_s"] = time.perf_counter() - t_prompt_start

            stop_reason = prompt_resp.stopReason
            accumulated_text = "".join(client.chunks)

        timings["total_wall_s"] = time.perf_counter() - t_wall_start

    print()
    print("=== Results ===")
    print(f"stopReason: {stop_reason!r}")
    print(f"session/update notifications received: {len(client.raw_updates)}")
    print(f"accumulated agent_message_chunk text: {accumulated_text!r}")
    print(f"non-empty: {bool(accumulated_text.strip())}")
    print()
    print("=== Timings ===")
    print(f"subprocess spawn:            {timings['spawn_s']:.3f}s")
    print(f"initialize+session/new:      {timings['handshake_s']:.3f}s")
    print(f"session/prompt round-trip:   {timings['prompt_round_trip_s']:.3f}s")
    print(f"total wall time:             {timings['total_wall_s']:.3f}s")
    print()

    if not accumulated_text.strip():
        print("WARNING: accumulated text is empty -- session/update chunks may not carry text as expected.")

    print("=== PromptResponse raw ===")
    print(prompt_resp.model_dump(by_alias=True))

    print()
    print("=== Raw session/update notifications ===")
    for i, upd in enumerate(client.raw_updates):
        print(f"[{i}] {upd}")


if __name__ == "__main__":
    asyncio.run(main())
