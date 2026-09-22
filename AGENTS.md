# AGENTS.md — for AI agents driving this pipeline via ACP

This repo exposes an automated multi-agent code-generation pipeline (plan → dispatch → specialist codegen → test/retry) as a standard **ACP Agent/server**: `pipeline/acp_server.py`. Spawn it, drive it via `initialize` → `new_session` → `prompt`, get generated code + tests written to disk on the server's machine.

Full human-readable reference: [docs/acp-integration.md](docs/acp-integration.md). This file is a terse action-oriented distillation — read that doc for anything not covered here.

## Critical gotchas (read before integrating)

1. **`stop_reason: "end_turn"` ≠ success.** There is no structured pass/fail field. Parse streamed `session_update` text for `"✅ Pipeline passed"` / `"❌ Pipeline failed"` to know the real outcome.
2. **Free-text prompt only.** No file/document attachment support. Non-text content blocks in `prompt` are silently ignored. If you need to pass file contents, inline them as plain text inside the prompt string itself.
3. **1 session = 1 run.** Calling `prompt` twice on the same `session_id` raises an error. New task → new `new_session` call.
4. **Long-running, no timeout semantics.** A run can take ~20s to several minutes. A delayed response is NOT a hang. Keep listening for streamed `session_update` chunks the whole time — that's the only progress signal until the final response.
5. **`session/cancel` is best-effort only.** It does NOT stop the run. It only changes how the final response is labeled (`stop_reason: "cancelled"`). The pipeline still runs to completion and still writes its output. Do not rely on it to halt work or side effects.
6. **Output is NOT returned inline.** Generated code/tests land in `runs/<run-id>/` on the machine running the server. To get the actual files, read that directory from the run's filesystem after `prompt` returns.

## Setup

```
git clone <this-repo> agents-pipeline
cd agents-pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Config file **must** be at `~/.config/Agents/config.json` on the machine running the server (falls back to repo's own `config.json` only if that's absent). Base it on `config.example.json`. Every `models.<role>.provider` must be the literal string `"acp"` — no other value is valid. Full field docs: docs/acp-integration.md §2.2.

Config is loaded once at server start — restart the server to pick up config changes.

## Spawn command

```
<venv-path>/venv/bin/python <repo-path>/pipeline/acp_server.py
```

Communicates over stdio (JSON-RPC framing on stdin/stdout).

## Handshake (Python, minimal working example)

```python
import asyncio
import acp


async def main() -> None:
    python_path = "/path/to/agents-pipeline/venv/bin/python"
    server_path = "/path/to/agents-pipeline/pipeline/acp_server.py"
    task_prompt = "build a cli that reverses a string"

    class Client:
        async def session_update(self, session_id, update, **kwargs):
            if getattr(update, "sessionUpdate", None) == "agent_message_chunk":
                text = getattr(update.content, "text", None)
                if text:
                    print(text, end="", flush=True)

        def on_connect(self, conn):
            return None

        # Other Client interface methods (request_permission, file I/O,
        # terminals, elicitation, ext_method/ext_notification) are not
        # used by this pipeline's server in v1 — a minimal client only
        # needs session_update and on_connect implemented.

    async with acp.spawn_agent_process(
        Client(), python_path, server_path, cwd="/tmp"
    ) as (conn, process):
        await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)

        session_resp = await conn.new_session(cwd="/tmp", mcp_servers=[])
        session_id = session_resp.session_id

        prompt_resp = await conn.prompt(
            session_id=session_id,
            prompt=[acp.text_block(task_prompt)],
        )

    print(f"\n\nstop_reason: {prompt_resp.stop_reason}")
    # stop_reason "end_turn" here still requires parsing streamed text
    # for pass/fail — see gotcha #1 above.


if __name__ == "__main__":
    asyncio.run(main())
```

Also see `spike/acp_server_test_client.py` in this repo for a fuller example client.

## Non-Python integrations

ACP is JSON-RPC over stdio — implementable in any language with a JSON-RPC + subprocess/stdio library. Spec: https://agentclientprotocol.com

## Where output lands

Each run creates `runs/<run-id>/` (relative to the server's cwd, e.g. `runs/0007-build-a-cli-that-reverses-a-string/`), containing:

- `product/` — generated code
- `tests/` — generated tests, plus `tests/results/` with per-attempt JSON results
- `manifest.json` — run status, timestamps, attempt count, original prompt
- `instructions.md` — human-readable run summary

This is on the server's own filesystem — not returned inline in the ACP response. Read it after `prompt` returns if you need the actual files.

---

For full details, edge cases, and the complete protocol contract, see [docs/acp-integration.md](docs/acp-integration.md) in this repo.
