# AGENTS.md — for AI agents driving this pipeline via ACP

This repo exposes an automated multi-agent code-generation pipeline (plan → dispatch → specialist codegen → test/retry) as a standard **ACP Agent/server**: `pipeline/acp_server.py`. Spawn it, drive it via `initialize` → `new_session` → `prompt`, get generated code + tests written to disk on the server's machine.

Task planning and run artifacts use the category-based multi-file model documented in [docs/task-workflow.md](docs/task-workflow.md). Historical runs are retained but are not migrated.

Full human-readable reference: [docs/acp-integration.md](docs/acp-integration.md). This file is a terse action-oriented distillation — read that doc for anything not covered here.

## Critical gotchas (read before integrating)

1. **`stop_reason: "end_turn"` ≠ success.** Query the advertised `_agents-pipeline/run/status` extension for structured outcome; `end_turn` only means the ACP prompt turn completed.
2. **Free-text prompt only.** No file/document attachment support. Non-text content blocks in `prompt` are silently ignored. If you need to pass file contents, inline them as plain text inside the prompt string itself.
3. **1 session = 1 run.** Calling `prompt` twice on the same `session_id` raises an error. New task → new `new_session` call.
4. **Long-running, no timeout semantics.** A run can take ~20s to several minutes. A delayed response is NOT a hang. Listen for streamed `session_update` text and query structured status for queued/running task progress.
5. **`session/cancel` does not stop work.** It sets `cancel_requested`, but neither removes a queued run nor stops an active run or its writes. After the run finishes, the turn/status is marked cancelled, retaining a known underlying outcome. Unexpected errors can still report `error`/`refusal`. Cancelling the ACP request itself is different: while queued it removes the queue entry; while active it waits for the blocking run and cleanup before releasing the slot.
6. **Output is NOT returned inline.** Read artifacts from the server's filesystem after `prompt` returns. The default is `<server-repo>/runs/<run-id>/`, not the client/session cwd; see [Where output lands](#where-output-lands).
7. **Prompts are serialized per server process.** Concurrent requests wait FIFO without blocking the event loop; a run holds the slot through planning, sequential dispatch (even independent tasks), tests/retries, progress delivery, and teardown of pooled internal ACP resources. Separate server processes are not coordinated.
8. **Progress extension contract.** Invoke `conn.ext_method("agents-pipeline/run/status", {"sessionId": session_id})`; ACP JSON-RPC method is `_agents-pipeline/run/status`. See [docs/acp-integration.md](docs/acp-integration.md#36-structured-run-status) for schema and counter meanings.

## Setup

```
git clone <this-repo> agents-pipeline
cd agents-pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Install **`opencode` with `acp` support** and ensure its executable is on the server process's `PATH`: internal model calls launch `opencode acp`. They use a temporary HOME/cwd and generated `OPENCODE_CONFIG`, not your normal home configuration. The executable must remain discoverable with that temporary HOME.

Config precedence on the server: `~/.config/Agents/config.json`, then the repo's `config.json` only if the first file is absent. Base it on [config.example.json](config.example.json); set `litellm_url`/`litellm_key` for your OpenAI-compatible proxy and `models.<role>.model` to models it exposes. Every role's `provider` must be the literal `"acp"`. Normal opencode settings/credentials are not a substitute. Full fields: [config reference](docs/acp-integration.md#22-create-your-config-file).

Config is loaded once at server start — restart the server to pick up config changes.

## Spawn command

```
<venv-path>/venv/bin/python <repo-path>/pipeline/acp_server.py
```

Communicates over stdio; reserve stdin/stdout for ACP JSON-RPC framing (no wrapper banners or logging to stdout).

## Handshake and live status (Python SDK 0.12.1)

```python
import asyncio
import acp


async def main() -> None:
    python_path = "/path/to/agents-pipeline/venv/bin/python"
    server_path = "/path/to/agents-pipeline/pipeline/acp_server.py"
    task_prompt = "build a cli that reverses a string"

    class Client:
        async def session_update(self, session_id, update, **kwargs):
            if isinstance(update, acp.schema.AgentMessageChunk):
                if isinstance(update.content, acp.schema.TextContentBlock):
                    print(update.content.text, end="", flush=True)

        def on_connect(self, conn):
            return None

        # This server does not request client permissions, file I/O or terminals.

    async with acp.spawn_agent_process(
        Client(), python_path, server_path, cwd="/tmp"
    ) as (conn, process):
        await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)

        session_resp = await conn.new_session(cwd="/tmp", mcp_servers=[])
        session_id = session_resp.session_id

        prompt_task = asyncio.create_task(
            conn.prompt(
                session_id=session_id,
                prompt=[acp.text_block(task_prompt)],
            )
        )
        while not prompt_task.done():
            status = await conn.ext_method(
                "agents-pipeline/run/status", {"sessionId": session_id}
            )
            print("\n[progress]", status)
            await asyncio.sleep(1)

        prompt_resp = await prompt_task
        run_status = await conn.ext_method(
            "agents-pipeline/run/status", {"sessionId": session_id}
        )

    print(f"\n\nstop_reason: {prompt_resp.stop_reason}")
    print(run_status)


if __name__ == "__main__":
    asyncio.run(main())
```

Adapt the paths before running. This invokes real model calls; the subprocess/session `cwd="/tmp"` does not relocate run output. Python updates use snake_case attributes (`session_update`); `sessionUpdate` is the JSON wire alias.

## Interpret status, not streamed text

Discover the extension at `agentCapabilities._meta["agents-pipeline"].runStatus.method` in `initialize`; it advertises `_agents-pipeline/run/status`. Pass the name **without** the leading underscore to Python's `conn.ext_method` (the SDK adds it); raw JSON-RPC uses the advertised name. Responses use `schema_version: 1`.

New-project mode is the default. To edit an existing project, configure the fresh session before `prompt`: `await conn.ext_method("agents-pipeline/session/configure", {"sessionId": session_id, "mode": "existing_project", "projectPath": "/absolute/server/path/to/project"})`. `projectPath` must exist on the server filesystem. Edits happen in place; failures do not roll them back. Check the run's `changes.json` for additions/modifications/deletions and partial-edit warnings. Full behavior: [existing-project session configuration](docs/acp-integration.md#existing-project-session-configuration).

- `status`: `created`, `queued`, `running`, `completed`, `failed`, or `cancelled`. **`completed` does not mean passed.**
- `outcome`: null until known, then `passed`, `failed`, `passed_with_test_failures` (advisory tests), `error`, `cancelled`, or `invalid_prompt`. A cancelled status can retain the underlying outcome; require `outcome == "passed"` for a clean pipeline pass and handle cancellation separately.
- `queue_position`: one-based among waiting runs; null when active or not queued. `current_task`: null when no specialist is active. `run_id`: null until artifacts exist; otherwise an ID, not an absolute path.
- `tasks_total`: planned work items. `tasks_started`/`tasks_completed`: specialist **attempts**, including retries; these can exceed `tasks_total`. `tasks_blocked` counts dependency-blocked tasks separately. Do not treat these counters as a completion percentage.

Query final status before closing the session or server. Session state is in memory, unavailable after `close_session` or a server restart; historical artifacts remain on disk. Full schema: [structured run status](docs/acp-integration.md#36-structured-run-status).

## Non-Python integrations

ACP is JSON-RPC over stdio — implementable in any language with a JSON-RPC + subprocess/stdio library. Spec: https://agentclientprotocol.com

## Where output lands

Run directories are `<runs-root>/<run-id>/`. `pipeline.runs_dir` resolves relative to the **server repo root** (default `runs/`); an absolute configured path is honored. Neither subprocess cwd nor `new_session.cwd` changes that root. Clients must know the server repo/config location: status supplies only `run_id`.

With default directory settings, a run contains:

- `product/` — generated code
- `tests/` — generated tests, plus `tests/results/` with per-attempt JSON results
- `design/plan.json` — version-2 categorized, multi-file ownership/dependency plan with statuses and attempt details
- `manifest.json` — run status, timestamps, attempt count, original prompt
- Existing-project runs also include `changes.json` — detected project-relative additions, modifications, deletions, and a warning when the run fails
- `instructions.md` — human-readable run summary

`pipeline.output_dir` and `pipeline.tests_dir` configure the product/test directories. These files are on the server's own filesystem — not returned inline in the ACP response. Read them after `prompt` returns.

---

For full details, edge cases, and the complete protocol contract, see [docs/acp-integration.md](docs/acp-integration.md) in this repo.
