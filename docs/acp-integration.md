# ACP Integration Guide

This document describes how to invoke this repository's automated multi-agent code-generation pipeline as an [Agent Client Protocol (ACP)](https://agentclientprotocol.com) Agent/server, for teams integrating it into their own automated flows.

## 1. What this is

This repository implements an automated multi-agent code-generation pipeline (category-based multi-file task planning → sequential task dispatch → specialist code generation → test execution/retry). That pipeline is exposed as a standard **ACP Agent** — a subprocess that any ACP-compliant client can spawn, initialize, and drive through a `new_session` → `prompt` handshake to run one end-to-end pipeline run, with progress streamed back live as the run executes. See [Task Planning And Execution](task-workflow.md) for task categories, ownership, retries, and artifacts.

If you already have (or are building) your own ACP client, you can drive this pipeline the same way you'd drive any other ACP agent (e.g. an editor plugin driving a coding assistant). If you don't, [section 4](#4-minimal-example-client-python) has a minimal working example.

## 2. Prerequisites / Setup (your side)

You run this pipeline yourself, using your own LLM provider credentials. There is no shared/remote instance — you spawn the server process locally (or in your own environment) and it makes its own LLM calls using config you supply.

### 2.1 Clone and install

```
git clone <this-repo> agents-pipeline
cd agents-pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

This installs, among other things, `agent-client-protocol==0.12.1` — the Python ACP library the server is built on.

Also install `opencode` with `acp` support and put its executable on the server process's `PATH`. Internal model calls spawn `opencode acp` with a temporary HOME/cwd and generated `OPENCODE_CONFIG` using this pipeline's proxy/model settings, not your normal home configuration or credentials. The executable must remain discoverable with that temporary HOME.

### 2.2 Create your config file

The server loads its config from **`~/.config/Agents/config.json`** first; if that file doesn't exist, it falls back to the repo's own `config.json` at the project root. Configure one of these locations on the machine/environment where you run the server; the home config takes precedence.

Base your config on `config.example.json` in this repo. Shape and fields:

```json
{
  "litellm_url": "https://your-litellm-proxy/v1",
  "litellm_key": "YOUR_API_KEY_HERE",
  "models": {
    "code":       {"model": "your-code-model",    "provider": "acp"},
    "default":    {"model": "your-default-model",  "provider": "acp"},
    "master":     {"model": "your-master-model",   "provider": "acp"},
    "design":     {"model": "your-design-model",   "provider": "acp"},
    "specialist": {"model": "your-specialist-model","provider": "acp"}
  },
  "pipeline": {
    "max_retries": 3,
    "runs_dir": "runs/",
    "output_dir": "product",
    "tests_dir": "tests",
    "tests_blocking": true
  },
  "languages": {
    "python": { "...": "see config.example.json for full test-runner config" },
    "javascript": { "...": "see config.example.json for full test-runner config" },
    "typescript": { "...": "see config.example.json for full test-runner config" }
  }
}
```

Field-by-field, what you need to fill in:

- **`litellm_url`** — your own LLM proxy/gateway base URL.
- **`litellm_key`** — your own API key for that proxy.
- **`models.<role>.model`** — the model name to use for each role (`code`, `default`, `master`, `design`, `specialist`). Set these to whatever models your proxy exposes.
- **`models.<role>.provider`** — must always be the literal string `"acp"` for every role. This is **not configurable**; there is no other valid value (older `chat_completions`/`responses` provider modes have been removed from this codebase). All model calls in this pipeline route through an ACP-based provider internally, regardless of which underlying LLM you point at via `litellm_url`.
- **`pipeline.max_retries`** — max dispatch/test retry attempts per run.
- **`pipeline.runs_dir`** — directory where per-run output directories are created; relative paths resolve from the server repo root, and absolute paths are honored. See [section 6](#6-where-output-lands).
- **`pipeline.output_dir`** — subdirectory name (inside each run dir) where generated product code is written.
- **`pipeline.tests_dir`** — subdirectory name (inside each run dir) where generated tests are written.
- **`pipeline.tests_blocking`** — whether test failures block a run from being reported as passed (`true`) or are advisory-only once all specialists have dispatched successfully (`false`).
- **`pipeline.max_prompt_content_chars`** *(optional, default `200000`)* — per-file truncation limit (in chars) for embedded text-resource attachments extracted from an ACP prompt (see `pipeline/acp_content.py`). Not yet wired into `acp_server.py`'s routing — prep work only.
- **`languages.*`** — per-language test-runner configuration (test command, file patterns, result format). Copy these as-is from `config.example.json` unless you need to change how tests are invoked for a given language.

### 2.3 Client-side requirements

- If you're writing an ACP client in **Python**, install `agent-client-protocol==0.12.1` (or a compatible version) — this is the same package this repo's server is built on, listed in `requirements.txt`.
- If your integration is in **another language**, ACP is a JSON-RPC-over-stdio protocol — it's implementable in any language with a JSON-RPC and subprocess/stdio library. See the spec at https://agentclientprotocol.com.

## 3. How to invoke it — the protocol contract

### 3.1 Spawn command

Spawn the server as a subprocess, communicating over stdio (stdin/stdout for JSON-RPC framing; the server's own diagnostic/error output does not go to stdout — see [section 3.5](#35-progress-streaming)):

```
<your-venv-path>/venv/bin/python <path-to-repo>/pipeline/acp_server.py
```

Use the interpreter from the venv you created in step 2.1 (it needs `agent-client-protocol` and this repo's other dependencies installed).

### 3.2 Handshake sequence

**`initialize`** — sent first, once per connection.

- Request: `protocol_version` (integer), optionally `client_capabilities`, `client_info`.
- Response (`InitializeResponse`): `protocol_version`, `agent_capabilities`, `agent_info` (name/version of this agent — currently `agents-pipeline` / `0.1.0`).

**`new_session`** — sent once per pipeline run you want to start.

- Request: `cwd` (a working directory string — required by the ACP schema, but not otherwise consumed by this pipeline's own logic beyond being recorded on the session), optionally `additional_directories`, `mcp_servers`.
- Response (`NewSessionResponse`): `session_id` — a fresh UUID-based identifier. You'll use this for the `prompt` call that follows.
- New-project mode is the default. To edit an existing project, configure the new session with the extension below before its `prompt` call.

**`prompt`** — sent once per session, with your task description.

- Request: `session_id` (from `new_session`), `prompt` (a list of content blocks — see [3.3](#33-prompt-contract-v1)).
- Response (`PromptResponse`): `stop_reason` — see [3.7](#37-final-response).

#### Existing-project session configuration

Existing-project mode is explicit and configured per session, leaving the current new-project behavior as the default. After `new_session`, but before its one `prompt`, call:

```python
await conn.ext_method(
    "agents-pipeline/session/configure",
    {
        "sessionId": session_id,
        "mode": "existing_project",
        "projectPath": "/absolute/path/on/the/server/to/project",
    },
)
```

The extension is advertised at `agentCapabilities._meta["agents-pipeline"].sessionConfig.method` as `_agents-pipeline/session/configure`; Python SDK callers omit the leading underscore in `conn.ext_method`. `projectPath` must be an absolute path to an existing directory on the **server's filesystem** and must not overlap the pipeline's configured run-artifact directory. The server resolves it to a real path and rejects invalid locations. The option is accepted only while the session is new; configuration after the prompt starts is rejected. A new session without this call continues to use the existing generated-project mode.

In existing-project mode, planning receives a bounded project tree/source context, and each specialist receives that context plus the current contents of its owned files (owned-file context is capped at 30,000 characters per file and 100,000 characters per task). Specialists update those files directly in the supplied project; they do not create an overlay or copy. Only planned owned paths may be written or intentionally deleted. Additions, modifications, and deletions are summarized in the run's `changes.json` and manifest. Symlink/path escapes, common generated/vendor directories, and overlap with the server's configured run-artifact directory are rejected. Change tracking excludes common generated/vendor/cache directories such as `.git`, `node_modules`, `venv`, build outputs, and test caches.

Tests run from the target project's root and against its existing test tree. Existing-project runs require at least one matching automated test and treat test failures as blocking even when `pipeline.tests_blocking` is `false`, so an unvalidated edit cannot be reported as a clean pass. Edits are not rolled back on failure; check `changes.json` for the detected change set and treat a failed run's target as potentially partially edited.

### 3.3 Prompt contract (v1)

**Free-text only.** The `prompt` request's list of content blocks is scanned for `TextContentBlock` entries; their `.text` values are joined together (space-separated) and passed to the pipeline as the task description (equivalent to typing a request into this pipeline's own interactive REPL).

**Other content block types are ignored in this version** — images, embedded resources/files, and any other non-text block type in the `prompt` list are silently skipped. This is a known v1 limitation. If no text content is found at all, the server responds with `stop_reason: "refusal"` (see [3.7](#37-final-response)) rather than starting a pipeline run.

File-attachment support (e.g. passing a design document alongside a task prompt) is a likely near-future addition, but is **not available yet**.

### 3.4 Session model

**1 ACP session = 1 pipeline run.** Each `new_session` call, optional existing-project configuration, and exactly one `prompt` call on that session's `session_id` produces one pipeline run. Calling `prompt` a second time on the same `session_id` raises an error (`invalid_params`, "session already used for a prompt"). To start another run, call `new_session` again to get a fresh `session_id`.

There is no multi-turn steering or follow-up prompting within a session.

### 3.5 Progress streaming

While a pipeline run executes — this can take anywhere from roughly 20 seconds to several minutes depending on task complexity — the server sends `session_update` notifications carrying `agent_message_chunk` updates. These chunks contain live console output: planning progress, per-task dispatch progress, and test results.

Listen for these updates to display human-readable output. For machine-readable, queryable state, use the extension in section 3.6.

### 3.6 Structured run status

The server advertises a custom status request through `agentCapabilities._meta["agents-pipeline"].runStatus`. Query it during or after a prompt turn using the ACP extension method `_agents-pipeline/run/status` and `{ "sessionId": "<session id>" }`.

With the Python SDK, call `await conn.ext_method("agents-pipeline/run/status", {"sessionId": session_id})`. The SDK adds the required leading underscore to the JSON-RPC method name. A response has this stable shape:

```json
{
  "schema_version": 1,
  "session_id": "<session id>",
  "run_id": "0001-build-a-cli",
  "status": "running",
  "phase": "dispatch",
  "outcome": null,
  "cancel_requested": false,
  "queue_position": null,
  "tasks_total": 3,
  "tasks_started": 2,
    "tasks_completed": 1,
    "tasks_blocked": 0,
    "current_task": {
      "id": "TASK-auth",
      "category": "feature",
      "status": "running",
      "attempt": 1
    },
  "attempt_round": 1,
  "max_attempts": 3,
  "tests_blocking": true,
  "mode": "existing_project",
  "project_path": "/absolute/path/on/the/server/to/project"
}
```

`status` is `created`, `queued`, `running`, `completed`, `failed`, or `cancelled`. **`completed` does not mean passed.** `phase` describes the current stage (`queued`, `planning`, `dispatch`, `testing`, `retrying`, `teardown`, and terminal/initial states). `outcome` is null until known, then `passed`, `failed`, `passed_with_test_failures`, `error`, `cancelled`, or `invalid_prompt`. A test failure with `tests_blocking: false` in new-project mode is reported as `passed_with_test_failures`, not as a clean pass. Existing-project mode always requires passing validation and reports validation failures as `failed`. Require `outcome == "passed"` for a clean pipeline pass and handle cancellation separately: cancelled status can retain a known underlying outcome. `PromptResponse.stop_reason` remains a protocol turn status, not the pipeline outcome.

`queue_position` is one-based among waiting runs and null for the active run or a run without a queue ticket. Task counts describe specialist attempts: `tasks_total` counts planned work items; `tasks_started` counts attempts that began; `tasks_completed` counts attempts that returned passed or failed; retries increment these counters rather than resetting them, so completed attempts may exceed `tasks_total`. Dependency-blocked tasks are counted separately and do not count as started/completed attempts. `current_task` is null when no specialist task is active. `run_id` becomes available after run artifacts are created; `session_id` is available from session creation and remains the correlation key while queued.

### 3.7 Final response

When `prompt` resolves, `PromptResponse.stop_reason` is one of:

- **`"end_turn"`** — the ACP prompt turn completed. Query structured run status to determine whether the pipeline passed, failed, or completed with advisory test failures.
- **`"refusal"`** — no usable text was found, or an unexpected pipeline error occurred. Status reports `invalid_prompt` or `error` respectively.
- **`"cancelled"`** — the ACP prompt turn was cancelled. See section 3.9; this does not imply early termination of the pipeline.

### 3.8 Run serialization

Prompts submitted to one server process are queued FIFO. Only one full pipeline run executes at a time, covering planning, sequential task dispatch, tests/retries, progress delivery, and run teardown. Independent work items also remain sequential. This is process-local coordination only; separate server processes do not share the queue.

### 3.9 Cancellation caveat

Calling `session/cancel` is **best-effort only**. It sets `cancel_requested`, but does not stop the active pipeline or remove a queued run. The pipeline completes, including file writes and teardown, before the prompt responds with `stop_reason: "cancelled"`; structured status becomes `cancelled` and retains a known pipeline outcome where available. Cancelling the ACP request itself while queued removes that queue entry. Cancelling it while active waits for the blocking pipeline to finish before releasing the queue slot.

An unexpected pipeline error can still produce `status: "failed"`, `outcome: "error"` and `stop_reason: "refusal"` even when cancellation was requested.

## 4. Minimal example client (Python)

This is a minimal, self-contained example showing the full round-trip: spawn → `initialize` → `new_session` → `prompt` with a free-text task → print streamed chunks as they arrive → print the final `stop_reason`. Adapt the paths and prompt text for your own use.

```python
import asyncio
import json
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

        run_status = await conn.ext_method(
            "agents-pipeline/run/status", {"sessionId": session_id}
        )

    print(f"\n\nstop_reason: {prompt_resp.stop_reason}")
    print(json.dumps(run_status, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
```

This example is Python-specific, but the underlying protocol — JSON-RPC over stdio — is language-agnostic; an equivalent client can be implemented in any language with JSON-RPC and subprocess support.

For live status polling while `prompt` is in flight, query the extension described in [section 3.6](#36-structured-run-status) from a concurrent task. Query final status before closing the session or server: session state is in memory and is discarded by `close_session` or a server restart; historical artifacts remain on disk.

## 5. Known limitations (v1)

- **Free-text prompts only** — no file/document attachment support yet.
- **Best-effort cancellation only** — no preemptive abort of an in-flight run.
- **1 session = 1 pipeline run** — no multi-turn steering or follow-up prompting within a session.
- **Config loaded once at server process start** — restart the server process to pick up changes to `~/.config/Agents/config.json`.
- **Structured run status is an extension** — clients should check the advertised `_agents-pipeline/run/status` method; it is not a field on standard `PromptResponse`.
- **Queue scope is one process** — separate server processes may run independently and are not coordinated.
- **Per-run ACP cleanup** — each run closes its pooled ACP connections/subprocesses, temporary directories, and background event loop before the next queued run starts.

## 6. Where output lands

Each pipeline run creates `<runs-root>/<run-id>/`. `pipeline.runs_dir` resolves relative to the server repo root (default `runs/`); an absolute configured path is honored. Neither subprocess cwd nor `new_session.cwd` relocates that root. Status returns `run_id`, not an absolute path, so clients must know the server repo/config location. With default `pipeline.output_dir` and `pipeline.tests_dir`, the directory contains:

- **`product/`** — the generated code.
- **`tests/`** — the generated tests, plus `tests/results/` with per-attempt JSON test-result files.
- **`manifest.json`** — run status (`passed`, `failed`, `passed_with_test_failures`, etc.), timestamps, attempt count, and the original prompt/label.
- **`design/plan.json`** — version-2 task plan, explicit category and file ownership, dependencies, current task status, attempt results, and failure details.
- **`instructions.md`** — a human-readable summary of the run.
- Existing-project runs also include **`changes.json`** with project-relative `added`, `modified`, and `deleted` paths, a partial-edit warning on failure, and the resolved `project_path`. Run metadata remains under the pipeline server's configured run directory; it is not written into the target project by the pipeline.
- In existing-project mode, actual product/test edits are in `project_path`; the run's `product/` and `tests/` directories remain part of the artifact layout and are not used as an overlay.

This directory lives **on the server's filesystem**, at the configured location — it is not returned inline through ACP. If your automated flow needs the actual generated files (not just the streamed text summary), read them from this run directory after the `prompt` call returns.

## 7. Support / contact

Questions about this integration: open an issue or contact the team maintaining this repository.
