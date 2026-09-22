# ACP Integration Guide

This document describes how to invoke this repository's automated multi-agent code-generation pipeline as an [Agent Client Protocol (ACP)](https://agentclientprotocol.com) Agent/server, for teams integrating it into their own automated flows.

## 1. What this is

This repository implements an automated multi-agent code-generation pipeline (planning → work-item dispatch → specialist code generation → test execution/retry). That pipeline is exposed as a standard **ACP Agent** — a subprocess that any ACP-compliant client can spawn, initialize, and drive through a `new_session` → `prompt` handshake to run one end-to-end pipeline run, with progress streamed back live as the run executes.

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

### 2.2 Create your config file

The server loads its config from **`~/.config/Agents/config.json`** first; if that file doesn't exist, it falls back to the repo's own `config.json` at the project root. **You must place your config at `~/.config/Agents/config.json`** on the machine/environment where you run the server — placing it anywhere else (or relying on the repo's own `config.json`) will not pick it up unless the repo's own `config.json` is what you intend to use.

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
- **`pipeline.runs_dir`** — directory (relative to the repo root) where per-run output directories are created. See [section 6](#6-where-output-lands).
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

**`prompt`** — sent once per session, with your task description.

- Request: `session_id` (from `new_session`), `prompt` (a list of content blocks — see [3.3](#33-prompt-contract-v1)).
- Response (`PromptResponse`): `stop_reason` — see [3.6](#36-final-response).

### 3.3 Prompt contract (v1)

**Free-text only.** The `prompt` request's list of content blocks is scanned for `TextContentBlock` entries; their `.text` values are joined together (space-separated) and passed to the pipeline as the task description (equivalent to typing a request into this pipeline's own interactive REPL).

**Other content block types are ignored in this version** — images, embedded resources/files, and any other non-text block type in the `prompt` list are silently skipped. This is a known v1 limitation. If no text content is found at all, the server responds with `stop_reason: "refusal"` (see [3.6](#36-final-response)) rather than starting a pipeline run.

File-attachment support (e.g. passing a design document alongside a task prompt) is a likely near-future addition, but is **not available yet**.

### 3.4 Session model

**1 ACP session = 1 pipeline run.** Each `new_session` call, followed by exactly one `prompt` call on that session's `session_id`, produces one pipeline run. Calling `prompt` a second time on the same `session_id` raises an error (`invalid_params`, "session already used for a prompt"). To start another run, call `new_session` again to get a fresh `session_id`.

There is no multi-turn steering or follow-up prompting within a session.

### 3.5 Progress streaming

While a pipeline run executes — this can take anywhere from roughly 20 seconds to several minutes depending on task complexity — the server sends `session_update` notifications on the connection, each carrying an `agent_message_chunk` update. These chunks contain the pipeline's live console output: planning progress, per-work-item dispatch progress, and test results, streamed as they're produced (not batched at the end).

Your client should listen for these notifications and surface them (display/relay/log) rather than waiting silently for the final `PromptResponse` — this is the only place run progress is visible before the run finishes.

### 3.6 Final response

When `prompt` resolves, `PromptResponse.stop_reason` is one of:

- **`"end_turn"`** — the pipeline ran to completion. This does **not** by itself mean the run succeeded — check the streamed `session_update` text for `"✅ Pipeline passed"` or `"❌ Pipeline failed"` to determine the actual outcome. There is no separate structured pass/fail field on the response; the outcome is communicated only via the streamed text.
- **`"refusal"`** — either no usable text was found in the prompt's content blocks, or an unexpected internal error occurred while running the pipeline. In the error case, an error message is streamed via `session_update` before this response is returned.
- **`"cancelled"`** — a `session/cancel` notification was received for this session before/during the run. See the cancellation caveat below — this does **not** mean the pipeline was stopped early.

### 3.7 Concurrency

Multiple sessions/prompts can run concurrently against the same server process — there is no artificial serialization forcing one run to wait for another.

### 3.8 Cancellation caveat

Calling `session/cancel` on a session is **best-effort only**. It does **not** preemptively stop an in-flight pipeline run — the pipeline runs to completion regardless. Cancellation only affects how the *eventual* response is labeled: if `cancel` was called before the run's `prompt` call returns, that call's `PromptResponse.stop_reason` will be reported as `"cancelled"` instead of `"end_turn"`, but the run itself will have executed in full (including any files it wrote). **True abort/early-termination behavior does not exist in this version.** Do not rely on cancel to stop resource usage or side effects from an in-flight run.

## 4. Minimal example client (Python)

This is a minimal, self-contained example showing the full round-trip: spawn → `initialize` → `new_session` → `prompt` with a free-text task → print streamed chunks as they arrive → print the final `stop_reason`. Adapt the paths and prompt text for your own use.

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


if __name__ == "__main__":
    asyncio.run(main())
```

This example is Python-specific, but the underlying protocol — JSON-RPC over stdio — is language-agnostic; an equivalent client can be implemented in any language with JSON-RPC and subprocess support.

## 5. Known limitations (v1)

- **Free-text prompts only** — no file/document attachment support yet.
- **Best-effort cancellation only** — no preemptive abort of an in-flight run.
- **1 session = 1 pipeline run** — no multi-turn steering or follow-up prompting within a session.
- **Config loaded once at server process start** — restart the server process to pick up changes to `~/.config/Agents/config.json`.
- **No structured pass/fail field** — the run outcome must be parsed from streamed text (`"✅ Pipeline passed"` / `"❌ Pipeline failed"`), not from `PromptResponse` fields.

## 6. Where output lands

Each pipeline run creates a new directory under this repo's `runs/` directory (e.g. `runs/0007-build-a-cli-that-reverses-a-string/`), containing:

- **`product/`** — the generated code.
- **`tests/`** — the generated tests, plus `tests/results/` with per-attempt JSON test-result files.
- **`manifest.json`** — run status (`passed`, `failed`, `passed_with_test_failures`, etc.), timestamps, attempt count, and the original prompt/label.
- **`instructions.md`** — a human-readable summary of the run.

This directory lives **on your own machine/environment**, alongside wherever you're running the pipeline server — it is not something you fetch remotely. If your automated flow needs the actual generated files (not just the streamed text summary), read them from this run directory after the `prompt` call returns.

## 7. Support / contact

Questions about this integration: open an issue or contact the team maintaining this repository.
