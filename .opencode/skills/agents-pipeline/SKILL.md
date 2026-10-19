---
name: agents-pipeline
description: This skill guides agents integrating with or driving this repository's ACP code-generation pipeline.
---

# Agents Pipeline ACP Client

- Start the server with the repository's virtualenv Python and `pipeline/acp_server.py`. Keep stdin/stdout reserved for ACP JSON-RPC; the process communicates over stdio.
- Follow the ACP handshake: `initialize`, then create a fresh `new_session` for each run, then send exactly one `prompt` containing free text. Non-text prompt blocks are ignored; inline any file contents as text.
- For existing-project edits, call `conn.ext_method("agents-pipeline/session/configure", {"sessionId": session_id, "mode": "existing_project", "projectPath": "/absolute/server/path/to/project"})` after `new_session` and before `prompt`. The path must exist on the server. Edits happen in place and are not rolled back; inspect `changes.json` and assume partial edits may remain after failure.
- Check the advertised `agents-pipeline/run/status` extension during and after the prompt; in the Python SDK, call `conn.ext_method("agents-pipeline/run/status", {"sessionId": session_id})` (omit the leading underscore; raw JSON-RPC uses the advertised method). Do not treat `stop_reason: "end_turn"` or `status: "completed"` as success; require `outcome == "passed"` for a clean pass. Handle cancellation separately because cancellation does not stop active work or undo writes.
- Read generated output from the server filesystem, not the ACP response. `run_id` is an ID, not a path; by default artifacts are under the server repository's `runs/<run-id>/`, subject to `pipeline.runs_dir`. In existing-project mode, edits are in the configured project directory.
- Configure the server with `~/.config/Agents/config.json` (takes precedence) or the repository-root `config.json` when the home config is absent. Set the LiteLLM proxy URL/key and model names; every model role must use `provider: "acp"`. Install `opencode` with ACP support on `PATH`: internal calls use a temporary HOME/config, not normal OpenCode settings or credentials.
- Review and validate generated changes yourself. This pipeline writes files locally; it does not create or submit pull requests automatically.

For the complete protocol, configuration, status schema, cancellation behavior, and artifact layout, see [the ACP integration guide](../../../docs/acp-integration.md). For task categories, file ownership, retries, and run-plan details, see [task planning and execution](../../../docs/task-workflow.md).
