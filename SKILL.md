---
name: agents-pipeline
description: This skill guides agents integrating with or driving this repository's ACP code-generation pipeline.
---

# Agents Pipeline ACP Client

- From the repository root, start the server with its virtualenv Python and `src/pipeline/acp_server.py`. Keep stdin/stdout reserved for ACP JSON-RPC; the process communicates over stdio.
- Follow the ACP handshake: `initialize`, then create a fresh `new_session` for each run, then send exactly one `prompt` containing free text. Non-text prompt blocks are ignored; inline any file contents as text.
- For existing-project edits, call `conn.ext_method("agents-pipeline/session/configure", {"sessionId": session_id, "mode": "existing_project", "projectPath": "/absolute/server/path/to/project"})` after `new_session` and before `prompt`. The path must exist on the server. Each complete, validated feature is published as a group using per-file replacements and best-effort rollback; feature-wide publication is not atomic, later task/run failures do not undo earlier successful features, and rollback failures may leave partial edits. Inspect `changes.json`.
- Check the advertised `agents-pipeline/run/status` extension during and after the prompt; in the Python SDK, call `conn.ext_method("agents-pipeline/run/status", {"sessionId": session_id})` (omit the leading underscore; raw JSON-RPC uses the advertised method). Schema 2 reports `outcome: "passed"` only when the implementation assignment completed, with `outcome_scope: "implementation"`. This is compatible with AI-SDLC's external Code bridge. `pipeline_outcome: "validation_passed_external_testing_pending"` and `external_testing_status: "pending"` separately record static success and unexecuted runtime tests. `end_turn` alone is not a successful assignment. Handle cancellation separately because cancellation does not stop active work or undo writes.
- Read generated output from the server filesystem, not the ACP response. Status exposes the absolute `run_dir` and finalized `implementation_result_path`; run artifacts otherwise default to the server repository's `runs/<run-id>/`, subject to `pipeline.runs_dir`. In existing-project mode, edits are in the configured project directory.
- Configure the server with `~/.config/Agents/config.json` (takes precedence) or the repository-root `config.json` when the home config is absent. Set the LiteLLM proxy URL/key and model names; every model role must use `provider: "acp"`. Install `opencode` with ACP support on `PATH`: internal calls use a temporary HOME/config, not normal OpenCode settings or credentials.
- Static validation does not import or execute generated code, install application dependencies, or run generated-project tests/builds. TypeScript/JavaScript/JSX syntax/type and unused-symbol coverage is limited and reported in the handoff. Pending file sources are kept under the run's `staging/` directory and are excluded from generated-project and external-testing inventory. In existing-project mode, published feature edits happen in place and may remain after later failure. Review `changes.json` and do not test directly in the original project directory.
- Review and validate generated changes yourself. This pipeline writes files locally; it does not create or submit pull requests automatically.

## AI-SDLC external implementer

Use the host's registered external-implementer route with `bind: git` for this
checkout, so ignored runtime directories (`runs/`, `venv/`, `node_modules/`)
are excluded from the skill fingerprint. Updated source/skill files require
the host's provider activation to be refreshed. Resolve these executable
and script paths against this registered skill directory before launching:

```text
venv/bin/python src/pipeline/acp_server.py
```

Configure method: `agents-pipeline/session/configure`.
Status method: `agents-pipeline/run/status`.
The host ACP client adds the leading underscore on the wire. Configure
`mode: "existing_project"` and `projectPath` equal to the host's Code worktree,
then submit the approved request once. Work4Me edits that worktree in place,
keeps run artifacts outside it, and does not commit or push.

After the host client reports `done`, use `provider_status.outcome == "passed"`
for Code-assignment completion. Read `implementation_result_path` for the
provider's task/coverage record. Its `tasks` have stable `T-001` ids in plan order,
the original `work_item_id`, explicit `done`/`in_progress`/`todo` status, and
published file paths. Copy only its declared `ac_coverage`; it includes exact
`AC-<number>` references attached to completed tasks. Do not infer new AC claims.
`lint` is `not_run` and `unit_tests` is null because this implementation pipeline
does not execute generated code.

The host session writes its own `tasks.md` and `code_manifest.json` in its Code
output folder and the host creates its index. Use `delivery_mode: "working_tree"`
and empty commit/PR values. For runtime evidence, require the separate isolated
testing result. The retired greenfield Work4Me importer assumes older paths,
console pass markers, and internal blocking tests; it is not the current bridge.

For the complete protocol, configuration, status schema, cancellation behavior, and artifact layout, see [the ACP integration guide](docs/acp-integration.md). For task categories, file ownership, retries, and run-plan details, see [task planning and execution](docs/task-workflow.md).
