# External Testing Agent

The external testing agent is a separate, manually delegated ACP process. It consumes a pipeline run's `testing-handoff.json`, snapshots the selected project, installs dependencies in Docker, executes recognized checks, and writes evidence beside the run artifacts. Per-file in-progress staging under the pipeline run's `staging/` directory is not part of the handoff or source snapshot; only published product/test files are inventoried. The implementation pipeline does not launch it, repair from its result, or update static-validation state.

## Prerequisites

- Docker Engine and the `docker` CLI must be available to the agent process.
- Install this repository's Python requirements in its virtual environment; `agent-client-protocol` is already included.
- The host needs network access to pull the configured official runtime image. Dependency downloads are allowed only during isolated setup.

Default images are explicit tags: `python:3.12.11-slim-bookworm` and `node:22.16.0-bookworm-slim`. They are not `latest` tags. Configure alternative exact official Docker Hub tags in `~/.config/Agents/testing-agent.json`:

```json
{
  "python_image": "python:3.12.11-slim-bookworm",
  "node_image": "node:22.16.0-bookworm-slim",
  "uv_version": "0.8.15",
  "setup_timeout_seconds": 900,
  "test_timeout_seconds": 900,
  "docker_timeout_seconds": 900,
  "memory_limit": "2g",
  "cpu_limit": 2,
  "pids_limit": 256
}
```

Setup commands, build/test commands, and individual Docker operations each
have a 15-minute maximum by default. Each command has its own deadline, so a
testing run with multiple setup/build/test commands can take longer overall.
Container/process cleanup keeps short bounded waits to avoid hanging shutdown.

The selected runtime must match recognized exact runtime declarations in `.python-version`, `.nvmrc`, or `.node-version`; simple minimum/runtime-major declarations are also checked. Unrecognized ranges are reported as a limitation. No host environment variables or credentials are passed into containers.

## Manual CLI invocation

For a generated-project run:

```sh
venv/bin/python src/testing_agent/cli.py \
  --handoff /absolute/path/to/runs/0001-example/testing-handoff.json
```

Existing-project runs require explicit authorization of the exact root recorded in both the handoff and `changes.json`:

```sh
venv/bin/python src/testing_agent/cli.py \
  --handoff /absolute/path/to/runs/0002-existing/testing-handoff.json \
  --project-root /absolute/path/to/edited-project
```

The handoff and run manifest must correlate by `run_id`, the pipeline manifest must still report `validation_passed_external_testing_pending`, and the handoff's external-testing status must be `pending`. Results are created in a unique `external-testing/<run-id>-<testing-run-id>/` directory. Existing result folders are never reused or overwritten.

## ACP invocation

Launch the server as a separate process:

```sh
venv/bin/python src/testing_agent/acp_server.py
```

Create an ACP session whose `cwd` is the run directory containing `testing-handoff.json`, then send exactly one prompt:

```text
run /absolute/path/to/runs/0001-example/testing-handoff.json
```

For existing-project mode, explicitly include the authorized path:

```text
run /absolute/path/to/runs/0002-existing/testing-handoff.json --project-root /absolute/path/to/edited-project
```

Quote path arguments containing spaces using shell-style quotes. The prompt is parsed as a small argument list; no host shell command is accepted. Progress is streamed through ACP `session_update`. Query `_agents-testing-agent/run/status` with `{"sessionId":"…"}` for structured status and the result path. `session/cancel` stops the active Docker container and records `cancelled`.

## Runtime and command policy

- **Python:** uses a root `uv.lock` with `uv sync --locked --all-groups --all-extras`, or installs the declared root `requirements.txt`, `requirements-dev.txt`, and `requirements-test.txt` files with pip. `poetry.lock` and `Pipfile.lock` are detected but currently return `setup_failed`. Recognized test files run with declared `pytest` or Python `unittest.TestCase` suites. A positive test count is required for `passed`.
- **Node.js / TypeScript:** requires a root `package.json` and `package-lock.json`; dependencies use `npm ci`. Other lockfile/package-manager combinations return `setup_failed`. If `packageManager` declares an npm version, the version in the configured runtime image must match. Standard test filenames plus a `test` npm script are required. Recognized Vitest, Jest, Mocha, Node test-runner, and TAP output must report a positive executed-test count before a pass is asserted.
- A root npm `build` script may also run and is recorded separately. A successful build alone results in `no_tests`, never `passed`.
- Commands declared in the handoff are advisory and are never executed as shell text. Node tests run only a directly declared Vitest, Jest, Mocha, TAP, or `node --test` command; npm shell wrappers are rejected. A recognized npm `build` script and npm install lifecycle hooks can still run arbitrary project code **inside Docker**.
- Missing manifests, unsupported lockfiles, incompatible runtime declarations, missing test runners, unavailable Docker, and setup timeouts are recorded as setup failures. Test command failures/timeouts are recorded in command evidence; user cancellation is `cancelled`.

The pipeline handoff reports `external_testing.lock_preparation`. `lock_pending`
means an npm project has no root `package-lock.json`; an operator must prepare
the lockfile in the isolated project workflow before delegating this tester.
The pipeline does not run `npm install`, resolve application dependencies, or
generate the lockfile. If a request explicitly selected pnpm, Yarn, or Bun, the
pipeline preserves that manager and reports `unsupported_package_manager`
because this tester currently supports npm only.

## Result and evidence

Each run writes:

- `testing-result.json` — schema version 1, pipeline `run_id`, unique `testing_run_id`, outcome, source SHA-256, runtime/image, setup status, command statuses/exit codes/durations/timeouts, test counts, cleanup status, limitations, and artifact paths.
- `report.md` — readable summary with the same correlation and evidence.
- `logs/` — bounded, redacted command output.
- `source-snapshot/` — the immutable sanitized source tree whose digest is reported.

Example result excerpt:

```json
{
  "schema_version": 1,
  "run_id": "0001-example",
  "testing_run_id": "<unique-id>",
  "outcome": "passed",
  "source_snapshot": {
    "path": "source-snapshot/",
    "sha256": "<digest>",
    "file_count": 8
  },
  "environment_setup": {
    "status": "succeeded",
    "image": "python:3.12.11-slim-bookworm",
    "network_policy": "bridge only during dependency setup; disconnected before build/test commands"
  },
  "test_evidence": {
    "status": "verified",
    "framework": "unittest",
    "test_count": 2,
    "passed_count": 2,
    "failed_count": 0
  }
}
```

Outcomes are `passed`, `tests_failed`, `setup_failed`, `no_tests`, or `cancelled`. A timeout remains visible as `timed_out` on its command record. The caller must consume the result explicitly and correlate both `run_id` and `source_snapshot.sha256`; no component imports the result or changes `manifest.json` / handoff status. The pipeline therefore continues to show external testing as pending until a future explicit result-recording workflow is implemented.

## Isolation limits

The source is copied to a disposable local snapshot, then streamed into the container through `docker exec -i tar`; the original project is never mounted. The container has a read-only root filesystem, a bounded tmpfs workspace, non-root UID, all Linux capabilities dropped, `no-new-privileges`, memory/CPU/PID limits, no host mounts, no Docker socket, and no privileged mode. It starts on Docker's bridge network for image/dependency setup, then disconnects before build/test commands. Containers are forcibly removed in `finally`; cleanup failure is recorded as `setup_failed`.

The snapshot omits common VCS, cache, virtualenv, dependency/vendor, build-output, dotenv, credential-file, private-key, and symlink entries. The policy is filename/path based and cannot identify every secret embedded in ordinary source/config files. Logs are capped at 64 KiB per command and redact common token/credential patterns, but redaction is heuristic.

Docker containers share the host kernel and Docker daemon; they are not a VM boundary. The daemon and its access permissions must be trusted. Dependency installers and npm lifecycle scripts receive network access during setup and may execute arbitrary code within the container. Tests/builds run with container networking disconnected, but malicious code can still attack the container/kernel boundary or exhaust resources up to configured limits. Review the source and use an appropriately isolated Docker host for untrusted projects.
