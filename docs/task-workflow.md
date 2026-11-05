# Task Planning And Execution

The pipeline plans cohesive architecture and feature tasks. A task may own
related frontend and backend files. It is split only when complexity or an
architecture boundary justifies separate ownership, never just to make one task
per file.

## Categories

| Category | Responsibility |
| --- | --- |
| `foundation` | Project setup and primitives used across modules. |
| `feature` | One cohesive end-user capability, potentially across frontend/backend. |
| `shared` | Cross-feature contracts, schemas, and utilities with one owner. |
| `integration` | Application composition, entry points, and wiring. |
| `validation` | Automated tests and validation assets, written under `tests_dir`. |

Workers are selected by category. Every category uses multi-file output. There
is no longer a `logic`/`ui`/`config` single-file task path.

## Common Task Model

Free-text planning (`runner.py` and ACP), approved design handoffs
(`ingest.py --mode handoff`), and file-manifest planning (LLM or external
strategy) all produce the same model:

```json
{
  "id": "TASK-auth",
  "category": "feature",
  "language": "typescript",
  "module": "authentication",
  "scope": "Sign-in and session renewal across the web client and API",
  "title": "Authentication",
  "description": "Implement the user-facing authentication capability.",
  "acceptance_criteria": ["A valid user can sign in and renew a session."],
  "requirement_refs": ["REQ-001", "TEST-004"],
  "owned_files": [
    {"path": "src/auth/session.ts", "language": "typescript", "area": "product"},
    {"path": "src/auth/SignIn.tsx", "language": "typescript", "area": "product"}
  ],
  "depends_on": ["TASK-foundation"],
  "expected_interfaces": ["Exports createSession(credentials) and renewSession(token)."]
}
```

`owned_files.path` is relative to its area. `area` is `product` or `tests`; the
configured directory is added once. A leading configured directory prefix is
normalized, so a validation file named `tests/foo.py` does not become
`tests/tests/foo.py`. Every task requires at least one owned file and acceptance
criterion. Paths are normalized and safe relative paths. A physical normalized
file has exactly one owner in a plan.

Tasks declare dependency ids and the expected interfaces they provide. Feature
and module scope, requirement references, complete file ownership, and
dependencies are explicit instead of being hidden in free-form descriptions.
The scheduler is sequential: a task waits for dependencies to pass; failed
dependencies leave consumers blocked.

## Worker Output And Safety

A WorkItem remains the single feature/owner and retains all of its owned paths,
acceptance criteria, dependencies, and ordering. Its specialist makes one
request per exact normalized owned path and returns that file's source, never a
JSON map or a model-selected path. Every request includes the complete WorkItem
plan, dependency context, existing-project context, and any accepted sibling
files already staged for that feature. It identifies one current target path;
earlier ACP turns and other owned paths are context only.

The response is framed with request-unique `BEGIN-OWNED-FILE:<nonce>` and
`END-OWNED-FILE:<nonce>` lines around raw source. The delivery layer removes
those framing lines and one separator newline, preserving quotes and meaningful
internal/trailing newlines. LF and CRLF framing are both accepted without
normalizing source line endings. A complete Markdown fence surrounding the entire
framed response is accepted and removed; an unterminated outer fence, mismatched
or stale nonce, or malformed framing is rejected. Fences inside the source body
are preserved as source. ASCII controls other than line feed, carriage return,
and tab are rejected. ACP must report `stopReason: "end_turn"`; empty, refused,
cancelled, and other known stop reasons are not accepted as complete source.
When a provider does not expose completion metadata, it remains
`unsupported` and file delivery rejects it rather than claiming completion.

An actually zero-byte file requires the exact request-specific
`ACCEPT-EMPTY-FILE:<nonce>` action as its entire body. Existing-project deletion
requires the exact request-specific `INTENTIONAL-DELETE-FILE:<nonce>` action;
this action is unavailable for generated projects. Missing, malformed, or
empty source never implies deletion. Nonce-bound markers keep actions distinct
from normal source and prevent previous-file responses from matching a new
request.

Accepted files live under `staging/<task-key>/` with schema-versioned
`stage.json`, source digest, file attempts, action, status, and bounded failure
details. Staging is outside `product/` and `tests/`, so pending/partial files
are not generated-project or external-testing handoff inventory. The complete
feature is checked for completeness, supported syntax/manifests, ownership, and
local relative Python imports against staged files plus existing and published
dependency files before any of that feature's actions are published. The
validator reads source only; it does not import or execute generated code,
install application dependencies, or launch tests/builds.

Rejected response bodies are retained separately under
`diagnostics/rejected/<task-key>/` as source-only evidence plus bounded framing
and completion metadata. The directory is owner-only, files are mode 0600, and
raw source retention has per-candidate and per-run byte budgets. Prompts,
provider logs, and configured API keys are not copied into these records. Stage
resets do not erase rejected-candidate evidence. Diagnostics are excluded from
product, tests, and external handoff inventories.

For new Node projects, planning requires exact ownership of root
`package.json` and `tsconfig.json` (TypeScript) or `jsconfig.json` (JavaScript).
The manifest task acceptance criteria cover the requested runtime, manager,
runtime dependencies, and test command. An explicitly requested manager is
preserved; npm is the default when none is named. Existing-project changes do
not require replacement scaffolding. The pipeline does not invent or resolve an
application lockfile. An npm project without `package-lock.json` is handed off
as `lock_pending`; external lock preparation is a separate manual step. The
external tester currently supports npm lockfiles only.

Publication uses per-file same-filesystem replacements and backups. A feature's
publication set is not filesystem-atomic. If any write/delete fails, changed
paths are best-effort restored and rollback failures are recorded in stage and
run artifacts; existing-project `changes.json` also records any remaining
partial changes. A later feature failure does not automatically roll back
previously published, validated features. Successful dependency outputs are
read only from published files.

Dependency outputs are loaded for every declared dependency, not only its first
file. Dispatch allows up to 40 files, 12,000 characters per file, and 80,000
characters total. Worker prompts show up to 20 files and 30,000 total
characters, with 3,000 characters per file. Any paths omitted or truncated by
these budgets are called out in the context.

## Retry Behavior

Deterministic static-validation errors map to the task owning the failed file.
Missing files from blocked or failed tasks are incomplete dispatch state, not
defects in implementation dependencies. A failing validation task is repaired
as that task; its implementation dependencies are not regenerated merely
because a test was missing or invalid. Successful dependents are regenerated
only when structured review identifies an affected contract. Dependents that
were blocked and never generated are dispatched after their selected failed
prerequisite is repaired. Unrelated successful tasks are not regenerated. Valid
staged sibling files remain available; only rejected, missing, stale, or
explicitly implicated files are regenerated. Review results report exact
owned paths and permitted downstream task ids. Reliable scoped findings
invalidate only those paths; unknown paths or unscoped findings use an explicit
conservative feature-wide fallback. Stage reuse requires the same plan identity, owned paths, feature
context, and existing-source snapshot; a mismatch resets that feature's stage.
`pipeline.max_retries` remains the bounded outer attempt budget; there is no
additional per-file retry loop. The optional feature-level self-check is one
bounded review call per complete staged feature attempt and sees staged source,
never unpublished product paths. Warning-only unused private declarations and
unsupported analysis limitations do not trigger repair retries. Runtime
tests/builds are external and do not participate in this retry loop. The legacy
`pipeline.tests_blocking` option is accepted but ignored.

## TypeScript/JavaScript Static Coverage

The pipeline-owned checker is pinned to TypeScript 6.0.3 in
`tools/typescript-checker/`. It uses a fixed `noEmit` compiler profile and a
restricted compiler host over the authorized source overlay (staged files,
published dependencies, and bounded project sources) plus TypeScript's trusted
declaration libraries. It ignores generated `tsconfig.json` options, plugins,
scripts, and package dependencies. Input is never executed, imported by Python,
installed, built, or tested. Limits are 30 seconds, 512 MiB V8 heap, 2,000
files, 4 MiB source text, and 200 diagnostics. Static results record
`typescript_status` (`checked`, `unsupported`, or `not_applicable`); missing
tooling and budget exhaustion are explicit limitations, not silent passes.
Provision the trusted checker with
`npm ci --ignore-scripts --prefix tools/typescript-checker`; this installs only
the lock-pinned pipeline tool and never resolves generated-project dependencies.

TS/TSX/JS/JSX/MJS/CJS syntax and parser-derived imports are inventoried.
Machine-extractable named interface/type contracts are checked with TypeScript
structural assignability in both directions. Free-text semantic requirements
remain review/acceptance concerns. Own-source TypeScript errors block when type
origins resolve. Unresolved external modules and Node builtin typings (including
`node:async_hooks`) contaminate their source and transitive dependents; semantic
diagnostics there are suppressed with explicit limitations. The checker does
not install `@types/node` or application dependencies. JavaScript/JSX
semantic checking and unused-symbol analysis are unsupported. TypeScript also
cannot establish domain meaning such as whether a UUID identifies a team or a
user.

## Progress Counters

The ACP run-status response reports `tasks_total`, `tasks_started`,
`tasks_completed`, and `tasks_blocked`. Total counts unique planned WorkItems.
Started/completed count specialist attempts, not unique WorkItems: retries add
attempts monotonically, so completed can exceed total. Blocked counts only
WorkItems explicitly reported blocked by dependency resolution; planned but
not-yet-dispatched WorkItems are not counted as blocked. The current task is
reported separately with its ID, category, status, and per-task attempt number.
See [ACP Integration](acp-integration.md#36-structured-run-status) for the
query method and complete response contract.

## Run Artifacts

`design/plan.json` is versioned and updated during execution:

```json
{
  "format_version": 2,
  "tasks": [
    {
      "id": "TASK-auth",
      "category": "feature",
      "status": "passed",
      "attempts": [
        {
            "number": 1,
            "status": "passed",
            "files_written": ["product/src/auth/session.ts"],
            "file_attempts": [
              {"path": "product/src/auth/session.ts", "number": 1, "status": "published", "digest": "<sha256>"}
            ],
            "result_summary": "Wrote 1 file(s)",
          "failure_details": ""
        }
      ]
    }
  ]
}
```

Each task status is `blocked`, `ready`, `running`, `passed`, or `failed`.
Attempt history records generated files, result summary, per-file attempt
records, and failure details. Task `passed` means that the complete owned
feature was deterministically checked and published; it does not mean the
generated project passed runtime tests. During dispatch, ACP `current_task.files`
reports per-file attempt, status, digest, failure, and available completion/stop
metadata while task-count and status fields retain their existing meaning.
`manifest.json` records
`validation_passed_external_testing_pending`, `validation_failed`, or a
pipeline/dispatch `failed` status, plus `validation_status`, warnings, and
analysis limitations. Schema version 3 also records plan/toolchain/config
compatibility identities and a published-file SHA-256 snapshot. Per-attempt
static findings (including parser-derived import inventory) and
`fail_report.json` are
under `validation/results/`. `tests/` contains generated test source only.
Every run also includes `testing-handoff.md` and `testing-handoff.json` for
isolated external runtime testing and the versioned result contract.

`implementation-result.json` exposes completed task and declared AC/file
coverage for an external Code-stage host. Its deterministic `T-001` ids map
back to original WorkItem ids without changing the execution plan. Runtime
test counts remain null and lint remains `not_run`. ACP schema 2 uses
`outcome: "passed"` for implementation completion and keeps the explicit
static outcome in `pipeline_outcome`; manifest and testing-handoff semantics
remain the same. In AI-SDLC existing-project mode the host owns its Code
artifacts and index; the provider updates only the assigned source worktree.

## Explicit Resume

Resume creates a child run; the source run's manifest, plan, stage, and
diagnostics remain unchanged. Preflight requires a finalized schema-3 source
run, exact task ids, compatible output/config/toolchain identity, unchanged
published source digests, and verified stages for passed dependencies. The
child imports only hash-verified published files and selected stage/attempt
history, skips paid planning, and dispatches the selected repair set plus
declared blocked/contract-affected closure. `resume-history.json` and manifest
`resume` provenance link the child to its source.

```sh
venv/bin/python src/pipeline/resume_cli.py \
  --run 0004-original-run \
  --task TASK-feature-auth
```

For existing-project runs, add `--project-root` with the exact original
project path. ACP clients configure `_agents-pipeline/run/resume` with
`sessionId`, `runId`, and exact `taskIds`, then send the prompt `resume` in
that session. A legacy run such as `0018` lacks the required schema/tooling/source
identity; it needs an authorized plan amendment before resume.

## Breaking Migration

New runs require the version-2 task schema and schema-3 run provenance. The previous `type` plus scalar
`output_path` plan format is not read or converted. Existing `runs/` outputs are
left untouched and remain available for inspection. Schema-2 plan artifacts
remain readable but cannot safely resume because their tooling/config/source
identities are missing. Start a new run through the current free-text, handoff,
or manifest planning path to generate a compatible plan. External file
manifests still use file entries, but are adapted and validated into the common
task model before dispatch.
