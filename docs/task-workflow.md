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
internal/trailing newlines. A complete Markdown fence surrounding the entire
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
A validation task's declared implementation dependencies are included when an
owned test/source asset needs repair. Dependents of changed tasks are also
retried because their outputs may be stale. Unrelated successful tasks are not
regenerated. Valid staged sibling files remain available on retries; rejected,
missing, stale, or explicitly implicated files are the only files regenerated.
An unowned/cross-file consistency failure without a safe file-specific mapping
invalidates the feature's staged set and carries bounded feedback into its next
request. Stage reuse requires the same plan identity, owned paths, feature
context, and existing-source snapshot; a mismatch resets that feature's stage.
`pipeline.max_retries` remains the bounded outer attempt budget; there is no
additional per-file retry loop. The optional feature-level self-check is one
bounded review call per complete staged feature attempt and sees staged source,
never unpublished product paths. Warning-only unused private declarations and
unsupported analysis limitations do not trigger repair retries. Runtime
tests/builds are external and do not participate in this retry loop. The legacy
`pipeline.tests_blocking` option is accepted but ignored.

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
analysis limitations. Per-attempt static findings and `fail_report.json` are
under `validation/results/`. `tests/` contains generated test source only.
Every run also includes `testing-handoff.md` and `testing-handoff.json` for
isolated external runtime testing and the versioned result contract.

## Breaking Migration

New runs require the version-2 task schema. The previous `type` plus scalar
`output_path` plan format is not read or converted. Existing `runs/` outputs are
left untouched and remain available for inspection. Start a new run through the
current free-text, handoff, or manifest planning path to generate a version-2
plan. External file manifests still use file entries, but are adapted and
validated into the common task model before dispatch.
