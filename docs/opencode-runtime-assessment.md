# OpenCode runtime assessment

Assessment date: 2026-11-05. Examined the current `feature/v4` working tree,
including its uncommitted changes. Installed OpenCode version: `1.19.0`.
This is an architecture assessment; the replacement runtime has not been built
or exercised against a model.

## Recommendation and objective

Use OpenCode as the coding agent runtime, with a primary coordinator delegating
cohesive tasks to tool-capable implementation and review agents. Keep the Python
application as a small supervisor that owns workspaces, run records, limits,
verification evidence, cancellation, and publication.

For its intended integration with `/home/nutthawat/web-sdlc-proposal`, this is
the **Code-stage provider's** supervisor. AI-SDLC remains responsible for the
SDLC state machine, approved Requirement/Design inputs, human checkpoints,
bound Code worktree, stage artifacts, and final acceptance. Work4Me's internal
coordinator and specialists operate within that Code assignment.

The intended workflow is: receive an objective, inspect the project, make and
revise a plan, delegate work, inspect actual changes, run isolated checks, and
direct repairs until the acceptance criteria pass or an explicit limit is
reached. Routine work between the coordinator and specialists should proceed
without a person relaying messages. Missing essential requirements or limits
that prevent completion must produce a recorded outcome rather than an
indefinite wait.

This is a substantial redesign of the execution core. The public ACP interface
and much of the run/test infrastructure can be retained. Changing ACP transport
to HTTP alone would not deliver the objective.

## Compatibility with web-sdlc-proposal

The host checkout was inspected on 2026-11-05. A headless OpenCode execution
core can remain compatible if Work4Me retains its external ACP facade. The
host's Code worker should not have to use OpenCode's HTTP API directly.

There are two integration paths in the host source:

- **Current external-implementer mechanism:** `application/agent_bridge.py`
  supplies AI-SDLC's detached `interfaces/acp_run.py` client, starts the server
  named by the provider's registered `SKILL.md`, configures
  `{"mode":"existing_project","projectPath":"<Code worktree>"}`, and reads
  the advertised status method. This matches the existing Work4Me configure
  extension. Work4Me edits the supplied worktree; the host session writes
  `tasks.md` and `code_manifest.json`, with the host generating its index.
  Registration/domain/prompt code and unit tests implement this mechanism,
  despite the tracked README still describing the general SkillAgentRunner
  direction as planned. Code establishes current support; a live Work4Me run
  through this route is not established by that support.
- **Older dedicated Work4Me bridge:** `runners/acp/work4me_code.py` and
  `work4me_handoff.py` still exist, but the tracked README labels this
  integration cancelled/superseded. It assumes greenfield provider-owned
  output and imports files into Code. Its assumptions do not match today's
  provider, so it should not define the replacement architecture.

Compatibility findings from the assessment, with the v4 alignment applied:

1. The current external-implementer instructions classify `done` with
   `provider_status.outcome == "passed"` as completed and every other outcome
   as failed. V4 now reports ACP schema-2 `outcome: "passed"` with
   `outcome_scope: "implementation"` only after successful finalization and
   teardown. `pipeline_outcome` retains
   `validation_passed_external_testing_pending`; external testing remains
   pending. The required `implementation-result.json` records host task ids,
   declared AC/file coverage, and null runtime test counts for the host's
   artifacts. SKILL.md documents this current contract.
2. The old bridge and standalone sender require
   `<WORK4ME_ROOT>/pipeline/acp_server.py`; this checkout's entry point is
   `src/pipeline/acp_server.py`. The provider skill now explicitly identifies
   that command and the configure/status methods for the current bridge.
3. The old bridge requires the console marker `✅ Pipeline passed`,
   `manifest.status == "passed"`, `tests_blocking == true`, and old internal
   test-result files. This provider deliberately emits different static-only
   success semantics. It also expects Python-only source/tests, task/AC
   sidecars, and particular output paths.
4. The generic ACP runner rejects Code when `execution_policy_version == 1`
   (managed commands), and rejects isolated artifact authoring version 1. It
   also lacks stage-MCP mounting/native policy attestation. These restrictions
   concern direct generic-ACP stage selection; the external-implementer route
   is a distinct host-session-plus-ACP mechanism and must be evaluated under
   its own grants.

For the replacement, preserve ACP initialization, a fresh external session per
host assignment, the existing configure/status methods, bounded progress, and
structured final evidence. Honor the supplied Code worktree and working-tree
delivery; do not commit, push, write the host's earlier-stage artifacts, or
override its human approvals. Keep requirements/AC references available so the
host can construct its required Code artifacts without inventing coverage.

Internal execution must also respect the host's command policy. Adding a
tool-capable nested agent does not automatically authorize arbitrary package
installation, shell commands, or runtime checks. Expose or negotiate the
allowed execution policy at the provider boundary. A provider's isolated
self-checks remain separate evidence from formal host Test-stage acceptance;
the host's present v1 flow stops after Code.

Focused host checks: 109 ACP transport, runner, stage-contract, and detached
client tests passed, plus 76 external-agent bridge/role tests (185 total).
This is offline contract evidence, not a live model-backed integration
acceptance. Provider-side regression tests additionally exercise the real
runner/finalization path with deterministic generation, including validation,
artifact/index/teardown failures and cancellation.

V4 alignment verification: the provider suite passed 278 tests, with four
opt-in Docker fixtures skipped. A real ACP stdio exchange using the host's
`AcpStdioClient` and deterministic generation accepted `outcome: "passed"`,
retained pending external testing, edited the supplied project, and validated
the exported task/coverage data with the host's `CodeManifestArtifact` schema.
This exercises the real protocol/client boundary without model calls; it does
not establish model-backed project-generation or host Code-gate acceptance.

## What the app actually does today

The app already uses headless OpenCode. It does not currently send its internal
model calls directly over a completions API. `acp_client.py` starts pooled
`opencode acp` processes, configures a LiteLLM-compatible provider, and sends
prompts through ACP. However:

- `_build_minimal_opencode_config()` disables all tools and denies file access,
  shell commands, and delegation.
- `_create_session()` starts each process and session in a temporary empty
  directory. The generated project's files are supplied as bounded prompt text.
- The pool groups sessions by run and role. File generation and self-review
  both use the run's `specialist` session; this is not a separate agent session
  for each task or an independent reviewer.
- `MasterAgent` emits a detailed task/file/dependency plan. `dispatch_work_items()`
  runs the tasks sequentially; Python owns the execution schedule.
- `run_file_delivery()` asks for one exact file per response, parses nonce-bound
  source framing, stages source, requests self-review, validates it, and
  publishes the feature's files.
- `runner.py` performs bounded repair rounds using Python's mapping of failures
  to planned tasks. It does not run generated-project tests or consume external
  test results to initiate repairs.
- The separate Docker tester only accepts a finalized handoff reporting
  `validation_passed_external_testing_pending`. It writes evidence without
  updating the implementation run's result.

The current inner flow is already automated; its limitation is the work it can
perform and the evidence it can react to. It has no general coordinator agent
that can inspect the live repository, execute tools, revise the task boundaries,
and continue from runtime failures.

Source anchors: `src/specialists/providers/acp_client.py` (configuration,
session pool, role keys), `src/specialists/file_delivery.py` (`run_file_delivery`),
`src/specialists/base.py` (`_self_check`), `src/pipeline/dispatch.py`
(`dispatch_work_items`), `src/pipeline/runner.py` (`_run_pipeline_body`), and
`src/testing_agent/contract.py` (`validate_handoff`).

The latest recorded URL-shortener run, `0018`, ended with seven tasks passed,
one failed, and six blocked. Its teams task failed on a mismatch between a token
record and the API view it produced. That is an example where a worker with
repository access and compiler/test feedback could inspect the actual types and
patch the implementation. OpenCode would provide that loop; its ability to fix
this particular defect still needs a measured experiment.

## Comparison with the coding-agent workflow

The useful comparison is the observable model/tool loop of this coding harness,
not a claim that OpenCode and this harness share an implementation. Official
OpenAI documentation describes delegated agent threads and programmatic
continuation; OpenCode provides its own mechanisms for these capabilities.

| Responsibility | Current work4me | Proposed workflow |
| --- | --- | --- |
| Understand the objective | Planning prompt produces a fixed task/file plan | Coordinator inspects the project and maintains a revisable plan |
| Give specialists context | Python embeds bounded source excerpts | Task brief plus requirements files; workers read/search the workspace |
| Implement a task | One complete source file per text response | Worker uses edit/patch tools across the cohesive task |
| Check the work | Static validator and same-session self-review | Static checks, separate reviewer, and isolated runtime evidence |
| Repair a failure | Map issues to owners and regenerate selected files | Coordinator assigns concrete findings; worker patches and verifies |
| Handle conversations | One pooled specialist history per run | Separate task sessions; continue the relevant worker when appropriate |
| Continue autonomously | Fixed dispatch and retry rounds | Model/tool loop plus supervisor-enforced completion and resource limits |
| Decide success | Static pass with external testing pending | App validates required checks against the exact candidate snapshot |
| Recover and report | Custom stage/attempt artifacts and selective resume | Durable run records plus OpenCode session IDs, snapshots, and evidence |

The coordinator replaces routine human interpretation of worker results. It
still needs concise task briefs, authoritative requirements, tools, and explicit
completion criteria. Moving those responsibilities into a prompt alone would
leave the application without a dependable success condition.

## Headless integration choices

OpenCode exposes two relevant paths:

1. **Keep ACP and enable real agents.** Official ACP documentation says OpenCode
   supports its file/terminal tools, custom agents, MCP, and permissions over
   ACP. This is the smallest transport change, but needs a workspace-backed
   session, agent configuration, appropriate capabilities, and lifecycle tests.
2. **Use the headless HTTP server.** `opencode serve` runs without the terminal
   UI. Its API supports sessions, child-session discovery, asynchronous prompts,
   SSE events, messages, diffs, and abort. A Python client can use HTTP directly
   or be generated from OpenAPI. Rewriting the app in TypeScript is unnecessary.
   The official JS/TS SDK can either start the runtime or attach to a server.

Prefer HTTP for a new runtime adapter because it exposes useful supervision and
observability APIs. Keep the application's external ACP interface if existing
clients depend on it. The OpenCode executable/server process is still required;
headless integration removes terminal interaction, not the runtime process.

The current SDK documentation also describes schema-validated output through a
`StructuredOutput` tool on its v2 client, with a bounded retry count. This can
simplify coordinator reports and plans. It validates output shape, not the
truth of completion claims or correctness of generated code. Check the pinned
runtime's API before depending on a documentation example.

OpenCode's native `task` tool can create child sessions and continue them using
`task_id`. In the `v1.19.0` implementation, child sessions reference their parent;
the normal task call waits for its result, and background execution requires
an experimental flag. It does not create a separate checkout for each child.
Start with foreground task execution and one active writer. Keep parallel
editing and experimental background execution out of the initial migration.

Expect different cost and latency behavior: the current adapter deliberately
minimizes tool context and completion overhead, whereas a coding agent may make
many model/tool steps for one task. The existing gateway and selected models
must support those tool calls correctly. Measure cost per verified outcome;
fewer lines of application harness do not imply fewer model tokens or fewer
implementation defects.

## Target architecture

The Python supervisor creates one disposable workspace for a run and starts or
attaches to a scoped OpenCode runtime in standalone mode. Under AI-SDLC it uses
the supplied Code worktree as the implementation target, with isolated copies
for permitted execution. Multiple agent sessions can live within that runtime;
there is no need for one process per file or model request.

The primary coordinator has access to requirements, read/search tools, and the
approved specialist agents. Implementation agents own cohesive tasks; a reviewer
examines the diff and acceptance criteria in a separate session. The coordinator
can continue a worker with concrete feedback. Work is sequential initially so
that changing shared files does not create concurrent write conflicts.

Keep application-owned tools small and typed, for example:

- `validate_candidate`: run deterministic source/contract checks over a snapshot.
- `test_candidate`: run approved checks in the isolated testing environment and
  return result paths, source digest, exit codes, and bounded logs.
- `record_decision`: store a bounded task outcome and its evidence references.

These could be MCP or OpenCode custom tools that call existing Python services.
They give the coordinator actions and evidence without making Python manage
every source response. Ordinary read/edit/search behavior stays with OpenCode.

If generated-project commands are allowed in a worker's shell, run that worker
inside the disposable execution environment. To preserve the current rule that
implementation never executes generated code, deny that shell access and route
all runtime commands through `test_candidate`. An OpenCode session, Git checkout,
and permission configuration do not by themselves provide OS isolation.

After an agent reports completion, the supervisor evaluates required evidence
for the current source digest. Passing evidence for an earlier snapshot is
stale after another edit. If work remains, it supplies concrete feedback and
continues the coordinator within the run budget. It publishes only a candidate
that satisfies the configured gates in standalone mode; under AI-SDLC it
reports the Code assignment's result and evidence for the host to accept.

For unattended execution, configure allowed operations and deny unavailable
ones in advance. Disable interactive question/plan-transition flows for routine
execution. Any unexpected permission request must receive a policy decision or
terminate as `needs_input`/`blocked`; leaving `ask` requests unanswered can stall
a headless run. The app should enforce wall time, cost/token limits when usage
is available, coordinator continuations, child count, repair rounds, and real
cancellation across the tracked session tree and test processes.

## Extent of the overhaul

| Area | Treatment |
| --- | --- |
| External ACP server, CLI/handoff inputs | Retain interfaces; replace execution adapter and map richer events/status |
| Provider pooling, prompt flattening, text accumulation | Replace with OpenCode runtime/session/event client |
| Master planning and dispatcher | Replace fixed orchestration with coordinator; retain useful task/requirement metadata |
| Specialist prompts | Convert to agent definitions and outcome-oriented task briefs |
| File framing, source parsing, per-file staging/self-check loop | Retire on the new path; use native edits, workspace diffs, and independent verification |
| Static validator and design contracts | Retain as evidence-producing tools where useful; supplement with compiler/runtime checks |
| Docker testing backend | Reuse execution/isolation code; add automatic candidate invocation and result consumption |
| Handoff contract | Extend or introduce a versioned candidate contract; existing tester only accepts finalized passing runs |
| Run manifests/progress | Retain concepts; add coordinator/child session IDs, usage, candidate digests, evidence, and richer states |
| Resume/publication | Redesign around session continuity and verified workspace snapshots; preserve historical artifacts |
| Tests | Replace source-delivery tests on the new path with lifecycle, policy, evidence, repair, and recovery tests |

The eight main current execution/planning/delivery modules total about 6,000
lines, including comments and legacy code (`runner`, `master`, `dispatch`,
`staging`, `base`, `file_delivery`, `llm_client`, and the ACP provider adapter).
That is an indication of the affected area, not a prediction of lines to
rewrite. Several surrounding modules and tests also require adaptation.

Rough effort for one developer familiar with this repository: a transport/tool
proof in 1-3 engineering days; a first complete autonomous path in roughly 1-2
weeks; and a migration with cancellation, restart/resume, evidence handling,
compatibility, and regression coverage in roughly 3-5 weeks total. These are
planning ranges, not measured delivery commitments. Parallel writing, additional
package managers, or stronger infrastructure isolation add scope.

## Treatment of the pending changes

- Preserve the TypeScript checker and design-contract work as optional validation
  tools and authoritative task context. Compiler checks and acceptance tests
  should carry more of the correctness burden once runtime execution is available.
- Reuse resume's source digests and provenance ideas. Its stage-import machinery
  is tied to the old executor and should not define the new session model.
- Reuse precise failure descriptions and affected-task metadata from targeted
  retries. Stop extending regeneration mechanics before measuring agent repairs.
- Replace rejected framed-response diagnostics with native tool/message events,
  bounded logs, and workspace diffs on the new path.
- Preserve explicit package-manager choices and lock readiness. Add lock
  preparation within the isolated environment if fully unattended Node testing
  is a requirement; the current tester cannot proceed without an npm lockfile.

## First experiment and migration sequence

1. Introduce an opt-in OpenCode executor alongside the existing path. Pin the
   OpenCode version and verify its documented API against its OpenAPI spec.
   Preserve the existing provider configuration while testing its tool support.
2. Give a coordinator one small repository task, an implementation agent, a
   separate reviewer, and the isolated testing tool. Seed a known cross-file
   type/interface defect so success requires reading, editing, and verifying.
3. Verify actual worker edits and that compiler/test failure returns to a worker
   for repair without a person forwarding results. Check that wrong success
   claims cannot bypass the application's evidence gate.
4. Exercise cancellation, run-limit exhaustion, denied permissions, a requested
   user clarification, worker failure, and a supervisor/runtime restart. Record
   outcomes rather than leaving the run waiting indefinitely.
5. Run the same fixed requirement and baseline with both executors. Record
   completion, acceptance evidence, human interventions, cost, duration, and
   failure causes. The historical `0018` artifacts are diagnostic context, not
   an equivalent current-toolchain baseline or automatically resumable input.
6. After the loop works, migrate the production execution path and retire the
   old file-delivery system. Add parallel writers only if measured throughput
   justifies separate workspace/merge coordination.

The experiment should prove autonomous, evidence-backed repair before investing
in a full replacement. OpenCode can take over the local agent mechanics; the
application continues to own the definition and proof of a successful run.

## Sources

- [OpenCode server](https://opencode.ai/docs/server/): headless HTTP runtime,
  OpenAPI, sessions, events, diffs, and cancellation.
- [OpenCode SDK](https://opencode.ai/docs/sdk/): attaching/starting a runtime and
  schema-validated output on the v2 client.
- [OpenCode agents](https://opencode.ai/docs/agents/): primary/subagents,
  permissions, compaction, and permitted delegation.
- [OpenCode ACP](https://opencode.ai/docs/acp/): supported tools and features over
  ACP.
- [OpenCode permissions](https://opencode.ai/docs/permissions/): allow/deny/ask,
  question tool, defaults, and stuck-loop behavior.
- [OpenCode custom tools](https://opencode.ai/docs/custom-tools/): typed tool
  definitions and calling external services/scripts.
- [OpenCode v1.19.0 task implementation](https://github.com/anomalyco/opencode/blob/v1.19.0/packages/opencode/src/tool/task.ts):
  child sessions, task continuation, foreground behavior, and experimental
  background support.
- [Official OpenAI documentation: subagents](https://developers.openai.com/codex/concepts/multi-agents):
  delegated tasks, parent coordination, and shared-workspace concerns.
- [Official OpenAI documentation: Codex SDK](https://developers.openai.com/codex/sdk):
  programmatic agent runs and continuing/resuming threads.
