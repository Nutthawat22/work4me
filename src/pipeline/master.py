"""
src/pipeline/master.py

MasterAgent: owns the user's intent AND all planning for a pipeline run
(Phase 2 of the ACP-native rearchitecture — see
dev-plans/agents/features/2026-09-14-acp-native-master-architecture.md).
Absorbs the three planning methods formerly on the separate
`DesignAgent` class (pipeline/design.py, deleted in this phase):

- decompose(user_prompt) and plan_handoff(design_text, requirements_text):
  both emit the same category-based, multi-file task model.
- decompose_features()/plan_layout(): optional file-manifest strategy that
  adapts architecture-grouped ownership into the same task model.

Also retains Master's pre-existing (Phase 3, pre-rearchitecture)
responsibilities: mapping static validation issues back to WorkItem ids
(map_validation_issues_to_work_items) and building the final FailReport
(build_fail_report) when retries are exhausted.

All LLM calls in this module route through
specialists.providers.acp_client.acp_call_with_retry directly (Option A:
same-session retry-with-error-feedback — see the design doc's
"Recommended Approach"), not through specialists.llm_client.call_llm's
provider-dispatch indirection. call_llm is a thin wrapper whose only
job was to resolve config["models"][role]["provider"] against a
multi-provider registry — moot now that PROVIDERS has exactly one entry
("acp", see specialists/providers/__init__.py). Calling
acp_call_with_retry directly is what actually gives Master's
highest-stakes JSON-emitting call (this module's whole reason for
existing, per the design doc's Accepted Risk section) the retry-with-
error-feedback safety net; routing through call_llm would silently lose
that behavior unless call_llm itself were rewritten to forward retry
parameters (parse_and_validate_fn, max_attempts) — more churn than this
phase calls for.
"""

import json
import os
import re

from pipeline.dispatch import CycleError, topological_sort
from pipeline.state import (
    AgentResult,
    FailReport,
    Feature,
    FileManifest,
    OwnedFile,
    ValidationIssue,
    ValidationResult,
    TaskAttempt,
    WorkItem,
)
from pipeline.task_validation import normalize_relative_path
from specialists.providers.acp_client import acp_call_with_retry

VALID_CATEGORIES: set[str] = {"foundation", "feature", "shared", "integration", "validation"}

# Longer timeout: plan_handoff feeds the full design doc (~100K+ chars),
# well past acp_call_with_retry's 60s default.
PLAN_HANDOFF_TIMEOUT_SECONDS = 240

# Longer timeout: decompose_features feeds the full design doc, like
# plan_handoff, so reuse the same generous budget.
DECOMPOSE_FEATURES_TIMEOUT_SECONDS = 240

# Longer timeout: decompose takes a short free-text prompt (much less
# input than plan_handoff/decompose_features's full design docs), so it
# doesn't need the full 240s budget, but the 60s acp_call_with_retry
# default is still risky for an LLM call producing a structured WorkItem
# array — split the difference.
DECOMPOSE_TIMEOUT_SECONDS = 120

# Retry-with-error-feedback attempt cap for every planning call in this
# module (see acp_call_with_retry's max_attempts). Not yet exposed as a
# config value — see the design doc's Open Questions > "Where does the
# retry-with-error-feedback loop's attempt cap live?".
PLANNING_MAX_ATTEMPTS = 3

FEATURE_REQUIRED_FIELDS = {
    "id",
    "title",
    "description",
    "acceptance_criteria",
    "language",
    "source_ids",
    "depends_on",
}

TASK_REQUIRED_FIELDS = {
    "id",
    "category",
    "language",
    "module",
    "scope",
    "title",
    "description",
    "acceptance_criteria",
    "requirement_refs",
    "owned_files",
    "depends_on",
    "expected_interfaces",
}


def _build_work_items_response_schema(name: str, valid_languages: set[str]) -> dict:
    """
    Build a prompt-embedded structured-output schema (see
    src/specialists/providers/acp_client.py's _build_schema_instruction) that
    describes a valid array of task objects, with "category" and
    "language" enum-constrained to VALID_CATEGORIES and the caller's
    valid_languages set respectively. ACP has no token-level schema
    enforcement mechanism (see the design doc's "Accepted Risk: No Hard
    Schema Enforcement, Anywhere") — this schema is only ever prepended
    to the prompt as a best-effort instruction; the actual reliability
    fallback is acp_call_with_retry's retry-with-error-feedback loop.

    The top-level shape is `{"items": [...]}` rather than a bare array —
    kept from the pre-existing (Responses-API-era) schema shape for
    continuity with _parse_json's unwrapping behavior, even though ACP
    itself does not require an object root.
    """
    file_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "language": {"type": "string", "enum": sorted(valid_languages)},
            "area": {"type": "string", "enum": ["product", "tests"]},
        },
        "required": ["path", "language", "area"],
        "additionalProperties": False,
    }
    task_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "category": {"type": "string", "enum": sorted(VALID_CATEGORIES)},
            "language": {"type": "string", "enum": sorted(valid_languages)},
            "module": {"type": "string"},
            "scope": {"type": "string"},
            "title": {"type": "string"},
            "description": {"type": "string"},
            "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
            "requirement_refs": {"type": "array", "items": {"type": "string"}},
            "owned_files": {"type": "array", "items": file_schema},
            "depends_on": {"type": "array", "items": {"type": "string"}},
            "expected_interfaces": {"type": "array", "items": {"type": "string"}},
        },
        "required": sorted(TASK_REQUIRED_FIELDS),
        "additionalProperties": False,
    }
    return {
        "name": name,
        "schema": {
            "type": "object",
            "properties": {
                  "items": {"type": "array", "items": task_schema},
            },
            "required": ["items"],
            "additionalProperties": False,
        },
    }


PLAN_HANDOFF_SYSTEM_PROMPT_TEMPLATE = """You are the MasterAgent in an autonomous coding pipeline, planning work \
for an already-approved technical design document handoff.

You are given the FULL text of an already-approved technical design document \
(component design, data design, runtime flows, security, deployment, \
acceptance/test design, implementation Work Packages, and end-to-end \
traceability). Your job is genuine task decomposition, not mechanical \
transcription: read and understand the design document's actual scope and \
complexity, then decide the RIGHT number of implementation WorkItems and \
what each one should own — this could be as few as 2 items for a small, \
narrow task, or 15+ for a large, many-component one. Judge the shape from \
the task itself, not from a fixed template that always produces the same \
count regardless of what's being built. Use the Component Design section's \
full descriptions (not just bare CMP-* IDs) to understand what each \
WorkItem must actually produce.

Output ONLY a JSON array (no prose, no markdown fences) where each element \
matches this exact schema:

{{
  "id": "WI-001",
  "type": "logic" | "ui" | "config" | "test" | "scaffold" | "integrate",
  "language": "python",
  "title": "one-line description",
  "description": "full task spec",
  "acceptance_criteria": ["criterion 1", "criterion 2"],
  "output_path": "relative/path/to/file.py",
  "depends_on": ["WI-000"]
}}

Rules:
- "type" must be exactly one of: logic, ui, config, test, scaffold, \
integrate. There is no "schema"/"migration" type — database schema/migration \
work belongs to the scaffold item's shared foundation (see rule 6).
- "language" must be exactly one of: {valid_languages}.
- "depends_on" lists ids of OTHER WorkItems IN THIS SAME ARRAY that must be \
completed first. Use [] if there are none.
- Every field is required on every item.
- Output must be a valid JSON array and nothing else.

Planning rules (must follow exactly):
1. COVER THE FULL SCOPE — do not invent components or behavior that are not \
named in the design document, but do not drop any of it either: every \
capability described across the design's Work Packages (WP-*) must be \
implemented by SOME WorkItem in your output. You decide how to group that \
scope into WorkItems — a single WP can become one WorkItem or several, and \
closely-related WPs can share a WorkItem, whatever genuinely reflects the \
task's natural shape. Do not force a rigid one-WP-to-one-WorkItem mapping if \
the actual work doesn't divide that way.
2. Order dependencies to match how the design's own components actually \
depend on each other (a WP that lists "Dependencies: WP-00X, WP-00Y, ..." \
means the WorkItem(s) covering it must depend, directly or transitively, on \
WorkItem(s) covering each of those dependency WPs). A WP with no listed \
dependencies has no cross-WP depends_on beyond the scaffold item (rule 6a).
3. Emit "type": "test" WorkItems covering the design's Test IDs and each \
WP's Completion criteria — decide the right granularity yourself (one test \
WorkItem per implementation WorkItem, or fewer test WorkItems covering \
multiple related implementation items, whichever better matches the actual \
test surface). Every test WorkItem's `depends_on` must include the \
implementation WorkItem(s) it exercises, and its `acceptance_criteria` \
should paraphrase the relevant Completion criteria and linked TEST-* \
entries' Expected/pass behavior.
4. Embed traceability in text, since the WorkItem schema has no dedicated \
field for it: prefix `title` with the most relevant WP id(s), e.g. "[WP-002] \
Request draft API", and include the relevant Source IDs, Test IDs, and \
Completion criteria verbatim in `description` (e.g. "Source IDs: REQ-003, \
... | Test IDs: TEST-001, ... | Completion criteria: ..."). If a WorkItem \
covers multiple WPs, list all of their ids.
5. Degrees of freedom — AUTONOMOUS, NO ESCALATION: this pipeline runs \
input-in/output-only with no human in the loop and cannot pause to ask a \
question. Implement ONLY what is described in the excerpt, but where the \
excerpt marks a value TECHNICAL_PROPOSAL, PROPOSED, or an OPEN risk (e.g. \
RISK-005) with a suggested/default value, USE that value as given — it is \
explicitly documented as a usable working default pending later \
confirmation, not a blocker. Do not invent a different value and do not \
skip the WorkItem. Record the decision for auditability: append a `Design \
notes:` line to `description` naming the proposal/risk id and the value \
used (e.g. "Design notes: NFR-005 session expiry is TECHNICAL_PROPOSAL, \
implemented as specified (12h) pending owner confirmation."). Never write \
"NEEDS ESCALATION" or similar — there is no one to escalate to.
6a. SCAFFOLD (runs FIRST) — Emit exactly ONE WorkItem of type "scaffold" \
with an id like "WI-000" and depends_on: []. It runs FIRST and produces the \
dependency manifest, build/compiler config, .gitignore, and the shared \
foundation modules (database setup/migrations, shared config loader, shared \
types) that other modules depend on. EVERY other implementation WorkItem \
(all non-scaffold, non-integrate items and their tests) MUST include this \
scaffold item's id in its depends_on. Its output_path is just a \
representative primary path used for labeling (e.g. "package.json"), \
because a scaffold specialist emits MULTIPLE files (a JSON files map), not \
one file. This structural invariant holds regardless of how many other \
WorkItems you decide the task needs.
6b. INTEGRATE (runs LAST) — Emit exactly ONE WorkItem of type "integrate" as \
the FINAL item, depending on every other implementation WorkItem (it does \
NOT need the test items in its depends_on). It writes the composition/wiring \
files: the server entry point mounting all routes, the frontend app shell \
mounting all screens, and any aggregation/index files. Like scaffold, an \
integrate specialist emits MULTIPLE files (a JSON files map), so its \
output_path is just a representative primary path used for labeling (e.g. \
"src/server/index.ts"). This structural invariant holds regardless of how \
many other WorkItems you decide the task needs.
7. RUNNABILITY — look up the design's named components against the \
Component Design section:
   a. If the design's components include the component described as a \
web/browser client (e.g. React Web Client), your plan must produce actual \
renderable UI ("type": "ui" WorkItem(s)) — real screens/pages/components an \
end user can open, not just supporting libraries (e.g. not just a \
translation dictionary or a types-only module with no rendered page).
   b. If the design's components include the component described as the \
HTTP server/adapter (e.g. Express/Bun HTTP Adapter), your plan must produce \
actual server route/endpoint wiring ("type": "logic" or "config" \
WorkItem(s)) that maps HTTP requests to the domain modules — not just a \
domain module with no server exposing it.
   c. The single "scaffold" item (rule 6a) and single "integrate" item \
(rule 6b) together handle project setup and final assembly — the scaffold \
produces the dependency manifest and build config up front, and the \
integrate item wires every domain module into a runnable app at the end. Do \
NOT emit separate ASSEMBLY-MANIFEST/ASSEMBLY-ENTRYPOINT config items for \
this; the scaffold + integrate pair replaces them.
   d. This rule does not apply if the design has no client/server \
components at all (e.g. a pure library/CLI design) — only apply (a)-(b) \
when the Component Design section actually describes a web client and/or \
HTTP server component. (Scaffold and integrate items should still be \
emitted per rules 6a/6b.)
 8. COMMONLY-FORGOTTEN DELIVERABLES — check the design excerpt's Data \
Design, Observability/Operations, and Deployment/Configuration sections \
(if present) and produce explicit WorkItems for what they describe, since \
these are easy to omit by only reading the Work Packages list:
   a. If the design defines data schemas (e.g. SCHEMA-* entries), these \
belong to the scaffold item's SHARED FOUNDATION (rule 6a) — the scaffold \
produces the actual schema/migration definitions as real, runnable \
artifacts (e.g. SQL migration files or an ORM schema/migration script), not \
just TypeScript/type-only interfaces describing the shape. Do not emit a \
separate logic WorkItem for base schema/migrations.
   b. If the design describes logging, correlation IDs, metrics, or \
health/readiness endpoints, include at least one WorkItem implementing \
basic structured logging and a health-check endpoint — do not leave this \
entirely implied by the server WorkItem(s).
   c. If the design describes a deployment/runtime environment (e.g. a \
specific process manager, container, or one-host deployment), include a \
WorkItem producing minimal deployment configuration (e.g. a Dockerfile or \
equivalent process/config file) reflecting that environment.
   d. Only produce these when the design excerpt actually describes the \
corresponding concern — do not invent database/deployment requirements \
for a design that doesn't call for them.
  """


TASK_PLANNING_PROMPT_TEMPLATE = """You are the MasterAgent for an autonomous coding pipeline.

Decompose the request/design into cohesive implementation tasks. Tasks may own
multiple related files and may span frontend/backend when that is one capability.
Split only for complexity or meaningful architecture boundaries, never one task
per file. All tasks use this schema:

{{"items":[{{"id":"TASK-auth","category":"feature","language":"typescript",
"module":"authentication","scope":"...","title":"...","description":"...",
"acceptance_criteria":["..."],"requirement_refs":["REQ-001"],
"owned_files":[{{"path":"src/auth.ts","language":"typescript","area":"product"}}],
"depends_on":[],"expected_interfaces":["exports authenticate(credentials)"]}}]}}

Categories and responsibility:
- foundation: project setup and primitives required across modules.
- feature: one cohesive user capability, including related frontend/backend files.
- shared: cross-feature contracts/utilities/schema with one clear owner.
- integration: application composition, entry points, and wiring.
- validation: tests or validation assets, owned under area "tests".

Rules:
- Use only categories above and languages from: {valid_languages}.
- Give every task a stable unique id, module, bounded scope, acceptance criteria,
  requirement references (empty only when none exist), and at least one owned file.
- Each file has a safe run-relative path, a configured language, and area
  "product" or "tests". Every path has exactly one owner across the plan.
- Dependencies reference task ids in this plan. State the concrete interfaces
  each task promises and expects so dependent tasks can integrate without guessing.
- Preserve all requested/design scope and trace requirement/test identifiers.
- Emit JSON only. Do not use legacy logic/ui/config/test/scaffold/integrate types.
"""


FEATURES_SYSTEM_PROMPT_TEMPLATE = """You are the FEATURE-DECOMPOSITION stage in an autonomous coding pipeline.

You are given the FULL text of an already-approved technical design \
document. Your job is to decompose it into a list of FEATURES — coherent \
vertical slices of end-user behavior (e.g. authentication, request \
authoring, approval workflow). A feature is NOT a file and NOT a low-level \
module; it is a unit of intent (the WHAT). The concrete file layout is \
decided by a later stage, so do NOT name files here.

Output ONLY a JSON object (no prose, no markdown fences) of this exact shape:

{{
  "features": [
    {{
      "id": "FEAT-auth",
      "title": "one-line description",
      "description": "full behavior spec",
      "acceptance_criteria": ["criterion 1", "criterion 2"],
      "language": "typescript",
      "source_ids": ["REQ-001", "AC-002", "TEST-003", "WP-001"],
      "depends_on": ["FEAT-other"]
    }}
  ]
}}

Rules:
- Each feature is a coherent vertical slice of behavior, not a file or a \
low-level module.
- "id" looks like "FEAT-auth" (a short kebab-case slug after "FEAT-").
- "language" must be exactly one of: {valid_languages}.
- "depends_on" lists ids of OTHER features in this same array. Use [] if none.
- "source_ids" carries REQ-*/AC-*/TEST-*/WP-* traceability back to the design.
- Do NOT drop scope: every Work Package's behavior in the design must map \
into some feature.
- Output must be a valid JSON object and nothing else.
"""


def _build_features_response_schema(valid_languages: set[str]) -> dict:
    """
    Build a prompt-embedded structured-output schema describing
    `{"features": [<feature_obj>]}`. Each feature_obj requires all
    fields, constrains "language" to the caller's valid_languages set,
    and forbids additional properties at every object level. See
    _build_work_items_response_schema's docstring for why this is a
    best-effort prompt instruction, not an enforced constraint, under ACP.
    """
    feature_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "title": {"type": "string"},
            "description": {"type": "string"},
            "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
            "language": {"type": "string", "enum": sorted(valid_languages)},
            "source_ids": {"type": "array", "items": {"type": "string"}},
            "depends_on": {"type": "array", "items": {"type": "string"}},
        },
        "required": sorted(FEATURE_REQUIRED_FIELDS),
        "additionalProperties": False,
    }
    return {
        "name": "decompose_features",
        "schema": {
            "type": "object",
            "properties": {
                "features": {"type": "array", "items": feature_schema},
            },
            "required": ["features"],
            "additionalProperties": False,
        },
    }


def _check_no_extra_json_data(stripped: str, exc: json.JSONDecodeError) -> None:
    """
    Detect the "two JSON documents concatenated with no separator"
    failure mode (e.g. a retry attempt whose response is the previous
    bad attempt's JSON immediately followed by a new one, with no
    comma/whitespace/newline between them). json.loads() surfaces this
    as a JSONDecodeError with msg "Extra data" at the offset where the
    first complete value ended — but that generic message doesn't tell
    a human (or a model reading it back as a retry failure_reason) what
    actually went wrong. If `exc` is such an error, raise a specific
    MasterPlanError instead of letting the generic one propagate.
    """
    if exc.msg != "Extra data":
        return

    try:
        _, end_index = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        return

    trailing = stripped[end_index:].strip()
    if trailing:
        raise MasterPlanError(
            "Response contained more than one JSON document (extra data "
            "after the first complete JSON value) — the model likely "
            "repeated/concatenated multiple attempts. This should not "
            "happen; if it recurs, the retry-feedback prompt may need "
            "further strengthening."
        )


class MasterPlanError(Exception):
    """
    Raised when planning returns malformed JSON, invalid task ownership,
    or an invalid dependency graph.
    """


class MasterAgent:
    def __init__(self, config: dict, user_prompt: str = ""):
        self.config = config
        self.intent: str = user_prompt

    def set_intent(self, user_prompt: str) -> None:
        """Update the held intent for a new pipeline run."""
        self.intent = user_prompt

    def decompose(
        self, user_prompt: str, project_context: str | None = None
    ) -> list[WorkItem]:
        """
        Ask the LLM to decompose user_prompt into WorkItems, parse and
        validate the JSON response, and return constructed WorkItem objects.

        Raises:
            MasterPlanError: if the response is not valid JSON or fails
                schema validation, on every retry attempt (see
                acp_call_with_retry).
        """
        valid_languages = set(self.config["languages"].keys())

        model_cfg = self.config["models"]["design"]
        model = model_cfg["model"]
        system_prompt = TASK_PLANNING_PROMPT_TEMPLATE.format(
            valid_languages=", ".join(sorted(valid_languages))
        )
        planning_input = user_prompt
        if project_context:
            system_prompt += (
                "\n\nThis is an existing-project change. Inspect the supplied project context, "
                "reuse its established structure and conventions, and plan only necessary "
                "project-relative additions, modifications, and intentional deletions. "
                "Do not plan replacement scaffolding for files that already serve the request. "
                "Path contract: every owned_files[].path is a slash-separated path relative "
                "to the project root, with no leading project/ label. A project-root file is "
                "difficulty.js; a nested file is src/difficulty.js. Use project/difficulty.js "
                "only when the actual owned file is inside a directory literally named project."
            )
            planning_input = (
                f"Requested change:\n{user_prompt}\n\n"
                f"{project_context}"
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": planning_input},
        ]
        response_schema = _build_work_items_response_schema("decompose_work_items", valid_languages)

        def _parse_and_validate(raw: str):
            work_items = self._validate_and_build(self._parse_json(raw), valid_languages)
            self._validate_task_plan(work_items)
            return work_items

        raw = acp_call_with_retry(
            messages, model, self.config,
            timeout=DECOMPOSE_TIMEOUT_SECONDS,
            response_schema=response_schema,
            parse_and_validate_fn=_parse_and_validate,
            max_attempts=PLANNING_MAX_ATTEMPTS,
        )

        if raw.startswith(f"[{model}] Error:"):
            raise MasterPlanError(raw)

        parsed = self._parse_json(raw)
        work_items = self._validate_and_build(parsed, valid_languages)
        self._validate_task_plan(work_items)
        return work_items

    def plan_handoff(
        self, design_text: str, requirements_text: str | None = None
    ) -> list[WorkItem]:
        """
        Plan dependency-ordered category tasks for an approved design handoff.
        It uses the same task schema and validation as free-text decomposition
        and does not mandate foundation or integration tasks.

        Args:
            design_text: The FULL text of the technical design document.
                This method does not read files — the caller owns loading
                the document.
            requirements_text: Optional full text of the source
                requirements document, if the caller wants additional
                REQ-*/AC-* context beyond what the design doc already
                restates.

        Returns:
            A validated list of the common multi-file task model, with
            requirement references and explicit per-file ownership.

        Raises:
            MasterPlanError: if the response, ownership set, or dependency
                graph is invalid.
        """
        valid_languages = set(self.config["languages"].keys())

        model_cfg = self.config["models"]["design"]
        model = model_cfg["model"]
        system_prompt = TASK_PLANNING_PROMPT_TEMPLATE.format(
            valid_languages=", ".join(sorted(valid_languages))
        )

        user_content = design_text
        if requirements_text:
            user_content = (
                f"{design_text}\n\n"
                f"--- Additional requirements excerpt (context only) ---\n"
                f"{requirements_text}"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        response_schema = _build_work_items_response_schema("plan_handoff_work_items", valid_languages)

        def _parse_and_validate(raw: str):
            work_items = self._validate_and_build(self._parse_json(raw), valid_languages)
            self._validate_task_plan(work_items)
            return work_items

        raw = acp_call_with_retry(
            messages, model, self.config,
            timeout=PLAN_HANDOFF_TIMEOUT_SECONDS,
            response_schema=response_schema,
            parse_and_validate_fn=_parse_and_validate,
            max_attempts=PLANNING_MAX_ATTEMPTS,
        )

        if raw.startswith(f"[{model}] Error:"):
            raise MasterPlanError(raw)

        parsed = self._parse_json(raw)
        work_items = self._validate_and_build(parsed, valid_languages)
        self._validate_task_plan(work_items)
        return work_items

    def decompose_features(
        self, design_text: str, requirements_text: str | None = None
    ) -> list[Feature]:
        """
        Decompose an approved technical design into a list of Features —
        coherent vertical slices of behavior (the WHAT), NOT files. This is
        the entry point for the file-manifest planning mode: the returned
        Features feed plan_file_manifest().

        Args:
            design_text: The FULL text of the technical design document.
                This method does not read files — the caller owns loading.
            requirements_text: Optional full text of the source
                requirements document, appended as extra context.

        Returns:
            A list[Feature].

        Raises:
            MasterPlanError: if the response is not valid JSON or fails
                schema validation.
        """
        valid_languages = set(self.config["languages"].keys())

        model_cfg = self.config["models"]["design"]
        model = model_cfg["model"]
        system_prompt = FEATURES_SYSTEM_PROMPT_TEMPLATE.format(
            valid_languages=", ".join(sorted(valid_languages))
        )

        user_content = design_text
        if requirements_text:
            user_content = (
                f"{design_text}\n\n"
                f"--- Additional requirements excerpt (context only) ---\n"
                f"{requirements_text}"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        response_schema = _build_features_response_schema(valid_languages)

        def _parse_and_validate(raw: str):
            return self._validate_and_build_features(self._parse_features_json(raw), valid_languages)

        raw = acp_call_with_retry(
            messages, model, self.config,
            timeout=DECOMPOSE_FEATURES_TIMEOUT_SECONDS,
            response_schema=response_schema,
            parse_and_validate_fn=_parse_and_validate,
            max_attempts=PLANNING_MAX_ATTEMPTS,
        )

        if raw.startswith(f"[{model}] Error:"):
            raise MasterPlanError(raw)

        parsed = self._parse_features_json(raw)
        return self._validate_and_build_features(parsed, valid_languages)

    def plan_layout(
        self,
        features: list[Feature],
        strategy: str = "llm_decided",
        external_layout: dict | None = None,
    ) -> list[WorkItem]:
        """
        Plan file/component layout for the given features, then adapt the
        result into dispatchable WorkItems. Two selectable strategies (see
        the design doc's Phase 4 — "file_manifest.py as a Selectable
        Strategy"):

        - "llm_decided" (default): calls file_manifest.plan_file_manifest()
          (one LLM call, itself retry-with-error-feedback wrapped — see
          plan_file_manifest's own docstring) to derive the file plan from
          features, exactly as src/pipeline/ingest.py's --mode manifest branch
          used to do by hand before this method existed.
        - "external": accepts a pre-built layout via `external_layout`
          (see the Implementation notes below for the exact shape
          expected) instead of making any LLM call for the layout step —
          still runs the result through the same
          group_into_work_items()/validate_file_manifest() pipeline as the
          llm_decided path, since those are pure, strategy-agnostic
          functions.

        Both strategies converge on the same group_into_work_items() ->
        file_groups_to_work_items() pipeline afterward — WorkItem output
        shape is identical regardless of which strategy produced the
        FileManifest.

        Local (not module-level) imports of pipeline.file_manifest below
        are required, not stylistic: file_manifest.py imports
        PLANNING_MAX_ATTEMPTS from this module at its own module level
        (see file_manifest.py's plan_file_manifest), so a module-level
        `from pipeline.file_manifest import ...` here would be a genuine
        circular import (verified: triggers ImportError on
        partially-initialized module). Deferring the import into this
        method's body avoids it, matching the existing precedent already
        used by pipeline/ingest.py's --mode manifest branch.

        Args:
            features: Feature list (e.g. from decompose_features()) that
                the layout must cover.
            strategy: "llm_decided" or "external".
            external_layout: Required when strategy="external". Minimal
                input contract (deliberately not over-built for a caller
                that doesn't exist yet): a dict of shape
                `{"files": [<file_spec_dict>, ...]}` where each
                file_spec_dict has the same keys FileSpec requires (path,
                role, language, contributing_features, requirements,
                depends_on_files) — the same shape
                file_manifest._parse_file_manifest_json/_build_file_spec
                already parse from LLM JSON responses. Reused directly here
                (via _build_file_spec) so there is exactly one code path
                turning a raw file-spec dict list into FileSpec objects
                regardless of strategy.

        Raises:
            MasterPlanError: if strategy is not one of "llm_decided"/
                "external", if strategy="external" and external_layout is
                None or malformed, or if the underlying FileManifest fails
                validate_file_manifest() (surfaced as
                FileManifestParseError from file_manifest.py, wrapped/
                re-raised as MasterPlanError for a consistent exception
                type at the MasterAgent boundary — callers of Master's
                other planning methods already only need to catch
                MasterPlanError, this should be no different).
        """
        from pipeline.file_manifest import (
            FileManifestParseError,
            _build_file_spec,
            file_groups_to_work_items,
            group_into_work_items,
            plan_file_manifest,
            validate_file_manifest,
        )

        if strategy == "llm_decided":
            try:
                manifest = plan_file_manifest(features, self.config)
            except FileManifestParseError as e:
                raise MasterPlanError(str(e)) from e

        elif strategy == "external":
            if external_layout is None:
                raise MasterPlanError(
                    "strategy='external' requires external_layout to be provided"
                )

            file_dicts = external_layout.get("files", [])
            try:
                manifest = FileManifest(
                    files=[_build_file_spec(idx, d) for idx, d in enumerate(file_dicts)]
                )
            except FileManifestParseError as e:
                raise MasterPlanError(str(e)) from e

            errors = validate_file_manifest(manifest, features)
            if errors:
                raise MasterPlanError(
                    "plan_layout(strategy='external') produced an invalid manifest:\n"
                    + "\n".join(errors)
                )

        else:
            raise MasterPlanError(
                f"Unknown layout strategy: {strategy!r}. Valid strategies: "
                f"'llm_decided', 'external'"
            )

        groups = group_into_work_items(manifest)
        work_items = file_groups_to_work_items(groups, manifest, features)
        self._validate_task_plan(work_items)
        return work_items

    @staticmethod
    def _parse_json(raw: str) -> list[dict]:
        """
        Parse raw into a list of WorkItem dicts. Accepts either a bare JSON
        array or a {"items": [...]} wrapper (the structured-output shape),
        unwrapping the latter transparently.
        """
        stripped = raw.strip()

        fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
        if fence_match:
            stripped = fence_match.group(1).strip()

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as e:
            _check_no_extra_json_data(stripped, e)
            raise MasterPlanError(f"Failed to parse MasterAgent response as JSON: {e}\nRaw response: {raw}")

        if isinstance(parsed, dict) and "items" in parsed:
            parsed = parsed["items"]

        if not isinstance(parsed, list):
            raise MasterPlanError(f"Expected a JSON array of WorkItems, got: {type(parsed).__name__}")

        return parsed

    @staticmethod
    def _validate_and_build(items: list[dict], valid_languages: set[str]) -> list[WorkItem]:
        work_items: list[WorkItem] = []

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                raise MasterPlanError(f"Task at index {idx} is not a JSON object: {item!r}")

            missing = TASK_REQUIRED_FIELDS - item.keys()
            if missing:
                raise MasterPlanError(
                    f"Task at index {idx} (id={item.get('id', '?')}) missing required fields: {sorted(missing)}"
                )

            category = item["category"]
            if not isinstance(category, str) or category not in VALID_CATEGORIES:
                raise MasterPlanError(
                    f"Task {item.get('id', '?')} has invalid category {category!r}, "
                    f"must be one of {sorted(VALID_CATEGORIES)}"
                )

            item_language = item["language"]
            if not isinstance(item_language, str) or item_language not in valid_languages:
                raise MasterPlanError(
                    f"Task {item.get('id', '?')} has invalid language '{item_language}', "
                    f"must be one of {sorted(valid_languages)}"
                )

            for string_field in ("id", "module", "scope", "title", "description"):
                if not isinstance(item[string_field], str) or not item[string_field].strip():
                    raise MasterPlanError(
                        f"Task {item.get('id', '?')} '{string_field}' must be a non-empty string"
                    )
            for list_field in (
                "acceptance_criteria", "requirement_refs", "depends_on", "expected_interfaces"
            ):
                value = item[list_field]
                if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                    raise MasterPlanError(f"Task {item['id']} '{list_field}' must be a list of strings")
            if not item["acceptance_criteria"]:
                raise MasterPlanError(f"Task {item['id']} must have acceptance criteria")
            if not isinstance(item["owned_files"], list) or not item["owned_files"]:
                raise MasterPlanError(f"Task {item['id']} must own at least one file")

            owned_files: list[OwnedFile] = []
            for file_idx, file_data in enumerate(item["owned_files"]):
                if not isinstance(file_data, dict):
                    raise MasterPlanError(f"Task {item['id']} owned_files[{file_idx}] must be an object")
                if not {"path", "language", "area"} <= set(file_data):
                    raise MasterPlanError(
                        f"Task {item['id']} owned_files[{file_idx}] requires path, language, and area"
                    )
                path = file_data["path"]
                file_language = file_data["language"]
                area = file_data["area"]
                if not isinstance(path, str):
                    raise MasterPlanError(f"Task {item['id']} owned file path must be a string")
                if not isinstance(file_language, str) or file_language not in valid_languages:
                    raise MasterPlanError(
                        f"Task {item['id']} owned file {path!r} has invalid language {file_language!r}"
                    )
                if area not in ("product", "tests"):
                    raise MasterPlanError(f"Task {item['id']} owned file {path!r} has invalid area {area!r}")
                try:
                    normalize_relative_path(path)
                except ValueError as exc:
                    raise MasterPlanError(f"Task {item['id']} owns unsafe path {path!r}: {exc}") from exc
                if (category == "validation") != (area == "tests"):
                    raise MasterPlanError(
                        f"Task {item['id']} category {category!r} must own files only in the "
                        f"{'tests' if category == 'validation' else 'product'} area"
                    )
                owned_files.append(OwnedFile(path=path, language=file_language, area=area))

            work_items.append(
                WorkItem(
                    id=item["id"],
                    category=category,  # type: ignore[arg-type]
                    language=item_language,
                    module=item["module"],
                    scope=item["scope"],
                    title=item["title"],
                    description=item["description"],
                    acceptance_criteria=item["acceptance_criteria"],
                    requirement_refs=item["requirement_refs"],
                    owned_files=owned_files,
                    depends_on=item["depends_on"],
                    expected_interfaces=item["expected_interfaces"],
                )
            )

        return work_items

    def _validate_task_plan(self, work_items: list[WorkItem]) -> None:
        if not work_items:
            raise MasterPlanError("Task plan must contain at least one task")
        ids = [item.id for item in work_items]
        if len(ids) != len(set(ids)):
            raise MasterPlanError("Task ids must be unique")

        owners: dict[tuple[str, str], str] = {}
        for item in work_items:
            if not isinstance(item.id, str) or not item.id.strip():
                raise MasterPlanError("Every task must have a non-empty id")
            if item.category not in VALID_CATEGORIES:
                raise MasterPlanError(f"Task {item.id} has invalid category {item.category!r}")
            if not isinstance(item.language, str) or item.language not in self.config["languages"]:
                raise MasterPlanError(f"Task {item.id} has invalid language {item.language!r}")
            for field_name in ("module", "scope", "title", "description"):
                value = getattr(item, field_name)
                if not isinstance(value, str) or not value.strip():
                    raise MasterPlanError(f"Task {item.id} must have a non-empty {field_name}")
            for field_name in (
                "acceptance_criteria", "requirement_refs", "depends_on", "expected_interfaces"
            ):
                values = getattr(item, field_name)
                if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                    raise MasterPlanError(f"Task {item.id} {field_name} must be a list of strings")
            if not item.acceptance_criteria:
                raise MasterPlanError(f"Task {item.id} must have acceptance criteria")
            if not item.owned_files:
                raise MasterPlanError(f"Task {item.id} must own at least one file")
            for owned_file in item.owned_files:
                if not isinstance(owned_file, OwnedFile):
                    raise MasterPlanError(f"Task {item.id} owned files must use the OwnedFile model")
                if owned_file.area not in ("product", "tests"):
                    raise MasterPlanError(f"Task {item.id} has invalid file area {owned_file.area!r}")
                if (item.category == "validation") != (owned_file.area == "tests"):
                    raise MasterPlanError(f"Task {item.id} category and file area do not match")
                if owned_file.language not in self.config["languages"]:
                    raise MasterPlanError(
                        f"Task {item.id} file {owned_file.path!r} has invalid language {owned_file.language!r}"
                    )
                try:
                    normalized = normalize_relative_path(owned_file.path)
                except ValueError as exc:
                    raise MasterPlanError(
                        f"Task {item.id} owns unsafe path {owned_file.path!r}: {exc}"
                    ) from exc
                area_key = "tests_dir" if owned_file.area == "tests" else "output_dir"
                default_dir = "tests" if area_key == "tests_dir" else "product"
                configured_dir = normalize_relative_path(
                    self.config.get("pipeline", {}).get(area_key, default_dir)
                )
                prefix = configured_dir + "/"
                if normalized.startswith(prefix):
                    try:
                        normalized = normalize_relative_path(normalized[len(prefix):])
                    except ValueError as exc:
                        raise MasterPlanError(
                            f"Task {item.id} owns unsafe path {owned_file.path!r}: {exc}"
                        ) from exc
                key = (configured_dir.casefold(), normalized.casefold())
                previous = owners.get(key)
                if previous is not None:
                    raise MasterPlanError(
                        f"File {configured_dir}/{normalized!s} has multiple task owners: "
                        f"{previous} and {item.id}"
                    )
                owners[key] = item.id

        try:
            topological_sort(work_items, strict=True)
        except CycleError as exc:
            raise MasterPlanError(f"invalid dependency graph: {exc}") from exc

        for item in work_items:
            item.status = "blocked" if item.depends_on else "ready"

    @staticmethod
    def _parse_features_json(raw: str) -> list[dict]:
        """
        Parse raw into a list of Feature dicts. Accepts either a
        {"features": [...]} wrapper (the structured-output shape) or a bare
        JSON array, stripping a ```json fence if present.
        """
        stripped = raw.strip()

        fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
        if fence_match:
            stripped = fence_match.group(1).strip()

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as e:
            _check_no_extra_json_data(stripped, e)
            raise MasterPlanError(
                f"Failed to parse decompose_features response as JSON: {e}\nRaw response: {raw}"
            )

        if isinstance(parsed, dict) and "features" in parsed:
            parsed = parsed["features"]

        if not isinstance(parsed, list):
            raise MasterPlanError(
                f"Expected a JSON array of Features, got: {type(parsed).__name__}"
            )

        return parsed

    @staticmethod
    def _validate_and_build_features(
        items: list[dict], valid_languages: set[str]
    ) -> list[Feature]:
        """Validate each Feature dict's required keys/types/language enum and
        construct Feature objects, raising MasterPlanError on problems."""
        features: list[Feature] = []

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                raise MasterPlanError(f"Feature at index {idx} is not a JSON object: {item!r}")

            missing = FEATURE_REQUIRED_FIELDS - item.keys()
            if missing:
                raise MasterPlanError(
                    f"Feature at index {idx} (id={item.get('id', '?')}) "
                    f"missing required fields: {sorted(missing)}"
                )

            item_language = item["language"]
            if item_language not in valid_languages:
                raise MasterPlanError(
                    f"Feature {item.get('id', '?')} has invalid language '{item_language}', "
                    f"must be one of {sorted(valid_languages)}"
                )

            for str_field in ("id", "title", "description"):
                if not isinstance(item[str_field], str):
                    raise MasterPlanError(
                        f"Feature {item.get('id', '?')} '{str_field}' must be a string"
                    )

            for list_field in ("acceptance_criteria", "source_ids", "depends_on"):
                if not isinstance(item[list_field], list):
                    raise MasterPlanError(
                        f"Feature {item.get('id', '?')} '{list_field}' must be a list"
                    )

            features.append(
                Feature(
                    id=item["id"],
                    title=item["title"],
                    description=item["description"],
                    acceptance_criteria=item["acceptance_criteria"],
                    language=item_language,
                    source_ids=item["source_ids"],
                    depends_on=item["depends_on"],
                )
            )

        return features

    def map_validation_issues_to_work_items(
        self,
        validation_result: ValidationResult,
        agent_results: list[AgentResult],
        work_items: list[WorkItem],
      ) -> dict[str, list[ValidationIssue]]:
        """
        Resolve failing output files to their owning tasks. Validation-task
        failures also retry their declared implementation dependencies; any
        changed task then pulls in downstream dependents whose outputs may be
        stale. Unrelated successful tasks stay out of the retry subset.

        Returns a map from retry task id to its directly associated validation
        failures. Tasks added for repair have empty lists.
        """
        by_id = {item.id: item for item in work_items}

        # Run-relative output path -> task id from every AgentResult.
        reverse_map: dict[str, str] = {}
        for result in agent_results:
            for path in result.files_written:
                reverse_map[os.path.normcase(os.path.normpath(path))] = result.work_item_id

        retry_failures: dict[str, list[ValidationIssue]] = {}

        for result in agent_results:
            item = by_id.get(result.work_item_id)
            if item is not None and item.status == "failed":
                retry_failures.setdefault(item.id, [])

        for failure in validation_result.errors:
            file_path = failure.file_path
            normalized = os.path.normcase(os.path.normpath(file_path))

            work_item_id = (
                failure.work_item_id
                if failure.work_item_id in by_id
                else reverse_map.get(normalized)
            )

            if work_item_id is None:
                # Fallback handles relative vs project-root-relative path
                # differences between validation findings and recorded outputs.
                matches = {
                    mapped_id
                    for mapped_path, mapped_id in reverse_map.items()
                    if normalized.endswith(mapped_path) or mapped_path.endswith(normalized)
                }
                if len(matches) == 1:
                    work_item_id = matches.pop()

            if work_item_id is None:
                # Couldn't resolve — skip retry-targeting for this failure,
                # but the raw finding still surfaces in the FailReport.
                continue

            retry_failures.setdefault(work_item_id, []).append(failure)
            item = by_id.get(work_item_id)
            if item is not None and item.category == "validation":
                for dep_id in item.depends_on:
                    retry_failures.setdefault(dep_id, [])

        changed = set(retry_failures)
        while changed:
            current = changed.pop()
            for item in work_items:
                if current in item.depends_on and item.id not in retry_failures:
                    retry_failures[item.id] = []
                    changed.add(item.id)

        return retry_failures

    def build_fail_report(
        self,
        run_number: int,
        validation_result: ValidationResult,
        unresolved_work_item_ids: list[str],
    ) -> FailReport:
        """
        Build the final FailReport after retries are exhausted.

        Note: the design doc's example shows an LLM-generated narrative
        summary. For this Phase 3 baseline, the summary is a deterministic
        templated string (no LLM round-trip) — see final report deviation
        notes.
        """
        summary = (
            f"{len(validation_result.errors)} static validation error(s) "
            f"across {len(unresolved_work_item_ids)} unresolved work "
            f"item(s) after exhausting retries."
        )

        return FailReport(
            run_number=run_number,
            failures=validation_result.errors,
            unresolved_items=unresolved_work_item_ids,
            summary=summary,
        )
