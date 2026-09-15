"""
pipeline/master.py

MasterAgent: owns the user's intent AND all planning for a pipeline run
(Phase 2 of the ACP-native rearchitecture — see
dev-plans/agents/features/2026-09-14-acp-native-master-architecture.md).
Absorbs the three planning methods formerly on the separate
`DesignAgent` class (pipeline/design.py, deleted in this phase):

- decompose(user_prompt): free-text path — one LLM call turns a short
  prompt into a flat WorkItem array.
- plan_handoff(design_text, requirements_text=None): package-aware path
  for approved technical-design handoffs. Renamed from decompose_handoff
  to reflect what changed in this phase: the LLM is no longer given a
  fixed WP-template to mechanically transcribe 1:1 — it genuinely
  decides the right number of implementation WorkItems and what each
  should own, based on the design document's actual scope/complexity.
  Two structural invariants are still enforced regardless of task shape
  (exactly one scaffold item first, exactly one integrate item last —
  see PLAN_HANDOFF_SYSTEM_PROMPT_TEMPLATE rules 6a/6b), because those
  are sound regardless of how many items the middle of the plan has.
- decompose_features(design_text, requirements_text): design doc ->
  Feature list (vertical slices of behavior, NOT files) — feeds
  file_manifest.py. Unchanged behavior this phase (Phase 4 of the
  design doc reframes file_manifest.py itself, not this method).

Also retains Master's pre-existing (Phase 3, pre-rearchitecture)
responsibilities: mapping test failures back to WorkItem ids
(map_failures_to_work_items) and building the final FailReport
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
from pipeline.state import AgentResult, FailReport, Feature, TestFailure, TestResult, WorkItem
from specialists.providers.acp_client import acp_call_with_retry

VALID_TYPES: set[str] = {"logic", "ui", "config", "test", "scaffold", "integrate"}

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

REQUIRED_FIELDS = {
    "id",
    "type",
    "language",
    "title",
    "description",
    "acceptance_criteria",
    "output_path",
    "depends_on",
}


def _build_work_items_response_schema(name: str, valid_languages: set[str]) -> dict:
    """
    Build a prompt-embedded structured-output schema (see
    specialists/providers/acp_client.py's _build_schema_instruction) that
    describes a valid array of WorkItem objects, with "type" and
    "language" enum-constrained to VALID_TYPES and the caller's
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
    work_item_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "type": {"type": "string", "enum": sorted(VALID_TYPES)},
            "language": {"type": "string", "enum": sorted(valid_languages)},
            "title": {"type": "string"},
            "description": {"type": "string"},
            "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
            "output_path": {"type": "string"},
            "depends_on": {"type": "array", "items": {"type": "string"}},
        },
        "required": sorted(REQUIRED_FIELDS),
        "additionalProperties": False,
    }
    return {
        "name": name,
        "schema": {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": work_item_schema},
            },
            "required": ["items"],
            "additionalProperties": False,
        },
    }


SYSTEM_PROMPT_TEMPLATE = """You are the MasterAgent in an autonomous coding pipeline.

Given a user's request, decompose it into a list of WorkItems for specialist \
agents to implement. Output ONLY a JSON array (no prose, no markdown fences) \
where each element matches this exact schema:

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
- "type" must be exactly one of: logic, ui, config, test, scaffold, integrate.
- "language" must be exactly one of: {valid_languages}.
- "depends_on" lists ids of other WorkItems in this same array that must be \
completed first. Use [] if there are none.
- Every field is required on every item.
- Output must be a valid JSON array and nothing else.
"""


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
    Raised when MasterAgent's planning LLM response cannot be turned
    into a valid plan — unparseable JSON, failed schema validation, or
    (for plan_handoff specifically) a dependency-cycle/dangling-reference
    graph. Renamed from pipeline/design.py's DesignParseError to match
    the class move; behavior/semantics are unchanged.
    """


class MasterAgent:
    def __init__(self, config: dict, user_prompt: str = ""):
        self.config = config
        self.intent: str = user_prompt

    def set_intent(self, user_prompt: str) -> None:
        """Update the held intent for a new pipeline run."""
        self.intent = user_prompt

    def decompose(self, user_prompt: str) -> list[WorkItem]:
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
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            valid_languages=", ".join(sorted(valid_languages))
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response_schema = _build_work_items_response_schema("decompose_work_items", valid_languages)

        def _parse_and_validate(raw: str):
            return self._validate_and_build(self._parse_json(raw), valid_languages)

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
        return self._validate_and_build(parsed, valid_languages)

    def plan_handoff(
        self, design_text: str, requirements_text: str | None = None
    ) -> list[WorkItem]:
        """
        Plan a dependency-ordered list of WorkItems for an approved
        technical-design handoff. Renamed from decompose_handoff: this is
        no longer a mechanical 1:1 translation of the design's authored
        Work Packages. Instead, the LLM is prompted to genuinely judge
        the right number of WorkItems and what each should own, based on
        the design document's actual scope and complexity — while still
        enforcing two structural invariants regardless of task shape
        (exactly one scaffold item first, exactly one integrate item
        last — see PLAN_HANDOFF_SYSTEM_PROMPT_TEMPLATE rules 6a/6b).

        Args:
            design_text: The FULL text of the technical design document.
                This method does not read files — the caller owns loading
                the document.
            requirements_text: Optional full text of the source
                requirements document, if the caller wants additional
                REQ-*/AC-* context beyond what the design doc already
                restates.

        Returns:
            A list[WorkItem] where every generated item's `depends_on`
            correctly encodes dependency order (so dispatch.py's
            topological sort can consume it directly), and traceability
            IDs (WP-*, REQ-*, TEST-*, AC-*) are embedded in
            `title`/`description` text.

        Raises:
            MasterPlanError: if the response is not valid JSON, fails
                schema validation, or — critically — if the generated
                WorkItems have dangling `depends_on` references or a
                dependency cycle (checked via
                pipeline.dispatch.topological_sort(strict=True)).
        """
        valid_languages = set(self.config["languages"].keys())

        model_cfg = self.config["models"]["design"]
        model = model_cfg["model"]
        system_prompt = PLAN_HANDOFF_SYSTEM_PROMPT_TEMPLATE.format(
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
            self._check_dependency_graph(work_items)
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
        self._check_dependency_graph(work_items)
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

    @staticmethod
    def _check_dependency_graph(work_items: list[WorkItem]) -> None:
        """
        Guard for plan_handoff: a missed or malformed depends_on edge
        passes schema validation but breaks dispatch.py's topological sort
        at runtime. Fail fast here with a clear message instead.
        """
        try:
            topological_sort(work_items, strict=True)
        except CycleError as e:
            raise MasterPlanError(
                f"plan_handoff produced an invalid dependency graph: {e}"
            ) from e

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
                raise MasterPlanError(f"WorkItem at index {idx} is not a JSON object: {item!r}")

            missing = REQUIRED_FIELDS - item.keys()
            if missing:
                raise MasterPlanError(
                    f"WorkItem at index {idx} (id={item.get('id', '?')}) missing required fields: {sorted(missing)}"
                )

            item_type = item["type"]
            if item_type not in VALID_TYPES:
                raise MasterPlanError(
                    f"WorkItem {item.get('id', '?')} has invalid type '{item_type}', "
                    f"must be one of {sorted(VALID_TYPES)}"
                )

            item_language = item["language"]
            if item_language not in valid_languages:
                raise MasterPlanError(
                    f"WorkItem {item.get('id', '?')} has invalid language '{item_language}', "
                    f"must be one of {sorted(valid_languages)}"
                )

            if not isinstance(item["acceptance_criteria"], list):
                raise MasterPlanError(
                    f"WorkItem {item.get('id', '?')} 'acceptance_criteria' must be a list"
                )

            if not isinstance(item["depends_on"], list):
                raise MasterPlanError(
                    f"WorkItem {item.get('id', '?')} 'depends_on' must be a list"
                )

            work_items.append(
                WorkItem(
                    id=item["id"],
                    type=item_type,  # type: ignore[arg-type]
                    language=item_language,
                    title=item["title"],
                    description=item["description"],
                    acceptance_criteria=item["acceptance_criteria"],
                    output_path=item["output_path"],
                    depends_on=item["depends_on"],
                )
            )

        return work_items

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

    def map_failures_to_work_items(
        self,
        test_result: TestResult,
        agent_results: list[AgentResult],
        work_items: list[WorkItem],
    ) -> dict[str, list[TestFailure]]:
        """
        Given a failed TestResult, resolve which WorkItem ids should be
        retried: the WorkItem that produced the failing test file, plus
        everything in that WorkItem's depends_on (either could be at
        fault).

        Convention (must match pipeline/test_runner.py): TestFailure.test_name
        is f"{file_path}::{classname}::{name}" — the file path is the first
        "::"-delimited segment.

        Returns:
            dict mapping work_item_id -> list of TestFailure that directly
            implicated it. WorkItem ids added to the retry set only via
            depends_on propagation (i.e. they didn't directly produce a
            failing file) map to an empty list — the caller can tell
            "retried because a dependent failed" from an empty list.
        """
        by_id = {item.id: item for item in work_items}

        # file_path (normalized) -> work_item_id, from every AgentResult's
        # files_written (paths are relative to project_root).
        reverse_map: dict[str, str] = {}
        for result in agent_results:
            for path in result.files_written:
                reverse_map[os.path.normpath(path)] = result.work_item_id

        retry_failures: dict[str, list[TestFailure]] = {}

        for failure in test_result.failures:
            file_path = failure.test_name.split("::", 1)[0]
            normalized = os.path.normpath(file_path)

            work_item_id = reverse_map.get(normalized)

            if work_item_id is None:
                # Fallback: endswith matching, handles relative vs
                # project-root-relative path discrepancies between how
                # pytest reports the file and how it's stored in the map.
                for mapped_path, mapped_id in reverse_map.items():
                    if normalized.endswith(mapped_path) or mapped_path.endswith(
                        normalized
                    ):
                        work_item_id = mapped_id
                        break

            if work_item_id is None:
                # Couldn't resolve — skip retry-targeting for this failure,
                # but the raw failure still surfaces in the FailReport via
                # test_result.failures regardless.
                continue

            retry_failures.setdefault(work_item_id, []).append(failure)
            item = by_id.get(work_item_id)
            if item is not None:
                for dep_id in item.depends_on:
                    retry_failures.setdefault(dep_id, [])

        return retry_failures

    def build_fail_report(
        self,
        run_number: int,
        test_result: TestResult,
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
            f"{test_result.failed} of {test_result.total} tests failed "
            f"across {len(unresolved_work_item_ids)} unresolved work "
            f"item(s) after exhausting retries."
        )

        return FailReport(
            run_number=run_number,
            failures=test_result.failures,
            unresolved_items=unresolved_work_item_ids,
            summary=summary,
        )
