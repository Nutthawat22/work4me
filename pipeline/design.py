"""
pipeline/design.py

DesignAgent: decomposes a user prompt into a list of WorkItems by asking
the LLM to output a JSON array matching the WorkItem schema.

Two decomposition paths are supported:

- decompose(user_prompt): free-text path — one LLM call turns a short
  prompt into a flat WorkItem array.
- decompose_handoff(design_text, requirements_text=None): package-aware
  path for approved technical-design handoffs that already contain an
  authored Work Package (WP) decomposition. Translates each WP (plus its
  linked TEST-* items) into one or more WorkItems, preserving WP
  dependency order via `depends_on` and embedding traceability IDs
  (REQ-*/AC-*/TEST-*/WP-*) in `title`/`description` text.

The full technical design document is passed as design_text so the LLM
can resolve component IDs (e.g. CMP-*) against the Component Design
section and produce runnable UI/server code rather than isolated
libraries.
"""

import json
import re

from pipeline.dispatch import CycleError, topological_sort
from pipeline.state import Feature, WorkItem
from specialists.llm_client import call_llm

VALID_TYPES: set[str] = {"logic", "ui", "config", "test", "scaffold", "integrate"}

# Longer timeout: decompose_handoff feeds the full design doc (~100K+ chars),
# well past call_llm's 60s default.
DECOMPOSE_HANDOFF_TIMEOUT_SECONDS = 240

# Longer timeout: decompose_features feeds the full design doc, like
# decompose_handoff, so reuse the same generous budget.
DECOMPOSE_FEATURES_TIMEOUT_SECONDS = 240

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
    Build a Responses-API-compatible structured-output schema (see
    specialists/providers/responses_api.py's response_schema parameter)
    that constrains the LLM's output to a valid array of WorkItem objects,
    with "type" and "language" enum-constrained to VALID_TYPES and the
    caller's valid_languages set respectively. Structured-output
    enforcement prevents missing required fields or out-of-enum values.

    The top-level shape is `{"items": [...]}` rather than a bare array,
    because strict-mode json_schema requires an object at the root;
    _parse_json unwraps it transparently. Strict mode also requires every
    property in "required" and additionalProperties: false at every level.
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

SYSTEM_PROMPT_TEMPLATE = """You are the DesignAgent in an autonomous coding pipeline.

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


HANDOFF_SYSTEM_PROMPT_TEMPLATE = """You are the DesignAgent in an autonomous coding pipeline, operating in \
WP-AWARE HANDOFF mode.

You are given the FULL text of an already-approved technical design document \
(component design, data design, runtime flows, security, deployment, \
acceptance/test design, implementation Work Packages, and end-to-end \
traceability). Your job is NOT to re-decompose the system from scratch. \
Instead, translate the design's own Work Packages (WP-*) into WorkItems for \
specialist agents, preserving the design's authored structure exactly — but \
use the Component Design section's full descriptions (not just bare CMP-* \
IDs) to understand what each WorkItem must actually produce.

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
work belongs to the scaffold item's shared foundation (see rule 7).
- "language" must be exactly one of: {valid_languages}.
- "depends_on" lists ids of OTHER WorkItems IN THIS SAME ARRAY that must be \
completed first. Use [] if there are none.
- Every field is required on every item.
- Output must be a valid JSON array and nothing else.

WP decomposition rules (must follow exactly):
1. Produce one or more WorkItems per Work Package (WP-*). Do not merge \
multiple WPs into one WorkItem, and do not invent WPs, components, or scope \
that are not named in the excerpt.
2. Preserve WP dependency order via `depends_on`: every WorkItem generated \
for a WP that has "Dependencies: WP-00X, WP-00Y, ..." must depend (directly \
or transitively through other same-WP items) on at least one WorkItem \
generated for EACH of those dependency WPs. A WP with "Dependencies: []" \
has no cross-WP depends_on.
3. Every WP that lists Test IDs must yield at least one WorkItem with \
"type": "test" whose `depends_on` includes the implementation WorkItem(s) \
for that same WP, and whose `acceptance_criteria` paraphrases that WP's \
Completion criteria and the linked TEST-* entries' Expected/pass behavior.
4. Embed traceability in text, since the WorkItem schema has no dedicated \
field for it: prefix `title` with the WP id, e.g. "[WP-002] Request draft \
API", and include the WP's Source IDs, Test IDs, and Completion criteria \
verbatim in `description` (e.g. "Source IDs: REQ-003, ... | Test IDs: \
TEST-001, ... | Completion criteria: ...").
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
6. Do not drop any WP present in the excerpt.
6a. SCAFFOLD (runs FIRST) — Emit exactly ONE WorkItem of type "scaffold" \
with an id like "WI-000" and depends_on: []. It runs FIRST and produces the \
dependency manifest, build/compiler config, .gitignore, and the shared \
foundation modules (database setup/migrations, shared config loader, shared \
types) that other modules depend on. EVERY other implementation WorkItem \
(logic/ui/config feature items and their tests) MUST include this scaffold \
item's id in its depends_on. Its output_path is just a representative primary \
path used for labeling (e.g. "package.json"), because a scaffold specialist \
emits MULTIPLE files (a JSON files map), not one file.
6b. INTEGRATE (runs LAST) — Emit exactly ONE WorkItem of type "integrate" as \
the FINAL item, depending on every other implementation WorkItem (all \
logic/ui/config feature items — it does NOT need the test items in its \
depends_on). It writes the composition/wiring files: the server entry point \
mounting all routes, the frontend app shell mounting all screens, and any \
aggregation/index files. Like scaffold, an integrate specialist emits \
MULTIPLE files (a JSON files map), so its output_path is just a \
representative primary path used for labeling (e.g. "src/server/index.ts").
7. RUNNABILITY — look up each WP's named components against the Component \
Design section:
   a. If a WP's components include the component described as a web/browser \
client (e.g. React Web Client), that WP must produce actual renderable UI \
("type": "ui" WorkItem(s)) — real screens/pages/components an end user can \
open, not just supporting libraries (e.g. not just a translation dictionary \
or a types-only module with no rendered page).
   b. If a WP's components include the component described as the HTTP \
server/adapter (e.g. Express/Bun HTTP Adapter), that WP must produce actual \
server route/endpoint wiring ("type": "logic" or "config" WorkItem(s)) that \
maps HTTP requests to the domain modules — not just a domain module with no \
server exposing it.
   c. The single "scaffold" item (rule 6a) and single "integrate" item \
(rule 6b) together handle project setup and final assembly — the scaffold \
produces the dependency manifest and build config up front, and the \
integrate item wires every domain module into a runnable app at the end. Do \
NOT emit separate ASSEMBLY-MANIFEST/ASSEMBLY-ENTRYPOINT config items for \
this; the scaffold + integrate pair replaces them.
   d. This rule does not apply if the design excerpt has no client/server \
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
    Build a Responses-API-compatible strict-mode structured-output schema
    constraining the LLM's output to `{"features": [<feature_obj>]}` (the
    object wrapper is required at the root by strict mode; the parser
    unwraps it). Each feature_obj requires all fields, constrains
    "language" to the caller's valid_languages set, and forbids additional
    properties at every object level.
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


class DesignParseError(Exception):
    """Raised when the DesignAgent's LLM response cannot be parsed into valid WorkItems."""


class DesignAgent:
    def __init__(self, config: dict):
        self.config = config

    def decompose(self, user_prompt: str) -> list[WorkItem]:
        """
        Ask the LLM to decompose user_prompt into WorkItems, parse and
        validate the JSON response, and return constructed WorkItem objects.

        Raises:
            DesignParseError: if the response is not valid JSON or fails
                schema validation.
        """
        valid_languages = set(self.config["languages"].keys())

        model_cfg = self.config["models"]["design"]
        model = model_cfg["model"]
        provider = model_cfg["provider"]
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            valid_languages=", ".join(sorted(valid_languages))
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response_schema = _build_work_items_response_schema("decompose_work_items", valid_languages)
        raw = call_llm(
            messages, model, self.config, provider=provider,
            response_schema=response_schema,
        )

        parsed = self._parse_json(raw)
        return self._validate_and_build(parsed, valid_languages)

    def decompose_handoff(
        self, design_text: str, requirements_text: str | None = None
    ) -> list[WorkItem]:
        """
        Translate an approved technical-design handoff's authored Work
        Package (WP) decomposition into a dependency-ordered list of
        WorkItems, honoring the design's own structure rather than
        re-decomposing from scratch.

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
            correctly encodes WP-to-WP dependency order (so dispatch.py's
            topological sort can consume it directly), and traceability
            IDs (WP-*, REQ-*, TEST-*, AC-*) are embedded in
            `title`/`description` text.

        Raises:
            DesignParseError: if the response is not valid JSON, fails
                schema validation, or — critically — if the generated
                WorkItems have dangling `depends_on` references or a
                dependency cycle (checked via
                pipeline.dispatch.topological_sort(strict=True)).
        """
        valid_languages = set(self.config["languages"].keys())

        model_cfg = self.config["models"]["design"]
        model = model_cfg["model"]
        provider = model_cfg["provider"]
        system_prompt = HANDOFF_SYSTEM_PROMPT_TEMPLATE.format(
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
        response_schema = _build_work_items_response_schema("decompose_handoff_work_items", valid_languages)
        raw = call_llm(
            messages, model, self.config, provider=provider,
            timeout=DECOMPOSE_HANDOFF_TIMEOUT_SECONDS,
            response_schema=response_schema,
        )

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
            DesignParseError: if the response is not valid JSON or fails
                schema validation.
        """
        valid_languages = set(self.config["languages"].keys())

        model_cfg = self.config["models"]["design"]
        model = model_cfg["model"]
        provider = model_cfg["provider"]
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
        raw = call_llm(
            messages, model, self.config, provider=provider,
            timeout=DECOMPOSE_FEATURES_TIMEOUT_SECONDS,
            response_schema=response_schema,
        )

        parsed = self._parse_features_json(raw)
        return self._validate_and_build_features(parsed, valid_languages)

    @staticmethod
    def _check_dependency_graph(work_items: list[WorkItem]) -> None:
        """
        Guard for decompose_handoff: a missed or malformed depends_on edge
        passes schema validation but breaks dispatch.py's topological sort
        at runtime. Fail fast here with a clear message instead.
        """
        try:
            topological_sort(work_items, strict=True)
        except CycleError as e:
            raise DesignParseError(
                f"decompose_handoff produced an invalid dependency graph: {e}"
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
            raise DesignParseError(f"Failed to parse DesignAgent response as JSON: {e}\nRaw response: {raw}")

        if isinstance(parsed, dict) and "items" in parsed:
            parsed = parsed["items"]

        if not isinstance(parsed, list):
            raise DesignParseError(f"Expected a JSON array of WorkItems, got: {type(parsed).__name__}")

        return parsed

    @staticmethod
    def _validate_and_build(items: list[dict], valid_languages: set[str]) -> list[WorkItem]:
        work_items: list[WorkItem] = []

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                raise DesignParseError(f"WorkItem at index {idx} is not a JSON object: {item!r}")

            missing = REQUIRED_FIELDS - item.keys()
            if missing:
                raise DesignParseError(
                    f"WorkItem at index {idx} (id={item.get('id', '?')}) missing required fields: {sorted(missing)}"
                )

            item_type = item["type"]
            if item_type not in VALID_TYPES:
                raise DesignParseError(
                    f"WorkItem {item.get('id', '?')} has invalid type '{item_type}', "
                    f"must be one of {sorted(VALID_TYPES)}"
                )

            item_language = item["language"]
            if item_language not in valid_languages:
                raise DesignParseError(
                    f"WorkItem {item.get('id', '?')} has invalid language '{item_language}', "
                    f"must be one of {sorted(valid_languages)}"
                )

            if not isinstance(item["acceptance_criteria"], list):
                raise DesignParseError(
                    f"WorkItem {item.get('id', '?')} 'acceptance_criteria' must be a list"
                )

            if not isinstance(item["depends_on"], list):
                raise DesignParseError(
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
            raise DesignParseError(
                f"Failed to parse decompose_features response as JSON: {e}\nRaw response: {raw}"
            )

        if isinstance(parsed, dict) and "features" in parsed:
            parsed = parsed["features"]

        if not isinstance(parsed, list):
            raise DesignParseError(
                f"Expected a JSON array of Features, got: {type(parsed).__name__}"
            )

        return parsed

    @staticmethod
    def _validate_and_build_features(
        items: list[dict], valid_languages: set[str]
    ) -> list[Feature]:
        """Validate each Feature dict's required keys/types/language enum and
        construct Feature objects, raising DesignParseError on problems."""
        features: list[Feature] = []

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                raise DesignParseError(f"Feature at index {idx} is not a JSON object: {item!r}")

            missing = FEATURE_REQUIRED_FIELDS - item.keys()
            if missing:
                raise DesignParseError(
                    f"Feature at index {idx} (id={item.get('id', '?')}) "
                    f"missing required fields: {sorted(missing)}"
                )

            item_language = item["language"]
            if item_language not in valid_languages:
                raise DesignParseError(
                    f"Feature {item.get('id', '?')} has invalid language '{item_language}', "
                    f"must be one of {sorted(valid_languages)}"
                )

            for str_field in ("id", "title", "description"):
                if not isinstance(item[str_field], str):
                    raise DesignParseError(
                        f"Feature {item.get('id', '?')} '{str_field}' must be a string"
                    )

            for list_field in ("acceptance_criteria", "source_ids", "depends_on"):
                if not isinstance(item[list_field], list):
                    raise DesignParseError(
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
