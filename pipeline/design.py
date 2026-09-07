"""
pipeline/design.py

DesignAgent: decomposes a user prompt into a list of WorkItems by asking
the LLM to output a JSON array matching the WorkItem schema.

Two decomposition paths are supported:

- decompose(user_prompt): the original free-text path — one LLM call
  turns a short prompt into a flat WorkItem array. Unchanged behavior.
- decompose_handoff(design_text, requirements_text=None): a package-aware
  path for approved technical-design handoffs that already contain an
  authored Work Package (WP) decomposition (see
  requirements/cra-approval-system-handoff/02-technical-design/04-technical-design.md,
  Section 19). Rather than re-decomposing from scratch, this path asks
  the LLM to translate each WP (plus its linked TEST-* items) into one
  or more WorkItems while preserving WP dependency order via
  `depends_on` and embedding traceability IDs (REQ-*/AC-*/TEST-*/WP-*)
  in `title`/`description` text.

  See ~/dev-plans/agents/analysis/2026-09-04-cra-handoff-capability-audit.md
  for the schema-mapping analysis this path implements. Key decisions
  taken from that audit:
  - No WorkItem schema change (audit's low-risk recommendation): WP/
    Source-ID/Test-ID/Completion-criteria/Parallelizable metadata has no
    dedicated field, so it's embedded as text rather than adding new
    fields.
  - SCHEMA-* (DB/migration) work has no dedicated WorkItem type; routed
    to "logic" (closest fit — see audit §4).
  - TypeScript cross-dependency: the design's stack (DEC-004) is
    React/Node/Bun/Express/TypeScript, but config.example.json's
    "languages" map has no "typescript" entry yet (that's owned by a
    separate handoff, HO-4, not yet done). decompose_handoff still emits
    language="typescript" per the design — _validate_and_build will
    reject it against today's real config until HO-4 lands. Tests for
    this path use a mock config with "typescript" registered; this is a
    known, documented blocking dependency, not a bug in this file.
  - Escalation (Section 21 degrees-of-freedom): the design forbids
    unilaterally changing behavior/permissions/data meaning/thresholds/
    ADRs, or accepting RISK-005. The pipeline has no escalation/pause
    primitive (audit §7, untriaged gap #11) — this path can only
    instruct the LLM not to invent scope and flag such items in
    generated descriptions; it cannot enforce a stop-and-ask workflow.

UPDATE (2026-09-04, post-run-0007 finding): decompose_handoff was
originally designed around a Section-16/19/20-only excerpt of the design
doc. That excerpt never included Section 7 (Component Design), so the
LLM had no definition of what "CMP-001" (React Web Client) or "CMP-002"
(Express/Bun HTTP Adapter) actually meant when a WP's Objective referred
to them by bare ID — it produced isolated backend/library modules with
no runnable UI or server wiring (run 0007: 5 files, no package.json, no
app entry point, no rendered UI). The caller (pipeline/ingest.py) now
feeds the FULL technical design document as design_text, and the prompt
below has explicit rules requiring WorkItems targeting CMP-001/CMP-002 to
produce actually-runnable UI/server code, not supporting libraries.
"""

import json
import re

from pipeline.dispatch import CycleError, topological_sort
from pipeline.state import WorkItem
from specialists.llm_client import call_llm

VALID_TYPES: set[str] = {"logic", "ui", "config", "test"}

# decompose_handoff's input can be ~100K+ chars (full design + requirements
# docs); observed real-world latency ~112s for the CRA handoff package,
# vs. call_llm's 60s default sized for decompose()'s much smaller free-text
# path. See decompose_handoff's call_llm invocation below.
DECOMPOSE_HANDOFF_TIMEOUT_SECONDS = 240

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
    that constrains the LLM's output to a valid array of WorkItem
    objects, with "type" and "language" enum-constrained to VALID_TYPES
    and the caller's actual valid_languages set respectively.

    UPDATE (2026-09-04, post-run "missing field on a different WorkItem
    index every time" finding): decompose_handoff's output is a single
    large JSON array (18-20+ items with long descriptions); the LLM
    would occasionally drop a required field or emit an invalid enum
    value on some item, non-deterministically — a known failure mode for
    "generate one huge valid JSON blob in one shot" tasks, not a bug in
    _validate_and_build (which correctly caught every occurrence). A
    prompt instruction ("every field is required") only reduces the
    likelihood; structured-output schema enforcement (verified working
    against this project's LiteLLM proxy/model via a manual test call)
    makes the model's actual token generation structurally incapable of
    omitting a required field or emitting an out-of-enum value, so
    _validate_and_build's field/type/language checks become a redundant
    safety net rather than the primary defense.

    Top-level shape is `{"items": [...]}` rather than a bare array,
    because OpenAI/Responses-API strict-mode json_schema requires an
    object at the root — _parse_json unwraps this transparently, so
    callers/tests that already expect a bare list are unaffected.

    Strict mode also requires every property to be listed in "required"
    (no true optional fields) and additionalProperties: false at every
    object level — both already true of WorkItem's fields, so no schema
    relaxation was needed to satisfy strict mode.
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
  "type": "logic" | "ui" | "config" | "test",
  "language": "python",
  "title": "one-line description",
  "description": "full task spec",
  "acceptance_criteria": ["criterion 1", "criterion 2"],
  "output_path": "relative/path/to/file.py",
  "depends_on": ["WI-000"]
}}

Rules:
- "type" must be exactly one of: logic, ui, config, test.
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
  "type": "logic" | "ui" | "config" | "test",
  "language": "python",
  "title": "one-line description",
  "description": "full task spec",
  "acceptance_criteria": ["criterion 1", "criterion 2"],
  "output_path": "relative/path/to/file.py",
  "depends_on": ["WI-000"]
}}

Rules:
- "type" must be exactly one of: logic, ui, config, test. There is no \
"schema"/"migration" type — route database schema/migration work to "logic".
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
   c. After all WPs are covered, if no WorkItem so far actually assembles \
the client and server into something a user could start and use end-to-end \
(e.g. a server that listens on a port and serves the client, wired to every \
domain module produced), add TWO final "type": "config" WorkItems — each \
depending on every other implementation WorkItem — because EACH WorkItem \
in this pipeline writes to exactly ONE output file, so a dependency \
manifest and an entry-point source file CANNOT be produced by the same \
WorkItem even though they are part of the same "assembly" concern:
      i. "[ASSEMBLY-MANIFEST] <short description>" whose `output_path` is \
the language's actual dependency manifest filename (e.g. "package.json" \
for a Node/Bun/TypeScript stack, "requirements.txt"/"pyproject.toml" for \
Python) and whose content lists every runtime/dev dependency implied by \
the stack and the other WorkItems produced (framework, test runner, \
build tool, etc.) plus the scripts needed to install and start the app \
(e.g. a Bun/npm "dev" or "start" script that runs the entry-point file).
      ii. "[ASSEMBLY-ENTRYPOINT] <short description>" whose `output_path` \
is the actual server/application entry-point source file (e.g. \
"src/server.ts") that wires every domain module together, depending on \
the manifest WorkItem from (i) as well as every implementation WorkItem.
   Do not skip either of these even if no single WP's Objective explicitly \
names them, since HTTP-adapter assembly is commonly a cross-cutting \
concern spanning multiple WPs rather than owned by one, and a dependency \
manifest is easy to forget entirely if only the entry-point file is \
considered.
   d. This rule does not apply if the design excerpt has no client/server \
components at all (e.g. a pure library/CLI design) — only apply (a)-(c) \
when the Component Design section actually describes a web client and/or \
HTTP server component.
8. COMMONLY-FORGOTTEN DELIVERABLES — check the design excerpt's Data \
Design, Observability/Operations, and Deployment/Configuration sections \
(if present) and produce explicit WorkItems for what they describe, since \
these are easy to omit by only reading the Work Packages list:
   a. If the design defines data schemas (e.g. SCHEMA-* entries), include \
a WorkItem that produces the actual schema/migration definition as a real, \
runnable artifact (e.g. SQL migration files or an ORM schema/migration \
script) — not just TypeScript/type-only interfaces describing the shape.
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
        re-decomposing from scratch. See module docstring for the full
        rationale and the audit this implements
        (~/dev-plans/agents/analysis/2026-09-04-cra-handoff-capability-audit.md).

        Args:
            design_text: The FULL text of the technical design document
                (callers, e.g. pipeline/ingest.py, are expected to pass
                the entire document — see module docstring's "UPDATE"
                note for why a Section-16/19/20-only excerpt was found to
                silently drop Section 7's Component Design definitions
                and caused non-runnable output in practice). This method
                does not itself read files — the caller owns loading the
                document.
            requirements_text: Optional full text of the source
                requirements document, if the caller wants additional
                REQ-*/AC-* context beyond what the design doc already
                restates.

        Returns:
            A list[WorkItem] where every generated item's `depends_on`
            correctly encodes WP-to-WP dependency order (so the existing
            dispatch.py topological sort can consume it directly), and
            traceability IDs (WP-*, REQ-*, TEST-*, AC-*) are embedded in
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
        # decompose_handoff feeds the FULL design (+ optional requirements)
        # document — observed ~112s for a ~142K-char CRA handoff input,
        # well past call_llm's 60s default (which is sized for the much
        # smaller decompose() free-text path). Use a longer timeout here
        # specifically; DECOMPOSE_HANDOFF_TIMEOUT_SECONDS is a module
        # constant so it's easy to tune without hunting through the method.
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

    @staticmethod
    def _check_dependency_graph(work_items: list[WorkItem]) -> None:
        """
        Post-generation guard for decompose_handoff: fanning out a WP's
        `Dependencies` into per-item `depends_on` edges is exactly the
        risk the audit flags (§6) as the weak point of WP-aware
        decomposition — a single missed or malformed edge would
        otherwise pass schema validation but break dispatch.py's
        topological sort at runtime, or silently under-order execution.
        Fail fast here with a clear message instead.
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
        Parse raw into a list of WorkItem dicts. Accepts two shapes:
        - A bare JSON array (the plain-text-prompt shape, used when no
          response_schema was requested, e.g. non-structured-output
          providers or tests feeding a hand-crafted response directly).
        - {"items": [...]} (what structured-output mode actually returns
          — see _build_work_items_response_schema's docstring for why
          the top-level object wrapper is required by strict-mode
          json_schema). Unwrapped transparently so callers/tests written
          against the bare-array shape don't need to change.
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
