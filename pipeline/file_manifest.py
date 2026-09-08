"""
pipeline/file_manifest.py

Additive, NOT-YET-WIRED scaffold for the feature-oriented / file-indexed
planning stage. Nothing in the live pipeline (dispatch.py, design.py,
runner.py) calls into this module yet — it exists so the planning flow

    Features  ->  FileManifest  ->  FileGroups

can be built up and tested in isolation before being wired in.

The stage works in three steps:

  1. plan_file_manifest(features, config)   -- turn feature intents into a
     file-indexed plan where every physical path appears exactly once with
     its requirements merged across all contributing features. This is a
     deliberate STUB: it will eventually make ONE structured LLM call, but
     that call is intentionally not implemented here.

  2. group_into_work_items(manifest)        -- cluster FileSpecs into
     FileGroups (cohesion units) that a single multi-file specialist call
     can author. Fully implemented, pure Python, no LLM.

  3. validate_file_manifest(manifest, ...)  -- check the one-author
     invariant and traceability. Fully implemented, pure Python, no LLM;
     returns a list of error strings (never raises), mirroring
     pipeline/config_validation.py.
"""

from __future__ import annotations

import json
import re

from pipeline.state import (
    Feature,
    FileGroup,
    FileManifest,
    FileSpec,
    WorkItem,
)
from specialists.llm_client import call_llm


# Longer timeout: the feature list plus the model's file-planning reasoning
# can be large, well past call_llm's 60s default.
PLAN_FILE_MANIFEST_TIMEOUT_SECONDS = 240

VALID_FILE_ROLES: list[str] = ["scaffold", "feature", "shared", "entrypoint"]

FILE_SPEC_REQUIRED_FIELDS = {
    "path",
    "role",
    "language",
    "contributing_features",
    "requirements",
    "depends_on_files",
}


class FileManifestParseError(Exception):
    """Raised when plan_file_manifest's LLM response cannot be parsed into a
    valid FileManifest (invalid JSON, schema failure, or validation errors)."""


# ── plan_file_manifest (REAL) ────────────────────────────────────────────────

def _build_file_manifest_response_schema(valid_languages: set[str]) -> dict:
    """
    Build a Responses-API-compatible strict-mode structured-output schema
    constraining the LLM's output to `{"files": [<file_spec>]}` (the object
    wrapper is required at the root by strict mode; the parser unwraps it).
    Each file_spec constrains "role" to the four FileRole values and
    "language" to the caller's valid_languages set, requires all six fields,
    and forbids additional properties at every object level.
    """
    file_spec_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "role": {"type": "string", "enum": VALID_FILE_ROLES},
            "language": {"type": "string", "enum": sorted(valid_languages)},
            "contributing_features": {"type": "array", "items": {"type": "string"}},
            "requirements": {"type": "array", "items": {"type": "string"}},
            "depends_on_files": {"type": "array", "items": {"type": "string"}},
        },
        "required": sorted(FILE_SPEC_REQUIRED_FIELDS),
        "additionalProperties": False,
    }
    return {
        "name": "file_manifest",
        "schema": {
            "type": "object",
            "properties": {
                "files": {"type": "array", "items": file_spec_schema},
            },
            "required": ["files"],
            "additionalProperties": False,
        },
    }


PLAN_FILE_MANIFEST_SYSTEM_PROMPT = """You are the file-planning stage in an autonomous coding pipeline.

You are given a list of feature intents (the WHAT — what each feature must \
do). Your job is to produce the file-indexed plan: the concrete set of \
files the application needs, each mapped back to the features it serves.

Output ONLY a JSON object (no prose, no markdown fences) of this exact shape:

{{
  "files": [
    {{
      "path": "relative/path/to/file.ts",
      "role": "scaffold" | "feature" | "shared" | "entrypoint",
      "language": "typescript",
      "contributing_features": ["FEAT-auth"],
      "requirements": ["file-scoped requirement bullet"],
      "depends_on_files": ["other/path/in/this/array.ts"]
    }}
  ]
}}

THE ONE-AUTHOR INVARIANT (most important): every physical file path appears \
EXACTLY ONCE in the files array. If multiple features need something in the \
same file (a router, an i18n bundle, a shared schema, an app shell), emit \
ONE FileSpec for that path and MERGE all their requirements into its \
`requirements`, listing every contributing feature id in \
`contributing_features`. Never emit the same path twice.

role meanings:
- "scaffold" = project skeleton / build config / shared foundation \
(package.json, tsconfig, db/config/shared types). May have empty \
contributing_features.
- "feature" = a file owned by a SINGLE feature.
- "shared" = a file MULTIPLE features contribute to — put every contributor \
in contributing_features.
- "entrypoint" = the composition/wiring files that assemble everything \
(server entry, app shell).

Rules:
- Every non-scaffold file must list at least one contributing feature, and \
every contributing feature id must be one of the given feature ids.
- Every feature provided must contribute to at least one file — do not drop \
a feature.
- depends_on_files must reference other paths present in the files array: no \
dangling references, no self-references, and no cycles. Use import/composition \
order — entrypoints depend on the feature/shared files they wire; feature \
files may depend on scaffold/shared foundation files.
- "language" must be exactly one of: {valid_languages}.
- Split each feature into cohesive small files rather than one monolithic \
file (e.g. a backend feature: route + service/logic + any local helper; a \
frontend feature: its screen(s)).
"""


def _render_features(features: list[Feature]) -> str:
    """Render features as a plain-text block for the user message: id, title,
    language, description, acceptance criteria, source ids, and depends_on."""
    blocks: list[str] = []
    for feat in features:
        lines = [
            f"Feature {feat.id}: {feat.title}",
            f"  language: {feat.language}",
            f"  description: {feat.description}",
        ]
        if feat.acceptance_criteria:
            lines.append("  acceptance_criteria:")
            lines.extend(f"    - {ac}" for ac in feat.acceptance_criteria)
        else:
            lines.append("  acceptance_criteria: (none)")
        lines.append(f"  source_ids: {', '.join(feat.source_ids) if feat.source_ids else '(none)'}")
        lines.append(f"  depends_on: {', '.join(feat.depends_on) if feat.depends_on else '(none)'}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def plan_file_manifest(features: list[Feature], config: dict) -> FileManifest:
    """
    Turn a list of feature intents into a file-indexed FileManifest via ONE
    structured LLM call that maps all features onto the concrete set of files
    the app needs.

    Enforces the one-author invariant: every physical path appears exactly
    once as a single FileSpec, with the requirements from every contributing
    feature MERGED into that spec (so a shared file such as a router, an i18n
    bundle, or a schema is never written twice by competing features). The
    result is validated with validate_file_manifest() before being returned.

    Raises:
        FileManifestParseError: if the response is not valid JSON, is
            structurally malformed, or fails manifest validation.
    """
    valid_languages = set(config["languages"].keys())

    model_cfg = config["models"]["design"]
    model = model_cfg["model"]
    provider = model_cfg["provider"]

    system_prompt = PLAN_FILE_MANIFEST_SYSTEM_PROMPT.format(
        valid_languages=", ".join(sorted(valid_languages))
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _render_features(features)},
    ]
    response_schema = _build_file_manifest_response_schema(valid_languages)
    raw = call_llm(
        messages, model, config, provider=provider,
        timeout=PLAN_FILE_MANIFEST_TIMEOUT_SECONDS,
        response_schema=response_schema,
    )

    if raw.startswith(f"[{model}] Error:"):
        raise FileManifestParseError(raw)

    file_dicts = _parse_file_manifest_json(raw)
    manifest = FileManifest(files=[_build_file_spec(idx, d) for idx, d in enumerate(file_dicts)])

    errors = validate_file_manifest(manifest, features)
    if errors:
        raise FileManifestParseError(
            "plan_file_manifest produced an invalid manifest:\n" + "\n".join(errors)
        )

    return manifest


def _parse_file_manifest_json(raw: str) -> list[dict]:
    """
    Parse raw into a list of FileSpec dicts. Accepts either a
    {"files": [...]} wrapper (the structured-output shape) or a bare JSON
    array (for robustness), stripping a ```json fence if present.
    """
    stripped = raw.strip()

    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fence_match:
        stripped = fence_match.group(1).strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as e:
        raise FileManifestParseError(
            f"Failed to parse plan_file_manifest response as JSON: {e}\nRaw response: {raw}"
        )

    if isinstance(parsed, dict) and "files" in parsed:
        parsed = parsed["files"]

    if not isinstance(parsed, list):
        raise FileManifestParseError(
            f"Expected a JSON object with 'files' array, got: {type(parsed).__name__}"
        )

    return parsed


def _build_file_spec(idx: int, item: dict) -> FileSpec:
    """Validate one FileSpec dict's keys and types, then construct a FileSpec.
    Raises FileManifestParseError on any structural problem."""
    if not isinstance(item, dict):
        raise FileManifestParseError(f"FileSpec at index {idx} is not a JSON object: {item!r}")

    missing = FILE_SPEC_REQUIRED_FIELDS - item.keys()
    if missing:
        raise FileManifestParseError(
            f"FileSpec at index {idx} (path={item.get('path', '?')}) "
            f"missing required fields: {sorted(missing)}"
        )

    if not isinstance(item["path"], str):
        raise FileManifestParseError(f"FileSpec at index {idx} 'path' must be a string")

    role = item["role"]
    if role not in VALID_FILE_ROLES:
        raise FileManifestParseError(
            f"FileSpec {item.get('path', '?')} has invalid role '{role}', "
            f"must be one of {VALID_FILE_ROLES}"
        )

    if not isinstance(item["language"], str):
        raise FileManifestParseError(f"FileSpec {item.get('path', '?')} 'language' must be a string")

    for list_field in ("contributing_features", "requirements", "depends_on_files"):
        if not isinstance(item[list_field], list):
            raise FileManifestParseError(
                f"FileSpec {item.get('path', '?')} '{list_field}' must be a list"
            )

    return FileSpec(
        path=item["path"],
        role=role,  # type: ignore[arg-type]
        language=item["language"],
        contributing_features=item["contributing_features"],
        requirements=item["requirements"],
        depends_on_files=item["depends_on_files"],
    )


# ── group_into_work_items (REAL) ─────────────────────────────────────────────

def _sanitize_path(path: str) -> str:
    """Turn a file path into an id fragment: '/' and '.' become '-', lowercased."""
    return path.replace("/", "-").replace(".", "-").lower()


def _dedup_preserve_order(items: list[str]) -> list[str]:
    """Return items with duplicates removed, preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def group_into_work_items(manifest: FileManifest) -> list[FileGroup]:
    """
    Cluster the FileSpecs of a FileManifest into FileGroups — cohesion units
    each authored by a single multi-file specialist call. Pure Python; no LLM.

    Clustering rules:
      - scaffold-role files            -> ONE group "GRP-scaffold".
      - feature-role, single feature   -> one group per feature id
                                          "GRP-feature-{featureid}" (local cohesion).
      - feature-role, MULTIPLE features -> convergent, so treated like shared:
                                          its own group, role "shared".
      - shared-role files              -> EACH its own group (single author,
                                          many contributors).
      - entrypoint-role files          -> EACH its own group.

    depends_on between groups is derived from FileSpec.depends_on_files:
    group A depends_on group B (A != B) if any file in A imports any file in B.

    Groups are returned ordered scaffold -> feature -> shared -> entrypoint,
    stable-by-id within each bucket.
    """
    # Bucket each spec into a group key.
    #   grouped_specs: group_id -> list[FileSpec]
    #   group_meta:    group_id -> (role, title)
    grouped_specs: dict[str, list[FileSpec]] = {}
    group_meta: dict[str, tuple[str, str]] = {}

    def _bucket(group_id: str, role: str, title: str, spec: FileSpec) -> None:
        grouped_specs.setdefault(group_id, []).append(spec)
        # First writer sets the title/role for the group.
        group_meta.setdefault(group_id, (role, title))

    for spec in manifest.files:
        if spec.role == "scaffold":
            _bucket("GRP-scaffold", "scaffold", "Project scaffold", spec)
        elif spec.role == "feature" and len(spec.contributing_features) == 1:
            feat = spec.contributing_features[0]
            _bucket(
                f"GRP-feature-{feat}",
                "feature",
                f"Feature: {feat}",
                spec,
            )
        elif spec.role == "shared" or (
            spec.role == "feature" and len(spec.contributing_features) > 1
        ):
            # shared files, and convergent feature files, get a solo group so
            # exactly one author owns the physical path.
            gid = f"GRP-shared-{_sanitize_path(spec.path)}"
            _bucket(gid, "shared", f"Shared: {spec.path}", spec)
        elif spec.role == "entrypoint":
            gid = f"GRP-entrypoint-{_sanitize_path(spec.path)}"
            _bucket(gid, "entrypoint", f"Entrypoint: {spec.path}", spec)
        else:
            # feature-role file with ZERO contributing features — degenerate,
            # but keep it addressable rather than silently dropping it.
            gid = f"GRP-shared-{_sanitize_path(spec.path)}"
            _bucket(gid, "shared", f"Shared: {spec.path}", spec)

    # Map each path -> its owning group id (for dependency resolution).
    path_to_group: dict[str, str] = {}
    for group_id, specs in grouped_specs.items():
        for spec in specs:
            path_to_group[spec.path] = group_id

    # Build the FileGroup objects.
    groups: dict[str, FileGroup] = {}
    for group_id, specs in grouped_specs.items():
        role, title = group_meta[group_id]
        # language: use the language of the files; if mixed, first is fine.
        language = specs[0].language

        files = sorted(spec.path for spec in specs)

        contributing: list[str] = []
        for spec in specs:
            contributing.extend(spec.contributing_features)
        contributing = sorted(set(contributing))

        requirements: list[str] = []
        for spec in specs:
            requirements.extend(spec.requirements)
        requirements = _dedup_preserve_order(requirements)

        # Cross-group dependencies from file-level imports.
        depends: set[str] = set()
        for spec in specs:
            for dep_path in spec.depends_on_files:
                dep_group = path_to_group.get(dep_path)
                if dep_group is not None and dep_group != group_id:
                    depends.add(dep_group)

        groups[group_id] = FileGroup(
            id=group_id,
            role=role,
            language=language,
            title=title,
            files=files,
            contributing_features=contributing,
            requirements=requirements,
            depends_on=sorted(depends),
        )

    # Order: scaffold -> feature -> shared -> entrypoint, stable-by-id within.
    role_order = {"scaffold": 0, "feature": 1, "shared": 2, "entrypoint": 3}
    return sorted(
        groups.values(),
        key=lambda g: (role_order.get(g.role, 99), g.id),
    )


# ── file_groups_to_work_items (adapter) ──────────────────────────────────────

# FileGroup.role -> WorkItemType. entrypoint groups wire everything together,
# so they become "integrate" work items.
_ROLE_TO_WORK_ITEM_TYPE = {
    "scaffold": "scaffold",
    "feature": "feature",
    "shared": "shared",
    "entrypoint": "integrate",
}


def file_groups_to_work_items(
    groups: list[FileGroup],
    manifest: FileManifest,
    language_default: str = "typescript",
) -> list[WorkItem]:
    """
    Adapt planning-artifact FileGroups into dispatchable WorkItems — the
    bridge from the file-manifest planning stage into the live pipeline.

    Each FileGroup maps to exactly one WorkItem. The description embeds the
    explicit list of files to create with per-file requirements (looked up
    from the manifest) plus the contributing features, so the multifile
    specialist knows exactly what to produce. Group ids double as WorkItem
    ids, so depends_on carries over unchanged. The adapter is purely
    structural — it does not synthesize test work items.

    Groups are returned in the same order group_into_work_items produced
    (scaffold -> feature -> shared -> entrypoint); dispatch's
    topological_sort will re-order by depends_on anyway.
    """
    spec_by_path: dict[str, FileSpec] = {spec.path: spec for spec in manifest.files}

    work_items: list[WorkItem] = []
    for group in groups:
        feats = ", ".join(group.contributing_features) if group.contributing_features else "(none)"
        lines = [
            f"Create exactly these files for group {group.id} "
            f"(contributing features: {feats}):"
        ]
        for path in group.files:
            spec = spec_by_path.get(path)
            reqs = "; ".join(spec.requirements) if (spec and spec.requirements) else "no specific requirements"
            lines.append(f"- {path}: {reqs}")
        description = "\n".join(lines)

        work_items.append(
            WorkItem(
                id=group.id,
                type=_ROLE_TO_WORK_ITEM_TYPE.get(group.role, "feature"),  # type: ignore[arg-type]
                language=group.language or language_default,
                title=group.title,
                description=description,
                acceptance_criteria=list(group.requirements),
                output_path=group.files[0] if group.files else "",
                depends_on=list(group.depends_on),
            )
        )

    return work_items


# ── validate_file_manifest (REAL) ────────────────────────────────────────────

def _detect_file_cycle(manifest: FileManifest) -> list[str]:
    """
    Return a list of paths involved in a dependency cycle over
    depends_on_files, or an empty list if the graph is acyclic. Small
    self-contained DFS — deliberately does NOT import dispatch.
    """
    # Adjacency limited to edges that point at real files in the manifest.
    known = {spec.path for spec in manifest.files}
    adj: dict[str, list[str]] = {
        spec.path: [d for d in spec.depends_on_files if d in known]
        for spec in manifest.files
    }

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {p: WHITE for p in adj}
    cycle: list[str] = []

    def dfs(node: str, stack: list[str]) -> bool:
        color[node] = GRAY
        stack.append(node)
        for nxt in adj.get(node, []):
            if color.get(nxt, BLACK) == GRAY:
                # Found a back-edge: extract the cycle from the stack.
                idx = stack.index(nxt)
                cycle.extend(stack[idx:])
                return True
            if color.get(nxt, BLACK) == WHITE and dfs(nxt, stack):
                return True
        stack.pop()
        color[node] = BLACK
        return False

    for path in adj:
        if color[path] == WHITE:
            if dfs(path, []):
                return cycle
    return []


def validate_file_manifest(manifest: FileManifest, features: list[Feature]) -> list[str]:
    """
    Validate a FileManifest against its feature set. Returns a list of
    human-readable error strings (empty list == valid). Never raises;
    mirrors the style of pipeline/config_validation.validate_config.

    Checks:
      1. No duplicate FileSpec.path (the one-author invariant).
      2. Every depends_on_files entry references a path in the manifest.
      3. No file depends on itself.
      4. Every Feature id contributes to at least one file (no dropped feature).
      5. Every contributing_features entry references a real Feature id.
      6. Every non-scaffold FileSpec has >= 1 contributing feature.
      7. No dependency cycle over depends_on_files.
    """
    errors: list[str] = []

    all_paths = [spec.path for spec in manifest.files]
    known_paths = set(all_paths)
    feature_ids = {f.id for f in features}

    # 1. Duplicate paths.
    seen: set[str] = set()
    reported_dupes: set[str] = set()
    for path in all_paths:
        if path in seen and path not in reported_dupes:
            errors.append(f"duplicate file path (violates one-author invariant): {path}")
            reported_dupes.add(path)
        seen.add(path)

    # 2 & 3. depends_on_files integrity.
    for spec in manifest.files:
        for dep in spec.depends_on_files:
            if dep == spec.path:
                errors.append(f"file depends on itself: {spec.path}")
            elif dep not in known_paths:
                errors.append(
                    f"dangling depends_on_files: {spec.path} -> {dep} (no such file in manifest)"
                )

    # 5. contributing_features reference real features; collect coverage.
    covered_features: set[str] = set()
    for spec in manifest.files:
        for feat in spec.contributing_features:
            if feat not in feature_ids:
                errors.append(
                    f"unknown feature id in contributing_features of {spec.path}: {feat}"
                )
            else:
                covered_features.add(feat)

    # 4. Dropped features.
    for feat_id in sorted(feature_ids):
        if feat_id not in covered_features:
            errors.append(f"feature contributes to no file (dropped feature): {feat_id}")

    # 6. Non-scaffold files must have >= 1 contributing feature.
    for spec in manifest.files:
        if spec.role != "scaffold" and not spec.contributing_features:
            errors.append(
                f"non-scaffold file has no contributing features: {spec.path}"
            )

    # 7. Cycle check.
    cycle = _detect_file_cycle(manifest)
    if cycle:
        errors.append("dependency cycle among files: " + " -> ".join(cycle))

    return errors
