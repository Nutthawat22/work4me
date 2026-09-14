"""
pipeline/ingest.py

Non-interactive entry point that feeds a structured requirements-handoff
package (e.g. requirements/cra-approval-system-handoff/) into the
pipeline, driving MasterAgent.plan_handoff() and then
runner.run_pipeline() without any REPL/input() interaction.

See ~/dev-plans/agents/analysis/2026-09-04-cra-handoff-capability-audit.md
for the gap analysis this implements, and pipeline/master.py's
plan_handoff() docstring for what design_text/requirements_text it
expects.

Usage:
    # Interactive: scans requirements/ and prompts you to pick a package.
    # For automated/scripted use, always pass --handoff explicitly instead.
    python pipeline/ingest.py --dry-run
    python pipeline/ingest.py

    # Explicit/scriptable (bypasses the interactive picker entirely):
    python pipeline/ingest.py --handoff requirements/cra-approval-system-handoff --dry-run
    python pipeline/ingest.py --handoff requirements/cra-approval-system-handoff
"""

import argparse
import hashlib
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pipeline.config_validation import validate_config
from pipeline.master import MasterAgent, MasterPlanError
from pipeline.file_manifest import FileManifestParseError
from pipeline.runner import load_config, run_pipeline

MANIFEST_FILENAME = "MANIFEST.sha256"
DESIGN_DOC_RELPATH = "02-technical-design/04-technical-design.md"
REQUIREMENTS_DOC_RELPATH = "01-requirements/02-requirements.md"
REQUIREMENTS_ROOT = os.path.join(PROJECT_ROOT, "requirements")

# NOTE (2026-09-04, post-run-0007 finding): this tool used to extract only
# Sections 16/19/20 (Tests, Work Packages, Traceability) from
# 04-technical-design.md and feed that excerpt to what was then
# DesignAgent.decompose_handoff() (since renamed to
# MasterAgent.plan_handoff() — see pipeline/master.py).
# That silently dropped Section 7 (Component Design — the ONLY place
# CMP-001 "React Web Client" and CMP-002 "Express/Bun HTTP Adapter" are
# actually defined), so the LLM saw bare "CMP-001"/"CMP-002" labels with
# no definition and produced isolated backend/library modules instead of
# a runnable UI+server app (see run 0007). The design doc is small enough
# (~44K chars) to pass in full — no excerpting needed. Feeding the whole
# document also gives the LLM Section 9 (Data Design), Section 10
# (Runtime Flows), Section 12 (Security), and Section 14 (Deployment),
# which were likewise being silently dropped and are relevant to
# generating correct, wired-together WorkItems.


class ManifestVerificationError(Exception):
    """Raised when MANIFEST.sha256 verification fails (missing file,
    hash mismatch, or unparsable manifest line) and --verify=fail (the
    default) is in effect."""


def parse_manifest(manifest_path: str) -> list[tuple[str, str]]:
    """
    Parse a MANIFEST.sha256 file in standard `sha256sum` format:
    "<64-hex-char-digest>  ./relative/path" (two spaces, sha256sum
    -c-compatible). Returns a list of (digest, relpath) tuples in file
    order. Blank lines are skipped.
    """
    entries: list[tuple[str, str]] = []
    with open(manifest_path, "r") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.rstrip("\n")
            if not stripped.strip():
                continue
            # sha256sum's default (non-binary) format is
            # "<digest>  <path>" (two spaces) or "<digest> <path>" (one
            # space) depending on mode marker; split on first run of
            # whitespace to be tolerant of either.
            match = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.+)$", stripped)
            if not match:
                raise ManifestVerificationError(
                    f"{manifest_path}:{lineno}: unparsable manifest line: {stripped!r}"
                )
            digest, relpath = match.group(1).lower(), match.group(2)
            entries.append((digest, relpath))
    return entries


def verify_manifest(package_dir: str) -> list[str]:
    """
    Recompute sha256 for every file listed in package_dir/MANIFEST.sha256
    and compare against the recorded digest.

    Returns:
        A list of human-readable mismatch/missing-file descriptions.
        Empty list means everything verified clean.

    Raises:
        ManifestVerificationError: if MANIFEST.sha256 itself is missing
            or unparsable — this is always fatal regardless of
            --verify mode, since there's nothing to check against.
    """
    manifest_path = os.path.join(package_dir, MANIFEST_FILENAME)
    if not os.path.isfile(manifest_path):
        raise ManifestVerificationError(f"{manifest_path} not found")

    problems: list[str] = []
    for expected_digest, relpath in parse_manifest(manifest_path):
        full_path = os.path.normpath(os.path.join(package_dir, relpath))
        if not os.path.isfile(full_path):
            problems.append(f"missing file: {relpath}")
            continue
        hasher = hashlib.sha256()
        with open(full_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        actual_digest = hasher.hexdigest()
        if actual_digest != expected_digest:
            problems.append(
                f"hash mismatch: {relpath} "
                f"(expected {expected_digest}, got {actual_digest})"
            )
    return problems


def build_design_spec(package_dir: str, include_requirements: bool) -> tuple[str, str | None]:
    """
    Load 04-technical-design.md IN FULL as design_text (all 22 sections —
    see module-level NOTE above for why this is no longer excerpted to
    just Sections 16/19/20), and optionally load 02-requirements.md in
    full as requirements_text — matching what
    MasterAgent.plan_handoff() expects (see its docstring).

    Returns:
        (design_text, requirements_text_or_None)
    """
    design_doc_path = os.path.join(package_dir, DESIGN_DOC_RELPATH)
    with open(design_doc_path, "r") as f:
        design_text = f.read()

    requirements_text = None
    if include_requirements:
        requirements_doc_path = os.path.join(package_dir, REQUIREMENTS_DOC_RELPATH)
        with open(requirements_doc_path, "r") as f:
            requirements_text = f.read()

    return design_text, requirements_text


def list_handoff_packages(requirements_root: str) -> list[str]:
    """
    List immediate subdirectories of requirements_root, sorted by name.
    Non-directory entries (stray files) are filtered out. Returns
    directory names only (not full paths).
    """
    if not os.path.isdir(requirements_root):
        return []
    return sorted(
        name
        for name in os.listdir(requirements_root)
        if os.path.isdir(os.path.join(requirements_root, name))
    )


def prompt_for_package(packages: list[str]) -> str:
    """
    Print a numbered menu of package names and prompt the user (via
    input()) to pick one. Re-prompts on invalid input (non-numeric,
    out of range). Returns the chosen package name (not full path).

    This is a stand-in for automated selection: a future non-interactive
    caller can bypass this entirely by passing --handoff explicitly.
    """
    print("Available requirements packages:")
    for i, name in enumerate(packages, start=1):
        print(f"  {i}. {name}")
    print()

    while True:
        choice = input(f"Select a package [1-{len(packages)}]: ").strip()
        if not choice.isdigit():
            print("Please enter a number.")
            continue
        index = int(choice)
        if not (1 <= index <= len(packages)):
            print(f"Please enter a number between 1 and {len(packages)}.")
            continue
        return packages[index - 1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Non-interactively feed a requirements-handoff package "
            "(e.g. requirements/cra-approval-system-handoff/) into the "
            "pipeline via MasterAgent.plan_handoff()."
        )
    )
    parser.add_argument(
        "--handoff",
        default=None,
        help="Path to the handoff package directory (contains README.md, "
        "MANIFEST.sha256, 01-requirements/, 02-technical-design/, etc). "
        "If omitted, scans requirements/ for subdirectories and prompts "
        "interactively to choose one (for automated/scripted use, always "
        "pass --handoff explicitly).",
    )
    parser.add_argument(
        "--verify",
        choices=["fail", "warn", "off"],
        default="fail",
        help="MANIFEST.sha256 verification mode. 'fail' (default): "
        "mismatches/missing files abort before any pipeline work. "
        "'warn': print mismatches but continue anyway. 'off': skip "
        "verification entirely.",
    )
    parser.add_argument(
        "--no-requirements",
        action="store_true",
        help="Do not pass the requirements doc as requirements_text "
        "(design_text — the full technical design doc — is always included).",
    )
    parser.add_argument(
        "--mode",
        choices=["handoff", "manifest"],
        default="handoff",
        help="Decomposition mode. 'handoff' (default): plan the design's "
        "scope into WorkItems via plan_handoff() (dynamic team "
        "composition — see pipeline/master.py). 'manifest': "
        "decompose into features, plan a file-indexed manifest, group into "
        "work items, then run the pipeline.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Assemble design_text/requirements_text and run manifest "
        "verification (if enabled), then print a summary and exit "
        "WITHOUT calling plan_handoff() or making any LLM call.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # See pipeline/runner.py's main() for why this is needed: without it,
    # progress output (manifest checks, MasterAgent/dispatch progress
    # printed inside run_pipeline) can sit in a full stdout buffer and
    # only appear once the whole process finishes, instead of streaming
    # live — especially likely here since this entry point is often run
    # non-interactively (stdout not a TTY).
    sys.stdout.reconfigure(line_buffering=True)

    args = parse_args(argv)

    if args.handoff is None:
        packages = list_handoff_packages(REQUIREMENTS_ROOT)
        if not packages:
            print(f"⚠️  No packages found under {REQUIREMENTS_ROOT}")
            print("Pass --handoff <dir> explicitly, or add a package directory there.")
            return 1
        if len(packages) == 1:
            print(f"Only one package found: {packages[0]}")
            choice = input("Use it? [Y/n]: ").strip().lower()
            if choice not in ("", "y", "yes"):
                print("Aborted.")
                return 1
            chosen = packages[0]
        else:
            chosen = prompt_for_package(packages)
        args.handoff = os.path.join(REQUIREMENTS_ROOT, chosen)
        print()

    package_dir = os.path.abspath(args.handoff)
    if not os.path.isdir(package_dir):
        print(f"⚠️  Handoff package directory not found: {package_dir}")
        return 1

    print("=" * 50)
    print("📦 Requirements-Package Ingestion")
    print("=" * 50)
    print(f"Package: {package_dir}")
    print(f"Verify mode: {args.verify}")
    print()

    if args.verify != "off":
        try:
            problems = verify_manifest(package_dir)
        except ManifestVerificationError as e:
            print(f"⚠️  Manifest verification error: {e}")
            return 1

        if problems:
            print(f"⚠️  Manifest verification found {len(problems)} problem(s):")
            for p in problems:
                print(f"  - {p}")
            if args.verify == "fail":
                print("\nAborting (--verify=fail). Use --verify=warn to continue anyway.")
                return 1
            print("\nContinuing anyway (--verify=warn).\n")
        else:
            print("✅ Manifest verification clean — all files match recorded digests.\n")

    try:
        design_text, requirements_text = build_design_spec(
            package_dir, include_requirements=not args.no_requirements
        )
    except (OSError, ValueError) as e:
        print(f"⚠️  Failed to assemble design spec: {e}")
        return 1

    print(f"design_text: {len(design_text)} chars (full technical design document)")
    if requirements_text is not None:
        print(f"requirements_text: {len(requirements_text)} chars (full requirements doc)")
    else:
        print("requirements_text: not included (--no-requirements)")
    print()

    if args.dry_run:
        print("─" * 50)
        if args.mode == "manifest":
            print("DRY RUN (manifest mode) — stopping before "
                  "decompose_features()/any LLM call.")
        else:
            print("DRY RUN — stopping before plan_handoff()/any LLM call.")
        print("─" * 50)
        print()
        if args.mode == "manifest":
            print("Mode: manifest (features → file manifest → groups → work items).")
            print()
        print("design_text preview (first 1000 chars):")
        print(design_text[:1000])
        if len(design_text) > 1000:
            print(f"... ({len(design_text) - 1000} more chars)")
        print()
        if requirements_text is not None:
            print("requirements_text preview (first 500 chars):")
            print(requirements_text[:500])
            if len(requirements_text) > 500:
                print(f"... ({len(requirements_text) - 500} more chars)")
            print()
        print("Dry run complete. No LLM call made, no run dir created.")
        return 0

    config, config_path = load_config()
    errors = validate_config(config)
    if errors:
        print(f"⚠️  Config validation failed (path: {config_path}):")
        for error in errors:
            print(f"  - {error}")
        return 1

    master = MasterAgent(config)

    label_base = os.path.basename(package_dir.rstrip(os.sep))

    if args.mode == "manifest":
        from pipeline.file_manifest import (
            file_groups_to_work_items,
            group_into_work_items,
            plan_file_manifest,
        )

        try:
            print("🧩 Decomposing into features...")
            features = master.decompose_features(design_text, requirements_text)
            print(f"   → {len(features)} feature(s)")
            print("🗂️  Planning file manifest...")
            manifest = plan_file_manifest(features, config)
            print(f"   → {len(manifest.files)} file(s)")
            groups = group_into_work_items(manifest)
            print(f"   → {len(groups)} group(s)")
            work_items = file_groups_to_work_items(groups, manifest)
        except (MasterPlanError, FileManifestParseError) as e:
            print(f"⚠️  manifest planning failed to produce a valid plan: {e}")
            return 1

        label = f"manifest-{label_base}"
        passed = run_pipeline(config, work_items=work_items, label=label)
        return 0 if passed else 1

    try:
        work_items = master.plan_handoff(design_text, requirements_text)
    except MasterPlanError as e:
        print(f"⚠️  plan_handoff() failed to produce a valid plan: {e}")
        return 1

    label = f"handoff-{label_base}"
    passed = run_pipeline(config, work_items=work_items, label=label)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
