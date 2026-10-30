"""Build external runtime-testing guidance without executing project commands."""

from __future__ import annotations

import ast
import json
import os
import re
from sys import stdlib_module_names
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.project_context import IGNORED_DIRECTORIES
from pipeline.project_paths import resolve_project_path
from pipeline.task_validation import normalize_relative_path

LOCKFILE_NAMES = {
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "bun.lock",
    "bun.lockb",
    "poetry.lock",
    "uv.lock",
    "Pipfile.lock",
    "Cargo.lock",
}
EXTERNAL_TEST_RESULT_CONTRACT = {
    "schema_version": 1,
    "required_fields": [
        "schema_version",
        "run_id",
        "outcome",
        "tested_copy_path",
        "environment_setup",
        "commands",
        "completed_at",
    ],
    "outcomes": ["passed", "code_or_test_failure", "environment_setup_failure", "not_run"],
    "environment_setup_status_values": ["succeeded", "failed", "not_attempted"],
    "command_status_values": ["passed", "failed", "not_run"],
    "secret_policy": "Do not include environment variable values, credentials, or secret-bearing logs.",
}
MANIFEST_NAMES = {"package.json", "pyproject.toml", "requirements.txt", "Pipfile", "Cargo.toml"}
ENV_NAME_PATTERNS = (
    re.compile(r"\bprocess\.env\.([A-Za-z_][A-Za-z0-9_]*)"),
    re.compile(r"\bimport\.meta\.env\.([A-Za-z_][A-Za-z0-9_]*)"),
    re.compile(r"\bos\.getenv\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]"),
    re.compile(r"\bos\.environ(?:\.get)?\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]"),
    re.compile(r"\bos\.environ\[\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*\]"),
)
JS_PACKAGE_PATTERN = re.compile(
    r"(?:\bfrom\s*|\bimport\s*|\brequire\s*\(\s*|\bimport\s*\(\s*)['\"]([^.'\"][^'\"]*)['\"]"
)
SECRET_URL_CREDENTIALS = re.compile(r"(https?://)[^/@\s]+@", re.IGNORECASE)
SECRET_ASSIGNMENT = re.compile(
    r"(?i)(password|passwd|token|secret|key|auth|bearer|credential)(\s*[=:]\s*)[^,\s\"']+"
)
TOKEN_LIKE_VALUE = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})\b")
HEADER_CREDENTIAL = re.compile(
    r"(?i)\b(authorization|proxy-authorization|cookie|set-cookie)(\s*[:=]\s*)[^\r\n,]+"
)
REQUIREMENT_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)(?:\[[^]]+\])?(.*)$")
REQUIREMENTS_MANIFEST = re.compile(r"^requirements(?:[-_.][A-Za-z0-9_.-]+)?\.txt$", re.IGNORECASE)
MAX_RUNTIME_VERSION_BYTES = 256
NODE_VERSION_DECLARATION = re.compile(
    r"(?:v?\d+(?:\.\d+){0,2}(?:[-+][A-Za-z0-9.-]+)?|node|stable|lts(?:/[A-Za-z0-9._-]+)?)",
    re.IGNORECASE,
)
PYTHON_VERSION_DECLARATION = re.compile(r"(?:\d+(?:\.\d+){0,2}(?:[-+][A-Za-z0-9.-]+)?|system)", re.IGNORECASE)


def _is_manifest_name(name: str) -> bool:
    return name in MANIFEST_NAMES or bool(REQUIREMENTS_MANIFEST.fullmatch(name))


def _walk_files(root: str) -> list[str]:
    paths: list[str] = []
    if not _safe_directory(root):
        return paths
    for current, directories, names in os.walk(root, followlinks=False):
        directories[:] = sorted(
            name for name in directories
            if name not in IGNORED_DIRECTORIES
            and not os.path.islink(os.path.join(current, name))
        )
        for name in sorted(names):
            full_path = os.path.join(current, name)
            if os.path.islink(full_path) or not os.path.isfile(full_path):
                continue
            paths.append(os.path.relpath(full_path, root).replace(os.sep, "/"))
    return paths


def _path_has_symlink_component(path: str) -> bool:
    absolute = os.path.abspath(path)
    current = os.path.abspath(os.sep)
    for component in Path(absolute).parts[1:]:
        current = os.path.join(current, component)
        if os.path.islink(current):
            return True
    return False


def _safe_directory(root: str) -> bool:
    absolute = os.path.abspath(root)
    return (
        not _path_has_symlink_component(absolute)
        and os.path.isdir(absolute)
        and os.path.commonpath((os.path.realpath(absolute), absolute)) == absolute
    )


def _canonical_run_directory(run_dir: str, configured_runs_dir: object) -> str | None:
    absolute_run_dir = os.path.abspath(run_dir)
    if os.path.islink(absolute_run_dir):
        return None
    if isinstance(configured_runs_dir, str) and configured_runs_dir:
        storage_dir = resolve_project_path(configured_runs_dir)
    else:
        storage_dir = os.path.dirname(absolute_run_dir)
    absolute_storage_dir = os.path.abspath(storage_dir)
    try:
        if os.path.commonpath((absolute_storage_dir, absolute_run_dir)) != absolute_storage_dir:
            return None
    except ValueError:
        return None
    canonical_storage_dir = os.path.realpath(absolute_storage_dir)
    expected_run_dir = os.path.join(
        canonical_storage_dir,
        os.path.relpath(absolute_run_dir, absolute_storage_dir),
    )
    canonical_run_dir = os.path.realpath(absolute_run_dir)
    if canonical_run_dir != expected_run_dir or not os.path.isdir(canonical_run_dir):
        return None
    return canonical_run_dir


def _canonical_generated_root(root: str, run_dir: str, canonical_run_dir: str) -> str | None:
    absolute_root = os.path.abspath(root)
    absolute_run_dir = os.path.abspath(run_dir)
    try:
        if os.path.commonpath((absolute_run_dir, absolute_root)) != absolute_run_dir:
            return None
    except ValueError:
        return None
    expected_root = os.path.join(
        canonical_run_dir,
        os.path.relpath(absolute_root, absolute_run_dir),
    )
    canonical_root = os.path.realpath(absolute_root)
    if canonical_root != expected_root or not os.path.isdir(canonical_root):
        return None
    return canonical_root


def _contains_symlink_entry(root: str) -> bool:
    for current, directories, names in os.walk(root, followlinks=False):
        if any(os.path.islink(os.path.join(current, name)) for name in directories + names):
            return True
    return False


def _safe_file_path(path: str, root: str) -> bool:
    absolute_root = os.path.abspath(root)
    absolute_path = os.path.abspath(path)
    try:
        inside_root = os.path.commonpath((absolute_root, absolute_path)) == absolute_root
    except ValueError:
        return False
    if not inside_root or not _safe_directory(absolute_root):
        return False
    if _path_has_symlink_component(absolute_path):
        return False
    try:
        return os.path.commonpath((os.path.realpath(absolute_root), os.path.realpath(absolute_path))) == os.path.realpath(absolute_root)
    except ValueError:
        return False


def _manifest_has_malformed_shapes(name: str, data: Any) -> bool:
    if name == "package.json":
        if not isinstance(data, dict):
            return True
        dependency_groups = ("dependencies", "devDependencies", "optionalDependencies", "peerDependencies")
        for group in (*dependency_groups, "scripts", "engines"):
            values = data.get(group, {})
            if not isinstance(values, dict):
                return True
            if any(not isinstance(key, str) or not isinstance(value, str) for key, value in values.items()):
                return True
        return False
    if not isinstance(data, dict):
        return True
    if name == "pyproject.toml":
        project = data.get("project", {})
        tool = data.get("tool", {})
        if not isinstance(project, dict) or not isinstance(tool, dict):
            return True
        declared = project.get("dependencies", [])
        optional = project.get("optional-dependencies", {})
        if not isinstance(declared, list) or any(not isinstance(value, str) for value in declared):
            return True
        if not isinstance(optional, dict) or any(
            not isinstance(values, list) or any(not isinstance(value, str) for value in values)
            for values in optional.values()
        ):
            return True
        poetry = tool.get("poetry", {})
        if not isinstance(poetry, dict):
            return True
        dependencies = poetry.get("dependencies", {})
        groups = poetry.get("group", {})
        return (
            not isinstance(dependencies, dict)
            or any(
                not isinstance(name, str)
                or not isinstance(specification, (str, dict))
                or isinstance(specification, dict)
                and "version" in specification
                and not isinstance(specification["version"], str)
                for name, specification in dependencies.items()
            )
            or not isinstance(groups, dict)
            or any(
                not isinstance(group, dict)
                or not isinstance(group.get("dependencies", {}), dict)
                or any(
                    not isinstance(name, str)
                    or not isinstance(specification, (str, dict))
                    or isinstance(specification, dict)
                    and "version" in specification
                    and not isinstance(specification["version"], str)
                    for name, specification in group.get("dependencies", {}).items()
                )
                for group in groups.values()
            )
        )
    if name == "Pipfile":
        requires = data.get("requires", {})
        return any(
            not isinstance(data.get(group, {}), dict)
            for group in ("packages", "dev-packages")
        ) or not isinstance(requires, dict)
    if name == "Cargo.toml":
        return any(
            not isinstance(data.get(group, {}), dict)
            for group in ("dependencies", "dev-dependencies", "build-dependencies")
        ) or not isinstance(data.get("package", {}), dict)
    return False


def _normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _safe_specification(specification: str) -> str:
    return _redact_secrets(SECRET_URL_CREDENTIALS.sub(r"\1[REDACTED]@", specification))


def _redact_secrets(value: str) -> str:
    value = HEADER_CREDENTIAL.sub(r"\1\2[REDACTED]", value)
    value = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [REDACTED]", value)
    value = SECRET_ASSIGNMENT.sub(r"\1\2[REDACTED]", value)
    return TOKEN_LIKE_VALUE.sub("[REDACTED]", value)


def _manifest_data(
    project_roots: list[str],
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, str]], list[str], list[str]]:
    manifests: list[dict[str, Any]] = []
    dependencies: dict[str, str] = {}
    dependency_records: list[dict[str, str]] = []
    lockfiles: list[str] = []
    warnings: list[str] = []

    def record_dependency(name: str, specification: str, group: str, source: str) -> None:
        safe_specification = _safe_specification(specification)
        dependencies.setdefault(name, safe_specification)
        dependency_records.append({
            "name": name,
            "version_specification": safe_specification,
            "group": group,
            "source": source,
        })

    seen: set[str] = set()
    for root in project_roots:
        for relative in _walk_files(root):
            basename = os.path.basename(relative)
            full_path = os.path.join(root, relative)
            absolute_display = os.path.join(root, relative)
            if basename in LOCKFILE_NAMES:
                lockfiles.append(absolute_display)
            if not _is_manifest_name(basename) or absolute_display in seen:
                continue
            seen.add(absolute_display)
            manifest: dict[str, Any] = {"path": absolute_display, "type": basename}
            try:
                if basename == "package.json":
                    with open(full_path, "r", encoding="utf-8") as source:
                        data = json.load(source)
                    if _manifest_has_malformed_shapes(basename, data):
                        manifest["read_status"] = "invalid_or_unsupported_shape"
                        warnings.append(f"{basename} contains malformed metadata shapes; some declarations may be omitted from the handoff.")
                    for group in ("dependencies", "devDependencies", "optionalDependencies", "peerDependencies"):
                        group_data = data.get(group, {}) if isinstance(data, dict) else {}
                        if isinstance(group_data, dict):
                            for name, version in group_data.items():
                                if isinstance(name, str) and isinstance(version, str):
                                    record_dependency(name, version, group, absolute_display)
                    engines = data.get("engines") if isinstance(data, dict) else None
                    if isinstance(engines, dict):
                        manifest["engines"] = {
                            name: _safe_specification(value)
                            for name, value in engines.items()
                            if name in {"node", "python"} and isinstance(value, str)
                        }
                elif basename == "pyproject.toml":
                    with open(full_path, "rb") as source:
                        data = tomllib.load(source)
                    if _manifest_has_malformed_shapes(basename, data):
                        manifest["read_status"] = "invalid_or_unsupported_shape"
                        warnings.append(f"{basename} contains malformed metadata shapes; some declarations may be omitted from the handoff.")
                    project = data.get("project", {})
                    declared = project.get("dependencies", []) if isinstance(project, dict) else []
                    optional = project.get("optional-dependencies", {}) if isinstance(project, dict) else {}
                    if isinstance(optional, dict):
                        for group_name, group_items in optional.items():
                            if isinstance(group_items, list):
                                for requirement in group_items:
                                    if isinstance(requirement, str):
                                        match = REQUIREMENT_NAME.match(requirement)
                                        if match:
                                            record_dependency(match.group(1), match.group(2).strip() or "unspecified", f"project.optional-dependencies.{group_name}", absolute_display)
                    if isinstance(declared, list):
                        for requirement in declared:
                            if isinstance(requirement, str):
                                match = REQUIREMENT_NAME.match(requirement)
                                if match:
                                    record_dependency(
                                        match.group(1),
                                        match.group(2).strip() or "unspecified",
                                        "project.dependencies",
                                        absolute_display,
                                    )
                    tools = data.get("tool", {})
                    poetry = tools.get("poetry", {}) if isinstance(tools, dict) else {}
                    if not isinstance(poetry, dict):
                        poetry = {}
                    poetry_groups = {"tool.poetry.dependencies": poetry.get("dependencies", {})}
                    groups = poetry.get("group", {})
                    if isinstance(groups, dict):
                        for group_name, group_data in groups.items():
                            if isinstance(group_data, dict):
                                poetry_groups[f"tool.poetry.group.{group_name}.dependencies"] = group_data.get("dependencies", {})
                    for group_name, entries in poetry_groups.items():
                        if isinstance(entries, dict):
                            for name, spec in entries.items():
                                if name == "python":
                                    continue
                                version = spec if isinstance(spec, str) else spec.get("version", "unspecified") if isinstance(spec, dict) else "unspecified"
                                record_dependency(name, str(version), group_name, absolute_display)
                    requires_python = project.get("requires-python") if isinstance(project, dict) else None
                    if not isinstance(requires_python, str):
                        poetry_dependencies = poetry.get("dependencies", {})
                        poetry_python = poetry_dependencies.get("python") if isinstance(poetry_dependencies, dict) else None
                        if isinstance(poetry_python, str):
                            requires_python = poetry_python
                        elif isinstance(poetry_python, dict):
                            requires_python = poetry_python.get("version")
                    manifest["requires_python"] = _safe_specification(requires_python) if isinstance(requires_python, str) else None
                elif REQUIREMENTS_MANIFEST.fullmatch(basename):
                    with open(full_path, "r", encoding="utf-8") as source:
                        lines = source.readlines()
                    for line in lines:
                        requirement = line.strip()
                        if not requirement or requirement.startswith(("#", "-")):
                            continue
                        match = REQUIREMENT_NAME.match(requirement)
                        if match:
                            record_dependency(
                                match.group(1),
                                match.group(2).strip() or "unspecified",
                                "requirements.txt",
                                absolute_display,
                            )
                elif basename == "Pipfile":
                    with open(full_path, "rb") as source:
                        data = tomllib.load(source)
                    if _manifest_has_malformed_shapes(basename, data):
                        manifest["read_status"] = "invalid_or_unsupported_shape"
                        warnings.append(f"{basename} contains malformed metadata shapes; some declarations may be omitted from the handoff.")
                    requires = data.get("requires", {})
                    python_version = (
                        requires.get("python_full_version", requires.get("python_version"))
                        if isinstance(requires, dict)
                        else None
                    )
                    manifest["requires_python"] = (
                        _safe_specification(python_version)
                        if isinstance(python_version, str)
                        else None
                    )
                    for group in ("packages", "dev-packages"):
                        entries = data.get(group, {})
                        if isinstance(entries, dict):
                            for name, spec in entries.items():
                                version = spec if isinstance(spec, str) else spec.get("version", "unspecified") if isinstance(spec, dict) else "unspecified"
                                record_dependency(name, str(version), group, absolute_display)
                elif basename == "Cargo.toml":
                    with open(full_path, "rb") as source:
                        data = tomllib.load(source)
                    if _manifest_has_malformed_shapes(basename, data):
                        manifest["read_status"] = "invalid_or_unsupported_shape"
                        warnings.append(f"{basename} contains malformed metadata shapes; some declarations may be omitted from the handoff.")
                    package = data.get("package", {})
                    rust_version = package.get("rust-version") if isinstance(package, dict) else None
                    manifest["requires_rust"] = (
                        _safe_specification(rust_version)
                        if isinstance(rust_version, str)
                        else None
                    )
                    for group_name in ("dependencies", "dev-dependencies", "build-dependencies"):
                        group = data.get(group_name, {})
                        if isinstance(group, dict):
                            for name, spec in group.items():
                                if isinstance(spec, str):
                                    record_dependency(name, spec, group_name, absolute_display)
                                elif isinstance(spec, dict):
                                    record_dependency(name, str(spec.get("version", "unspecified")), group_name, absolute_display)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, tomllib.TOMLDecodeError, TypeError, ValueError):
                manifest["read_status"] = "invalid_or_unreadable"
                warnings.append(f"{basename} could not be safely parsed for the handoff.")
            manifests.append(manifest)
    dependency_records.sort(key=lambda item: (item["name"].casefold(), item["source"], item["group"]))
    return manifests, dependencies, dependency_records, sorted(set(lockfiles)), sorted(set(warnings))


def _runtime_requirements(
    project_roots: list[str], manifests: list[dict[str, Any]]
) -> tuple[list[dict[str, str]], list[str]]:
    runtimes: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    warnings: list[str] = []
    for root in project_roots:
        for name, runtime in ((".nvmrc", "Node.js"), (".node-version", "Node.js"), (".python-version", "Python")):
            path = os.path.join(root, name)
            if not os.path.lexists(path):
                continue
            if not _safe_file_path(path, root) or not os.path.isfile(path):
                warnings.append(f"{name} was not read because its path is unsafe or is not a regular file.")
                continue
            try:
                with open(path, "rb") as source:
                    encoded = source.read(MAX_RUNTIME_VERSION_BYTES + 1)
                if len(encoded) > MAX_RUNTIME_VERSION_BYTES:
                    warnings.append(f"{name} exceeds the runtime declaration size limit; its contents were omitted.")
                    continue
                text = encoded.decode("utf-8")
            except (OSError, UnicodeDecodeError):
                warnings.append(f"{name} is unreadable or is not valid UTF-8; its contents were omitted.")
                continue
            declarations = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
            validator = NODE_VERSION_DECLARATION if runtime == "Node.js" else PYTHON_VERSION_DECLARATION
            if not declarations or any(not validator.fullmatch(version) for version in declarations):
                warnings.append(f"{name} does not contain a recognized runtime version declaration; its contents were omitted.")
                continue
            for version in declarations:
                value = (runtime, version)
                if value not in seen:
                    runtimes.append({"runtime": runtime, "version": version, "source": path})
                    seen.add(value)
    for manifest in manifests:
        path = manifest["path"]
        if manifest["type"] == "pyproject.toml" and manifest.get("requires_python"):
            value = ("Python", str(manifest["requires_python"]))
            if value not in seen:
                runtimes.append({"runtime": value[0], "version": value[1], "source": path})
                seen.add(value)
        if manifest["type"] == "Pipfile" and manifest.get("requires_python"):
            value = ("Python", str(manifest["requires_python"]))
            if value not in seen:
                runtimes.append({"runtime": value[0], "version": value[1], "source": path})
                seen.add(value)
        if manifest["type"] == "Cargo.toml" and manifest.get("requires_rust"):
            value = ("Rust", str(manifest["requires_rust"]))
            if value not in seen:
                runtimes.append({"runtime": value[0], "version": value[1], "source": path})
                seen.add(value)
        if manifest["type"] == "package.json":
            for runtime_name, version in manifest.get("engines", {}).items():
                value = ("Node.js" if runtime_name == "node" else "Python", version)
                if value not in seen:
                    runtimes.append({"runtime": value[0], "version": value[1], "source": path})
                    seen.add(value)
    return runtimes, sorted(set(warnings))


def _commands(project_roots: list[str], config: dict) -> list[dict[str, str]]:
    commands: list[dict[str, str]] = []
    for root in project_roots:
        for relative in _walk_files(root):
            if os.path.basename(relative) != "package.json":
                continue
            full_path = os.path.join(root, relative)
            try:
                data = json.loads(Path(full_path).read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            scripts = data.get("scripts", {}) if isinstance(data, dict) else {}
            if not isinstance(scripts, dict):
                continue
            folder = os.path.dirname(full_path)
            lock_names = set(os.listdir(folder)) if os.path.isdir(folder) else set()
            manager = "pnpm" if "pnpm-lock.yaml" in lock_names else "yarn" if "yarn.lock" in lock_names else "bun" if {"bun.lock", "bun.lockb"} & lock_names else "npm"
            for script_name in ("test", "build", "typecheck", "check", "lint"):
                if script_name in scripts:
                    commands.append({"kind": script_name, "command": f"{manager} run {script_name}", "source": full_path})
    for language, language_config in config.get("languages", {}).items():
        if not isinstance(language_config, dict) or not isinstance(language_config.get("test_command"), list):
            continue
        command_tokens = [str(token) for token in language_config["test_command"]]
        safe_tokens: list[str] = []
        redact_next = False
        for token in command_tokens:
            if redact_next:
                safe_tokens.append("[REDACTED]")
                redact_next = False
                continue
            if token == "{tests_dir}":
                safe_tokens.append("<configured-tests-dir>")
                continue
            if token == "{junit_path}" or token == "{result_path}":
                safe_tokens.append("<external-test-report-path>")
                continue
            if re.search(
                r"(?i)^-{0,2}[A-Za-z0-9_.-]*(?:password|passwd|token|secret|key|auth|bearer|credential)$",
                token,
            ):
                safe_tokens.append(token)
                redact_next = True
            else:
                safe_tokens.append(_redact_secrets(token))
        command = " ".join(safe_tokens)
        command = re.sub(r"\{[^}]+\}", "<external-test-argument>", command)
        if command:
            commands.append({"kind": "pipeline-configured-test", "language": language, "command": command, "source": "pipeline configuration"})
    return commands


def _changed_files(run_dir: str, project_root: str | None, output_dir: str, tests_dir: str) -> dict[str, list[str]]:
    if project_root:
        changes_path = os.path.join(run_dir, "changes.json")
        try:
            data = json.loads(Path(changes_path).read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("changes metadata must be an object")
            return {key: list(data.get(key, [])) for key in ("added", "modified", "deleted")}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            return {"added": [], "modified": [], "deleted": []}
    if not output_dir or not tests_dir:
        return {"added": [], "modified": [], "deleted": []}
    files = []
    for area in (output_dir, tests_dir):
        for relative in _walk_files(os.path.join(run_dir, area)):
            files.append(f"{area}/{relative}")
    return {"added": sorted(files), "modified": [], "deleted": []}


def _validation_findings(run_dir: str) -> dict[str, Any]:
    attempts_dir = os.path.join(run_dir, "validation", "results")
    attempts = sorted(
        (name for name in os.listdir(attempts_dir) if name.startswith("attempt-") and name.endswith(".json")),
        key=lambda name: int(re.search(r"\d+", name).group()) if re.search(r"\d+", name) else 0,
    ) if os.path.isdir(attempts_dir) else []
    if not attempts:
        return {"status": "not_reached", "checked_files": 0, "errors": [], "warnings": [], "limitations": []}
    try:
        result = json.loads(Path(attempts_dir, attempts[-1]).read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("validation result must be an object")
        result["status"] = "passed" if result.get("passed") else "failed"
        return result
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return {"status": "unavailable", "checked_files": 0, "errors": [], "warnings": [], "limitations": []}


def _required_environment(project_roots: list[str], changed: dict[str, list[str]]) -> list[str]:
    names: set[str] = set()
    for root in project_roots:
        for relative in _walk_files(root):
            path = os.path.join(root, relative)
            basename = os.path.basename(relative)
            if basename.startswith(".env") and basename not in {".env.example", ".env.sample", ".env.template"}:
                continue
            if basename in {".env.example", ".env.sample", ".env.template"}:
                try:
                    for line in Path(path).read_text(encoding="utf-8").splitlines():
                        match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
                        if match:
                            names.add(match.group(1))
                except (OSError, UnicodeDecodeError):
                    pass
            if Path(relative).suffix.lower() not in {".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}:
                continue
            try:
                content = Path(path).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for pattern in ENV_NAME_PATTERNS:
                names.update(pattern.findall(content))
    return sorted(names)


def _unresolved_dependencies(project_roots: list[str], dependencies: dict[str, str]) -> list[str]:
    declared = {_normalized_name(name) for name in dependencies}
    imports: set[str] = set()
    for root in project_roots:
        project_paths = set(_walk_files(root))
        local_roots = {Path(path).parts[0] for path in project_paths if Path(path).parts}
        for relative in sorted(project_paths):
            suffix = Path(relative).suffix.lower()
            if suffix not in {".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}:
                continue
            try:
                content = Path(root, relative).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            names: set[str] = set()
            if suffix == ".py":
                try:
                    tree = ast.parse(content)
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        names.update(alias.name.split(".", 1)[0] for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                        names.add(node.module.split(".", 1)[0])
                from sys import stdlib_module_names
                names.difference_update(stdlib_module_names)
            else:
                names.update(match.group(1).split("/", 1)[0] if not match.group(1).startswith("@") else "/".join(match.group(1).split("/")[:2]) for match in JS_PACKAGE_PATTERN.finditer(content))
                names.difference_update({"node:assert", "node:fs", "node:path", "node:crypto", "node:util", "node:os", "node:http", "node:url", "node:events"})
            for name in names:
                if name not in local_roots and _normalized_name(name) not in declared:
                    imports.add(name)
    return sorted(imports)


def _compose_services(project_roots: list[str]) -> tuple[list[str], list[str]]:
    files: list[str] = []
    services: set[str] = set()
    for root in project_roots:
        for relative in _walk_files(root):
            if os.path.basename(relative) not in {"compose.yaml", "compose.yml", "docker-compose.yaml", "docker-compose.yml"}:
                continue
            path = os.path.join(root, relative)
            files.append(path)
            try:
                lines = Path(path).read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            in_services = False
            for line in lines:
                if re.match(r"^services\s*:\s*(?:#.*)?$", line):
                    in_services = True
                    continue
                if in_services and line and not line[0].isspace() and not line.lstrip().startswith("#"):
                    in_services = False
                match = re.match(r"^  ([A-Za-z0-9_.-]+)\s*:\s*(?:#.*)?$", line)
                if in_services and match:
                    services.add(match.group(1))
    return files, sorted(services)


def build_testing_handoff(config: dict, run_dir: str) -> dict[str, Any]:
    pipeline = config.get("pipeline", {})
    project_root = pipeline.get("_existing_project_path")
    output_dir = pipeline.get("output_dir", "product")
    tests_dir = pipeline.get("tests_dir", "tests")
    path_warnings: list[str] = []
    canonical_run_dir = _canonical_run_directory(run_dir, pipeline.get("runs_dir"))
    inventory_run_dir = canonical_run_dir or run_dir
    if canonical_run_dir is None:
        path_warnings.append(
            "Run directory is outside the authorized runs storage; run inventory may be incomplete."
        )
    if project_root:
        canonical_project_root = os.path.realpath(project_root)
        if os.path.isdir(canonical_project_root):
            project_roots = [canonical_project_root]
        else:
            project_roots = []
            path_warnings.append(
                "Configured existing-project root is unavailable; project inventory may be incomplete."
            )
    else:
        try:
            output_dir = normalize_relative_path(output_dir)
            tests_dir = normalize_relative_path(tests_dir)
            if canonical_run_dir is None:
                raise ValueError("run directory is outside the authorized runs storage")
            project_roots = []
            for area in (output_dir, tests_dir):
                root = _canonical_generated_root(
                    os.path.join(run_dir, area), run_dir, canonical_run_dir
                )
                if root is None:
                    path_warnings.append(
                        "Configured product/test output path is unavailable or unsafe; "
                        "project inventory may be incomplete."
                    )
                    continue
                project_roots.append(root)
        except (TypeError, ValueError):
            output_dir = ""
            tests_dir = ""
            project_roots = []
            path_warnings.append("Configured product/test output path is unsafe or invalid; project files were not scanned for this handoff.")
    if any(_contains_symlink_entry(root) for root in project_roots):
        path_warnings.append(
            "Symbolic-link entries were skipped to keep project inventory within authorized roots."
        )
    manifests, dependencies, dependency_records, lockfiles, manifest_warnings = _manifest_data(project_roots)
    changes = _changed_files(inventory_run_dir, project_root, output_dir, tests_dir)
    partial_edit_warning = None
    if project_root:
        try:
            change_data = json.loads(Path(run_dir, "changes.json").read_text(encoding="utf-8"))
            if isinstance(change_data, dict):
                partial_edit_warning = change_data.get("warning")
            else:
                path_warnings.append("Existing-project change metadata has an invalid shape; partial-edit details may be unavailable.")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
    validation = _validation_findings(inventory_run_dir)
    manifest_path = os.path.join(inventory_run_dir, "manifest.json")
    try:
        run_manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if not isinstance(run_manifest, dict):
            path_warnings.append("Run manifest has an invalid shape; some handoff metadata may be unavailable.")
            run_manifest = {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        run_manifest = {}
    plan_path = os.path.join(inventory_run_dir, "design", "plan.json")
    try:
        plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
        if not isinstance(plan, dict):
            path_warnings.append("Plan metadata has an invalid shape; runtime language declarations may be unavailable.")
            plan = {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        plan = {}
    tasks = plan.get("tasks", [])
    if not isinstance(tasks, list):
        path_warnings.append("Plan task metadata has an invalid shape; runtime language declarations may be unavailable.")
        tasks = []
    runtimes = sorted({
        entry.get("language", "unknown")
        for task in tasks
        if isinstance(task, dict) and isinstance(task.get("owned_files", []), list)
        for entry in task.get("owned_files", [])
        if isinstance(entry, dict)
    })
    compose_files, services = _compose_services(project_roots)
    project_files = {
          "run_dir": os.path.realpath(inventory_run_dir),
        "project_root": os.path.realpath(project_root) if project_root else os.path.realpath(os.path.join(run_dir, output_dir)),
        "product_dir": os.path.realpath(os.path.join(run_dir, output_dir)) if not project_root and output_dir else None if not project_root else os.path.realpath(project_root),
        "tests_dir": os.path.realpath(os.path.join(run_dir, tests_dir)) if not project_root and tests_dir else None if not project_root else os.path.realpath(project_root),
          "validation_results": os.path.join(inventory_run_dir, "validation", "results"),
          "changes_json": os.path.join(inventory_run_dir, "changes.json") if project_root else None,
    }
    runtime_requirements, runtime_warnings = _runtime_requirements(project_roots, manifests)
    result: dict[str, Any] = {
        "schema_version": 1,
          "run_id": run_manifest.get("run_id", os.path.basename(inventory_run_dir.rstrip(os.sep))),
          "pipeline_status": pipeline.get(
              "_testing_handoff_pipeline_status",
              run_manifest.get("status", "in_progress"),
          ),
        "project_mode": "existing_project" if project_root else "new_project",
        "paths": project_files,
          "changed_files": changes,
          "runtimes": runtimes,
          "runtime_version_requirements": runtime_requirements,
        "dependency_manifests": manifests,
        "lockfiles": lockfiles,
        "declared_dependencies": dependency_records,
        "unresolved_or_undeclared_dependencies": _unresolved_dependencies(project_roots, dependencies),
        "test_and_build_commands": _commands(project_roots, config),
        "required_environment_variable_names": _required_environment(project_roots, changes),
        "environment_variable_detection": "Names are detected conservatively from dotenv example filenames and Python/JavaScript/TypeScript source patterns; dynamic lookups may be missed.",
        "service_configuration_files": compose_files,
        "declared_services": services,
        "validation": validation,
          "handoff_warnings": sorted(set(path_warnings + manifest_warnings + runtime_warnings)),
          "external_testing": {
              "status": "pending",
              "result_file": "external-test-result.json",
              "contract_version": 1,
              "result_contract": EXTERNAL_TEST_RESULT_CONTRACT,
        },
        "partial_edits_possible": bool(run_manifest.get("partial_edits_possible", False)),
        "partial_edit_warning": partial_edit_warning,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return result


def _markdown(metadata: dict[str, Any]) -> str:
    lines = [
        f"# External Testing Handoff — {metadata['run_id']}",
        "",
        "## Pipeline status",
        "",
        f"- Static pipeline status: `{metadata['pipeline_status']}`",
        "- Runtime/build tests: **pending external execution**. Static validation is not runtime test evidence.",
        f"- Mode: `{metadata['project_mode']}`",
        f"- Run artifacts: `{metadata['paths']['run_dir']}`",
        f"- Project source: `{metadata['paths']['project_root']}`",
        "",
        "## Changed files",
        "",
    ]
    changed = metadata["changed_files"]
    for group in ("added", "modified", "deleted"):
        lines.append(f"### {group.title()}")
        lines.extend(f"- `{path}`" for path in changed.get(group, []))
        if not changed.get(group):
            lines.append("- None recorded")
        lines.append("")
    if metadata["partial_edits_possible"]:
        lines.extend([
            "> Existing-project edits were made in place. The pipeline did not roll them back; partial changes may remain. Review `changes.json` before testing or applying changes.",
            "",
        ])
    lines.extend(["## Dependency and runtime sources", ""])
    for runtime in metadata["runtime_version_requirements"]:
        lines.append(f"- {runtime['runtime']} `{runtime['version']}` (declared in `{runtime['source']}`)")
    if not metadata["runtime_version_requirements"]:
        lines.append("- No runtime version was declared in recognized version files/manifests.")
    lines.extend(["", "Dependency manifests (source of declared package specifications):"])
    lines.extend(f"- `{item['path']}` ({item['type']})" for item in metadata["dependency_manifests"])
    if not metadata["dependency_manifests"]:
        lines.append("- No recognized dependency manifest found.")
    lines.append("Lockfiles (source of resolved package versions):")
    lines.extend(f"- `{path}`" for path in metadata["lockfiles"])
    if not metadata["lockfiles"]:
        lines.append("- No recognized lockfile found; do not infer pinned versions.")
    lines.extend(["", "Declared dependencies:"])
    lines.extend(
        f"- `{item['name']}`: `{item['version_specification']}` ({item['group']}; source: `{item['source']}`)"
        for item in metadata["declared_dependencies"]
    )
    if not metadata["declared_dependencies"]:
        lines.append("- None extracted from recognized manifests.")
    lines.extend(["", "Imports requiring dependency review:"])
    lines.extend(f"- `{name}` (not matched to a declared dependency; review aliases/local paths)" for name in metadata["unresolved_or_undeclared_dependencies"])
    if not metadata["unresolved_or_undeclared_dependencies"]:
        lines.append("- None detected by the conservative import scan.")
    lines.extend(["", "## Commands discovered", ""])
    lines.extend(f"- `{item['command']}` ({item['kind']}; source: `{item['source']}`)" for item in metadata["test_and_build_commands"])
    if not metadata["test_and_build_commands"]:
        lines.append("- No test/build command was declared in recognized project scripts/configuration or pipeline config.")
    lines.extend(["", "## Environment and services", ""])
    lines.append("Environment variable names only: " + (", ".join(f"`{name}`" for name in metadata["required_environment_variable_names"]) or "none detected"))
    lines.append("Service configuration files: " + (", ".join(f"`{path}`" for path in metadata["service_configuration_files"]) or "none detected"))
    lines.append("Declared compose services: " + (", ".join(f"`{name}`" for name in metadata["declared_services"]) or "none detected"))
    lines.extend(["", "## Static validation", "", f"- Status: `{metadata['validation'].get('status', 'unknown')}`", f"- Files checked: {metadata['validation'].get('checked_files', 0)}"])
    for error in metadata["validation"].get("errors", []):
        lines.append(f"- Error `{error.get('file_path', '?')}`: {error.get('error_output', '')}")
    for warning in metadata["validation"].get("warnings", []):
        lines.append(f"- Warning `{warning.get('file_path', '?')}`: {warning.get('error_output', '')}")
    for limitation in metadata["validation"].get("limitations", []):
        lines.append(f"- Limitation: {limitation}")
    lines.extend([
        "",
        "## External test procedure",
        "",
        "1. Create an isolated copy/worktree of the project files. In existing-project mode, test the copied tree; do not test by editing or running commands in the original project directory.",
        "   For new-project runs, stage `product/` and generated `tests/` files together as needed; they are separate artifact directories.",
        "2. Set up runtimes and dependencies from the listed manifests and lockfiles in that isolated environment. Keep credentials out of logs and result metadata.",
        "3. Run the discovered test/build commands, recording exact commands and exit codes. Do not report runtime success from pipeline completion or static validation alone.",
        "4. Save a JSON result matching the contract below as `external-test-result.json` next to this handoff, then update `external_testing.status` to `reported` in `testing-handoff.json`.",
        "",
        "### External result contract (version 1)",
        "",
        "`outcome` must be one of `passed`, `code_or_test_failure`, `environment_setup_failure`, or `not_run`. Setup failures (runtime/dependency/service provisioning) are separate from failures after commands could run.",
        "",
        "```json",
        json.dumps({
            "schema_version": 1,
            "run_id": metadata["run_id"],
            "outcome": "not_run",
            "tested_copy_path": "",
            "environment_setup": {"status": "not_attempted", "summary": ""},
            "commands": [{"command": "", "status": "not_run", "exit_code": None, "summary": ""}],
            "completed_at": None,
        }, indent=2),
        "```",
        "",
        "Do not include environment variable values, access tokens, or secret-bearing logs in the result.",
    ])
    return "\n".join(lines) + "\n"


def write_testing_handoff(config: dict, run_dir: str) -> tuple[str, str]:
    metadata = build_testing_handoff(config, run_dir)
    json_path = os.path.join(run_dir, "testing-handoff.json")
    markdown_path = os.path.join(run_dir, "testing-handoff.md")
    with open(json_path, "w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2)
    with open(markdown_path, "w", encoding="utf-8") as output:
        output.write(_markdown(metadata))
    return markdown_path, json_path
