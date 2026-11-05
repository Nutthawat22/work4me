"""Deterministic, non-executing validation of generated project files."""

from __future__ import annotations

import ast
import json
import os
import re
import stat
import tomllib
from collections.abc import Mapping, Set
from pathlib import Path

from pipeline.project_context import IGNORED_DIRECTORIES, snapshot_project_files, summarize_project_changes
from pipeline.state import AgentResult, OwnedFile, ValidationIssue, ValidationResult, WorkItem
from pipeline.task_validation import normalize_relative_path
from pipeline.typescript_checker import check_typescript

JS_SOURCE_SUFFIXES = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")
REQUIREMENTS_MANIFEST = re.compile(r"^requirements(?:[-_.][A-Za-z0-9_.-]+)?\.txt$", re.IGNORECASE)


def _planned_path(
    owned_file: OwnedFile,
    config: dict,
    *,
    existing_project: bool,
) -> str:
    try:
        configured_dir = normalize_relative_path(
            config["pipeline"]["tests_dir" if owned_file.area == "tests" else "output_dir"]
        )
        path = normalize_relative_path(owned_file.path)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid owned path {owned_file.path!r}: {error}") from error
    if not existing_project and path.startswith(configured_dir + "/"):
        path = path[len(configured_dir) + 1 :]
    return path if existing_project else f"{configured_dir}/{path}"


def _walk_files(root: str) -> list[str]:
    found: list[str] = []
    if not os.path.isdir(root):
        return found
    for current, directories, names in os.walk(root, followlinks=False):
        directories[:] = sorted(
            name
            for name in directories
            if name not in IGNORED_DIRECTORIES
            and not os.path.islink(os.path.join(current, name))
        )
        for name in sorted(names):
            path = os.path.join(current, name)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            found.append(os.path.relpath(path, root).replace(os.sep, "/"))
    return found


def _has_symlink_component(root: str, relative_path: str) -> bool:
    current = root
    for component in relative_path.split("/"):
        current = os.path.join(current, component)
        if os.path.islink(current):
            return True
    return False


def _safe_local_target(path: str, root: str, *, allow_directory: bool = False) -> bool:
    real_root = os.path.realpath(root)
    relative = os.path.relpath(path, real_root)
    if relative == os.curdir or relative == os.pardir or relative.startswith(os.pardir + os.sep):
        return False
    if _has_symlink_component(real_root, relative):
        return False
    return os.path.isfile(path) or (allow_directory and os.path.isdir(path))


def _pyproject_shape_error(data: object) -> str | None:
    if not isinstance(data, dict):
        return "top-level value must be a table"
    project = data.get("project", {})
    if not isinstance(project, dict):
        return "project must be a table"
    dependencies = project.get("dependencies", [])
    if not isinstance(dependencies, list) or any(not isinstance(value, str) for value in dependencies):
        return "project.dependencies must be an array of strings"
    optional = project.get("optional-dependencies", {})
    if not isinstance(optional, dict) or any(
        not isinstance(values, list) or any(not isinstance(value, str) for value in values)
        for values in optional.values()
    ):
        return "project.optional-dependencies must map names to arrays of strings"
    tools = data.get("tool", {})
    if not isinstance(tools, dict):
        return "tool must be a table"
    poetry = tools.get("poetry", {})
    if not isinstance(poetry, dict):
        return "tool.poetry must be a table"
    if not isinstance(poetry.get("dependencies", {}), dict):
        return "tool.poetry.dependencies must be a table"
    groups = poetry.get("group", {})
    if not isinstance(groups, dict) or any(
        not isinstance(group, dict) or not isinstance(group.get("dependencies", {}), dict)
        for group in groups.values()
    ):
        return "tool.poetry.group entries must contain dependency tables"
    return None


def _other_toml_shape_error(name: str, data: object) -> str | None:
    if not isinstance(data, dict):
        return "top-level value must be a table"
    if name == "Pipfile":
        if any(not isinstance(data.get(group, {}), dict) for group in ("packages", "dev-packages", "requires")):
            return "packages, dev-packages, and requires must be tables"
    elif name == "Cargo.toml":
        if not isinstance(data.get("package", {}), dict):
            return "package must be a table"
        if any(
            not isinstance(data.get(group, {}), dict)
            for group in ("dependencies", "dev-dependencies", "build-dependencies")
        ):
            return "dependency groups must be tables"
    return None


def _parse_jsonc(text: str) -> object:
    without_comments: list[str] = []
    index = 0
    quoted = False
    escaped = False
    while index < len(text):
        current = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if quoted:
            without_comments.append(current)
            if escaped:
                escaped = False
            elif current == "\\":
                escaped = True
            elif current == '"':
                quoted = False
        elif current == '"':
            quoted = True
            without_comments.append(current)
        elif current == "/" and following == "/":
            while index < len(text) and text[index] != "\n":
                index += 1
            if index < len(text):
                without_comments.append("\n")
        elif current == "/" and following == "*":
            index += 2
            while index < len(text) and text[index:index + 2] != "*/":
                if text[index] == "\n":
                    without_comments.append("\n")
                index += 1
            index += 1
        else:
            without_comments.append(current)
        index += 1

    source = "".join(without_comments)
    output: list[str] = []
    quoted = False
    escaped = False
    for index, current in enumerate(source):
        if quoted:
            output.append(current)
            if escaped:
                escaped = False
            elif current == "\\":
                escaped = True
            elif current == '"':
                quoted = False
        elif current == '"':
            quoted = True
            output.append(current)
        elif current == ",":
            following = index + 1
            while following < len(source) and source[following].isspace():
                following += 1
            if following >= len(source) or source[following] not in "]}":
                output.append(current)
        else:
            output.append(current)
    return json.loads("".join(output))


def _python_local_target_exists(
    source_path: str,
    node: ast.ImportFrom,
    root: str,
    virtual_paths: set[str] | None = None,
) -> bool:
    if node.level <= 0:
        return True
    source = Path(source_path)
    package_parts = list(source.parent.parts)
    ascent = node.level - 1
    if ascent > len(package_parts):
        return False
    package_parts = package_parts[: len(package_parts) - ascent]
    module_parts = node.module.split(".") if node.module else []
    parent = os.path.join(root, *package_parts, *module_parts)
    available = virtual_paths or set()

    def exists(candidate: str) -> bool:
        relative = os.path.relpath(candidate, root).replace(os.sep, "/").casefold()
        return relative in available or _safe_local_target(candidate, root)

    if exists(parent + ".py") or exists(
        os.path.join(parent, "__init__.py")
    ):
        return True
    if node.module and (
        _safe_local_target(parent, root, allow_directory=True)
        or any(path.startswith(os.path.relpath(parent, root).replace(os.sep, "/").casefold() + "/") for path in available)
    ):
        # A namespace package may have no __init__.py; its contents cannot be
        # proven statically, so an existing directory is treated as a possible target.
        return True
    if node.module is None:
        return any(
            exists(os.path.join(parent, alias.name, "__init__.py"))
            or exists(os.path.join(parent, alias.name + ".py"))
            or _safe_local_target(os.path.join(parent, alias.name), root, allow_directory=True)
            or any(
                path.startswith(
                    os.path.relpath(os.path.join(parent, alias.name), root)
                    .replace(os.sep, "/")
                    .casefold()
                    + "/"
                )
                for path in available
            )
            for alias in node.names
        )
    return False


def _unused_private_declarations(tree: ast.Module, source_path: str) -> list[str]:
    if Path(source_path).name == "__init__.py":
        return []
    exported: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name) and statement.target.id == "__all__":
            return []
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
            if not isinstance(statement.value, (ast.List, ast.Tuple, ast.Set)):
                return []
            if not all(
                isinstance(element, ast.Constant) and isinstance(element.value, str)
                for element in statement.value.elts
            ):
                return []
            exported.update(element.value for element in statement.value.elts)

    loaded_names = {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    string_names = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    warnings: list[str] = []
    for statement in tree.body:
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        name = statement.name.casefold()
        callback_like = (
            name.startswith(("_on_", "_handle_", "_callback_", "_route_", "_hook_", "_fixture_", "_event_", "_listener_"))
            or name.endswith(("_callback", "_handler", "_route", "_hook", "_fixture", "_listener", "_entrypoint"))
        )
        if (
            not statement.name.startswith("_")
            or statement.name in exported
            or statement.name in string_names
            or statement.decorator_list
            or callback_like
        ):
            continue
        if statement.name not in loaded_names:
            warnings.append(statement.name)
    return warnings


class StaticValidator:
    """Check expected outputs, manifests, Python syntax, and conservative imports."""

    def run(
        self,
        config: dict,
        work_items: list[WorkItem],
        run_dir: str,
        agent_results: list[AgentResult] | None = None,
        *,
        staged_files: Mapping[str, str] | None = None,
        staged_deletions: Set[str] | None = None,
        dependency_files: Mapping[str, str] | None = None,
        contracts_complete: bool = True,
    ) -> ValidationResult:
        project_root = config["pipeline"].get("_existing_project_path")
        existing_project = bool(project_root)
        root = os.path.realpath(project_root if existing_project else run_dir)
        staged_by_key = {
            path.replace("\\", "/").casefold(): content
            for path, content in (staged_files or {}).items()
        }
        deleted_keys = {
            path.replace("\\", "/").casefold()
            for path in (staged_deletions or set())
        }
        virtual_paths = set(staged_by_key)
        dependency_by_key = {
            path.replace("\\", "/").casefold(): content
            for path, content in (dependency_files or {}).items()
        }
        virtual_paths.update(dependency_by_key)

        def source_content(relative: str) -> str:
            staged = staged_by_key.get(relative.replace("\\", "/").casefold())
            if staged is not None:
                if Path(relative).suffix.lower() in JS_SOURCE_SUFFIXES and len(staged) > 1024 * 1024:
                    raise ValueError("TypeScript source exceeds the 1 MiB checker file budget")
                return staged
            dependency = dependency_by_key.get(relative.replace("\\", "/").casefold())
            if dependency is not None:
                if Path(relative).suffix.lower() in JS_SOURCE_SUFFIXES and len(dependency) > 1024 * 1024:
                    raise ValueError("TypeScript dependency source exceeds the 1 MiB checker file budget")
                return dependency
            if Path(relative).suffix.lower() in JS_SOURCE_SUFFIXES:
                full_path = os.path.join(root, relative)
                if os.path.commonpath((root, os.path.realpath(full_path))) != root:
                    raise OSError("source path escapes its authorized root")
                descriptor = os.open(full_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(descriptor, "rb") as source:
                    if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                        raise OSError("source is not a regular file")
                    encoded = source.read(1024 * 1024 + 1)
                if len(encoded) > 1024 * 1024:
                    raise ValueError("TypeScript source exceeds the 1 MiB checker file budget")
                return encoded.decode("utf-8")
            with open(os.path.join(root, relative), "r", encoding="utf-8") as source:
                return source.read()
        expected: dict[str, str] = {}
        expected_files: dict[str, str] = {}
        errors: list[ValidationIssue] = []
        warnings: list[ValidationIssue] = []
        checked_files = 0
        new_node_project = not existing_project and any(
            owned.area == "product"
            and (
                owned.language in {"javascript", "typescript"}
                or owned.path.casefold().endswith(JS_SOURCE_SUFFIXES)
            )
            for item in work_items for owned in item.owned_files
        )
        latest_results = {result.work_item_id: result for result in agent_results or []}
        incomplete_owner_ids = {
            work_item_id
            for work_item_id, result in latest_results.items()
            if not result.success
        }

        def add_error(path: str, message: str, work_item_id: str = "") -> None:
            errors.append(ValidationIssue(path, work_item_id, message))

        for item in work_items:
            for owned_file in item.owned_files:
                try:
                    relative = _planned_path(owned_file, config, existing_project=existing_project)
                except ValueError as error:
                    add_error(owned_file.path, str(error), item.id)
                    continue
                key = relative.casefold()
                if key in expected:
                    add_error(relative, "Path is owned more than once in the plan", item.id)
                    continue
                expected[key] = item.id
                expected_files[key] = relative
                full_path = os.path.join(root, relative)
                if _has_symlink_component(root, relative):
                    add_error(relative, "Expected owned path is a symbolic link", item.id)

        item_by_path = {path.casefold(): item_id for path, item_id in expected.items()}
        for result in agent_results or []:
            for reported_path in (*result.files_written, *result.files_deleted):
                normalized = reported_path.replace("\\", "/")
                if normalized.casefold() not in item_by_path:
                    add_error(normalized, "Specialist reported changing an unowned file", result.work_item_id)

        intentional_deletions = (
            {
                path.replace("\\", "/").casefold()
                for result in latest_results.values()
                for path in result.files_deleted
            }
            if existing_project
            else set()
        )
        for key, relative in expected_files.items():
            full_path = os.path.join(root, relative)
            if _has_symlink_component(root, relative):
                continue
            if (
                not os.path.isfile(full_path)
                and key not in staged_by_key
                and key not in intentional_deletions
                and key not in deleted_keys
            ):
                if expected[key] in incomplete_owner_ids:
                    continue
                add_error(relative, "Expected owned file is missing", expected[key])

        if existing_project:
            baseline = config["pipeline"].get("_existing_project_baseline")
            if isinstance(baseline, dict):
                protected_root = config["pipeline"].get("_protected_run_root")
                excluded = (protected_root,) if isinstance(protected_root, str) and protected_root else ()
                current = snapshot_project_files(root, excluded)
                changes = summarize_project_changes(baseline, current)
                for relative in (*changes["added"], *changes["modified"], *changes["deleted"]):
                    if relative.casefold() not in item_by_path:
                        add_error(relative, "Existing-project change is outside planned file ownership")

        if existing_project or staged_files is not None:
            project_files = [
              path for path in expected_files.values()
              if (
                  os.path.isfile(os.path.join(root, path))
                  or path.casefold() in staged_by_key
              )
              and path.casefold() not in deleted_keys
              and not _has_symlink_component(root, path)
            ]
        else:
            scan_roots = []
            for key in ("output_dir", "tests_dir"):
                try:
                    relative_area = normalize_relative_path(config["pipeline"][key])
                except (KeyError, TypeError, ValueError) as error:
                    add_error(key, f"Invalid configured output directory: {error}")
                    continue
                scan_roots.append(os.path.join(root, relative_area))
            project_files = []
            for scan_root in scan_roots:
                for relative in _walk_files(scan_root):
                    absolute = os.path.join(scan_root, relative)
                    project_relative = os.path.relpath(absolute, root).replace(os.sep, "/")
                    project_files.append(project_relative)
                    if project_relative.casefold() not in item_by_path:
                        add_error(project_relative, "Generated file is not owned by a planned task")
            for relative in staged_by_key:
                if relative not in {path.casefold() for path in project_files}:
                    project_files.append(relative)
                if relative not in item_by_path:
                    add_error(relative, "Staged file is not owned by a planned task")
            project_files = [path for path in project_files if path.casefold() not in deleted_keys]

        manifest_limitations: list[str] = []
        for relative in sorted(set(project_files)):
            path = os.path.join(root, relative)
            owner_id = item_by_path.get(relative.casefold(), "")
            suffix = Path(relative).suffix.lower()
            try:
                content = source_content(relative)
            except (OSError, UnicodeDecodeError, ValueError) as error:
                if relative.casefold() in item_by_path:
                    if isinstance(error, ValueError) and Path(relative).suffix.lower() in JS_SOURCE_SUFFIXES:
                        manifest_limitations.append(str(error))
                    else:
                        add_error(relative, f"Cannot read expected source file: {type(error).__name__}", owner_id)
                continue
            checked_files += 1

            if suffix == ".py":
                try:
                    tree = ast.parse(content, filename=relative)
                except SyntaxError as error:
                    add_error(relative, f"Python syntax error at line {error.lineno}: {error.msg}", owner_id)
                    continue
                try:
                    compile(content, relative, "exec")
                except SyntaxError as error:
                    add_error(relative, f"Python compile error at line {error.lineno}: {error.msg}", owner_id)
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.level and not _python_local_target_exists(
                        relative, node, root, virtual_paths
                    ):
                        add_error(relative, f"Relative Python import target not found: {'.' * node.level}{node.module or ''}", owner_id)
                for name in _unused_private_declarations(tree, relative):
                    warnings.append(
                        ValidationIssue(
                            relative,
                            owner_id,
                            f"Private top-level declaration appears unused in this module: {name}",
                            severity="warning",
                        )
                    )
        for relative in sorted(set(project_files)):
            path = os.path.join(root, relative)
            if os.path.basename(relative) == "package.json":
                try:
                    with open(path, "r", encoding="utf-8") as source:
                        manifest = json.load(source)
                    if not isinstance(manifest, dict):
                        raise ValueError("top-level value must be an object")
                    for group in ("dependencies", "devDependencies", "optionalDependencies", "peerDependencies"):
                        values = manifest.get(group, {})
                        if not isinstance(values, dict) or any(
                            not isinstance(name, str) or not isinstance(version, str)
                            for name, version in values.items()
                        ):
                            raise ValueError(f"{group} must map package names to version strings")
                    for group in ("scripts", "engines"):
                        values = manifest.get(group, {})
                        if not isinstance(values, dict) or any(
                            not isinstance(name, str) or not isinstance(value, str)
                            for name, value in values.items()
                        ):
                            raise ValueError(f"{group} must map names to strings")
                    package_manager = manifest.get("packageManager")
                    if package_manager is not None and not isinstance(package_manager, str):
                        raise ValueError("packageManager must be a string when declared")
                    output_root = normalize_relative_path(config["pipeline"].get("output_dir", "product"))
                    if new_node_project and relative.casefold() == f"{output_root}/package.json".casefold():
                        scripts = manifest.get("scripts", {})
                        if not isinstance(scripts, dict) or not isinstance(scripts.get("test"), str) or not scripts["test"].strip():
                            add_error(relative, "New Node project package.json must declare a non-empty test script", item_by_path.get(relative.casefold(), ""))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                    add_error(relative, f"Invalid package.json manifest ({type(error).__name__})", item_by_path.get(relative.casefold(), ""))
            elif os.path.basename(relative) in {"tsconfig.json", "jsconfig.json"}:
                try:
                    compiler_config = _parse_jsonc(source_content(relative))
                    if not isinstance(compiler_config, dict):
                        raise ValueError("top-level value must be an object")
                    options = compiler_config.get("compilerOptions", {})
                    if not isinstance(options, dict):
                        raise ValueError("compilerOptions must be an object")
                    unsafe = [key for key in ("extends", "references") if key in compiler_config]
                    unsafe.extend(
                        f"compilerOptions.{key}"
                        for key in ("plugins", "incremental", "composite", "tsBuildInfoFile")
                        if key in options and options[key]
                    )
                    if unsafe:
                        add_error(relative, "Compiler config uses unsupported external/build/plugin settings: " + ", ".join(unsafe), item_by_path.get(relative.casefold(), ""))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                    add_error(relative, f"Invalid compiler configuration ({type(error).__name__})", item_by_path.get(relative.casefold(), ""))
            elif os.path.basename(relative) in {"pyproject.toml", "Pipfile", "Cargo.toml"}:
                try:
                    with open(path, "rb") as source:
                        manifest = tomllib.load(source)
                    basename = os.path.basename(relative)
                    shape_error = (
                        _pyproject_shape_error(manifest)
                        if basename == "pyproject.toml"
                        else _other_toml_shape_error(basename, manifest)
                    )
                    if shape_error:
                        add_error(relative, f"Invalid {basename} metadata shape: {shape_error}", item_by_path.get(relative.casefold(), ""))
                except (OSError, tomllib.TOMLDecodeError) as error:
                    add_error(relative, f"Invalid TOML manifest ({type(error).__name__})", item_by_path.get(relative.casefold(), ""))
            elif REQUIREMENTS_MANIFEST.fullmatch(os.path.basename(relative)):
                try:
                    with open(path, "r", encoding="utf-8") as source:
                        lines = source.readlines()
                except (OSError, UnicodeDecodeError) as error:
                    add_error(relative, f"Cannot read requirements manifest: {error}", item_by_path.get(relative.casefold(), ""))
                    continue
                requirement_pattern = re.compile(
                    r"^[A-Za-z0-9][A-Za-z0-9_.-]*(?:\[[A-Za-z0-9_,.-]+\])?"
                    r"(?:\s*(?:===|==|~=|!=|<=|>=|<|>)\s*[A-Za-z0-9][A-Za-z0-9._+!*]*)?"
                    r"(?:\s*,\s*(?:===|==|~=|!=|<=|>=|<|>)\s*[A-Za-z0-9][A-Za-z0-9._+!*]*)*"
                    r"(?:\s*;\s*[^#]+)?(?:\s*#.*)?$"
                )
                uncertain_specification = re.compile(
                    r"^[A-Za-z0-9][A-Za-z0-9_.-]*(?:\[[A-Za-z0-9_,.-]+\])?\s*(?:===|==|~=|!=|<=|>=|<|>)"
                )
                for line_number, line in enumerate(lines, start=1):
                    entry = line.strip()
                    if not entry or entry.startswith("#"):
                        continue
                    if entry.startswith("-") or " @ " in entry or "://" in entry:
                        manifest_limitations.append(
                            f"requirements.txt entry at line {line_number} uses an include, option, or direct URL not parsed by static validation."
                        )
                    elif not requirement_pattern.fullmatch(entry):
                        if uncertain_specification.match(entry):
                            manifest_limitations.append(
                                f"requirements.txt entry at line {line_number} uses a version form not parsed by static validation."
                            )
                        else:
                            add_error(
                                relative,
                                f"Invalid requirements.txt entry syntax at line {line_number}",
                                item_by_path.get(relative.casefold(), ""),
                            )

        limitations: list[str] = list(manifest_limitations)
        if any(Path(path).suffix.lower() == ".py" for path in project_files):
            limitations.append(
                "Python absolute imports are not resolved against third-party packages; relative `from . import name` may expose package attributes and is not fully verified."
            )

        typescript_sources: dict[str, str] = {}
        source_paths = set(project_files)
        source_paths.update((dependency_files or {}).keys())
        source_paths.update((staged_files or {}).keys())
        if existing_project:
            source_paths.update(
                path for path in _walk_files(root)
                if Path(path).suffix.lower() in JS_SOURCE_SUFFIXES
                and not any(
                    part in Path(path).name.casefold()
                    for part in ("secret", "credential", "private_key", ".env")
                )
            )
        else:
            for area_key in ("output_dir", "tests_dir"):
                try:
                    area = normalize_relative_path(config["pipeline"][area_key])
                except (KeyError, TypeError, ValueError):
                    continue
                area_root = os.path.join(root, area)
                source_paths.update(
                    f"{area}/{path}"
                    for path in _walk_files(area_root)
                    if Path(path).suffix.lower() in JS_SOURCE_SUFFIXES
                )

        snapshot_chars = 0
        snapshot_over_budget = False
        for relative in sorted(source_paths, key=str.casefold):
            if Path(relative).suffix.lower() not in JS_SOURCE_SUFFIXES:
                continue
            normalized = relative.replace("\\", "/")
            if normalized.casefold() in deleted_keys or _has_symlink_component(root, normalized):
                continue
            try:
                staged = staged_by_key.get(normalized.casefold())
                dependency = dependency_by_key.get(normalized.casefold())
                if staged is not None:
                    content = staged
                elif dependency is not None:
                    content = dependency
                else:
                    full_path = os.path.join(root, normalized)
                    if os.path.commonpath((root, os.path.realpath(full_path))) != root:
                        continue
                    descriptor = os.open(full_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                    with os.fdopen(descriptor, "rb") as source:
                        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                            continue
                        encoded = source.read(1024 * 1024 + 1)
                    if len(encoded) > 1024 * 1024:
                        snapshot_over_budget = True
                        break
                    content = encoded.decode("utf-8")
            except (OSError, UnicodeDecodeError, ValueError):
                continue
            snapshot_chars += len(content.encode("utf-8"))
            if snapshot_chars > 4 * 1024 * 1024:
                snapshot_over_budget = True
                break
            typescript_sources[normalized] = content

        assigned_contracts = {
            (contract.get("name", ""), contract.get("declaration", "")): contract
            for item in work_items
            for contract in item.design_contracts
        }
        contract_inventory = {
            (contract.get("name", ""), contract.get("declaration", "")): contract
            for item in work_items
            for contract in item.design_contract_inventory
        }
        if not contract_inventory:
            contract_inventory = assigned_contracts
        contracts_by_key = {**contract_inventory, **assigned_contracts}
        if snapshot_over_budget:
            typescript_result = {
                "status": "unsupported",
                "diagnostics": [],
                "imports": [],
                "contract_findings": [],
                "limitations": [
                    "TypeScript checking skipped: a file exceeds the 1 MiB limit or the authorized source snapshot exceeds the 4 MiB limit."
                ],
            }
        else:
            typescript_result = check_typescript(
                typescript_sources,
                list(contracts_by_key.values()),
                require_all_contracts=contracts_complete,
                required_contract_names={name for name, _ in assigned_contracts},
            )

        typescript_status = str(typescript_result.get("status", "unsupported"))
        import_inventory = [
            entry for entry in typescript_result.get("imports", [])
            if isinstance(entry, dict)
            and isinstance(entry.get("source"), str)
            and isinstance(entry.get("specifier"), str)
        ]
        for diagnostic in typescript_result.get("diagnostics", []):
            if not isinstance(diagnostic, dict):
                continue
            path = str(diagnostic.get("path", ""))
            owner_id = item_by_path.get(path.casefold(), "")
            message = str(diagnostic.get("message", "TypeScript diagnostic"))
            line = diagnostic.get("line")
            prefix = f"line {line}: " if isinstance(line, int) else ""
            add_error(
                path or "<typescript>",
                f"TypeScript {diagnostic.get('kind', 'diagnostic')} error: {prefix}{message}",
                owner_id,
            )
        for finding in typescript_result.get("contract_findings", []):
            if not isinstance(finding, dict):
                continue
            path = str(finding.get("path", ""))
            owner_id = item_by_path.get(path.casefold(), "")
            if not owner_id:
                owner_id = next((
                    item.id for item in work_items
                    if any(contract.get("name") == finding.get("name") for contract in item.design_contracts)
                ), "")
                path = path or next((
                    candidate for candidate in expected_files.values()
                    if item_by_path.get(candidate.casefold()) == owner_id
                    and Path(candidate).suffix.lower() in JS_SOURCE_SUFFIXES
                ), "")
            add_error(
                path or "<typescript-contract>",
                f"Design contract conformance error ({finding.get('source_ref', 'planning input')}): {finding.get('message', 'contract mismatch')}",
                owner_id,
            )
        limitations.extend(
            item for item in typescript_result.get("limitations", []) if isinstance(item, str)
        )
        if typescript_sources:
            limitations.append("TypeScript/JavaScript unused-symbol analysis is unsupported.")
        if typescript_status == "unsupported" and typescript_sources:
            limitations.append(
                "TypeScript syntax/type validation status is unsupported; install the pinned trusted pipeline TypeScript toolchain to enable it."
            )
        limitations.append(
            "Static validation does not import or execute project code, install application dependencies, or run generated-project tests/builds."
        )
        return ValidationResult(
            passed=not errors,
            checked_files=checked_files,
            errors=errors,
            warnings=warnings,
            limitations=limitations,
            typescript_status=typescript_status,
            import_inventory=import_inventory,
        )
