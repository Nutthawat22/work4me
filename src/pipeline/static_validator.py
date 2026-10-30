"""Deterministic, non-executing validation of generated project files."""

from __future__ import annotations

import ast
import json
import os
import re
import tomllib
from pathlib import Path

from pipeline.project_context import IGNORED_DIRECTORIES, snapshot_project_files, summarize_project_changes
from pipeline.state import AgentResult, OwnedFile, ValidationIssue, ValidationResult, WorkItem
from pipeline.task_validation import normalize_relative_path

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


def _python_local_target_exists(source_path: str, node: ast.ImportFrom, root: str) -> bool:
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
    if _safe_local_target(parent + ".py", root) or _safe_local_target(
        os.path.join(parent, "__init__.py"), root
    ):
        return True
    if node.module and _safe_local_target(parent, root, allow_directory=True):
        # A namespace package may have no __init__.py; its contents cannot be
        # proven statically, so an existing directory is treated as a possible target.
        return True
    if node.module is None:
        return any(
            _safe_local_target(os.path.join(parent, alias.name, "__init__.py"), root)
            or _safe_local_target(os.path.join(parent, alias.name + ".py"), root)
            or _safe_local_target(os.path.join(parent, alias.name), root, allow_directory=True)
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
    ) -> ValidationResult:
        project_root = config["pipeline"].get("_existing_project_path")
        existing_project = bool(project_root)
        root = os.path.realpath(project_root if existing_project else run_dir)
        expected: dict[str, str] = {}
        expected_files: dict[str, str] = {}
        errors: list[ValidationIssue] = []
        warnings: list[ValidationIssue] = []
        checked_files = 0

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
        latest_results = {result.work_item_id: result for result in agent_results or []}
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
            if not os.path.isfile(full_path) and key not in intentional_deletions:
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

        if existing_project:
            project_files = [
              path for path in expected_files.values()
              if os.path.isfile(os.path.join(root, path))
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

        manifest_limitations: list[str] = []
        for relative in sorted(set(project_files)):
            path = os.path.join(root, relative)
            owner_id = item_by_path.get(relative.casefold(), "")
            suffix = Path(relative).suffix.lower()
            try:
                with open(path, "r", encoding="utf-8") as source:
                    content = source.read()
            except (OSError, UnicodeDecodeError) as error:
                if relative.casefold() in item_by_path:
                    add_error(relative, f"Cannot read expected source file: {error}", owner_id)
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
                    if isinstance(node, ast.ImportFrom) and node.level and not _python_local_target_exists(relative, node, root):
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
                except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                    add_error(relative, f"Invalid package.json manifest ({type(error).__name__})", item_by_path.get(relative.casefold(), ""))
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
        if any(Path(path).suffix.lower() in {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"} for path in project_files):
            limitations.extend(
                [
                    "TypeScript/JavaScript/JSX syntax, type correctness, and local import targets are not parsed or resolved; no fixed parser is available in the pipeline toolchain.",
                    "TypeScript/JavaScript unused-symbol analysis is unsupported.",
                ]
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
        )
