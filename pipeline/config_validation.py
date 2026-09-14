"""
pipeline/config_validation.py

Validates a loaded config dict against the shape required by the
pipeline (see config.example.json). Keeps load_config() in runner.py
free of validation logic — this module is the single place that knows
what keys/types every part of the pipeline actually reads.
"""

VALID_RESULT_FORMATS = {"junit", "jest_json", "vitest_json"}
VALID_PROVIDERS = {"acp"}

REQUIRED_TOP_LEVEL: dict[str, type] = {
    "litellm_url": str,
    "litellm_key": str,
    "models": dict,
    "pipeline": dict,
    "languages": dict,
}

REQUIRED_MODEL_KEYS: tuple = ("design", "specialist")

REQUIRED_MODEL_ENTRY_KEYS: dict[str, type] = {
    "model": str,
    "provider": str,
}

REQUIRED_PIPELINE_KEYS: dict[str, type] = {
    "max_retries": int,
    "runs_dir": str,
    "output_dir": str,
    "tests_dir": str,
}

REQUIRED_LANGUAGE_KEYS: dict[str, tuple] = {
    "test_command": (list,),
    "test_file_patterns": (list,),
    "result_format": (str,),
    "file_extension": (str, list),
    "test_framework_name": (str,),
}


def _article(t: type) -> str:
    return "an" if t.__name__[0].lower() in "aeiou" else "a"


def _type_name(t: type) -> str:
    return t.__name__


def validate_config(config: dict) -> list[str]:
    """
    Validate config against the required shape read by pipeline/*.py and
    specialists/*.py.

    Returns a list of human-readable error strings; empty list means the
    config is valid. Does not raise — callers decide how to surface
    errors (see main() in pipeline/runner.py).
    """
    errors: list[str] = []

    if not isinstance(config, dict):
        return [f"Config root must be a JSON object, got {type(config).__name__}"]

    for key, expected_type in REQUIRED_TOP_LEVEL.items():
        if key not in config:
            errors.append(f"Missing required key: {key}")
        elif not isinstance(config[key], expected_type):
            errors.append(
                f"{key} must be {_article(expected_type)} {_type_name(expected_type)}, "
                f"got {type(config[key]).__name__}"
            )

    # models.*
    models = config.get("models")
    if isinstance(models, dict):
        for key in REQUIRED_MODEL_KEYS:
            if key not in models:
                errors.append(f"Missing required key: models.{key}")
                continue

            model_entry = models[key]
            if not isinstance(model_entry, dict):
                errors.append(
                    f"models.{key} must be an object, got {type(model_entry).__name__}"
                )
                continue

            for entry_key, expected_type in REQUIRED_MODEL_ENTRY_KEYS.items():
                if entry_key not in model_entry:
                    errors.append(f"Missing required key: models.{key}.{entry_key}")
                    continue
                elif not isinstance(model_entry[entry_key], expected_type):
                    errors.append(
                        f"models.{key}.{entry_key} must be {_article(expected_type)} "
                        f"{_type_name(expected_type)}, got {type(model_entry[entry_key]).__name__}"
                    )

            provider = model_entry.get("provider")
            if isinstance(provider, str) and provider not in VALID_PROVIDERS:
                errors.append(
                    f"models.{key}.provider must be one of {sorted(VALID_PROVIDERS)}, "
                    f"got {provider!r}"
                )

    # pipeline.*
    pipeline_cfg = config.get("pipeline")
    if isinstance(pipeline_cfg, dict):
        for key, expected_type in REQUIRED_PIPELINE_KEYS.items():
            if key not in pipeline_cfg:
                errors.append(f"Missing required key: pipeline.{key}")
            elif not isinstance(pipeline_cfg[key], expected_type) or (
                expected_type is int and isinstance(pipeline_cfg[key], bool)
            ):
                errors.append(
                    f"pipeline.{key} must be {_article(expected_type)} {_type_name(expected_type)}, "
                    f"got {type(pipeline_cfg[key]).__name__}"
                )

    # languages.*
    languages = config.get("languages")
    if isinstance(languages, dict):
        if not languages:
            errors.append("languages must be a non-empty dict (at least one language entry)")
        for lang_name, lang_cfg in languages.items():
            if not isinstance(lang_cfg, dict):
                errors.append(
                    f"languages.{lang_name} must be an object, got {type(lang_cfg).__name__}"
                )
                continue

            for key, expected_types in REQUIRED_LANGUAGE_KEYS.items():
                if key not in lang_cfg:
                    errors.append(f"Missing required key: languages.{lang_name}.{key}")
                    continue

                value = lang_cfg[key]
                if not isinstance(value, expected_types):
                    type_names = " or ".join(_type_name(t) for t in expected_types)
                    errors.append(
                        f"languages.{lang_name}.{key} must be {type_names}, "
                        f"got {type(value).__name__}"
                    )

            result_format = lang_cfg.get("result_format")
            if isinstance(result_format, str) and result_format not in VALID_RESULT_FORMATS:
                errors.append(
                    f"languages.{lang_name}.result_format must be one of "
                    f"{sorted(VALID_RESULT_FORMATS)}, got {result_format!r}"
                )

    return errors
