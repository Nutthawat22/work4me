"""
tests/pipeline/test_config_validation.py

Tests for pipeline/config_validation.py's validate_config(), focused on
provider validation (see specialists/providers/__init__.py's PROVIDERS
registry and VALID_PROVIDERS here).
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline.config_validation import validate_config


def _base_config() -> dict:
    return {
        "litellm_url": "https://litellm.example.test/v1",
        "litellm_key": "sk-test-key",
        "models": {
            "design": {"model": "gpt-5.6-luna", "provider": "acp"},
            "specialist": {"model": "kimi-k2.7-code", "provider": "acp"},
        },
        "pipeline": {
            "max_retries": 3,
            "runs_dir": "runs/",
            "output_dir": "product",
            "tests_dir": "tests",
        },
        "languages": {
            "python": {
                "test_command": ["pytest", "{tests_dir}", "--junitxml={junit_path}"],
                "test_file_patterns": ["test_*.py", "*_test.py"],
                "result_format": "junit",
                "file_extension": ".py",
                "test_framework_name": "pytest",
            },
        },
    }


def test_acp_is_accepted_as_a_valid_provider():
    config = _base_config()
    config["models"]["specialist"]["provider"] = "acp"

    errors = validate_config(config)

    assert errors == []


def test_unknown_provider_is_still_rejected():
    config = _base_config()
    config["models"]["specialist"]["provider"] = "not-a-real-provider"

    errors = validate_config(config)

    assert any("provider must be one of" in e for e in errors)
