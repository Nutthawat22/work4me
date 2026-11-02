"""Load non-secret Docker runner settings from the user's config directory."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from testing_agent.runner import AgentSettings


def load_settings(config_path: str | None = None) -> AgentSettings:
    path = Path(config_path).expanduser() if config_path else Path.home() / ".config" / "Agents" / "testing-agent.json"
    if not path.exists():
        settings = AgentSettings()
        settings.validate()
        return settings
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read testing-agent config {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("testing-agent config must be a JSON object")
    allowed = {item.name for item in fields(AgentSettings)}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown testing-agent config keys: {', '.join(unknown)}")
    try:
        settings = AgentSettings(**value)
    except TypeError as error:
        raise ValueError(f"Invalid testing-agent config: {error}") from error
    settings.validate()
    return settings
