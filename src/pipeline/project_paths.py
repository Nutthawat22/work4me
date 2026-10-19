"""Repository-root paths shared by the pipeline modules."""

import os
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])


def resolve_project_path(path: str) -> str:
    """Resolve a configured path against the repository root."""
    expanded_path = os.path.expanduser(path)
    if not os.path.isabs(expanded_path):
        expanded_path = os.path.join(PROJECT_ROOT, expanded_path)
    return os.path.abspath(expanded_path)
