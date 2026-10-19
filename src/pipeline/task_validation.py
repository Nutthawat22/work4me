"""Validation shared by task planning and specialist output handling."""

from __future__ import annotations

import ntpath
import posixpath


def normalize_relative_path(path: str) -> str:
    """Return a canonical slash-separated relative path or raise ValueError."""
    candidate = path.replace("\\", "/")
    drive, _ = ntpath.splitdrive(candidate)
    if not candidate or "\x00" in candidate or drive or candidate.startswith("/"):
        raise ValueError(f"path must be non-empty and relative: {path!r}")
    parts = candidate.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"path contains an unsafe component: {path!r}")
    return posixpath.normpath(candidate)


def truncate_with_marker(content: str, limit: int, marker: str) -> str:
    """Keep truncation text inside the supplied character budget."""
    limit = max(0, limit)
    if len(content) <= limit:
        return content
    if limit <= len(marker):
        return marker[:limit]
    return content[:limit - len(marker)] + marker
