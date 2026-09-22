"""
pipeline/acp_content.py

Standalone, NOT-YET-WIRED-IN utility for extracting file-like content
from an ACP `prompt` request's list of content blocks (see
pipeline/acp_server.py's module docstring for the surrounding ACP
Agent/server context, and its `_extract_prompt_text()` for the
existing free-text-only extraction this module deliberately does not
touch).

Why this exists / why it's separate: pipeline/acp_server.py's
`prompt()` currently reads ONLY `TextContentBlock` entries
(_extract_prompt_text) — v1 scope, per that module's docstring. This
module is prep work for a later version that also reads embedded
text-resource attachments (e.g. a client sending a design doc's full
text alongside a free-text instruction). It is intentionally kept
separate and NOT called from acp_server.py yet: the external team that
owns the ACP client side hasn't confirmed what shape of content blocks
they'll actually send, so wiring this into the routing/dispatch path
now would risk building against a guessed contract. This module is
therefore independently importable/testable with zero dependency on
acp_server.py or any live ACP connection.

v1 scope of THIS module (do not expand without checking the design
doc / flagging first):

  - TextContentBlock (type="text") -> contributes to the returned
    `text` field, using the exact same matching/joining logic as
    pipeline/acp_server.py's `_extract_prompt_text()` (duck-typed on
    `.type == "text"` and a non-None `.text`, joined with a single
    space). This module does not import from acp_server.py to avoid a
    reverse dependency; the logic is duplicated deliberately, matching
    that function's exact behavior.
  - EmbeddedResourceContentBlock (type="resource") wrapping a
    TextResourceContents (duck-typed: has `.text`) -> becomes an
    AttachedFile, with `.text` truncated to `max_chars` (see
    pipeline/dispatch.py's `_read_dependency_content()` /
    MAX_DEPENDENCY_FILE_CHARS for the truncation convention this
    mirrors: cut off cleanly and append "\n... (truncated)").
  - EmbeddedResourceContentBlock wrapping a BlobResourceContents
    (duck-typed: has `.blob` instead of `.text`) -> binary/image data,
    no usable text. Skipped; recorded in `skipped_blocks` for
    observability, not treated as an error.
  - ResourceLink (type="resource_link") -> a pointer only, no inline
    content. Fetching it would require a client filesystem callback
    this module has no access to and no opinion about yet — explicitly
    OUT OF SCOPE. Skipped; recorded in `skipped_blocks`.
  - ImageContentBlock / AudioContentBlock / any other
    unrecognized/malformed block -> skipped; recorded in
    `skipped_blocks`. This function never raises on a block it doesn't
    understand — every attribute access uses `getattr(..., None)`,
    matching `_extract_prompt_text()`'s defensive style.

Explicitly NOT done here (by design, for this task):

  - No config reading. `extract_prompt_content()` takes `max_chars` as
    a required, explicit parameter — the caller (whoever eventually
    wires this in) is responsible for pulling
    `config["pipeline"].get("max_prompt_content_chars",
    DEFAULT_MAX_PROMPT_CONTENT_CHARS)` and passing it in. This keeps
    this module a pure function with no config-fixture dependency,
    trivially unit-testable in isolation.
  - No wiring into pipeline/acp_server.py's `prompt()` routing. This
    module is a standalone utility until the external team confirms
    what content-block shapes they'll actually send.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Forward-looking guardrail against a runaway/malicious/oversized
# embedded-resource text payload, NOT a currently-active constraint --
# see the design discussion this constant was born from: Claude Sonnet
# 5 (the model actually configured for this pipeline) has a
# ~2.5M-char context window, and the largest real design doc seen in
# this project so far is ~44K chars. This default is intentionally
# generous relative to real usage; callers may override it via
# `extract_prompt_content(prompt, max_chars=...)`.
DEFAULT_MAX_PROMPT_CONTENT_CHARS = 200_000


@dataclass
class AttachedFile:
    """One embedded text-resource attachment extracted from a prompt's
    content blocks. `name` is derived from the resource's `.uri` (last
    path segment) since EmbeddedResourceContentBlock carries no
    separate display name the way ResourceLink does."""

    name: str
    uri: str | None
    mime_type: str | None
    text: str
    truncated: bool


@dataclass
class PromptContent:
    """Result of extracting everything usable out of an ACP prompt's
    content blocks: joined free text, extracted text-file attachments,
    and a log of anything skipped (for observability/debugging — not
    an error condition)."""

    text: str
    files: list[AttachedFile] = field(default_factory=list)
    skipped_blocks: list[str] = field(default_factory=list)


def _block_type_label(block: Any) -> str:
    """Best-effort human-readable type label for a skipped block, used
    in `skipped_blocks` notes. Falls back to the block's class name if
    it has no `.type` attribute at all (e.g. a bare malformed object)."""
    block_type = getattr(block, "type", None)
    if block_type:
        return str(block_type)
    return type(block).__name__


def _basename_from_uri(uri: str | None) -> str:
    if not uri:
        return "unnamed"
    # Deliberately simple: split on both URI and filesystem-style
    # separators without pulling in urllib.parse/os.path, since a URI
    # here may be a bare file path, a file:// URI, or any other
    # client-defined scheme -- we only want the trailing segment for a
    # display name, not full URI parsing.
    trailing = uri.rstrip("/").rsplit("/", 1)[-1]
    return trailing or "unnamed"


def extract_prompt_content(prompt: list[Any], max_chars: int) -> PromptContent:
    """
    Extract free text and embedded text-file attachments from an ACP
    prompt's list of content blocks.

    Never raises on a malformed/unexpected block -- every attribute
    access is defensive (`getattr(..., None)`), matching
    pipeline/acp_server.py's `_extract_prompt_text()` style. Blocks
    this function doesn't understand or explicitly defers (ResourceLink,
    ImageContentBlock, AudioContentBlock, blob-only embedded resources,
    or anything else) are skipped and logged in `skipped_blocks` rather
    than causing an error.

    Args:
        prompt: the ACP prompt's list of content blocks (real
            `acp.schema` objects or duck-typed equivalents -- this
            function only ever uses getattr()).
        max_chars: per-file truncation limit for embedded text
            resources. Required and explicit -- this module does not
            read config itself (see module docstring); callers should
            pass `config["pipeline"].get("max_prompt_content_chars",
            DEFAULT_MAX_PROMPT_CONTENT_CHARS)` or similar.

    Returns:
        A PromptContent with `text` (space-joined TextContentBlock
        text, "" if none), `files` (one AttachedFile per embedded text
        resource, in prompt order), and `skipped_blocks` (one
        human-readable note per skipped block, in prompt order).
    """
    text_parts: list[str] = []
    files: list[AttachedFile] = []
    skipped_blocks: list[str] = []

    for block in prompt:
        block_type = getattr(block, "type", None)
        text_attr = getattr(block, "text", None)

        if text_attr is not None and block_type == "text":
            if text_attr:
                text_parts.append(text_attr)
            continue

        if block_type == "resource":
            resource = getattr(block, "resource", None)
            resource_text = getattr(resource, "text", None)
            resource_blob = getattr(resource, "blob", None)

            if resource_text is not None:
                uri = getattr(resource, "uri", None)
                mime_type = getattr(resource, "mime_type", None)
                truncated = len(resource_text) > max_chars
                file_text = resource_text[:max_chars] + "\n... (truncated)" if truncated else resource_text
                files.append(
                    AttachedFile(
                        name=_basename_from_uri(uri),
                        uri=uri,
                        mime_type=mime_type,
                        text=file_text,
                        truncated=truncated,
                    )
                )
                continue

            if resource_blob is not None:
                uri = getattr(resource, "uri", None)
                skipped_blocks.append(f"resource (blob/binary, no usable text): uri={uri!r}")
                continue

            # `.resource` present but neither text nor blob attr found
            # -- unrecognized resource shape, skip defensively.
            uri = getattr(resource, "uri", None)
            skipped_blocks.append(f"resource (unrecognized resource contents shape): uri={uri!r}")
            continue

        if block_type == "resource_link":
            name = getattr(block, "name", None)
            uri = getattr(block, "uri", None)
            skipped_blocks.append(
                f"resource_link (unsupported/deferred -- pointer only, no fetch callback): "
                f"name={name!r} uri={uri!r}"
            )
            continue

        if block_type in ("image", "audio"):
            uri = getattr(block, "uri", None)
            skipped_blocks.append(f"{block_type} (unsupported/deferred): uri={uri!r}")
            continue

        # Anything else: unrecognized or malformed block (e.g. no
        # `.type` at all). Skip gracefully rather than crash.
        skipped_blocks.append(f"{_block_type_label(block)} (unrecognized block, skipped)")

    return PromptContent(
        text=" ".join(text_parts).strip(),
        files=files,
        skipped_blocks=skipped_blocks,
    )
