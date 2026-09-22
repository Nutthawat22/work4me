"""
tests/pipeline/test_acp_content.py

Unit tests for pipeline/acp_content.py's extract_prompt_content(). This
module is standalone (not wired into pipeline/acp_server.py's routing
yet), so these tests use plain duck-typed fakes (SimpleNamespace) for
ACP content blocks rather than importing the real acp.schema classes --
matching the fake-block convention already used in
tests/pipeline/test_acp_server.py (e.g. its use of acp.text_block /
acp.image_block helpers), but going one level more minimal since this
module never imports acp.schema itself.
"""

import os
import sys
from types import SimpleNamespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from pipeline.acp_content import AttachedFile, PromptContent, extract_prompt_content


def _text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def _embedded_text_resource_block(uri: str, text: str, mime_type: str | None = None):
    resource = SimpleNamespace(uri=uri, text=text, mime_type=mime_type)
    return SimpleNamespace(type="resource", resource=resource)


def _embedded_blob_resource_block(uri: str, blob: str, mime_type: str | None = None):
    resource = SimpleNamespace(uri=uri, blob=blob, mime_type=mime_type)
    return SimpleNamespace(type="resource", resource=resource)


def _resource_link_block(name: str, uri: str, mime_type: str | None = None):
    return SimpleNamespace(type="resource_link", name=name, uri=uri, mime_type=mime_type)


def _image_block(uri: str | None = None):
    return SimpleNamespace(type="image", uri=uri)


DEFAULT_MAX_CHARS = 200_000


# ── text-only prompts ────────────────────────────────────────────────────────

def test_text_only_prompt_populates_text_and_leaves_files_and_skipped_empty():
    blocks = [_text_block("hello"), _text_block("world")]

    result = extract_prompt_content(blocks, max_chars=DEFAULT_MAX_CHARS)

    assert result.text == "hello world"
    assert result.files == []
    assert result.skipped_blocks == []


# ── embedded text resource ──────────────────────────────────────────────────

def test_embedded_text_resource_produces_attached_file():
    blocks = [_embedded_text_resource_block("file:///tmp/design.md", "some design content", mime_type="text/markdown")]

    result = extract_prompt_content(blocks, max_chars=DEFAULT_MAX_CHARS)

    assert result.text == ""
    assert result.skipped_blocks == []
    assert len(result.files) == 1

    f = result.files[0]
    assert isinstance(f, AttachedFile)
    assert f.name == "design.md"
    assert f.uri == "file:///tmp/design.md"
    assert f.mime_type == "text/markdown"
    assert f.text == "some design content"
    assert f.truncated is False


def test_embedded_text_resource_longer_than_max_chars_is_truncated():
    long_text = "x" * 100
    blocks = [_embedded_text_resource_block("file:///tmp/big.txt", long_text)]

    result = extract_prompt_content(blocks, max_chars=10)

    assert len(result.files) == 1
    f = result.files[0]
    assert f.truncated is True
    assert f.text.endswith("\n... (truncated)")
    assert f.text.startswith("x" * 10)
    # capped: original 10 chars + the truncation marker, nothing more of the original content
    assert f.text == "x" * 10 + "\n... (truncated)"


# ── blob resource (skipped) ──────────────────────────────────────────────────

def test_blob_resource_is_skipped_not_crashed():
    blocks = [_embedded_blob_resource_block("file:///tmp/image.png", "base64stuff", mime_type="image/png")]

    result = extract_prompt_content(blocks, max_chars=DEFAULT_MAX_CHARS)

    assert result.files == []
    assert len(result.skipped_blocks) == 1
    assert "blob" in result.skipped_blocks[0].lower() or "binary" in result.skipped_blocks[0].lower()


# ── resource_link (skipped) ──────────────────────────────────────────────────

def test_resource_link_is_skipped_with_descriptive_note():
    blocks = [_resource_link_block("design.md", "file:///tmp/design.md", mime_type="text/markdown")]

    result = extract_prompt_content(blocks, max_chars=DEFAULT_MAX_CHARS)

    assert result.files == []
    assert len(result.skipped_blocks) == 1
    note = result.skipped_blocks[0]
    assert "resource_link" in note
    assert "design.md" in note


# ── unrecognized/malformed block ────────────────────────────────────────────

def test_malformed_block_with_no_type_is_skipped_gracefully():
    blocks = [object()]

    result = extract_prompt_content(blocks, max_chars=DEFAULT_MAX_CHARS)

    assert result.text == ""
    assert result.files == []
    assert len(result.skipped_blocks) == 1


def test_image_block_is_skipped():
    blocks = [_image_block(uri="file:///tmp/photo.png")]

    result = extract_prompt_content(blocks, max_chars=DEFAULT_MAX_CHARS)

    assert result.files == []
    assert len(result.skipped_blocks) == 1
    assert "image" in result.skipped_blocks[0]


# ── mixed prompt ─────────────────────────────────────────────────────────────

def test_mixed_prompt_handles_all_block_kinds_in_one_call():
    blocks = [
        _text_block("please review"),
        _embedded_text_resource_block("file:///tmp/notes.txt", "notes content"),
        _resource_link_block("design.md", "file:///tmp/design.md"),
        _text_block("thanks"),
    ]

    result = extract_prompt_content(blocks, max_chars=DEFAULT_MAX_CHARS)

    assert result.text == "please review thanks"
    assert len(result.files) == 1
    assert result.files[0].name == "notes.txt"
    assert result.files[0].text == "notes content"
    assert len(result.skipped_blocks) == 1
    assert "resource_link" in result.skipped_blocks[0]


# ── empty prompt ─────────────────────────────────────────────────────────────

def test_empty_prompt_list_returns_all_empty_prompt_content():
    result = extract_prompt_content([], max_chars=DEFAULT_MAX_CHARS)

    assert isinstance(result, PromptContent)
    assert result.text == ""
    assert result.files == []
    assert result.skipped_blocks == []
