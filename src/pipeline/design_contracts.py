"""Extract bounded, exact TypeScript interface contracts from planning input."""

from __future__ import annotations

import re

CONTRACT_START = re.compile(
    r"(?m)^[ \t]*(?:(?:export|declare)\s+)*(?:interface|type)\s+"
    r"([A-Za-z_$][A-Za-z0-9_$]*)[^\n{]*\{"
)
MAX_CONTRACTS = 20
MAX_DECLARATION_CHARS = 10000


def _matching_brace_end(text: str, opening: int) -> int | None:
    depth = 0
    quote: str | None = None
    escaped = False
    line_comment = False
    block_comment = False
    index = opening
    while index < len(text):
        current = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if line_comment:
            if current == "\n":
                line_comment = False
        elif block_comment:
            if current == "*" and following == "/":
                block_comment = False
                index += 1
        elif quote:
            if escaped:
                escaped = False
            elif current == "\\":
                escaped = True
            elif current == quote:
                quote = None
        elif current == "/" and following == "/":
            line_comment = True
            index += 1
        elif current == "/" and following == "*":
            block_comment = True
            index += 1
        elif current in {"'", '"', "`"}:
            quote = current
        elif current == "{":
            depth += 1
        elif current == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return None


def extract_typescript_contracts(text: str, source_label: str) -> list[dict[str, str]]:
    contracts: list[dict[str, str]] = []
    for match in CONTRACT_START.finditer(text):
        if len(contracts) >= MAX_CONTRACTS:
            break
        end = _matching_brace_end(text, match.end() - 1)
        if end is None or end - match.start() > MAX_DECLARATION_CHARS:
            continue
        start_line = text.count("\n", 0, match.start()) + 1
        end_line = text.count("\n", 0, end) + 1
        contracts.append({
            "name": match.group(1),
            "source_ref": f"{source_label} lines {start_line}-{end_line}",
            "declaration": text[match.start():end].strip(),
        })
    return contracts


def merge_contracts(*groups: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        for contract in group:
            key = (contract.get("name", ""), contract.get("declaration", ""))
            if key in seen or not key[0] or not key[1]:
                continue
            seen.add(key)
            merged.append(contract)
            if len(merged) >= MAX_CONTRACTS:
                return merged
    return merged
