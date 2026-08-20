"""
pipeline/design.py

DesignAgent: decomposes a user prompt into a list of WorkItems by asking
the LLM to output a JSON array matching the WorkItem schema.
"""

import json
import re

from pipeline.state import WorkItem
from specialists.llm_client import call_llm

VALID_TYPES: set[str] = {"logic", "ui", "config", "test"}

REQUIRED_FIELDS = {
    "id",
    "type",
    "language",
    "title",
    "description",
    "acceptance_criteria",
    "output_path",
    "depends_on",
}

SYSTEM_PROMPT_TEMPLATE = """You are the DesignAgent in an autonomous coding pipeline.

Given a user's request, decompose it into a list of WorkItems for specialist \
agents to implement. Output ONLY a JSON array (no prose, no markdown fences) \
where each element matches this exact schema:

{{
  "id": "WI-001",
  "type": "logic" | "ui" | "config" | "test",
  "language": "python",
  "title": "one-line description",
  "description": "full task spec",
  "acceptance_criteria": ["criterion 1", "criterion 2"],
  "output_path": "relative/path/to/file.py",
  "depends_on": ["WI-000"]
}}

Rules:
- "type" must be exactly one of: logic, ui, config, test.
- "language" must be exactly one of: {valid_languages}.
- "depends_on" lists ids of other WorkItems in this same array that must be \
completed first. Use [] if there are none.
- Every field is required on every item.
- Output must be a valid JSON array and nothing else.
"""


class DesignParseError(Exception):
    """Raised when the DesignAgent's LLM response cannot be parsed into valid WorkItems."""


class DesignAgent:
    def __init__(self, config: dict):
        self.config = config

    def decompose(self, user_prompt: str) -> list[WorkItem]:
        """
        Ask the LLM to decompose user_prompt into WorkItems, parse and
        validate the JSON response, and return constructed WorkItem objects.

        Raises:
            DesignParseError: if the response is not valid JSON or fails
                schema validation.
        """
        valid_languages = set(self.config["languages"].keys())

        model_cfg = self.config["models"]["design"]
        model = model_cfg["model"]
        provider = model_cfg["provider"]
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            valid_languages=", ".join(sorted(valid_languages))
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw = call_llm(messages, model, self.config, provider=provider)

        parsed = self._parse_json(raw)
        return self._validate_and_build(parsed, valid_languages)

    @staticmethod
    def _parse_json(raw: str) -> list[dict]:
        stripped = raw.strip()

        fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
        if fence_match:
            stripped = fence_match.group(1).strip()

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise DesignParseError(f"Failed to parse DesignAgent response as JSON: {e}\nRaw response: {raw}")

        if not isinstance(parsed, list):
            raise DesignParseError(f"Expected a JSON array of WorkItems, got: {type(parsed).__name__}")

        return parsed

    @staticmethod
    def _validate_and_build(items: list[dict], valid_languages: set[str]) -> list[WorkItem]:
        work_items: list[WorkItem] = []

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                raise DesignParseError(f"WorkItem at index {idx} is not a JSON object: {item!r}")

            missing = REQUIRED_FIELDS - item.keys()
            if missing:
                raise DesignParseError(
                    f"WorkItem at index {idx} (id={item.get('id', '?')}) missing required fields: {sorted(missing)}"
                )

            item_type = item["type"]
            if item_type not in VALID_TYPES:
                raise DesignParseError(
                    f"WorkItem {item.get('id', '?')} has invalid type '{item_type}', "
                    f"must be one of {sorted(VALID_TYPES)}"
                )

            item_language = item["language"]
            if item_language not in valid_languages:
                raise DesignParseError(
                    f"WorkItem {item.get('id', '?')} has invalid language '{item_language}', "
                    f"must be one of {sorted(valid_languages)}"
                )

            if not isinstance(item["acceptance_criteria"], list):
                raise DesignParseError(
                    f"WorkItem {item.get('id', '?')} 'acceptance_criteria' must be a list"
                )

            if not isinstance(item["depends_on"], list):
                raise DesignParseError(
                    f"WorkItem {item.get('id', '?')} 'depends_on' must be a list"
                )

            work_items.append(
                WorkItem(
                    id=item["id"],
                    type=item_type,  # type: ignore[arg-type]
                    language=item_language,
                    title=item["title"],
                    description=item["description"],
                    acceptance_criteria=item["acceptance_criteria"],
                    output_path=item["output_path"],
                    depends_on=item["depends_on"],
                )
            )

        return work_items
