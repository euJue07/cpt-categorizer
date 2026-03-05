import re
from typing import Any


SNAKE_CASE_SANITIZER = re.compile(r"[^a-z0-9]+")


def to_snake_case(value: str) -> str:
    normalized = SNAKE_CASE_SANITIZER.sub("_", value.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        return ""
    if normalized[0].isdigit():
        normalized = f"v_{normalized}"
    return normalized


class NormalizerAgent:
    """Normalizes tag payload keys and values into canonical form."""

    def normalize_tag(self, tag: dict[str, Any]) -> dict[str, Any]:
        dimensions = tag.get("dimensions", {})
        actual = dimensions.get("actual", {})
        proposed = dimensions.get("proposed", {})

        normalized_actual = {
            to_snake_case(str(dimension)): self._normalize_value_items(values)
            for dimension, values in actual.items()
        }
        normalized_existing = {
            to_snake_case(str(dimension)): self._normalize_value_items(values)
            for dimension, values in proposed.get("existing_dimensions", {}).items()
        }
        normalized_new = {
            to_snake_case(str(dimension)): self._normalize_value_items(values)
            for dimension, values in proposed.get("new_dimensions", {}).items()
        }

        return {
            "section": str(tag.get("section", "")).strip().lower(),
            "subsection": str(tag.get("subsection", "")).strip().lower(),
            "confidence": float(tag.get("confidence", 0.0)),
            "dimensions": {
                "actual": {k: v for k, v in normalized_actual.items() if k},
                "proposed": {
                    "existing_dimensions": {
                        k: v for k, v in normalized_existing.items() if k
                    },
                    "new_dimensions": {k: v for k, v in normalized_new.items() if k},
                },
            },
        }

    def _normalize_value_items(self, values: Any) -> list[dict[str, Any]]:
        if not isinstance(values, list):
            return []
        normalized_items: list[dict[str, Any]] = []
        for value in values:
            if not isinstance(value, dict):
                continue
            normalized_value = to_snake_case(str(value.get("value", "")))
            if not normalized_value:
                continue
            normalized_items.append(
                {
                    "value": normalized_value,
                    "confidence": float(value.get("confidence", 0.0)),
                }
            )
        return normalized_items
