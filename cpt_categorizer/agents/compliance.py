from __future__ import annotations

from typing import Any

from cpt_categorizer.agents.normalizer import to_snake_case
from cpt_categorizer.schema_contract import SchemaContract


def empty_dimensions() -> dict[str, Any]:
    return {
        "actual": {},
        "proposed": {
            "existing_dimensions": {},
            "new_dimensions": {},
        },
    }


class SchemaComplianceAgent:
    """Validates and repairs tag payloads against schema contract."""

    def __init__(self, schema_contract: SchemaContract, mode: str = "balanced"):
        if mode not in {"lenient", "balanced", "strict"}:
            raise ValueError(f"Unsupported compliance mode: {mode}")
        self.schema_contract = schema_contract
        self.mode = mode

    def validate_tag(self, tag: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
        warnings: list[str] = []
        section = str(tag.get("section", "")).strip().lower()
        subsection = str(tag.get("subsection", "")).strip().lower()
        confidence = float(tag.get("confidence", 0.0))

        if not section:
            return self._handle_fatal("missing section", warnings)

        if section != "others" and section not in self.schema_contract.sections:
            return self._handle_fatal(f"unknown section '{section}'", warnings)

        if section == "others":
            subsection = "others"
            return (
                {
                    "section": "others",
                    "subsection": subsection,
                    "confidence": confidence,
                    "dimensions": empty_dimensions(),
                },
                warnings,
            )

        allowed_subsections = set(self.schema_contract.allowed_subsections(section))
        if subsection not in allowed_subsections:
            return self._handle_fatal(
                f"subsection '{subsection}' not in section '{section}'", warnings
            )

        normalized_dimensions = self._validate_dimensions(
            section=section,
            subsection=subsection,
            dimensions=tag.get("dimensions", {}),
            warnings=warnings,
        )

        return (
            {
                "section": section,
                "subsection": subsection,
                "confidence": confidence,
                "dimensions": normalized_dimensions,
            },
            warnings,
        )

    def _validate_dimensions(
        self,
        section: str,
        subsection: str,
        dimensions: Any,
        warnings: list[str],
    ) -> dict[str, Any]:
        if not isinstance(dimensions, dict):
            warnings.append("dimensions payload is not an object; replaced with defaults")
            return empty_dimensions()

        allowed_dimensions = set(self.schema_contract.allowed_dimensions(section, subsection))
        actual = dimensions.get("actual", {})
        proposed = dimensions.get("proposed", {})

        normalized_actual: dict[str, list[dict[str, Any]]] = {}
        if isinstance(actual, dict):
            for dimension, values in actual.items():
                dim_key = to_snake_case(str(dimension))
                if str(dimension) != dim_key:
                    warnings.append(
                        f"normalized dimension key '{dimension}' -> '{dim_key}'"
                    )
                if dim_key not in allowed_dimensions:
                    warnings.append(f"dropped non-allowed dimension '{dimension}'")
                    continue
                valid_values = self._filter_values(
                    values=values,
                    dimension=dim_key,
                    known_values=set(self.schema_contract.dimensions[dim_key]["values"]),
                    warnings=warnings,
                    require_known=True,
                )
                if valid_values:
                    normalized_actual[dim_key] = valid_values

        existing_in = proposed.get("existing_dimensions", {}) if isinstance(proposed, dict) else {}
        new_in = proposed.get("new_dimensions", {}) if isinstance(proposed, dict) else {}
        normalized_existing: dict[str, list[dict[str, Any]]] = {}
        normalized_new: dict[str, list[dict[str, Any]]] = {}

        if isinstance(existing_in, dict):
            for dimension, values in existing_in.items():
                dim_key = to_snake_case(str(dimension))
                if str(dimension) != dim_key:
                    warnings.append(
                        f"normalized proposed dimension key '{dimension}' -> '{dim_key}'"
                    )
                if dim_key not in allowed_dimensions:
                    warnings.append(f"dropped non-allowed proposed dimension '{dimension}'")
                    continue
                valid_values = self._filter_values(
                    values=values,
                    dimension=dim_key,
                    known_values=set(self.schema_contract.dimensions[dim_key]["values"]),
                    warnings=warnings,
                    require_known=False,
                )
                if valid_values:
                    normalized_existing[dim_key] = valid_values

        if isinstance(new_in, dict):
            for dimension, values in new_in.items():
                dim_key = to_snake_case(str(dimension))
                if str(dimension) != dim_key:
                    warnings.append(
                        f"normalized new dimension key '{dimension}' -> '{dim_key}'"
                    )
                if not dim_key:
                    continue
                valid_values = self._filter_values(
                    values=values,
                    dimension=dim_key,
                    known_values=set(),
                    warnings=warnings,
                    require_known=False,
                )
                if valid_values:
                    normalized_new[dim_key] = valid_values

        return {
            "actual": normalized_actual,
            "proposed": {
                "existing_dimensions": normalized_existing,
                "new_dimensions": normalized_new,
            },
        }

    def _filter_values(
        self,
        values: Any,
        dimension: str,
        known_values: set[str],
        warnings: list[str],
        require_known: bool,
    ) -> list[dict[str, Any]]:
        if not isinstance(values, list):
            return []

        filtered: list[dict[str, Any]] = []
        for item in values:
            if not isinstance(item, dict):
                continue
            confidence = float(item.get("confidence", 0.0))
            if confidence < 0.5:
                warnings.append(f"dropped low-confidence value in '{dimension}'")
                continue
            raw_value = str(item.get("value", ""))
            value = to_snake_case(raw_value)
            if raw_value != value:
                warnings.append(
                    f"normalized value '{raw_value}' -> '{value}' in '{dimension}'"
                )
            if not value:
                continue
            if require_known and value not in known_values:
                warnings.append(f"dropped unknown enum value '{value}' in '{dimension}'")
                continue
            filtered.append({"value": value, "confidence": confidence})
        return filtered

    def _handle_fatal(
        self, message: str, warnings: list[str]
    ) -> tuple[dict[str, Any] | None, list[str]]:
        warnings.append(message)
        if self.mode == "strict":
            return None, warnings
        if self.mode == "lenient":
            return (
                {
                    "section": "others",
                    "subsection": "others",
                    "confidence": 0.0,
                    "dimensions": empty_dimensions(),
                },
                warnings,
            )
        return None, warnings
