from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from cpt_categorizer.config.directory import SCHEMA_DIR


@dataclass(frozen=True)
class SchemaContract:
    sections: dict[str, Any]
    subsections: dict[str, Any]
    dimensions: dict[str, Any]
    version: str

    def validate(self) -> None:
        for section, section_spec in self.sections.items():
            subsections = section_spec.get("subsections", [])
            if not isinstance(subsections, list):
                raise ValueError(f"{section}: 'subsections' must be a list")

            if section not in self.subsections:
                raise ValueError(f"Section '{section}' missing from subsections schema")

            known_subsections = self.subsections[section]
            for subsection in subsections:
                if subsection not in known_subsections:
                    raise ValueError(
                        f"Subsection '{subsection}' listed in sections but missing from "
                        f"subsections[{section}]"
                    )

        for section, subsection_map in self.subsections.items():
            if section not in self.sections:
                raise ValueError(f"Section '{section}' in subsections not found in sections")

            for subsection, subsection_spec in subsection_map.items():
                dimensions = subsection_spec.get("dimensions", [])
                if not isinstance(dimensions, list):
                    raise ValueError(
                        f"{section}.{subsection}: 'dimensions' must be a list"
                    )
                for dimension in dimensions:
                    if dimension not in self.dimensions:
                        raise ValueError(
                            f"{section}.{subsection}: unknown dimension '{dimension}'"
                        )

    def allowed_subsections(self, section: str) -> list[str]:
        section_spec = self.sections.get(section, {})
        subsection_ids = section_spec.get("subsections", [])
        return list(subsection_ids) if isinstance(subsection_ids, list) else []

    def allowed_dimensions(self, section: str, subsection: str) -> list[str]:
        subsection_spec = self.subsections.get(section, {}).get(subsection, {})
        dimensions = subsection_spec.get("dimensions", [])
        return list(dimensions) if isinstance(dimensions, list) else []


_CACHED_CONTRACT: SchemaContract | None = None


def _read_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return data


def _schema_version(
    sections: dict[str, Any], subsections: dict[str, Any], dimensions: dict[str, Any]
) -> str:
    payload = json.dumps(
        {
            "sections": sections,
            "subsections": subsections,
            "dimensions": dimensions,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def load_schema_contract(use_cache: bool = True) -> SchemaContract:
    global _CACHED_CONTRACT

    if use_cache and _CACHED_CONTRACT is not None:
        return _CACHED_CONTRACT

    sections = _read_json(SCHEMA_DIR / "sections.json")
    subsections = _read_json(SCHEMA_DIR / "subsections.json")
    dimensions = _read_json(SCHEMA_DIR / "dimensions.json")
    version = _schema_version(sections, subsections, dimensions)

    contract = SchemaContract(
        sections=sections,
        subsections=subsections,
        dimensions=dimensions,
        version=version,
    )
    contract.validate()

    _CACHED_CONTRACT = contract
    return contract
