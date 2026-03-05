"""Persist section and subsection tagging caches for cross-run API savings."""

from __future__ import annotations

import json
from pathlib import Path


def _normalize_section_value(v: object) -> list[tuple[str, float]]:
    """Ensure section cache value is list of [section, confidence]."""
    if not isinstance(v, list):
        return []
    out: list[tuple[str, float]] = []
    for item in v:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            s, c = item[0], item[1]
            if isinstance(s, str):
                try:
                    out.append((s, float(c)))
                except (TypeError, ValueError):
                    pass
    return out


def _normalize_sections(data: object) -> dict[str, list[tuple[str, float]]]:
    """Parse sections dict from JSON: str -> list of [section, confidence]."""
    if not isinstance(data, dict):
        return {}
    return {str(k): _normalize_section_value(v) for k, v in data.items() if isinstance(k, str)}


def _normalize_subsections(data: object) -> dict[str, list[tuple[str, float]]]:
    """Parse subsections dict from JSON: "section|normalized_text" -> list of [subsection, confidence]."""
    if not isinstance(data, dict):
        return {}
    return {str(k): _normalize_section_value(v) for k, v in data.items() if isinstance(k, str)}


class TaggingCache:
    """Shared cache for section and subsection tagging; load on init, persist after updates."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._sections: dict[str, list[tuple[str, float]]] = {}
        self._subsections: dict[str, list[tuple[str, float]]] = {}

    @property
    def sections(self) -> dict[str, list[tuple[str, float]]]:
        """Live dict for section cache (normalized text -> list of (section, confidence))."""
        return self._sections

    @property
    def subsections(self) -> dict[str, list[tuple[str, float]]]:
        """Live dict for subsection cache ("section|normalized_text" -> list of (subsection, confidence))."""
        return self._subsections

    def load(self) -> None:
        """Load from JSON file; if missing or invalid, leave sections/subsections empty."""
        self._sections = {}
        self._subsections = {}
        if not self._path.exists():
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(data, dict):
            return
        self._sections = _normalize_sections(data.get("sections"))
        self._subsections = _normalize_subsections(data.get("subsections"))

    def persist(self) -> None:
        """Atomic write of sections and subsections to the cache file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sections": {k: list(v) for k, v in self._sections.items()},
            "subsections": {k: list(v) for k, v in self._subsections.items()},
        }
        raw = json.dumps(payload, indent=2)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(raw)
        tmp.replace(self._path)
