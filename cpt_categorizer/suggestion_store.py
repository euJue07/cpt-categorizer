"""Persist and query suggestions (section/subsection/dimension) for suggestors and governors."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from cpt_categorizer.config.directory import SUGGESTIONS_PATH

SuggestionType = Literal["section", "subsection", "dimension"]
SuggestionStatus = Literal["pending", "accepted", "rejected", "duplicate"]


def load(path: Path | None = None) -> list[dict[str, Any]]:
    """Load all suggestions from the JSON file. Returns [] if file is missing or empty."""
    p = path if path is not None else SUGGESTIONS_PATH
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{p.name} must be a JSON array")
    return list(data)


def _save(path: Path, suggestions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(suggestions, indent=2)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(raw)
    tmp.replace(path)


def find_by_type(path: Path | None, suggestion_type: SuggestionType) -> list[dict[str, Any]]:
    """Return suggestions matching the given type."""
    all_suggestions = load(path)
    return [s for s in all_suggestions if s.get("type") == suggestion_type]


def find_by_type_key(
    path: Path | None,
    suggestion_type: SuggestionType,
    suggested_key: str,
    context: str | None = None,
) -> list[dict[str, Any]]:
    """Return suggestions matching type and key; if context is given, filter by context as well."""
    all_suggestions = load(path)
    out = [
        s
        for s in all_suggestions
        if s.get("type") == suggestion_type and s.get("suggested_key") == suggested_key
    ]
    if context is not None:
        out = [s for s in out if s.get("context") == context]
    return out


def find_by_status(path: Path | None, status: SuggestionStatus) -> list[dict[str, Any]]:
    """Return suggestions with the given status."""
    all_suggestions = load(path)
    return [s for s in all_suggestions if s.get("status") == status]


def append(path: Path | None, suggestion: dict[str, Any]) -> None:
    """Append one suggestion; set created_at and id if missing. Persists with atomic write."""
    p = path if path is not None else SUGGESTIONS_PATH
    suggestions = load(p)
    record = dict(suggestion)
    if "id" not in record:
        record["id"] = str(uuid.uuid4())
    if "created_at" not in record:
        record["created_at"] = datetime.now(tz=timezone.utc).isoformat()
    suggestions.append(record)
    _save(p, suggestions)


def update_status(
    path: Path | None,
    suggestion_id: str,
    status: SuggestionStatus,
) -> bool:
    """Update status of the suggestion with the given id. Returns True if found and updated."""
    p = path if path is not None else SUGGESTIONS_PATH
    suggestions = load(p)
    for s in suggestions:
        if s.get("id") == suggestion_id:
            s["status"] = status
            _save(p, suggestions)
            return True
    return False
