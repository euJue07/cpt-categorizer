"""Tests for suggestion_store load/save/query/append/update_status."""

from pathlib import Path
from unittest.mock import patch

import pytest

from cpt_categorizer.suggestion_store import (
    append,
    find_by_status,
    find_by_type,
    find_by_type_key,
    load,
    update_status,
)


def test_load_missing_file_returns_empty_list(tmp_path: Path) -> None:
    path = tmp_path / "suggestions.json"
    assert not path.exists()
    assert load(path) == []


def test_load_empty_file_returns_empty_list(tmp_path: Path) -> None:
    path = tmp_path / "suggestions.json"
    path.write_text("[]")
    assert load(path) == []


def test_load_valid_file_returns_list(tmp_path: Path) -> None:
    path = tmp_path / "suggestions.json"
    path.write_text('[{"type": "section", "suggested_key": "foo", "status": "pending"}]')
    got = load(path)
    assert len(got) == 1
    assert got[0]["type"] == "section"
    assert got[0]["suggested_key"] == "foo"


def test_load_invalid_not_array_raises(tmp_path: Path) -> None:
    path = tmp_path / "suggestions.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="must be a JSON array"):
        load(path)


def test_append_creates_file_and_sets_id_and_created_at(tmp_path: Path) -> None:
    path = tmp_path / "suggestions.json"
    suggestion = {
        "type": "section",
        "suggested_key": "new_section",
        "suggested_description": "A new section",
        "status": "pending",
        "source": "batch_1",
    }
    append(path, suggestion)
    assert path.exists()
    got = load(path)
    assert len(got) == 1
    assert got[0]["type"] == "section"
    assert got[0]["suggested_key"] == "new_section"
    assert "id" in got[0]
    assert "created_at" in got[0]


def test_append_and_reload_persists(tmp_path: Path) -> None:
    path = tmp_path / "suggestions.json"
    append(path, {"type": "subsection", "suggested_key": "sub", "status": "pending"})
    again = load(path)
    assert len(again) == 1
    assert again[0]["type"] == "subsection"
    assert again[0]["suggested_key"] == "sub"


def test_find_by_type(tmp_path: Path) -> None:
    path = tmp_path / "suggestions.json"
    append(path, {"type": "section", "suggested_key": "a", "status": "pending"})
    append(path, {"type": "subsection", "suggested_key": "b", "status": "pending"})
    append(path, {"type": "section", "suggested_key": "c", "status": "pending"})
    sections = find_by_type(path, "section")
    assert len(sections) == 2
    assert {s["suggested_key"] for s in sections} == {"a", "c"}
    subsections = find_by_type(path, "subsection")
    assert len(subsections) == 1
    assert subsections[0]["suggested_key"] == "b"


def test_find_by_type_key(tmp_path: Path) -> None:
    path = tmp_path / "suggestions.json"
    append(
        path,
        {
            "type": "subsection",
            "suggested_key": "same_key",
            "context": "imaging",
            "status": "pending",
        },
    )
    append(
        path,
        {
            "type": "subsection",
            "suggested_key": "same_key",
            "context": "lab",
            "status": "pending",
        },
    )
    append(
        path,
        {
            "type": "subsection",
            "suggested_key": "other",
            "context": "imaging",
            "status": "pending",
        },
    )
    matches = find_by_type_key(path, "subsection", "same_key")
    assert len(matches) == 2
    matches_with_context = find_by_type_key(path, "subsection", "same_key", context="imaging")
    assert len(matches_with_context) == 1
    assert matches_with_context[0]["context"] == "imaging"


def test_find_by_status(tmp_path: Path) -> None:
    path = tmp_path / "suggestions.json"
    append(path, {"type": "section", "suggested_key": "p1", "status": "pending"})
    append(path, {"type": "section", "suggested_key": "a1", "status": "accepted"})
    append(path, {"type": "section", "suggested_key": "p2", "status": "pending"})
    pending = find_by_status(path, "pending")
    assert len(pending) == 2
    assert {s["suggested_key"] for s in pending} == {"p1", "p2"}
    accepted = find_by_status(path, "accepted")
    assert len(accepted) == 1
    assert accepted[0]["suggested_key"] == "a1"


def test_update_status_by_id(tmp_path: Path) -> None:
    path = tmp_path / "suggestions.json"
    append(path, {"type": "dimension", "suggested_key": "dim", "status": "pending"})
    data = load(path)
    sid = data[0]["id"]
    ok = update_status(path, sid, "accepted")
    assert ok is True
    updated = load(path)
    assert updated[0]["status"] == "accepted"


def test_update_status_unknown_id_returns_false(tmp_path: Path) -> None:
    path = tmp_path / "suggestions.json"
    append(path, {"type": "section", "suggested_key": "x", "status": "pending"})
    ok = update_status(path, "nonexistent-id", "accepted")
    assert ok is False
    assert load(path)[0]["status"] == "pending"


def test_append_creates_interim_dir_if_missing(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "nested" / "suggestions.json"
    assert not path.parent.exists()
    append(path, {"type": "section", "suggested_key": "k", "status": "pending"})
    assert path.exists()
    assert load(path)[0]["suggested_key"] == "k"


def test_load_uses_default_path_when_path_is_none(tmp_path: Path) -> None:
    """When path=None, load() uses SUGGESTIONS_PATH."""
    store_path = tmp_path / "suggestions.json"
    store_path.write_text("[]")
    with patch("cpt_categorizer.suggestion_store.SUGGESTIONS_PATH", store_path):
        assert load(None) == []


def test_append_uses_default_path_when_path_is_none(tmp_path: Path) -> None:
    """When path=None, append() uses SUGGESTIONS_PATH."""
    store_path = tmp_path / "suggestions.json"
    with patch("cpt_categorizer.suggestion_store.SUGGESTIONS_PATH", store_path):
        append(None, {"type": "section", "suggested_key": "default_path", "status": "pending"})
    assert store_path.exists()
    got = load(store_path)
    assert len(got) == 1
    assert got[0]["suggested_key"] == "default_path"


def test_update_status_uses_default_path_when_path_is_none(tmp_path: Path) -> None:
    """When path=None, update_status() uses SUGGESTIONS_PATH."""
    store_path = tmp_path / "suggestions.json"
    append(store_path, {"type": "section", "suggested_key": "x", "status": "pending"})
    sid = load(store_path)[0]["id"]
    with patch("cpt_categorizer.suggestion_store.SUGGESTIONS_PATH", store_path):
        ok = update_status(None, sid, "accepted")
    assert ok is True
    assert load(store_path)[0]["status"] == "accepted"


def test_append_preserves_existing_id_and_created_at(tmp_path: Path) -> None:
    """When suggestion already has id and created_at, append does not overwrite them."""
    path = tmp_path / "suggestions.json"
    custom_id = "custom-uuid-123"
    custom_created = "2020-01-01T00:00:00+00:00"
    suggestion = {
        "type": "section",
        "suggested_key": "preserved",
        "status": "pending",
        "id": custom_id,
        "created_at": custom_created,
    }
    append(path, suggestion)
    got = load(path)
    assert len(got) == 1
    assert got[0]["id"] == custom_id
    assert got[0]["created_at"] == custom_created
