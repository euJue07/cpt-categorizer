"""Tests for tagging_cache load/persist and round-trip."""

from pathlib import Path

import pytest

from cpt_categorizer.tagging_cache import TaggingCache


def test_load_missing_file_leaves_empty_sections_and_subsections(tmp_path: Path) -> None:
    path = tmp_path / "tagging_cache.json"
    assert not path.exists()
    cache = TaggingCache(path)
    cache.load()
    assert cache.sections == {}
    assert cache.subsections == {}


def test_persist_creates_parent_dir_if_missing(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "nested" / "tagging_cache.json"
    assert not path.parent.exists()
    cache = TaggingCache(path)
    cache.sections["chest x-ray"] = [("imaging", 0.9)]
    cache.persist()
    assert path.exists()
    cache2 = TaggingCache(path)
    cache2.load()
    assert cache2.sections == {"chest x-ray": [("imaging", 0.9)]}


def test_round_trip_sections_and_subsections(tmp_path: Path) -> None:
    path = tmp_path / "tagging_cache.json"
    cache = TaggingCache(path)
    cache.sections["normalized desc"] = [("imaging", 0.95), ("lab", 0.6)]
    cache.subsections["imaging|chest x-ray"] = [("xray", 0.9)]
    cache.persist()

    cache2 = TaggingCache(path)
    cache2.load()
    assert cache2.sections == {"normalized desc": [("imaging", 0.95), ("lab", 0.6)]}
    assert cache2.subsections == {"imaging|chest x-ray": [("xray", 0.9)]}


def test_load_invalid_json_leaves_empty_dicts(tmp_path: Path) -> None:
    path = tmp_path / "tagging_cache.json"
    path.write_text("not valid json {")
    cache = TaggingCache(path)
    cache.load()
    assert cache.sections == {}
    assert cache.subsections == {}


def test_load_non_object_root_leaves_empty_dicts(tmp_path: Path) -> None:
    path = tmp_path / "tagging_cache.json"
    path.write_text("[1, 2, 3]")
    cache = TaggingCache(path)
    cache.load()
    assert cache.sections == {}
    assert cache.subsections == {}
