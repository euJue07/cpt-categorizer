"""Tests for SectionSuggestorAgent: store hit, new suggestion, duplicate key."""

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cpt_categorizer.agents.section_suggestor import SectionSuggestorAgent
from cpt_categorizer.agents.section_suggestor import _context_for_description
from cpt_categorizer.suggestion_store import append, find_by_type, load


def _context_for_description_test(text: str) -> str:
    """Same as agent's internal context computation for test setup."""
    normalized = text.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _make_mock_response(suggested_key: str, suggested_description: str):
    """Build a mock OpenAI response with function_call.arguments for suggest_section."""
    args = json.dumps({"suggested_key": suggested_key, "suggested_description": suggested_description})
    choice = MagicMock()
    choice.message.function_call.arguments = args
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 8
    usage.total_tokens = 18
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    resp.model = "gpt-4o"
    return resp


@pytest.fixture
def section_schema():
    return {"anesthesiology": {"description": "Anesthesia"}, "imaging": {"description": "Imaging"}}


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "suggestions.json"


def test_suggest_section_returns_none_for_empty_description(section_schema, store_path):
    agent = SectionSuggestorAgent(section_schema=section_schema, store_path=store_path)
    assert agent.suggest_section("") is None
    assert agent.suggest_section("   ") is None


def test_suggest_section_store_hit_by_context(section_schema, store_path):
    """Pre-populate store with suggestion for context; suggest_section returns it and skips LLM."""
    description = "Rare procedure XYZ"
    context = _context_for_description_test(description)
    existing = {
        "type": "section",
        "suggested_key": "rare_procedure",
        "suggested_description": "Rare procedure section",
        "context": context,
        "status": "pending",
        "source": "test",
    }
    append(store_path, existing)

    mock_client = MagicMock()
    agent = SectionSuggestorAgent(
        section_schema=section_schema,
        store_path=store_path,
        client=mock_client,
    )

    with patch("cpt_categorizer.agents.section_suggestor.log_agent_usage") as log_usage:
        result = agent.suggest_section(description, source="test")

    assert result is not None
    assert result["suggested_key"] == "rare_procedure"
    assert result["context"] == context
    mock_client.chat.completions.create.assert_not_called()
    log_usage.assert_called_once()
    call_kw = log_usage.call_args.kwargs
    assert call_kw["description"] == "section_suggest_store_hit"
    assert call_kw["prompt_tokens"] == 0
    assert call_kw["total_tokens"] == 0


def test_suggest_section_new_suggestion_appends_to_store(section_schema, store_path):
    """Empty store; mock LLM returns key + description; one new suggestion appended and returned."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response(
        "wound_care",
        "Wound care and dressing services",
    )
    agent = SectionSuggestorAgent(
        section_schema=section_schema,
        store_path=store_path,
        client=mock_client,
    )

    with patch("cpt_categorizer.agents.section_suggestor.log_agent_usage"):
        result = agent.suggest_section("Wound debridement and dressing", source="batch_1")

    assert result is not None
    assert result["type"] == "section"
    assert result["suggested_key"] == "wound_care"
    assert result["suggested_description"] == "Wound care and dressing services"
    assert result["status"] == "pending"
    assert result["source"] == "batch_1"
    assert result["context"] == _context_for_description("Wound debridement and dressing")
    assert "id" in result
    assert "created_at" in result

    stored = load(store_path)
    assert len(stored) == 1
    assert stored[0]["suggested_key"] == "wound_care"

    # Usage logged with non-zero tokens
    assert mock_client.chat.completions.create.called


def test_suggest_section_duplicate_key_returns_existing_no_append(section_schema, store_path):
    """Store already has suggestion with key 'foo'. Mock LLM returns 'foo'; return existing, do not append."""
    # Pre-populate with key "foo" (different context so we don't hit context-first path)
    other_context = hashlib.sha256(b"other description").hexdigest()
    append(
        store_path,
        {
            "type": "section",
            "suggested_key": "foo",
            "suggested_description": "Existing foo section",
            "context": other_context,
            "status": "pending",
            "source": "prior",
        },
    )
    initial_count = len(load(store_path))

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response("foo", "New foo description")
    agent = SectionSuggestorAgent(
        section_schema=section_schema,
        store_path=store_path,
        client=mock_client,
    )

    with patch("cpt_categorizer.agents.section_suggestor.log_agent_usage") as log_usage:
        result = agent.suggest_section("Some procedure that will suggest foo", source="batch_2")

    assert result is not None
    assert result["suggested_key"] == "foo"
    assert result["suggested_description"] == "Existing foo section"
    assert result["context"] == other_context
    # No second record appended
    assert len(load(store_path)) == initial_count
    log_usage.assert_called()
    # Last call should be duplicate_key with 0 tokens
    last_kw = log_usage.call_args.kwargs
    assert last_kw["description"] == "section_suggest_duplicate_key"
    assert last_kw["prompt_tokens"] == 0
    assert last_kw["total_tokens"] == 0


def test_context_for_description_stable():
    """Exported helper produces same context for same normalized description."""
    assert _context_for_description("  My CPT  ") == _context_for_description("my cpt")
    assert _context_for_description("My CPT") == _context_for_description("my cpt")
