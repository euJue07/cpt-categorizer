"""Tests for SubsectionSuggestorAgent: store hit, new suggestion, duplicate key."""

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cpt_categorizer.agents.subsection_suggestor import SubsectionSuggestorAgent
from cpt_categorizer.agents.subsection_suggestor import _context_for_subsection
from cpt_categorizer.suggestion_store import append, load


def _context_for_subsection_test(section: str, text: str) -> str:
    """Same as agent's internal context computation for test setup."""
    normalized = text.strip().lower()
    desc_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{section}:{desc_hash}"


def _make_mock_response(suggested_key: str, suggested_description: str):
    """Build a mock OpenAI response with function_call.arguments for suggest_subsection."""
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
    return {
        "anesthesiology": {
            "description": "Anesthesia",
            "subsections": ["general_anesthesia", "regional_anesthesia"],
        },
        "dental": {
            "description": "Dental",
            "subsections": ["dental_diagnostics", "preventive_dental"],
        },
    }


@pytest.fixture
def subsection_schema():
    return {
        "anesthesiology": {
            "general_anesthesia": {"description": "General anesthesia"},
            "regional_anesthesia": {"description": "Regional anesthesia"},
        },
        "dental": {
            "dental_diagnostics": {"description": "Dental diagnostics"},
            "preventive_dental": {"description": "Preventive dental"},
        },
    }


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "suggestions.json"


def test_suggest_subsection_returns_none_for_invalid_inputs(section_schema, subsection_schema, store_path):
    agent = SubsectionSuggestorAgent(
        section_schema=section_schema,
        subsection_schema=subsection_schema,
        store_path=store_path,
    )
    assert agent.suggest_subsection("dental", "") is None
    assert agent.suggest_subsection("dental", "   ") is None
    assert agent.suggest_subsection("", "Some procedure") is None
    assert agent.suggest_subsection("   ", "Some procedure") is None
    assert agent.suggest_subsection("others", "Some procedure") is None
    assert agent.suggest_subsection("OTHERS", "Some procedure") is None


def test_suggest_subsection_store_hit_by_context(section_schema, subsection_schema, store_path):
    """Pre-populate store with suggestion for context; suggest_subsection returns it and skips LLM."""
    section = "dental"
    description = "Rare dental procedure XYZ"
    context = _context_for_subsection_test(section, description)
    existing = {
        "type": "subsection",
        "suggested_key": "rare_dental",
        "suggested_description": "Rare dental subsection",
        "context": context,
        "parent_section": section,
        "status": "pending",
        "source": "test",
    }
    append(store_path, existing)

    mock_client = MagicMock()
    agent = SubsectionSuggestorAgent(
        section_schema=section_schema,
        subsection_schema=subsection_schema,
        store_path=store_path,
        client=mock_client,
    )

    with patch("cpt_categorizer.agents.subsection_suggestor.log_agent_usage") as log_usage:
        result = agent.suggest_subsection(section, description, source="test")

    assert result is not None
    assert result["suggested_key"] == "rare_dental"
    assert result["context"] == context
    assert result["parent_section"] == section
    mock_client.chat.completions.create.assert_not_called()
    log_usage.assert_called_once()
    call_kw = log_usage.call_args.kwargs
    assert call_kw["description"] == "subsection_suggest_store_hit"
    assert call_kw["prompt_tokens"] == 0
    assert call_kw["total_tokens"] == 0


def test_suggest_subsection_new_suggestion_appends_to_store(section_schema, subsection_schema, store_path):
    """Empty store; mock LLM returns key + description; one new suggestion appended and returned."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response(
        "oral_surgery",
        "Oral surgery procedures",
    )
    agent = SubsectionSuggestorAgent(
        section_schema=section_schema,
        subsection_schema=subsection_schema,
        store_path=store_path,
        client=mock_client,
    )

    with patch("cpt_categorizer.agents.subsection_suggestor.log_agent_usage"):
        result = agent.suggest_subsection(
            "dental",
            "Tooth extraction and surgical removal",
            source="batch_1",
        )

    assert result is not None
    assert result["type"] == "subsection"
    assert result["suggested_key"] == "oral_surgery"
    assert result["suggested_description"] == "Oral surgery procedures"
    assert result["status"] == "pending"
    assert result["source"] == "batch_1"
    assert result["parent_section"] == "dental"
    assert result["context"] == _context_for_subsection("dental", "Tooth extraction and surgical removal")
    assert "id" in result
    assert "created_at" in result

    stored = load(store_path)
    assert len(stored) == 1
    assert stored[0]["suggested_key"] == "oral_surgery"
    assert stored[0]["parent_section"] == "dental"

    assert mock_client.chat.completions.create.called


def test_suggest_subsection_duplicate_key_returns_existing_no_append(section_schema, subsection_schema, store_path):
    """Store already has subsection suggestion with key 'foo' under same section. Mock LLM returns 'foo'; return existing, do not append."""
    section = "dental"
    # Different context so store-hit-by-context doesn't fire
    other_context = f"{section}:{hashlib.sha256(b'other description').hexdigest()}"
    append(
        store_path,
        {
            "type": "subsection",
            "suggested_key": "foo",
            "suggested_description": "Existing foo subsection",
            "context": other_context,
            "parent_section": section,
            "status": "pending",
            "source": "prior",
        },
    )
    initial_count = len(load(store_path))

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response("foo", "New foo subsection")
    agent = SubsectionSuggestorAgent(
        section_schema=section_schema,
        subsection_schema=subsection_schema,
        store_path=store_path,
        client=mock_client,
    )

    with patch("cpt_categorizer.agents.subsection_suggestor.log_agent_usage") as log_usage:
        result = agent.suggest_subsection(
            section,
            "Some procedure that will suggest foo",
            source="batch_2",
        )

    assert result is not None
    assert result["suggested_key"] == "foo"
    assert result["suggested_description"] == "Existing foo subsection"
    assert result["parent_section"] == section
    assert len(load(store_path)) == initial_count
    log_usage.assert_called()
    last_kw = log_usage.call_args.kwargs
    assert last_kw["description"] == "subsection_suggest_duplicate_key"
    assert last_kw["prompt_tokens"] == 0
    assert last_kw["total_tokens"] == 0


def test_context_for_subsection_stable():
    """Context is stable for same section + normalized description."""
    assert _context_for_subsection("dental", "  My CPT  ") == _context_for_subsection("dental", "my cpt")
    assert _context_for_subsection("dental", "My CPT") == _context_for_subsection("dental", "my cpt")
    assert _context_for_subsection("dental", "x") != _context_for_subsection("anesthesiology", "x")
