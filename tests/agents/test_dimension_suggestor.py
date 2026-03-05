"""Tests for DimensionSuggestorAgent: invalid inputs, store hit, new existing_dim, new new_dim, duplicate key."""

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cpt_categorizer.agents.dimension_suggestor import DimensionSuggestorAgent
from cpt_categorizer.agents.dimension_suggestor import _context_for_dimension
from cpt_categorizer.suggestion_store import append, load


def _context_for_dimension_test(section: str, subsection: str, dim_key: str, text: str) -> str:
    """Same as agent's internal context computation for test setup."""
    normalized = text.strip().lower()
    desc_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{section}:{subsection}:{dim_key}:{desc_hash}"


def _make_mock_response(suggested_description: str):
    """Build a mock OpenAI response with function_call.arguments for suggest_dimension_description."""
    args = json.dumps({"suggested_description": suggested_description})
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
def dimension_schema():
    return {
        "analyte": {"description": "Analyte", "values": ["glucose", "creatinine"]},
        "anatomic_site": {"description": "Anatomic site", "values": ["abdomen", "chest"]},
    }


@pytest.fixture
def subsection_schema():
    return {
        "imaging": {
            "ct_scan": {"description": "CT scan", "dimensions": ["anatomic_site", "contrast"]},
            "ultrasound": {"description": "Ultrasound", "dimensions": ["anatomic_site"]},
        },
    }


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "suggestions.json"


def test_suggest_dimensions_returns_empty_for_invalid_inputs(dimension_schema, subsection_schema, store_path):
    agent = DimensionSuggestorAgent(
        dimension_schema=dimension_schema,
        subsection_schema=subsection_schema,
        store_path=store_path,
    )
    desc = "Some CPT procedure"
    proposed_empty = {"existing_dimensions": {}, "new_dimensions": {}}
    assert agent.suggest_dimensions("imaging", "ct_scan", desc, proposed_empty) == []
    assert agent.suggest_dimensions("imaging", "ct_scan", "", {"existing_dimensions": {"x": [{"value": "y"}]}}) == []
    assert agent.suggest_dimensions("imaging", "ct_scan", "   ", {"new_dimensions": {"view": [{"value": "coronal"}]}}) == []
    assert agent.suggest_dimensions("", "ct_scan", desc, {"new_dimensions": {"view": [{"value": "coronal"}]}}) == []
    assert agent.suggest_dimensions("imaging", "", desc, {"new_dimensions": {"view": [{"value": "coronal"}]}}) == []
    assert agent.suggest_dimensions("others", "ct_scan", desc, {"new_dimensions": {"view": [{"value": "coronal"}]}}) == []
    assert agent.suggest_dimensions("imaging", "others", desc, {"new_dimensions": {"view": [{"value": "coronal"}]}}) == []


def test_suggest_dimensions_store_hit_by_context(dimension_schema, subsection_schema, store_path):
    """Pre-populate store with dimension suggestion for context; suggest_dimensions returns it and skips LLM."""
    section = "imaging"
    subsection = "ct_scan"
    dimension_key = "view"
    description = "CT abdomen with coronal view"
    proposed = {"existing_dimensions": {}, "new_dimensions": {dimension_key: [{"value": "coronal", "confidence": 0.9}]}}
    context = _context_for_dimension_test(section, subsection, dimension_key, description)
    existing = {
        "type": "dimension",
        "suggested_key": dimension_key,
        "suggested_values": ["coronal"],
        "suggested_description": "Imaging view orientation",
        "context": context,
        "parent_section": section,
        "parent_subsection": subsection,
        "status": "pending",
        "source": "test",
    }
    append(store_path, existing)

    mock_client = MagicMock()
    agent = DimensionSuggestorAgent(
        dimension_schema=dimension_schema,
        subsection_schema=subsection_schema,
        store_path=store_path,
        client=mock_client,
    )

    with patch("cpt_categorizer.agents.dimension_suggestor.log_agent_usage") as log_usage:
        result = agent.suggest_dimensions(section, subsection, description, proposed, source="test")

    assert len(result) == 1
    assert result[0]["suggested_key"] == dimension_key
    assert result[0]["context"] == context
    assert result[0]["parent_section"] == section
    assert result[0]["parent_subsection"] == subsection
    mock_client.chat.completions.create.assert_not_called()
    log_usage.assert_called_once()
    call_kw = log_usage.call_args.kwargs
    assert call_kw["description"] == "dimension_suggest_store_hit"
    assert call_kw["prompt_tokens"] == 0
    assert call_kw["total_tokens"] == 0


def test_suggest_dimensions_new_existing_dimension_no_llm(dimension_schema, subsection_schema, store_path):
    """Proposed has existing_dimensions only; no store match; one record appended, no LLM call."""
    section = "imaging"
    subsection = "ct_scan"
    description = "CT with triple-phase protocol"
    proposed = {
        "existing_dimensions": {
            "contrast": [{"value": "triple_phase", "confidence": 0.8}],
        },
        "new_dimensions": {},
    }

    mock_client = MagicMock()
    agent = DimensionSuggestorAgent(
        dimension_schema=dimension_schema,
        subsection_schema=subsection_schema,
        store_path=store_path,
        client=mock_client,
    )

    with patch("cpt_categorizer.agents.dimension_suggestor.log_agent_usage"):
        result = agent.suggest_dimensions(section, subsection, description, proposed, source="batch_1")

    assert len(result) == 1
    assert result[0]["type"] == "dimension"
    assert result[0]["suggested_key"] == "contrast"
    assert result[0]["suggested_values"] == ["triple_phase"]
    assert "suggested_description" not in result[0] or result[0].get("suggested_description") == ""
    assert result[0]["status"] == "pending"
    assert result[0]["parent_section"] == section
    assert result[0]["parent_subsection"] == subsection
    assert "id" in result[0]
    assert "created_at" in result[0]

    stored = load(store_path)
    assert len(stored) == 1
    assert stored[0]["suggested_key"] == "contrast"
    assert stored[0]["suggested_values"] == ["triple_phase"]

    mock_client.chat.completions.create.assert_not_called()


def test_suggest_dimensions_new_dimension_calls_llm(dimension_schema, subsection_schema, store_path):
    """Proposed has new_dimensions; no store match; mock LLM returns suggested_description; one record appended."""
    section = "imaging"
    subsection = "ct_scan"
    description = "CT with coronal view"
    proposed = {
        "existing_dimensions": {},
        "new_dimensions": {"view": [{"value": "coronal", "confidence": 0.9}]},
    }

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response(
        "Imaging plane or view orientation (e.g. axial, coronal, sagittal)."
    )
    agent = DimensionSuggestorAgent(
        dimension_schema=dimension_schema,
        subsection_schema=subsection_schema,
        store_path=store_path,
        client=mock_client,
    )

    with patch("cpt_categorizer.agents.dimension_suggestor.log_agent_usage"):
        result = agent.suggest_dimensions(section, subsection, description, proposed, source="batch_1")

    assert len(result) == 1
    assert result[0]["type"] == "dimension"
    assert result[0]["suggested_key"] == "view"
    assert result[0]["suggested_values"] == ["coronal"]
    assert result[0]["suggested_description"] == "Imaging plane or view orientation (e.g. axial, coronal, sagittal)."
    assert result[0]["status"] == "pending"
    assert result[0]["parent_section"] == section
    assert result[0]["parent_subsection"] == subsection

    stored = load(store_path)
    assert len(stored) == 1
    assert stored[0]["suggested_key"] == "view"
    assert stored[0]["suggested_description"] == "Imaging plane or view orientation (e.g. axial, coronal, sagittal)."

    mock_client.chat.completions.create.assert_called_once()


def test_suggest_dimensions_duplicate_key_returns_existing(dimension_schema, subsection_schema, store_path):
    """Store already has dimension suggestion with key 'view' under same section/subsection. Mock LLM returns description; we detect duplicate and return existing, do not append."""
    section = "imaging"
    subsection = "ct_scan"
    dimension_key = "view"
    other_context = _context_for_dimension_test(section, subsection, dimension_key, "other description")
    append(
        store_path,
        {
            "type": "dimension",
            "suggested_key": dimension_key,
            "suggested_values": ["sagittal"],
            "suggested_description": "Existing view dimension",
            "context": other_context,
            "parent_section": section,
            "parent_subsection": subsection,
            "status": "pending",
            "source": "prior",
        },
    )
    initial_count = len(load(store_path))

    proposed = {
        "existing_dimensions": {},
        "new_dimensions": {dimension_key: [{"value": "coronal", "confidence": 0.9}]},
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response("New view description")
    agent = DimensionSuggestorAgent(
        dimension_schema=dimension_schema,
        subsection_schema=subsection_schema,
        store_path=store_path,
        client=mock_client,
    )

    with patch("cpt_categorizer.agents.dimension_suggestor.log_agent_usage") as log_usage:
        result = agent.suggest_dimensions(
            section,
            subsection,
            "CT with coronal view",
            proposed,
            source="batch_2",
        )

    assert len(result) == 1
    assert result[0]["suggested_key"] == dimension_key
    assert result[0]["suggested_description"] == "Existing view dimension"
    assert result[0]["parent_section"] == section
    assert result[0]["parent_subsection"] == subsection
    assert len(load(store_path)) == initial_count

    log_usage.assert_called()
    last_kw = log_usage.call_args.kwargs
    assert last_kw["description"] == "dimension_suggest_duplicate_key"
    assert last_kw["prompt_tokens"] == 0
    assert last_kw["total_tokens"] == 0


def test_context_for_dimension_stable():
    """Context is stable for same section + subsection + dimension_key + normalized description."""
    assert _context_for_dimension("imaging", "ct_scan", "view", "  My CPT  ") == _context_for_dimension(
        "imaging", "ct_scan", "view", "my cpt"
    )
    assert _context_for_dimension("imaging", "ct_scan", "view", "x") != _context_for_dimension(
        "imaging", "ct_scan", "contrast", "x"
    )
    assert _context_for_dimension("imaging", "ct_scan", "view", "x") != _context_for_dimension(
        "lab", "chemistry", "view", "x"
    )
