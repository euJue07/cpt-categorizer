"""Tests for DimensionGovernorAgent: duplicate, store reuse, novel+LLM, empty pending."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cpt_categorizer.agents.dimension_governor import DimensionGovernorAgent
from cpt_categorizer.suggestion_store import append, load


@pytest.fixture
def dimension_schema():
    return {
        "analyte": {
            "description": "Analyte measured",
            "values": ["glucose", "creatinine", "hemoglobin"],
        },
        "method": {
            "description": "Method used",
            "values": ["pcr", "elisa"],
        },
    }


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "suggestions.json"


def test_resolve_key_in_schema_all_values_in_schema_marks_duplicate_no_llm(dimension_schema, store_path):
    """Pending dimension with suggested_key in schema and all suggested_values already in schema -> duplicate, no LLM."""
    append(
        store_path,
        {
            "type": "dimension",
            "suggested_key": "analyte",
            "suggested_values": ["glucose", "creatinine"],
            "context": "ctx1",
            "parent_section": "lab",
            "parent_subsection": "chemistry",
            "status": "pending",
            "source": "test",
        },
    )
    stored = load(store_path)
    pending_id = stored[0]["id"]

    mock_client = MagicMock()
    agent = DimensionGovernorAgent(
        dimension_schema=dimension_schema,
        store_path=store_path,
        client=mock_client,
    )

    with patch("cpt_categorizer.agents.dimension_governor.log_agent_usage") as log_usage:
        results = agent.resolve_pending_dimensions()

    assert len(results) == 1
    assert results[0]["status"] == "duplicate"
    mock_client.chat.completions.create.assert_not_called()
    log_usage.assert_called_once()
    call_kw = log_usage.call_args.kwargs
    assert call_kw["description"] == "dimension_governor_duplicate"
    assert call_kw["prompt_tokens"] == 0
    assert call_kw["total_tokens"] == 0

    after = load(store_path)
    row = next(s for s in after if s["id"] == pending_id)
    assert row["status"] == "duplicate"


def test_resolve_key_in_schema_new_values_not_duplicate(dimension_schema, store_path):
    """suggested_key in schema but at least one suggested_value not in schema -> not duplicate; LLM called."""
    append(
        store_path,
        {
            "type": "dimension",
            "suggested_key": "analyte",
            "suggested_values": ["glucose", "new_analyte_xyz"],
            "context": "ctx1",
            "parent_section": "lab",
            "parent_subsection": "chemistry",
            "status": "pending",
            "source": "test",
        },
    )
    stored = load(store_path)
    pending_id = stored[0]["id"]

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.function_call.arguments = json.dumps({"accept": True})
    mock_response.usage.prompt_tokens = 22
    mock_response.usage.completion_tokens = 4
    mock_response.usage.total_tokens = 26
    mock_response.model = "gpt-4o"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    agent = DimensionGovernorAgent(
        dimension_schema=dimension_schema,
        store_path=store_path,
        client=mock_client,
    )

    with patch("cpt_categorizer.agents.dimension_governor.log_agent_usage") as log_usage:
        results = agent.resolve_pending_dimensions()

    assert len(results) == 1
    assert results[0]["status"] == "accepted"
    mock_client.chat.completions.create.assert_called_once()
    call_kw = log_usage.call_args.kwargs
    assert call_kw["description"] == "dimension_governor"

    after = load(store_path)
    row = next(s for s in after if s["id"] == pending_id)
    assert row["status"] == "accepted"


def test_resolve_new_dimension_key_not_duplicate(dimension_schema, store_path):
    """suggested_key not in schema (new dimension key) -> not duplicate from schema; LLM called."""
    append(
        store_path,
        {
            "type": "dimension",
            "suggested_key": "sample_source",
            "suggested_values": ["venous", "arterial"],
            "suggested_description": "Source of sample",
            "context": "ctx1",
            "parent_section": "lab",
            "parent_subsection": "chemistry",
            "status": "pending",
            "source": "test",
        },
    )
    pending_id = load(store_path)[0]["id"]

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.function_call.arguments = json.dumps({"accept": True})
    mock_response.usage.prompt_tokens = 25
    mock_response.usage.completion_tokens = 3
    mock_response.usage.total_tokens = 28
    mock_response.model = "gpt-4o"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    agent = DimensionGovernorAgent(
        dimension_schema=dimension_schema,
        store_path=store_path,
        client=mock_client,
    )

    with patch("cpt_categorizer.agents.dimension_governor.log_agent_usage") as log_usage:
        results = agent.resolve_pending_dimensions()

    assert len(results) == 1
    assert results[0]["status"] == "accepted"
    mock_client.chat.completions.create.assert_called_once()
    assert log_usage.call_args.kwargs["description"] == "dimension_governor"
    assert load(store_path)[0]["status"] == "accepted"


def test_resolve_same_key_parent_section_subsection_accepted_in_store_reuses_no_llm(
    dimension_schema, store_path
):
    """One pending; another record with same key+parent_section+parent_subsection has status accepted; reuse, no LLM."""
    append(
        store_path,
        {
            "type": "dimension",
            "suggested_key": "sample_type",
            "suggested_values": ["serum", "plasma"],
            "context": "ctx_old",
            "parent_section": "lab",
            "parent_subsection": "chemistry",
            "status": "accepted",
            "source": "prior",
        },
    )
    append(
        store_path,
        {
            "type": "dimension",
            "suggested_key": "sample_type",
            "suggested_values": ["serum"],
            "context": "ctx_new",
            "parent_section": "lab",
            "parent_subsection": "chemistry",
            "status": "pending",
            "source": "test",
        },
    )
    stored = load(store_path)
    pending_id = next(s for s in stored if s.get("status") == "pending")["id"]

    mock_client = MagicMock()
    agent = DimensionGovernorAgent(
        dimension_schema=dimension_schema,
        store_path=store_path,
        client=mock_client,
    )

    with patch("cpt_categorizer.agents.dimension_governor.log_agent_usage") as log_usage:
        results = agent.resolve_pending_dimensions()

    assert len(results) == 1
    assert results[0]["status"] == "accepted"
    mock_client.chat.completions.create.assert_not_called()
    call_kw = log_usage.call_args.kwargs
    assert call_kw["description"] == "dimension_governor_store_reuse"
    assert call_kw["prompt_tokens"] == 0

    after = load(store_path)
    row = next(s for s in after if s["id"] == pending_id)
    assert row["status"] == "accepted"


def test_resolve_same_key_parent_section_subsection_rejected_in_store_reuses_no_llm(
    dimension_schema, store_path
):
    """One pending; another record with same key+parent_section+parent_subsection has status rejected; reuse, no LLM."""
    append(
        store_path,
        {
            "type": "dimension",
            "suggested_key": "redundant_dim",
            "suggested_values": ["x", "y"],
            "context": "ctx_old",
            "parent_section": "imaging",
            "parent_subsection": "mri",
            "status": "rejected",
            "source": "prior",
        },
    )
    append(
        store_path,
        {
            "type": "dimension",
            "suggested_key": "redundant_dim",
            "suggested_values": ["z"],
            "context": "ctx_new",
            "parent_section": "imaging",
            "parent_subsection": "mri",
            "status": "pending",
            "source": "test",
        },
    )
    stored = load(store_path)
    pending_id = next(s for s in stored if s.get("status") == "pending")["id"]

    mock_client = MagicMock()
    agent = DimensionGovernorAgent(
        dimension_schema=dimension_schema,
        store_path=store_path,
        client=mock_client,
    )

    with patch("cpt_categorizer.agents.dimension_governor.log_agent_usage") as log_usage:
        results = agent.resolve_pending_dimensions()

    assert len(results) == 1
    assert results[0]["status"] == "rejected"
    mock_client.chat.completions.create.assert_not_called()
    call_kw = log_usage.call_args.kwargs
    assert call_kw["description"] == "dimension_governor_store_reuse"

    after = load(store_path)
    row = next(s for s in after if s["id"] == pending_id)
    assert row["status"] == "rejected"


def test_resolve_novel_suggestion_calls_llm_updates_status(dimension_schema, store_path):
    """One pending dimension not duplicate and no accepted/rejected for same key+parent; LLM called, status updated."""
    append(
        store_path,
        {
            "type": "dimension",
            "suggested_key": "analyte",
            "suggested_values": ["new_analyte_abc"],
            "context": "ctx1",
            "parent_section": "lab",
            "parent_subsection": "chemistry",
            "status": "pending",
            "source": "test",
        },
    )
    stored = load(store_path)
    pending_id = stored[0]["id"]

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.function_call.arguments = json.dumps({"accept": True})
    mock_response.usage.prompt_tokens = 20
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 25
    mock_response.model = "gpt-4o"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    agent = DimensionGovernorAgent(
        dimension_schema=dimension_schema,
        store_path=store_path,
        client=mock_client,
    )

    with patch("cpt_categorizer.agents.dimension_governor.log_agent_usage") as log_usage:
        results = agent.resolve_pending_dimensions()

    assert len(results) == 1
    assert results[0]["status"] == "accepted"
    mock_client.chat.completions.create.assert_called_once()
    call_kw = log_usage.call_args.kwargs
    assert call_kw["description"] == "dimension_governor"
    assert call_kw["prompt_tokens"] == 20
    assert call_kw["completion_tokens"] == 5
    assert call_kw["total_tokens"] == 25

    after = load(store_path)
    row = next(s for s in after if s["id"] == pending_id)
    assert row["status"] == "accepted"


def test_resolve_novel_suggestion_llm_reject(dimension_schema, store_path):
    """Novel suggestion; LLM returns accept=False; status set to rejected."""
    append(
        store_path,
        {
            "type": "dimension",
            "suggested_key": "vague_dim",
            "suggested_values": ["vague_value"],
            "context": "ctx1",
            "parent_section": "lab",
            "parent_subsection": "chemistry",
            "status": "pending",
            "source": "test",
        },
    )
    pending_id = load(store_path)[0]["id"]

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.function_call.arguments = json.dumps({"accept": False})
    mock_response.usage.prompt_tokens = 18
    mock_response.usage.completion_tokens = 3
    mock_response.usage.total_tokens = 21
    mock_response.model = "gpt-4o"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    agent = DimensionGovernorAgent(
        dimension_schema=dimension_schema,
        store_path=store_path,
        client=mock_client,
    )

    with patch("cpt_categorizer.agents.dimension_governor.log_agent_usage"):
        results = agent.resolve_pending_dimensions()

    assert len(results) == 1
    assert results[0]["status"] == "rejected"
    assert load(store_path)[0]["status"] == "rejected"


def test_resolve_empty_pending_returns_without_llm(dimension_schema, store_path):
    """No pending dimension suggestions; resolve returns empty list, no LLM call."""
    append(
        store_path,
        {
            "type": "dimension",
            "suggested_key": "analyte",
            "suggested_values": ["glucose"],
            "context": "ctx",
            "parent_section": "lab",
            "parent_subsection": "chemistry",
            "status": "accepted",
            "source": "prior",
        },
    )

    mock_client = MagicMock()
    agent = DimensionGovernorAgent(
        dimension_schema=dimension_schema,
        store_path=store_path,
        client=mock_client,
    )

    with patch("cpt_categorizer.agents.dimension_governor.log_agent_usage") as log_usage:
        results = agent.resolve_pending_dimensions()

    assert results == []
    mock_client.chat.completions.create.assert_not_called()
    log_usage.assert_not_called()


def test_resolve_multiple_pending_mixed_paths(dimension_schema, store_path):
    """Two pending: one duplicate (key+all values in schema), one novel; only novel gets LLM."""
    append(
        store_path,
        {
            "type": "dimension",
            "suggested_key": "analyte",
            "suggested_values": ["glucose", "creatinine"],
            "context": "c1",
            "parent_section": "lab",
            "parent_subsection": "chemistry",
            "status": "pending",
            "source": "test",
        },
    )
    append(
        store_path,
        {
            "type": "dimension",
            "suggested_key": "method",
            "suggested_values": ["new_method_xyz"],
            "context": "c2",
            "parent_section": "lab",
            "parent_subsection": "immunology",
            "status": "pending",
            "source": "test",
        },
    )
    stored = load(store_path)
    id_dup = next(s for s in stored if s["suggested_key"] == "analyte")["id"]
    id_novel = next(s for s in stored if s["suggested_key"] == "method")["id"]

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.function_call.arguments = json.dumps({"accept": True})
    mock_response.usage.prompt_tokens = 15
    mock_response.usage.completion_tokens = 2
    mock_response.usage.total_tokens = 17
    mock_response.model = "gpt-4o"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    agent = DimensionGovernorAgent(
        dimension_schema=dimension_schema,
        store_path=store_path,
        client=mock_client,
    )

    with patch("cpt_categorizer.agents.dimension_governor.log_agent_usage") as log_usage:
        results = agent.resolve_pending_dimensions()

    assert len(results) == 2
    by_id = {r["id"]: r for r in results}
    assert by_id[id_dup]["status"] == "duplicate"
    assert by_id[id_novel]["status"] == "accepted"
    assert mock_client.chat.completions.create.call_count == 1
    assert log_usage.call_count == 2
    descriptions = [c.kwargs["description"] for c in log_usage.call_args_list]
    assert "dimension_governor_duplicate" in descriptions
    assert "dimension_governor" in descriptions
