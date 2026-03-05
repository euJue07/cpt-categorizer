"""Tests for SectionGovernorAgent: duplicate, store reuse, novel+LLM, empty pending."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cpt_categorizer.agents.section_governor import SectionGovernorAgent
from cpt_categorizer.suggestion_store import append, load


@pytest.fixture
def section_schema():
    return {
        "anesthesiology": {"description": "Anesthesia"},
        "imaging": {"description": "Imaging"},
    }


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "suggestions.json"


def test_resolve_key_in_schema_marks_duplicate_no_llm(section_schema, store_path):
    """One pending section suggestion with suggested_key = schema key; governor marks duplicate, no LLM."""
    append(
        store_path,
        {
            "type": "section",
            "suggested_key": "anesthesiology",
            "suggested_description": "Anesthesia services",
            "context": "ctx1",
            "status": "pending",
            "source": "test",
        },
    )
    stored = load(store_path)
    pending_id = stored[0]["id"]

    mock_client = MagicMock()
    agent = SectionGovernorAgent(
        section_schema=section_schema,
        store_path=store_path,
        client=mock_client,
    )

    with patch("cpt_categorizer.agents.section_governor.log_agent_usage") as log_usage:
        results = agent.resolve_pending_sections()

    assert len(results) == 1
    assert results[0]["status"] == "duplicate"
    mock_client.chat.completions.create.assert_not_called()
    log_usage.assert_called_once()
    call_kw = log_usage.call_args.kwargs
    assert call_kw["description"] == "section_governor_duplicate"
    assert call_kw["prompt_tokens"] == 0
    assert call_kw["total_tokens"] == 0

    # Store updated
    after = load(store_path)
    row = next(s for s in after if s["id"] == pending_id)
    assert row["status"] == "duplicate"


def test_resolve_same_key_accepted_in_store_reuses_no_llm(section_schema, store_path):
    """One pending suggestion; another record with same key has status accepted; governor reuses, no LLM."""
    append(
        store_path,
        {
            "type": "section",
            "suggested_key": "wound_care",
            "suggested_description": "Wound care (accepted)",
            "context": "ctx_old",
            "status": "accepted",
            "source": "prior",
        },
    )
    append(
        store_path,
        {
            "type": "section",
            "suggested_key": "wound_care",
            "suggested_description": "Wound care (pending)",
            "context": "ctx_new",
            "status": "pending",
            "source": "test",
        },
    )
    stored = load(store_path)
    pending_row = next(s for s in stored if s.get("status") == "pending")
    pending_id = pending_row["id"]

    mock_client = MagicMock()
    agent = SectionGovernorAgent(
        section_schema=section_schema,
        store_path=store_path,
        client=mock_client,
    )

    with patch("cpt_categorizer.agents.section_governor.log_agent_usage") as log_usage:
        results = agent.resolve_pending_sections()

    assert len(results) == 1
    assert results[0]["status"] == "accepted"
    mock_client.chat.completions.create.assert_not_called()
    log_usage.assert_called_once()
    call_kw = log_usage.call_args.kwargs
    assert call_kw["description"] == "section_governor_store_reuse"
    assert call_kw["prompt_tokens"] == 0

    after = load(store_path)
    row = next(s for s in after if s["id"] == pending_id)
    assert row["status"] == "accepted"


def test_resolve_same_key_rejected_in_store_reuses_no_llm(section_schema, store_path):
    """One pending suggestion; another record with same key has status rejected; governor reuses, no LLM."""
    append(
        store_path,
        {
            "type": "section",
            "suggested_key": "misc_other",
            "suggested_description": "Misc (rejected)",
            "context": "ctx_old",
            "status": "rejected",
            "source": "prior",
        },
    )
    append(
        store_path,
        {
            "type": "section",
            "suggested_key": "misc_other",
            "suggested_description": "Misc (pending)",
            "context": "ctx_new",
            "status": "pending",
            "source": "test",
        },
    )
    stored = load(store_path)
    pending_id = next(s for s in stored if s.get("status") == "pending")["id"]

    mock_client = MagicMock()
    agent = SectionGovernorAgent(
        section_schema=section_schema,
        store_path=store_path,
        client=mock_client,
    )

    with patch("cpt_categorizer.agents.section_governor.log_agent_usage") as log_usage:
        results = agent.resolve_pending_sections()

    assert len(results) == 1
    assert results[0]["status"] == "rejected"
    mock_client.chat.completions.create.assert_not_called()
    call_kw = log_usage.call_args.kwargs
    assert call_kw["description"] == "section_governor_store_reuse"

    after = load(store_path)
    row = next(s for s in after if s["id"] == pending_id)
    assert row["status"] == "rejected"


def test_resolve_novel_suggestion_calls_llm_updates_status(section_schema, store_path):
    """One pending suggestion with key not in schema and no accepted/rejected same-key; LLM called, status updated."""
    append(
        store_path,
        {
            "type": "section",
            "suggested_key": "wound_care",
            "suggested_description": "Wound care and dressing",
            "context": "ctx1",
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

    agent = SectionGovernorAgent(
        section_schema=section_schema,
        store_path=store_path,
        client=mock_client,
    )

    with patch("cpt_categorizer.agents.section_governor.log_agent_usage") as log_usage:
        results = agent.resolve_pending_sections()

    assert len(results) == 1
    assert results[0]["status"] == "accepted"
    mock_client.chat.completions.create.assert_called_once()
    call_kw = log_usage.call_args.kwargs
    assert call_kw["description"] == "section_governor"
    assert call_kw["prompt_tokens"] == 20
    assert call_kw["completion_tokens"] == 5
    assert call_kw["total_tokens"] == 25

    after = load(store_path)
    row = next(s for s in after if s["id"] == pending_id)
    assert row["status"] == "accepted"


def test_resolve_novel_suggestion_llm_reject(section_schema, store_path):
    """Novel suggestion; LLM returns accept=False; status set to rejected."""
    append(
        store_path,
        {
            "type": "section",
            "suggested_key": "vague_section",
            "suggested_description": "Vague",
            "context": "ctx1",
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

    agent = SectionGovernorAgent(
        section_schema=section_schema,
        store_path=store_path,
        client=mock_client,
    )

    with patch("cpt_categorizer.agents.section_governor.log_agent_usage"):
        results = agent.resolve_pending_sections()

    assert len(results) == 1
    assert results[0]["status"] == "rejected"
    assert load(store_path)[0]["status"] == "rejected"


def test_resolve_empty_pending_returns_without_llm(section_schema, store_path):
    """No pending section suggestions; resolve returns empty list, no LLM call."""
    append(
        store_path,
        {
            "type": "section",
            "suggested_key": "wound_care",
            "suggested_description": "Wound",
            "context": "ctx",
            "status": "accepted",
            "source": "prior",
        },
    )

    mock_client = MagicMock()
    agent = SectionGovernorAgent(
        section_schema=section_schema,
        store_path=store_path,
        client=mock_client,
    )

    with patch("cpt_categorizer.agents.section_governor.log_agent_usage") as log_usage:
        results = agent.resolve_pending_sections()

    assert results == []
    mock_client.chat.completions.create.assert_not_called()
    log_usage.assert_not_called()


def test_resolve_multiple_pending_mixed_paths(section_schema, store_path):
    """Two pending: one duplicate (key in schema), one novel; only novel gets LLM."""
    append(
        store_path,
        {
            "type": "section",
            "suggested_key": "imaging",
            "suggested_description": "Imaging (duplicate)",
            "context": "c1",
            "status": "pending",
            "source": "test",
        },
    )
    append(
        store_path,
        {
            "type": "section",
            "suggested_key": "cardiology",
            "suggested_description": "Cardiology",
            "context": "c2",
            "status": "pending",
            "source": "test",
        },
    )
    stored = load(store_path)
    id_dup = next(s for s in stored if s["suggested_key"] == "imaging")["id"]
    id_novel = next(s for s in stored if s["suggested_key"] == "cardiology")["id"]

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.function_call.arguments = json.dumps({"accept": True})
    mock_response.usage.prompt_tokens = 15
    mock_response.usage.completion_tokens = 2
    mock_response.usage.total_tokens = 17
    mock_response.model = "gpt-4o"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    agent = SectionGovernorAgent(
        section_schema=section_schema,
        store_path=store_path,
        client=mock_client,
    )

    with patch("cpt_categorizer.agents.section_governor.log_agent_usage") as log_usage:
        results = agent.resolve_pending_sections()

    assert len(results) == 2
    by_id = {r["id"]: r for r in results}
    assert by_id[id_dup]["status"] == "duplicate"
    assert by_id[id_novel]["status"] == "accepted"
    assert mock_client.chat.completions.create.call_count == 1
    assert log_usage.call_count == 2
    descriptions = [c.kwargs["description"] for c in log_usage.call_args_list]
    assert "section_governor_duplicate" in descriptions
    assert "section_governor" in descriptions
