import json
from unittest.mock import patch

import pytest

from cpt_categorizer.agents.compliance import SchemaComplianceAgent
from cpt_categorizer.agents.normalizer import NormalizerAgent
from cpt_categorizer.agents.parsing import ParsingAgent
from cpt_categorizer.agents.tagging import DimensionTaggingAgent
from cpt_categorizer.agents.tagging import SectionTaggingAgent
from cpt_categorizer.agents.tagging import SubsectionTaggingAgent
from cpt_categorizer.agents.tagging import TaggingAgent
from cpt_categorizer.pipeline import process_row
from cpt_categorizer.schema_contract import load_schema_contract
from cpt_categorizer.tagging_cache import TaggingCache


class MockResponse:
    def __init__(self, arguments_json: str):
        self.choices = [
            type(
                "Choice",
                (),
                {
                    "message": type(
                        "Message",
                        (),
                        {
                            "function_call": type(
                                "FuncCall", (), {"arguments": arguments_json}
                            )()
                        },
                    )()
                },
            )()
        ]
        self.usage = type(
            "usage",
            (),
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )()
        self.model = "gpt-4o"


def build_mock_response(obj: dict | str) -> MockResponse:
    if isinstance(obj, dict):
        return MockResponse(json.dumps(obj))
    return MockResponse(obj)


@pytest.fixture
def schema_contract():
    return load_schema_contract(use_cache=False)


@pytest.fixture
def tagging_agent(schema_contract):
    return TaggingAgent(
        section_schema=schema_contract.sections,
        subsection_schema=schema_contract.subsections,
        dimension_schema=schema_contract.dimensions,
    )


@pytest.mark.schema_validation
def test_schema_contract_loads_and_has_version(schema_contract):
    assert schema_contract.sections
    assert schema_contract.subsections
    assert schema_contract.dimensions
    assert isinstance(schema_contract.version, str)
    assert len(schema_contract.version) == 12


@pytest.mark.schema_validation
def test_schema_contract_cross_references_are_valid(schema_contract):
    for section, section_spec in schema_contract.sections.items():
        for subsection in section_spec.get("subsections", []):
            assert subsection in schema_contract.subsections[section]

    for section, subsection_map in schema_contract.subsections.items():
        for subsection_spec in subsection_map.values():
            for dimension in subsection_spec.get("dimensions", []):
                assert dimension in schema_contract.dimensions


@pytest.mark.generate_tags
@patch.object(DimensionTaggingAgent, "_call_openai_completion")
@patch.object(SubsectionTaggingAgent, "_call_openai_completion")
@patch.object(SectionTaggingAgent, "_call_openai_completion")
def test_generate_tags_returns_dimensions_contract(
    mock_section_call, mock_subsection_call, mock_dimension_call, tagging_agent
):
    mock_section_call.side_effect = [
        build_mock_response(
            {"sections": [{"section": "imaging", "confidence": 0.95}]}
        ),
    ]
    mock_subsection_call.side_effect = [
        build_mock_response(
            {"subsections": [{"subsection": "xray", "confidence": 0.9}]}
        ),
    ]
    mock_dimension_call.side_effect = [
        build_mock_response(
            {
                "actual": {"contrast": [{"value": "with_contrast", "confidence": 0.9}]},
                "proposed": {
                    "existing_dimensions": {},
                    "new_dimensions": {},
                },
            }
        ),
    ]

    tags = tagging_agent.generate_tags("Chest X-ray with contrast")
    assert len(tags) == 1
    tag = tags[0]
    assert set(tag.keys()) == {"section", "subsection", "confidence", "dimensions"}
    assert set(tag["dimensions"].keys()) == {"actual", "proposed"}
    assert set(tag["dimensions"]["proposed"].keys()) == {
        "existing_dimensions",
        "new_dimensions",
    }


@pytest.mark.generate_tags
@patch.object(DimensionTaggingAgent, "_call_openai_completion")
@patch.object(SubsectionTaggingAgent, "_call_openai_completion")
@patch.object(SectionTaggingAgent, "_call_openai_completion")
def test_generate_tags_filters_low_dimension_confidence(
    mock_section_call, mock_subsection_call, mock_dimension_call, tagging_agent
):
    mock_section_call.side_effect = [
        build_mock_response(
            {"sections": [{"section": "imaging", "confidence": 0.95}]}
        ),
    ]
    mock_subsection_call.side_effect = [
        build_mock_response(
            {"subsections": [{"subsection": "xray", "confidence": 0.9}]}
        ),
    ]
    mock_dimension_call.side_effect = [
        build_mock_response(
            {
                "actual": {
                    "contrast": [
                        {"value": "with_contrast", "confidence": 0.49},
                        {"value": "without_contrast", "confidence": 0.8},
                    ]
                },
                "proposed": {
                    "existing_dimensions": {},
                    "new_dimensions": {},
                },
            }
        ),
    ]
    tags = tagging_agent.generate_tags("Chest X-ray")
    actual = tags[0]["dimensions"]["actual"]["contrast"]
    assert len(actual) == 1
    assert actual[0]["value"] == "without_contrast"


@pytest.mark.generate_tags
@patch.object(SectionTaggingAgent, "_call_openai_completion")
def test_tag_entry_others_keeps_full_dimensions_shape(mock_section_call, tagging_agent):
    mock_section_call.side_effect = [
        build_mock_response(
            {"sections": [{"section": "others", "confidence": 0.7}]}
        )
    ]
    result = tagging_agent.tag_entry("Unknown service", code="1000")
    assert len(result["tags"]) == 1
    tag = result["tags"][0]
    assert tag["subsection"] == "others"
    assert tag["dimensions"] == {
        "actual": {},
        "proposed": {"existing_dimensions": {}, "new_dimensions": {}},
    }


@pytest.mark.schema_validation
def test_compliance_agent_balanced_mode_normalizes_and_warns(schema_contract):
    compliance = SchemaComplianceAgent(schema_contract=schema_contract, mode="balanced")
    input_tag = {
        "section": "imaging",
        "subsection": "xray",
        "confidence": 0.9,
        "dimensions": {
            "actual": {
                "Contrast": [
                    {"value": "With Contrast", "confidence": 0.9},
                ]
            },
            "proposed": {
                "existing_dimensions": {},
                "new_dimensions": {},
            },
        },
    }

    normalized, warnings = compliance.validate_tag(input_tag)
    assert normalized is not None
    assert "contrast" in normalized["dimensions"]["actual"]
    assert normalized["dimensions"]["actual"]["contrast"][0]["value"] == "with_contrast"
    assert warnings


@pytest.mark.generate_tags
def test_process_row_emits_schema_version(schema_contract):
    """process_row uses step-by-step tagging; stub must implement classify_sections/subsections/dimensions."""

    class StubTaggingAgent:
        def classify_sections(self, text_description: str, confidence_threshold: float = 0.5):
            return [("imaging", 0.9)]

        def classify_subsections(
            self, section: str, text_description: str, confidence_threshold: float = 0.5
        ):
            return [("xray", 0.9)]

        def classify_dimensions(
            self, section: str, subsection: str, text_description: str
        ):
            return {
                "actual": {
                    "Contrast": [{"value": "With Contrast", "confidence": 0.9}]
                },
                "proposed": {
                    "existing_dimensions": {},
                    "new_dimensions": {},
                },
            }

    category_rows = []
    dimension_rows = []
    process_row(
        row={"CPTDesc": "Chest Xray", "CPTDescKey": "X-1"},
        idx=1,
        parsing_agent=ParsingAgent(),
        tagging_agent=StubTaggingAgent(),
        normalizer_agent=NormalizerAgent(),
        compliance_agent=SchemaComplianceAgent(schema_contract=schema_contract, mode="balanced"),
        schema_version=schema_contract.version,
        col_name_desc="CPTDesc",
        col_name_key="CPTDescKey",
        category_rows=category_rows,
        dimension_rows=dimension_rows,
    )

    assert len(category_rows) == 1
    assert category_rows[0]["SchemaVersion"] == schema_contract.version
    assert len(dimension_rows) == 1
    assert dimension_rows[0]["SchemaVersion"] == schema_contract.version


# --- Unit tests for SectionTaggingAgent, SubsectionTaggingAgent, DimensionTaggingAgent ---


@pytest.fixture
def minimal_section_schema():
    return {"imaging": {"description": "Imaging"}, "lab": {"description": "Lab"}}


@pytest.fixture
def minimal_subsection_schema():
    return {
        "imaging": {"xray": {"description": "X-ray"}, "mri": {"description": "MRI"}},
        "lab": {"chemistry": {"description": "Chemistry", "dimensions": ["analyte"]}},
    }


@pytest.fixture
def minimal_dimension_schema():
    return {"analyte": {"description": "Analyte", "values": ["glucose", "creatinine"]}}


@pytest.mark.generate_tags
@patch.object(SectionTaggingAgent, "_call_openai_completion")
def test_section_tagging_agent_classify_sections_returns_pairs_filters_threshold(
    mock_call, minimal_section_schema
):
    """SectionTaggingAgent.classify_sections returns list of (section, confidence); filters by threshold."""
    mock_call.return_value = build_mock_response(
        {
            "sections": [
                {"section": "imaging", "confidence": 0.95},
                {"section": "lab", "confidence": 0.4},
            ]
        }
    )
    agent = SectionTaggingAgent(
        section_schema=minimal_section_schema,
        schema_version="test",
    )
    result = agent.classify_sections("Chest X-ray", confidence_threshold=0.5)
    assert result == [("imaging", 0.95)]
    mock_call.assert_called_once()


@pytest.mark.generate_tags
@patch.object(SectionTaggingAgent, "_call_openai_completion")
def test_section_tagging_agent_classify_sections_caches_by_normalized_text(
    mock_call, minimal_section_schema
):
    """Same normalized text returns cached result; API called once."""
    mock_call.return_value = build_mock_response(
        {"sections": [{"section": "imaging", "confidence": 0.9}]}
    )
    agent = SectionTaggingAgent(
        section_schema=minimal_section_schema,
        schema_version="test",
    )
    r1 = agent.classify_sections("Chest X-ray")
    r2 = agent.classify_sections("  CHEST X-RAY  ")
    assert r1 == r2 == [("imaging", 0.9)]
    assert mock_call.call_count == 1


@pytest.mark.generate_tags
@patch.object(SectionTaggingAgent, "_call_openai_completion")
def test_section_tagging_agent_classify_sections_others(mock_call, minimal_section_schema):
    """SectionTaggingAgent can return 'others' when LLM returns it."""
    mock_call.return_value = build_mock_response(
        {"sections": [{"section": "others", "confidence": 0.8}]}
    )
    agent = SectionTaggingAgent(
        section_schema=minimal_section_schema,
        schema_version="test",
    )
    result = agent.classify_sections("Unknown procedure")
    assert result == [("others", 0.8)]


@pytest.mark.generate_tags
def test_subsection_tagging_agent_classify_subsections_others_returns_without_llm(
    minimal_section_schema, minimal_subsection_schema
):
    """SubsectionTaggingAgent when section is 'others' returns [('others', 1.0)] without API call."""
    agent = SubsectionTaggingAgent(
        section_schema=minimal_section_schema,
        subsection_schema=minimal_subsection_schema,
        schema_version="test",
    )
    with patch.object(SubsectionTaggingAgent, "_call_openai_completion") as mock_call:
        result = agent.classify_subsections("others", "Some text")
    assert result == [("others", 1.0)]
    mock_call.assert_not_called()


@pytest.mark.generate_tags
@patch.object(SubsectionTaggingAgent, "_call_openai_completion")
def test_subsection_tagging_agent_classify_subsections_returns_pairs_filters_threshold(
    mock_call, minimal_section_schema, minimal_subsection_schema
):
    """SubsectionTaggingAgent.classify_subsections returns (subsection, confidence); filters by threshold."""
    section_schema_with_subs = {
        "imaging": {"description": "Imaging", "subsections": ["xray", "mri"]},
    }
    mock_call.return_value = build_mock_response(
        {
            "subsections": [
                {"subsection": "xray", "confidence": 0.9},
                {"subsection": "mri", "confidence": 0.3},
            ]
        }
    )
    agent = SubsectionTaggingAgent(
        section_schema=section_schema_with_subs,
        subsection_schema={"imaging": {"xray": {}, "mri": {}}},
        schema_version="test",
    )
    result = agent.classify_subsections("imaging", "Chest X-ray", confidence_threshold=0.5)
    assert result == [("xray", 0.9)]
    mock_call.assert_called_once()


@pytest.mark.generate_tags
@patch.object(SubsectionTaggingAgent, "_call_openai_completion")
def test_subsection_tagging_agent_classify_subsections_caches_by_section_and_text(
    mock_call, minimal_section_schema, minimal_subsection_schema
):
    """Cache key includes section; same section+text returns cached result."""
    minimal_section_schema["imaging"] = {"description": "Imaging", "subsections": ["xray"]}
    minimal_subsection_schema["imaging"] = {"xray": {"description": "X-ray"}}
    mock_call.return_value = build_mock_response(
        {"subsections": [{"subsection": "xray", "confidence": 0.9}]}
    )
    agent = SubsectionTaggingAgent(
        section_schema=minimal_section_schema,
        subsection_schema=minimal_subsection_schema,
        schema_version="test",
    )
    agent.classify_subsections("imaging", "Chest X-ray")
    agent.classify_subsections("imaging", "  chest x-ray  ")
    assert mock_call.call_count == 1


@pytest.mark.generate_tags
def test_dimension_tagging_agent_classify_dimensions_others_returns_empty_without_llm(
    minimal_subsection_schema, minimal_dimension_schema
):
    """DimensionTaggingAgent when section or subsection is 'others' returns empty_dimensions() without API call."""
    agent = DimensionTaggingAgent(
        subsection_schema=minimal_subsection_schema,
        dimension_schema=minimal_dimension_schema,
        schema_version="test",
    )
    with patch.object(DimensionTaggingAgent, "_call_openai_completion") as mock_call:
        r1 = agent.classify_dimensions("others", "xray", "Some text")
        r2 = agent.classify_dimensions("imaging", "others", "Some text")
    from cpt_categorizer.agents.tagging import empty_dimensions

    assert r1 == r2 == empty_dimensions()
    mock_call.assert_not_called()


@pytest.mark.generate_tags
@patch.object(DimensionTaggingAgent, "_call_openai_completion")
def test_dimension_tagging_agent_classify_dimensions_returns_actual_and_proposed(
    mock_call, minimal_subsection_schema, minimal_dimension_schema
):
    """DimensionTaggingAgent.classify_dimensions returns dict with actual and proposed (existing_dimensions, new_dimensions)."""
    subsection_schema = {
        "lab": {"chemistry": {"description": "Chemistry", "dimensions": ["analyte"]}},
    }
    mock_call.return_value = build_mock_response(
        {
            "actual": {"analyte": [{"value": "glucose", "confidence": 0.9}]},
            "proposed": {
                "existing_dimensions": {},
                "new_dimensions": {"sample_type": [{"value": "serum", "confidence": 0.8}]},
            },
        }
    )
    agent = DimensionTaggingAgent(
        subsection_schema=subsection_schema,
        dimension_schema=minimal_dimension_schema,
        schema_version="test",
    )
    result = agent.classify_dimensions("lab", "chemistry", "Glucose test")
    assert set(result.keys()) == {"actual", "proposed"}
    assert set(result["proposed"].keys()) == {"existing_dimensions", "new_dimensions"}
    assert result["actual"].get("analyte")
    assert result["actual"]["analyte"][0]["value"] == "glucose"
    assert result["proposed"]["new_dimensions"].get("sample_type")
    assert result["proposed"]["new_dimensions"]["sample_type"][0]["value"] == "serum"


@pytest.mark.generate_tags
@patch.object(DimensionTaggingAgent, "_call_openai_completion")
def test_dimension_tagging_agent_classify_dimensions_filters_low_confidence(
    mock_call, minimal_subsection_schema, minimal_dimension_schema
):
    """DimensionTaggingAgent filters dimension values by min_confidence 0.5."""
    subsection_schema = {
        "lab": {"chemistry": {"description": "Chemistry", "dimensions": ["analyte"]}},
    }
    mock_call.return_value = build_mock_response(
        {
            "actual": {
                "analyte": [
                    {"value": "glucose", "confidence": 0.9},
                    {"value": "creatinine", "confidence": 0.3},
                ]
            },
            "proposed": {"existing_dimensions": {}, "new_dimensions": {}},
        }
    )
    agent = DimensionTaggingAgent(
        subsection_schema=subsection_schema,
        dimension_schema=minimal_dimension_schema,
        schema_version="test",
    )
    result = agent.classify_dimensions("lab", "chemistry", "Lab panel")
    assert len(result["actual"]["analyte"]) == 1
    assert result["actual"]["analyte"][0]["value"] == "glucose"


# --- Cache-backed agents (persist tagging cache) ---


@pytest.mark.generate_tags
@patch.object(SectionTaggingAgent, "_call_openai_completion")
def test_section_tagging_agent_with_cache_loads_on_init_and_skips_api_when_cached(
    mock_call, minimal_section_schema, tmp_path
):
    """SectionTaggingAgent with cache uses pre-populated cache; no API call for cached key."""
    cache_path = tmp_path / "tagging_cache.json"
    cache = TaggingCache(cache_path)
    cache.sections["chest x-ray"] = [("imaging", 0.9)]
    cache.persist()

    agent = SectionTaggingAgent(
        section_schema=minimal_section_schema,
        schema_version="test",
        cache=cache,
    )
    result = agent.classify_sections("Chest X-ray")
    assert result == [("imaging", 0.9)]
    mock_call.assert_not_called()


@pytest.mark.generate_tags
@patch.object(SectionTaggingAgent, "_call_openai_completion")
def test_section_tagging_agent_with_cache_persists_after_new_result(
    mock_call, minimal_section_schema, tmp_path
):
    """SectionTaggingAgent with cache persists to file after API result."""
    mock_call.return_value = build_mock_response(
        {"sections": [{"section": "lab", "confidence": 0.85}]}
    )
    cache_path = tmp_path / "tagging_cache.json"
    cache = TaggingCache(cache_path)
    cache.load()

    agent = SectionTaggingAgent(
        section_schema=minimal_section_schema,
        schema_version="test",
        cache=cache,
    )
    agent.classify_sections("Blood draw")
    assert cache_path.exists()
    cache2 = TaggingCache(cache_path)
    cache2.load()
    assert "blood draw" in cache2.sections
    assert cache2.sections["blood draw"] == [("lab", 0.85)]


@pytest.mark.generate_tags
@patch.object(SubsectionTaggingAgent, "_call_openai_completion")
def test_subsection_tagging_agent_with_cache_loads_on_init_and_skips_api_when_cached(
    mock_call, minimal_section_schema, minimal_subsection_schema, tmp_path
):
    """SubsectionTaggingAgent with cache uses pre-populated cache; no API call for cached key."""
    minimal_section_schema["imaging"] = {"description": "Imaging", "subsections": ["xray"]}
    minimal_subsection_schema["imaging"] = {"xray": {"description": "X-ray"}}
    cache_path = tmp_path / "tagging_cache.json"
    cache = TaggingCache(cache_path)
    cache.subsections["imaging|chest x-ray"] = [("xray", 0.9)]
    cache.persist()

    agent = SubsectionTaggingAgent(
        section_schema=minimal_section_schema,
        subsection_schema=minimal_subsection_schema,
        schema_version="test",
        cache=cache,
    )
    result = agent.classify_subsections("imaging", "Chest X-ray")
    assert result == [("xray", 0.9)]
    mock_call.assert_not_called()


@pytest.mark.generate_tags
@patch.object(SubsectionTaggingAgent, "_call_openai_completion")
def test_subsection_tagging_agent_with_cache_persists_after_new_result(
    mock_call, minimal_section_schema, minimal_subsection_schema, tmp_path
):
    """SubsectionTaggingAgent with cache persists to file after API result."""
    minimal_section_schema["imaging"] = {"description": "Imaging", "subsections": ["xray"]}
    minimal_subsection_schema["imaging"] = {"xray": {"description": "X-ray"}}
    mock_call.return_value = build_mock_response(
        {"subsections": [{"subsection": "xray", "confidence": 0.88}]}
    )
    cache_path = tmp_path / "tagging_cache.json"
    cache = TaggingCache(cache_path)
    cache.load()

    agent = SubsectionTaggingAgent(
        section_schema=minimal_section_schema,
        subsection_schema=minimal_subsection_schema,
        schema_version="test",
        cache=cache,
    )
    agent.classify_subsections("imaging", "Chest X-ray")
    assert cache_path.exists()
    cache2 = TaggingCache(cache_path)
    cache2.load()
    assert "imaging|chest x-ray" in cache2.subsections
    assert cache2.subsections["imaging|chest x-ray"] == [("xray", 0.88)]


@pytest.mark.generate_tags
@patch.object(DimensionTaggingAgent, "_call_openai_completion")
@patch.object(SubsectionTaggingAgent, "_call_openai_completion")
@patch.object(SectionTaggingAgent, "_call_openai_completion")
def test_tagging_agent_with_cache_path_loads_and_persists_section_cache(
    mock_section_call,
    mock_subsection_call,
    mock_dimension_call,
    schema_contract,
    tmp_path,
):
    """TaggingAgent with cache_path loads on init and persists section cache after classify."""
    mock_section_call.return_value = build_mock_response(
        {"sections": [{"section": "imaging", "confidence": 0.9}]}
    )
    mock_subsection_call.return_value = build_mock_response(
        {"subsections": [{"subsection": "xray", "confidence": 0.9}]}
    )
    mock_dimension_call.return_value = build_mock_response(
        {
            "actual": {},
            "proposed": {"existing_dimensions": {}, "new_dimensions": {}},
        }
    )
    cache_path = tmp_path / "tagging_cache.json"
    agent = TaggingAgent(
        section_schema=schema_contract.sections,
        subsection_schema=schema_contract.subsections,
        dimension_schema=schema_contract.dimensions,
        schema_version=schema_contract.version,
        cache_path=cache_path,
    )
    agent.classify_sections("Chest X-ray")
    assert cache_path.exists()
    cache = TaggingCache(cache_path)
    cache.load()
    assert "chest x-ray" in cache.sections
    assert cache.sections["chest x-ray"] == [("imaging", 0.9)]
