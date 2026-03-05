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
    class StubTaggingAgent:
        def tag_entry(self, text_description: str, code: str):
            return {
                "code": code,
                "description": text_description,
                "tags": [
                    {
                        "section": "imaging",
                        "subsection": "xray",
                        "confidence": 0.9,
                        "dimensions": {
                            "actual": {
                                "Contrast": [
                                    {"value": "With Contrast", "confidence": 0.9}
                                ]
                            },
                            "proposed": {
                                "existing_dimensions": {},
                                "new_dimensions": {},
                            },
                        },
                    }
                ],
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
