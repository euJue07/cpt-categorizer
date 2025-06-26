import os
from openai import RateLimitError
import time
import json
import pytest
from unittest.mock import patch, Mock
from openai import BadRequestError

from cpt_categorizer.agents.tagging import TaggingAgent
from cpt_categorizer.config.directory import SCHEMA_DIR


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
        self.model = "gpt-4"


# Utility function to build a MockResponse from dict or str
def build_mock_response(obj: dict | str) -> MockResponse:
    if isinstance(obj, dict):
        return MockResponse(json.dumps(obj))
    return MockResponse(obj)


# Helper: Validate tag dictionary structure
def assert_valid_tag_format(tag):
    assert isinstance(tag, dict), "Each tag must be a dictionary"
    assert "section" in tag, "Missing 'section' key"
    assert "subsection" in tag, "Missing 'subsection' key"
    assert "confidence" in tag, "Missing 'confidence' key"
    assert isinstance(tag["confidence"], float), "Confidence must be a float"
    assert 0 <= tag["confidence"] <= 1, "Confidence must be in [0, 1]"
    assert "details" in tag, "Missing 'details' key"
    assert isinstance(tag["details"], dict), "Details must be a dict"


# --- Edge case: Empty GPT response ---
@pytest.mark.generate_tags
@patch.object(TaggingAgent, "_call_openai_completion")
def test_generate_tags_returns_empty_for_no_valid_tags(mock_call, tagging_agent):
    """
    Test how generate_tags handles an empty GPT response.
    The mocked API returns empty lists, simulating no tags found.
    """
    # Mocking empty responses for both section and subsection calls
    mock_call.side_effect = [build_mock_response("[]"), build_mock_response("[]")]
    tags = tagging_agent.generate_tags("Unknown procedure")
    # Verify that the output is an empty list when no tags are found
    assert isinstance(tags, list)
    assert tags == [], f"Expected empty tag list, got {tags}"
    for tag in tags:
        assert_valid_tag_format(tag)


# --- Edge case: Malformed GPT response ---
@pytest.mark.generate_tags
@patch.object(TaggingAgent, "_call_openai_completion")
def test_generate_tags_handles_malformed_response(mock_call, tagging_agent):
    """
    Test how generate_tags handles a malformed GPT response.
    The mocked API returns invalid JSON and None to simulate parsing errors.
    """
    # Mocking a malformed JSON response followed by None
    mock_call.side_effect = [build_mock_response("{not a valid json}"), None]
    tags = tagging_agent.generate_tags("Some malformed input")
    # Verify that the output is an empty list for malformed input
    assert isinstance(tags, list)
    assert tags == [], f"Expected empty tag list for malformed input, got {tags}"
    for tag in tags:
        assert_valid_tag_format(tag)


# --- Parametrized multi-input test ---
@pytest.mark.generate_tags
@pytest.mark.parametrize(
    "description, expected_section, expected_subsection, section_conf, sub_conf",
    [
        ("Chest X-ray", "imaging", "xray", 0.9, 0.8),
        ("Appendectomy", "others", "others", 0.95, 0.85),
        ("Complete Blood Count", "laboratory", "hematology", 0.92, 0.9),
    ],
)
@patch.object(TaggingAgent, "_call_openai_completion")
def test_generate_tags_with_various_valid_inputs(
    mock_call,
    tagging_agent,
    description,
    expected_section,
    expected_subsection,
    section_conf,
    sub_conf,
):
    """
    Parametrized test to verify generate_tags with multiple input descriptions.
    Each input has expected section, subsection, and confidence scores.
    """
    # Mock responses return the expected section and subsection with given confidence, plus details
    mock_call.side_effect = [
        build_mock_response(
            f'{{"sections": [{{"section": "{expected_section}", "confidence": {section_conf}}}]}}'
        ),
        build_mock_response(
            f'{{"subsections": [{{"subsection": "{expected_subsection}", "confidence": {sub_conf}}}]}}'
        ),
        build_mock_response('[{"details": {}}]'),
    ]
    tags = tagging_agent.generate_tags(description)
    # Verify that the output is a list with one tag matching expected values
    assert isinstance(tags, list)
    assert len(tags) == 1
    tag = tags[0]
    assert tag["section"] == expected_section, (
        f"Expected section {expected_section}, got {tag['section']}"
    )
    assert tag["subsection"] == expected_subsection, (
        f"Expected subsection {expected_subsection}, got {tag['subsection']}"
    )
    for tag in tags:
        assert isinstance(tag, dict), "Each tag must be a dictionary"
        assert "section" in tag, "Missing 'section' key"
        assert "subsection" in tag, "Missing 'subsection' key"
        assert "confidence" in tag, "Missing 'confidence' key"
        assert isinstance(tag["confidence"], float), "Confidence must be a float"
        assert 0 <= tag["confidence"] <= 1, "Confidence must be in [0, 1]"
        assert "dimensions" in tag, "Missing 'dimensions' key"
        assert "actual" in tag["dimensions"], "Missing 'actual' key in dimensions"
        assert isinstance(tag["dimensions"]["actual"], dict), "'actual' must be a dict"


# --- Multi-tag response simulation ---


@pytest.mark.schema_validation
@patch.object(TaggingAgent, "_call_openai_completion")
def test_schema_generates_valid_openai_function(
    mock_call, tagging_agent, sample_schema
):
    """
    Ensure each section's subsection function schema is valid JSON Schema.
    This protects against OpenAI's BadRequestError for invalid function spec.
    """
    sections_schema = sample_schema[0]
    for section in sections_schema:
        try:
            spec = tagging_agent._get_subsection_function_specification(section)
            assert isinstance(spec, dict)
            assert spec["parameters"]["type"] == "object"
            enum_values = spec["parameters"]["properties"]["subsections"]["items"][
                "properties"
            ]["subsection"]["enum"]
            assert isinstance(enum_values, list), f"Enum not list for {section}"
            assert all(isinstance(val, str) for val in enum_values), (
                f"Invalid enum contents for {section}"
            )
        except BadRequestError as e:
            pytest.fail(f"Section {section} produces an invalid schema: {e}")


@pytest.fixture
def sample_schema():
    with open(SCHEMA_DIR / "sections.json") as f:
        sections_schema = json.load(f)

    with open(SCHEMA_DIR / "subsections.json") as f:
        subsections_schema = json.load(f)

    with open(SCHEMA_DIR / "dimensions.json") as f:
        dimensions_schema = json.load(f)

    return (
        sections_schema,
        subsections_schema,
        dimensions_schema,
    )


# New fixture: tagging_agent
@pytest.fixture
def tagging_agent(sample_schema):
    return TaggingAgent(
        section_schema=sample_schema[0],
        subsection_schema=sample_schema[1],
        dimension_schema=sample_schema[2],
    )


@pytest.mark.schema_validation
def test_schema_structure_consistency(sample_schema):
    """
    Test if section, subsection, and dimension references across all schemas are internally consistent.
    """
    sections_schema, subsections_schema, dim_val_schema = sample_schema

    # 1. Every section in dimension schema must exist in service group schema
    for section in subsections_schema:
        assert section in sections_schema, (
            f"Missing section in sections_schema: {section}"
        )

    # 2. Every subsection in dimension schema must exist in service group schema
    for section, subsection_dict in subsections_schema.items():
        expected_subsections = set(sections_schema[section]["subsections"])
        for subsection in subsection_dict:
            assert subsection in expected_subsections, (
                f"Subsection '{subsection}' not found in section '{section}'"
            )

    # 3. Every dimension used in dimension schema must exist in dimension value schema
    for section_subsections in subsections_schema.values():
        for subsection_data in section_subsections.values():
            for dim in subsection_data.get("dimensions", []):
                assert dim in dim_val_schema, (
                    f"Dimension '{dim}' missing in dimension value schema"
                )


@pytest.mark.generate_tags
@patch.object(TaggingAgent, "_call_openai_completion")
def test_generate_tags_filters_out_invalid_subsections(mock_call, sample_schema):
    (
        sections_schema,
        dim_schema,
        dim_val_schema,
    ) = sample_schema

    mock_call.side_effect = [
        build_mock_response(
            '{"sections": [{"section": "laboratory", "confidence": 0.95}]}'
        ),
        build_mock_response(
            '{"subsections": [{"subsection": "ultrasound", "confidence": 0.9}]}'
        ),
    ]

    agent = TaggingAgent(
        section_schema=sections_schema,
        subsection_schema=dim_schema,
        dimension_schema=dim_val_schema,
    )

    tags = agent.generate_tags("Subsection from wrong section test")
    assert tags == [], f"Expected no tags for mismatched subsection, got: {tags}"
    for tag in tags:
        assert_valid_tag_format(tag)


# --- New test: OpenAI called only once if section is invalid ---
@pytest.mark.generate_tags
@patch.object(TaggingAgent, "_call_openai_completion")
def test_generate_tags_skips_subsection_when_section_invalid(mock_call, sample_schema):
    """
    Test that OpenAI API is only called once when classify_sections is enough to determine no valid section.
    """
    (
        sections_schema,
        dim_schema,
        dim_val_schema,
    ) = sample_schema

    # Simulate classify_sections returns an invalid section
    mock_call.side_effect = [
        build_mock_response(
            '{"sections": [{"section": "nonexistent", "confidence": 0.8}]}'
        )
    ]

    agent = TaggingAgent(
        section_schema=sections_schema,
        subsection_schema=dim_schema,
        dimension_schema=dim_val_schema,
    )

    tags = agent.generate_tags("Invalid section only")
    assert tags == [], "Expected no tags due to invalid section"
    assert mock_call.call_count == 1, (
        f"Expected only one OpenAI call, got {mock_call.call_count}"
    )
    for tag in tags:
        assert_valid_tag_format(tag)


@pytest.mark.generate_tags
@patch.object(TaggingAgent, "_call_openai_completion")
def test_generate_tags_empty_input_returns_empty(mock_call, sample_schema):
    mock_call.side_effect = [build_mock_response("[]")]
    (
        sections_schema,
        dim_schema,
        dim_val_schema,
    ) = sample_schema

    agent = TaggingAgent(
        section_schema=sections_schema,
        subsection_schema=dim_schema,
        dimension_schema=dim_val_schema,
    )

    tags = agent.generate_tags("")
    assert tags == [], "Expected empty list for empty description"
    for tag in tags:
        assert_valid_tag_format(tag)


# --- New test: section but no subsection ---
@pytest.mark.generate_tags
@patch.object(TaggingAgent, "_call_openai_completion")
def test_generate_tags_with_valid_section_and_no_subsection_fallbacks_to_others(
    mock_call, sample_schema
):
    (
        sections_schema,
        dim_schema,
        dim_val_schema,
    ) = sample_schema

    mock_call.side_effect = [
        build_mock_response(
            '{"sections": [{"section": "imaging", "confidence": 0.9}]}'
        ),
        build_mock_response('{"subsections": []}'),
    ]

    agent = TaggingAgent(
        section_schema=sections_schema,
        subsection_schema=dim_schema,
        dimension_schema=dim_val_schema,
    )

    tags = agent.generate_tags("Imaging without specific procedure")
    assert tags == [], f"Expected no tags due to empty subsection, got: {tags}"
    for tag in tags:
        assert_valid_tag_format(tag)


@pytest.mark.generate_tags
@patch.object(TaggingAgent, "_call_openai_completion")
def test_generate_tags_handles_others_as_section(mock_call, sample_schema):
    """
    Simulate GPT returning 'others' as a valid section.
    Should default to section=others, subsection=others, confidence from section.
    """
    (
        sections_schema,
        dim_schema,
        dim_val_schema,
    ) = sample_schema

    mock_call.side_effect = [
        build_mock_response(
            '{"sections": [{"section": "others", "confidence": 0.95}]}'
        ),
        build_mock_response("[]"),
        build_mock_response('[{"details": {}}]'),
    ]

    agent = TaggingAgent(
        section_schema=sections_schema,
        subsection_schema=dim_schema,
        dimension_schema=dim_val_schema,
    )

    tags = agent.generate_tags("General unknown service")
    assert isinstance(tags, list), "Expected tag list"
    assert len(tags) == 1, f"Expected 1 tag, got {len(tags)}"
    tag = tags[0]
    assert tag["section"] == "others", (
        f"Expected section 'others', got {tag['section']}"
    )
    assert tag["subsection"] == "others", (
        f"Expected subsection 'others', got {tag['subsection']}"
    )
    for tag in tags:
        assert isinstance(tag, dict), "Each tag must be a dictionary"
        assert "section" in tag, "Missing 'section' key"
        assert "subsection" in tag, "Missing 'subsection' key"
        assert "confidence" in tag, "Missing 'confidence' key"
        assert isinstance(tag["confidence"], float), "Confidence must be a float"
        assert 0 <= tag["confidence"] <= 1, "Confidence must be in [0, 1]"
        assert "dimensions" in tag, "Missing 'dimensions' key"
        assert "actual" in tag["dimensions"], "Missing 'actual' key in dimensions"
        assert isinstance(tag["dimensions"]["actual"], dict), "'actual' must be a dict"


# --- Group all classify_sections tests together ---


@pytest.mark.classify_sections
@patch.object(TaggingAgent, "_call_openai_completion")
def test_classify_sections_with_malformed_json(mock_call, sample_schema):
    """
    Test classify_sections with a malformed JSON response.
    Ensures graceful fallback and no crash.
    """
    mock_call.return_value = build_mock_response(
        "{invalid json"
    )  # Intentionally bad JSON

    agent = TaggingAgent(*sample_schema)
    result = agent.classify_sections("malformed json input")

    assert isinstance(result, list), "Expected result to be a list"
    assert result == [], f"Expected empty list on malformed input, got {result}"


@pytest.mark.classify_sections
@patch("time.sleep", return_value=None)
@patch.object(TaggingAgent, "_call_openai_completion")
def test_classify_sections_handles_rate_limit_error(
    mock_call, mock_sleep, sample_schema
):
    """
    Simulate RateLimitError raised during classify_sections.
    Ensure agent retries and then returns empty list after failures.
    """

    mock_response = Mock()
    mock_response.status_code = 429
    mock_call.side_effect = RateLimitError(
        message="Rate limit exceeded",
        response=mock_response,
        body={"error": {"message": "Rate limit exceeded"}},
    )

    agent = TaggingAgent(*sample_schema)
    result = agent.classify_sections("trigger rate limit")

    assert result == [], "Expected empty result after RateLimitError"
    assert mock_call.call_count >= 1, "Expected at least one retry attempt"


@pytest.mark.classify_sections
@pytest.mark.parametrize(
    "threshold, returned_confidence, should_include",
    [
        (0.5, 0.5, True),
        (0.5, 0.49, False),
        (0.8, 0.75, False),
        (0.0, 0.0, True),
        (1.0, 1.0, True),
    ],
)
@patch.object(TaggingAgent, "_call_openai_completion")
def test_classify_sections_confidence_threshold_behavior(
    mock_call, sample_schema, threshold, returned_confidence, should_include
):
    mock_call.return_value = build_mock_response(
        json.dumps(
            {"sections": [{"section": "laboratory", "confidence": returned_confidence}]}
        )
    )

    agent = TaggingAgent(*sample_schema)
    result = agent.classify_sections("threshold test", confidence_threshold=threshold)

    if should_include:
        assert ("laboratory", returned_confidence) in result, (
            f"Expected laboratory section to be included at threshold {threshold}"
        )
    else:
        assert result == [], f"Expected result to be empty at threshold {threshold}"


@pytest.mark.classify_sections
@patch.object(TaggingAgent, "_call_openai_completion")
def test_classify_sections_cache_behavior(mock_call, sample_schema):
    """
    Verify that repeated calls to classify_sections for the same input use cached results.
    """
    mock_call.return_value = build_mock_response(
        json.dumps({"sections": [{"section": "imaging", "confidence": 0.9}]})
    )

    agent = TaggingAgent(*sample_schema)

    # First call: should trigger OpenAI
    result1 = agent.classify_sections("MRI of head")
    assert result1 == [("imaging", 0.9)]

    # Second call: should use cache, not call OpenAI again
    result2 = agent.classify_sections("MRI of head")
    assert result2 == [("imaging", 0.9)]
    assert mock_call.call_count == 1, "Expected second call to hit cache, not OpenAI"


@pytest.mark.classify_sections
@patch.object(TaggingAgent, "_call_openai_completion")
def test_classify_sections_with_duplicate_entries(mock_call, sample_schema):
    """
    Test classify_sections handles duplicate section entries and deduplicates them.
    """
    mock_call.return_value = build_mock_response(
        json.dumps(
            {
                "sections": [
                    {"section": "laboratory", "confidence": 0.9},
                    {"section": "laboratory", "confidence": 0.9},
                ]
            }
        )
    )
    agent = TaggingAgent(*sample_schema)
    result = agent.classify_sections("lab with duplicates")
    assert result == [("laboratory", 0.9)], (
        f"Expected deduplicated result, got {result}"
    )


# --- Group all classify_dimensions tests together ---


@pytest.mark.classify_dimensions
@patch.object(TaggingAgent, "_call_openai_completion")
def test_classify_dimensions_without_schema_returns_empty(mock_call, sample_schema):
    """
    If no dimension schema is provided, classify_dimensions should return an empty dict.
    """
    sections_schema, _, _ = sample_schema
    mock_call.return_value = build_mock_response(
        json.dumps(
            {
                "dimensions": {
                    "anesthesia_type": ["general"],
                    "irrelevant_key": ["noise"],
                }
            }
        )
    )

    agent = TaggingAgent(section_schema=sections_schema)

    result = agent.classify_dimensions(
        "procedures", "anesthesiology", "General anesthesia"
    )
    assert set(result.keys()) <= {"actual", "proposed"}, (
        "Unexpected top-level keys in dimension output"
    )
    assert result.get("actual") == {}, "Expected empty 'actual' if no schema is defined"
    assert result.get("proposed") == {
        "existing_dimensions": {},
        "new_dimensions": {},
    }, "Expected empty but structured 'proposed' if no schema is defined"


@pytest.mark.schema_validation
def test_schema_files_exist_and_loadable():
    """
    Ensure core schema files exist in the SCHEMA_DIR and are loadable as JSON.
    """
    required_files = ["sections.json", "subsections.json", "dimensions.json"]
    for file_name in required_files:
        path = SCHEMA_DIR / file_name
        assert path.exists(), f"Missing schema file: {file_name}"
        try:
            with open(path) as f:
                data = json.load(f)
                assert isinstance(data, dict), (
                    f"{file_name} should contain a JSON object"
                )
        except Exception as e:
            pytest.fail(f"Failed to load or parse {file_name}: {e}")
