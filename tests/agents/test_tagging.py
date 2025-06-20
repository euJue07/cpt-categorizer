import json
import pytest
from unittest.mock import patch
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


# --- Basic single tag test with mock ---
@patch.object(TaggingAgent, "_call_openai_completion")
def test_generate_tags_with_mock(mock_call, sample_schema):
    """
    Test that generate_tags correctly processes a basic single tag response.
    The mocked API returns one section and one subsection with confidence scores.
    """
    (
        service_group_schema,
        service_group_dimension_schema,
        service_group_dimension_value_schema,
    ) = sample_schema

    # Mock return values for classify_sections and classify_subsections
    # First call: classify_sections returns imaging with confidence 0.95
    # Second call: classify_subsections returns ultrasound with confidence 0.9
    mock_call.side_effect = [
        MockResponse('{"sections": [{"section": "imaging", "confidence": 0.95}]}'),
        MockResponse(
            '{"subsections": [{"subsection": "ultrasound", "confidence": 0.9}]}'
        ),
    ]

    agent = TaggingAgent(
        service_group_schema=service_group_schema,
        service_group_dimension_schema=service_group_dimension_schema,
        service_group_dimension_value_schema=service_group_dimension_value_schema,
    )

    tags = agent.generate_tags("Chest X-ray")
    # Verify that the output is a list containing one tag
    assert isinstance(tags, list)
    assert len(tags) == 1
    tag = tags[0]
    # Verify that the tag contains the expected section and subsection
    assert tag["section"] == "imaging", (
        f"Expected section 'imaging', got {tag['section']}"
    )
    assert tag["subsection"] == "ultrasound", (
        f"Expected subsection 'ultrasound', got {tag['subsection']}"
    )


# --- Edge case: Empty GPT response ---
@patch.object(TaggingAgent, "_call_openai_completion")
def test_generate_tags_empty(mock_call, sample_schema):
    """
    Test how generate_tags handles an empty GPT response.
    The mocked API returns empty lists, simulating no tags found.
    """
    (
        service_group_schema,
        service_group_dimension_schema,
        service_group_dimension_value_schema,
    ) = sample_schema
    # Mocking empty responses for both section and subsection calls
    mock_call.side_effect = [MockResponse("[]"), MockResponse("[]")]

    agent = TaggingAgent(
        service_group_schema=service_group_schema,
        service_group_dimension_schema=service_group_dimension_schema,
        service_group_dimension_value_schema=service_group_dimension_value_schema,
    )

    tags = agent.generate_tags("Unknown procedure")
    # Verify that the output is an empty list when no tags are found
    assert isinstance(tags, list)
    assert tags == [], f"Expected empty tag list, got {tags}"


# --- Edge case: Malformed GPT response ---
@patch.object(TaggingAgent, "_call_openai_completion")
def test_generate_tags_malformed(mock_call, sample_schema):
    """
    Test how generate_tags handles a malformed GPT response.
    The mocked API returns invalid JSON and None to simulate parsing errors.
    """
    (
        service_group_schema,
        service_group_dimension_schema,
        service_group_dimension_value_schema,
    ) = sample_schema
    # Mocking a malformed JSON response followed by None
    mock_call.side_effect = [MockResponse("{not a valid json}"), None]

    agent = TaggingAgent(
        service_group_schema=service_group_schema,
        service_group_dimension_schema=service_group_dimension_schema,
        service_group_dimension_value_schema=service_group_dimension_value_schema,
    )

    tags = agent.generate_tags("Some malformed input")
    # Verify that the output is an empty list for malformed input
    assert isinstance(tags, list)
    assert tags == [], f"Expected empty tag list for malformed input, got {tags}"


# --- Parametrized multi-input test ---
@pytest.mark.parametrize(
    "description, expected_section, expected_subsection, section_conf, sub_conf",
    [
        ("Chest X-ray", "imaging", "xray", 0.9, 0.8),
        ("Appendectomy", "procedures", "major_procedures", 0.95, 0.85),
        ("Complete Blood Count", "laboratory", "hematology", 0.92, 0.9),
    ],
)
@patch.object(TaggingAgent, "_call_openai_completion")
def test_generate_tags_parametrized(
    mock_call,
    sample_schema,
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
    (
        service_group_schema,
        service_group_dimension_schema,
        service_group_dimension_value_schema,
    ) = sample_schema

    # Mock responses return the expected section and subsection with given confidence
    mock_call.side_effect = [
        MockResponse(
            f'{{"sections": [{{"section": "{expected_section}", "confidence": {section_conf}}}]}}'
        ),
        MockResponse(
            f'{{"subsections": [{{"subsection": "{expected_subsection}", "confidence": {sub_conf}}}]}}'
        ),
    ]

    agent = TaggingAgent(
        service_group_schema=service_group_schema,
        service_group_dimension_schema=service_group_dimension_schema,
        service_group_dimension_value_schema=service_group_dimension_value_schema,
    )

    tags = agent.generate_tags(description)
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


# --- Multi-tag response simulation ---
@patch.object(TaggingAgent, "_call_openai_completion")
def test_generate_tags_multiple_results(mock_call, sample_schema):
    """
    Test generate_tags with simulated multiple sections and subsections returned.
    The mocked API returns multiple tags with varying confidence scores.
    """
    (
        service_group_schema,
        service_group_dimension_schema,
        service_group_dimension_value_schema,
    ) = sample_schema

    # Simulate multiple sections and subsections from GPT
    mock_call.side_effect = [
        MockResponse(
            '{"sections": [{"section": "imaging", "confidence": 0.9}, {"section": "laboratory", "confidence": 0.8}]}'
        ),  # classify_sections
        MockResponse(
            '{"subsections": [{"subsection": "ultrasound", "confidence": 0.85}, {"subsection": "ct_scan", "confidence": 0.7}]}'
        ),  # classify_subsections for "imaging"
        MockResponse(
            '{"subsections": [{"subsection": "hematology", "confidence": 0.9}]}'
        ),  # classify_subsections for "laboratory"
    ]

    agent = TaggingAgent(
        service_group_schema=service_group_schema,
        service_group_dimension_schema=service_group_dimension_schema,
        service_group_dimension_value_schema=service_group_dimension_value_schema,
    )

    tags = agent.generate_tags("Chest imaging and CBC")
    # Verify that the output is a list containing all combined tags
    assert isinstance(tags, list)
    assert len(tags) == 3

    expected = {
        ("imaging", "ultrasound"): 0.9 * 0.85,
        ("imaging", "ct_scan"): 0.9 * 0.7,
        ("laboratory", "hematology"): 0.8 * 0.9,
    }

    for tag in tags:
        key = (tag["section"], tag["subsection"])
        # Verify that each tag key is expected and present in the results
        assert key in expected, f"Unexpected tag key: {key}, got tags: {tags}"


@patch.object(TaggingAgent, "_call_openai_completion")
def test_schema_generates_valid_openai_function(mock_call, sample_schema):
    """
    Ensure each section's subsection function schema is valid JSON Schema.
    This protects against OpenAI's BadRequestError for invalid function spec.
    """
    (
        service_group_schema,
        _,
        _,
    ) = sample_schema

    agent = TaggingAgent(
        service_group_schema=service_group_schema,
    )

    for section in service_group_schema["sections"]:
        try:
            spec = agent._get_subsection_function_specification(section)
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
    with open(SCHEMA_DIR / "service_group_schema.json") as f:
        service_group_schema = json.load(f)

    with open(SCHEMA_DIR / "service_group_dimension_schema.json") as f:
        service_group_dimension_schema = json.load(f)

    with open(SCHEMA_DIR / "service_group_dimension_value_schema.json") as f:
        service_group_dimension_value_schema = json.load(f)

    return (
        service_group_schema,
        service_group_dimension_schema,
        service_group_dimension_value_schema,
    )


@patch.object(TaggingAgent, "_call_openai_completion")
def test_openai_called_once_for_multiple_sections(mock_call, sample_schema):
    """
    Test that OpenAI is called once for sections and once per valid section for subsections,
    when multiple valid sections are returned. Verifies correct number of calls for multi-section input.
    """
    (
        service_group_schema,
        dim_schema,
        dim_val_schema,
    ) = sample_schema

    mock_call.side_effect = [
        MockResponse(
            '{"sections": [{"section": "imaging", "confidence": 0.8}, {"section": "laboratory", "confidence": 0.7}]}'
        ),
        MockResponse(
            '{"subsections": [{"subsection": "ultrasound", "confidence": 0.6}]}'
        ),
        MockResponse(
            '{"subsections": [{"subsection": "hematology", "confidence": 0.65}]}'
        ),
    ]

    agent = TaggingAgent(
        service_group_schema=service_group_schema,
        service_group_dimension_schema=dim_schema,
        service_group_dimension_value_schema=dim_val_schema,
    )

    tags = agent.generate_tags("Valid multiple sections test")
    assert mock_call.call_count == 3, (
        f"Expected 3 OpenAI calls (1 for sections, 2 for subsections), got {mock_call.call_count}"
    )


@patch.object(TaggingAgent, "_call_openai_completion")
def test_generate_tags_with_wrong_subsection(mock_call, sample_schema):
    (
        service_group_schema,
        dim_schema,
        dim_val_schema,
    ) = sample_schema

    mock_call.side_effect = [
        MockResponse('{"sections": [{"section": "laboratory", "confidence": 0.95}]}'),
        MockResponse(
            '{"subsections": [{"subsection": "ultrasound", "confidence": 0.9}]}'
        ),
    ]

    agent = TaggingAgent(
        service_group_schema=service_group_schema,
        service_group_dimension_schema=dim_schema,
        service_group_dimension_value_schema=dim_val_schema,
    )

    tags = agent.generate_tags("Subsection from wrong section test")
    assert tags == [], f"Expected no tags for mismatched subsection, got: {tags}"


# --- New test: OpenAI called only once if section is invalid ---
@patch.object(TaggingAgent, "_call_openai_completion")
def test_openai_called_once(mock_call, sample_schema):
    """
    Test that OpenAI API is only called once when classify_sections is enough to determine no valid section.
    """
    (
        service_group_schema,
        dim_schema,
        dim_val_schema,
    ) = sample_schema

    # Simulate classify_sections returns an invalid section
    mock_call.side_effect = [
        MockResponse('{"sections": [{"section": "nonexistent", "confidence": 0.8}]}')
    ]

    agent = TaggingAgent(
        service_group_schema=service_group_schema,
        service_group_dimension_schema=dim_schema,
        service_group_dimension_value_schema=dim_val_schema,
    )

    tags = agent.generate_tags("Invalid section only")
    assert tags == [], "Expected no tags due to invalid section"
    assert mock_call.call_count == 1, (
        f"Expected only one OpenAI call, got {mock_call.call_count}"
    )


@patch.object(TaggingAgent, "_call_openai_completion")
def test_generate_tags_with_empty_string(mock_call, sample_schema):
    mock_call.side_effect = [MockResponse("[]")]
    (
        service_group_schema,
        dim_schema,
        dim_val_schema,
    ) = sample_schema

    agent = TaggingAgent(
        service_group_schema=service_group_schema,
        service_group_dimension_schema=dim_schema,
        service_group_dimension_value_schema=dim_val_schema,
    )

    tags = agent.generate_tags("")
    assert tags == [], "Expected empty list for empty description"


# --- New test: section but no subsection ---
@patch.object(TaggingAgent, "_call_openai_completion")
def test_generate_tags_with_section_but_no_subsection(mock_call, sample_schema):
    (
        service_group_schema,
        dim_schema,
        dim_val_schema,
    ) = sample_schema

    mock_call.side_effect = [
        MockResponse('{"sections": [{"section": "imaging", "confidence": 0.9}]}'),
        MockResponse('{"subsections": []}'),
    ]

    agent = TaggingAgent(
        service_group_schema=service_group_schema,
        service_group_dimension_schema=dim_schema,
        service_group_dimension_value_schema=dim_val_schema,
    )

    tags = agent.generate_tags("Imaging without specific procedure")
    assert tags == [], "Expected no tags if no subsections are returned"
