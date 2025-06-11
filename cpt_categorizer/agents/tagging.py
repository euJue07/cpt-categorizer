import json
from dataclasses import dataclass
from typing import Optional, Dict
from cpt_categorizer.agents.parsing import ParsingResult
from config.directory import SCHEMA_DIR
import openai

# Load category and subcategory mappings
with open(SCHEMA_DIR / "cpt_section_subsection.json") as f:
    SECTION_SUBSECTION_SCHEMA = json.load(f)

# Load detail dimensions
with open(SCHEMA_DIR / "cpt_detail.json") as f:
    DETAIL_SCHEMA = json.load(f)

# Extract category enums
CATEGORY_ENUM = list(SECTION_SUBSECTION_SCHEMA.keys())


@dataclass
class TaggingResult:
    category: str
    subcategory: str
    confidence: Optional[float] = None
    details: Optional[Dict[str, str]] = None


# === Step 1: Section Classification ===

# Prompt template for classifying the CPT description into a top-level Section.
SECTION_CLASSIFIER_PROMPT = f"""You are a CPT classification assistant.

Given a cleaned CPT description, choose the best top-level Section that fits the service.

Use one of the following Sections:
{chr(10).join(f"- {section}" for section in CATEGORY_ENUM)}

If uncertain, return 'UNCAT' and explain in the note field.
"""

SECTION_FUNCTION_SPEC = {
    "name": "select_section",
    "description": "Classifies a CPT description into one top-level Section",
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": CATEGORY_ENUM,
                "description": "Top-level Section",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence score for the classification",
            },
        },
        "required": ["category", "confidence"],
    },
}


def tag_section(
    parsed: ParsingResult, client: Optional[openai.OpenAI] = None
) -> tuple[str, float]:
    """
    Classifies the cleaned CPT description into a top-level Section using the OpenAI API.
    Returns the chosen category and confidence score.
    """
    client = client or openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content": SECTION_CLASSIFIER_PROMPT},
            {"role": "user", "content": f"Cleaned CPT description:\n{parsed.cleaned}"},
        ],
        functions=[SECTION_FUNCTION_SPEC],
        function_call={"name": "select_section"},
    )
    args = json.loads(response.choices[0].message.function_call.arguments)
    return args["category"], args.get("confidence", 0.0)


# === Step 2: Subsection Classification ===


def get_subsection_prompt(section: str) -> str:
    """
    Returns a prompt string to classify the subsection within a given section.
    """
    subdescs = SECTION_SUBSECTION_SCHEMA.get(section, {}).get("subsections", {})
    formatted = "\n".join(f"- {key}: {desc}" for key, desc in subdescs.items())
    return f"""You are a CPT classification assistant.

Given a cleaned CPT description, and the chosen Section: {section}, select the most appropriate Subsection.

Choose from the following options:
{formatted}

If uncertain, return 'UNCAT' and explain in the note field.
"""


def get_subsection_function_spec(section: str) -> dict:
    """
    Returns a function spec with an enum of valid subsection choices for the given section.
    """
    enum_values = list(
        SECTION_SUBSECTION_SCHEMA.get(section, {}).get("subsections", {}).keys()
    )
    return {
        "name": f"select_subsection_for_{section.lower()}",
        "description": f"Classifies CPT description into a subsection under {section}",
        "parameters": {
            "type": "object",
            "properties": {
                "subcategory": {
                    "type": "string",
                    "enum": enum_values,
                    "description": f"Subsection under {section}",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence score for the classification",
                },
            },
            "required": ["subcategory", "confidence"],
        },
    }


def tag_subsection(
    section: str, parsed: ParsingResult, client: Optional[openai.OpenAI] = None
) -> tuple[str, float]:
    """
    Classifies the cleaned CPT description into a subsection under the given section using the OpenAI API.
    Returns the chosen subcategory and confidence score.
    """
    client = client or openai.OpenAI()
    prompt = get_subsection_prompt(section)
    spec = get_subsection_function_spec(section)
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Cleaned CPT description:\n{parsed.cleaned}"},
        ],
        functions=[spec],
        function_call={"name": spec["name"]},
    )
    args = json.loads(response.choices[0].message.function_call.arguments)
    return args["subcategory"], args.get("confidence", 0.0)


def tag_section_and_subsection(
    parsed: ParsingResult, client: Optional[openai.OpenAI] = None
) -> TaggingResult:
    """
    Performs hierarchical classification of a CPT description:
    first into a Section, then into a Subsection.
    Returns a TaggingResult with combined confidence.
    """
    category, section_conf = tag_section(parsed, client)
    subcategory, sub_conf = tag_subsection(category, parsed, client)
    return TaggingResult(
        category=category,
        subcategory=subcategory,
        confidence=min(section_conf, sub_conf),
        details=None,
    )


# === TODO: Next Steps ===
# - Implement detail tagging logic to extract structured dimension:value pairs
# - Load allowed detail dimensions from DETAIL_SCHEMA and match tokens from the cleaned description
# - Return those details as part of the TaggingResult
# - Add unit tests for tag_section, tag_subsection, and tag_section_and_subsection
# - Consider fallback logic if confidence is below a threshold (e.g., mark as UNCAT)
