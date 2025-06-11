import json
import openai
from typing import Optional
from openai import OpenAIError
from cpt_categorizer.config.openai import OPENAI_MODEL, get_openai_client

SYSTEM_PROMPT = """You are a medical text cleaning assistant for healthcare procedure codes (CPTs).
Your job is to clean up messy, inconsistent medical procedure descriptions for downstream categorization.

Follow these rules:
1. Expand known medical abbreviations when clear (e.g., PT -> prothrombin time), but flag if ambiguous.
2. Remove administrative codes, filler prefixes, suffixes, or version indicators (e.g., "Code 1012", "v2").
3. Retain medically relevant phrases like "with contrast", "2D echo", "MRI of brain".
4. Fix case: use sentence-style capitalization, but preserve acronyms (e.g., HIV, CT, MRI).
5. Fix punctuation and remove unneeded quotes or parentheticals unless medically relevant.
6. Do not assign any structure — return only the cleaned and informative phrase.

Return only the cleaned text via function call: `clean_cpt_description`.
"""

FUNCTION_SPEC = {
    "name": "clean_cpt_description",
    "description": "Returns a cleaned CPT text after parsing a messy description.",
    "parameters": {
        "type": "object",
        "properties": {
            "cleaned": {
                "type": "string",
                "description": "The cleaned and normalized CPT description.",
            }
        },
        "required": ["cleaned"],
    },
}


def parse(raw_text: str, client: Optional[openai.OpenAI] = None) -> str:
    """
    Clean and normalize a raw CPT description using OpenAI function calling.

    Args:
        raw_text (str): The unprocessed CPT description.
        client (Optional[openai.OpenAI]): Optionally provide an OpenAI client.

    Returns:
        str: Cleaned medical description.
    """
    if client is None:
        client = get_openai_client()

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw_text},
            ],
            temperature=0,
            tools=[{"type": "function", "function": FUNCTION_SPEC}],
            tool_choice="auto",
        )

        tool_call = response.choices[0].message.tool_calls[0]
        raw_args = tool_call.function.arguments
        parsed_args = json.loads(raw_args)
        cleaned = parsed_args["cleaned"]

        return cleaned.strip()

    except (OpenAIError, KeyError, IndexError, json.JSONDecodeError) as e:
        raise RuntimeError(f"OpenAI function call failed: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error in parsing: {e}")
