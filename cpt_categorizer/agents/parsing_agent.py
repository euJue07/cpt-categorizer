import json
import openai
import time
from datetime import datetime, timedelta, timezone
from typing import Optional
from openai import OpenAIError
from cpt_categorizer.config.openai import (
    OPENAI_MODEL,
    get_openai_client,
    GPT4O_COST_INPUT,
    GPT4O_COST_OUTPUT,
)
from cpt_categorizer.utils.logging_utils import log_usage_row

PH_TZ = timezone(timedelta(hours=8))


# System prompt guiding the model on how to clean medical CPT descriptions
SYSTEM_PROMPT = """You are a medical text cleaning assistant for healthcare procedure codes (CPTs).
Your job is to clean up messy, inconsistent medical procedure descriptions for downstream categorization.

Follow these rules:
1. Expand known medical abbreviations when clear (e.g., PT -> prothrombin time), but flag if ambiguous.
2. Remove administrative codes, filler prefixes, suffixes, or version indicators (e.g., "Code 1012", "v2").
3. Retain medically relevant phrases like "with contrast", "2D echo", "MRI of brain".
4. Lower cased, but preserve acronyms (e.g., HIV, CT, MRI).
5. Fix punctuation and remove unneeded quotes or parentheticals unless medically relevant.
6. Do not assign any structure — return only the cleaned and informative phrase.

Return only the cleaned text via function call: `clean_cpt_description`.
"""

# Specification for the OpenAI function call to parse and clean CPT descriptions
FUNCTION_SPEC = {
    "name": "clean_cpt_description",
    "description": "Returns a cleaned CPT text after parsing a messy description.",
    "parameters": {
        "type": "object",
        "properties": {
            "cleaned": {
                "type": "string",
                "description": "The cleaned and normalized CPT description.",
            },
            "confidence_score": {
                "type": "number",
                "description": "A score from 0 to 1 indicating the model's confidence in the correctness of the cleaned description.",
                "minimum": 0,
                "maximum": 1,
            },
            "is_ambiguous": {
                "type": "boolean",
                "description": "True if the input description is ambiguous or unclear; otherwise, False.",
            },
        },
        "required": ["cleaned", "confidence_score", "is_ambiguous"],
    },
}


# Helper function to compute the cost of the OpenAI API call based on token usage
def compute_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * GPT4O_COST_INPUT) + (completion_tokens * GPT4O_COST_OUTPUT)


# Main parsing function that uses OpenAI's function calling to clean CPT descriptions
def parse(raw_text: str, client: Optional[openai.OpenAI] = None) -> dict:
    """
    Clean and normalize a raw CPT description using OpenAI function calling.

    Args:
        raw_text (str): The unprocessed CPT description.
        client (Optional[openai.OpenAI]): Optionally provide an OpenAI client.

    Returns:
        dict: Dictionary with keys 'cleaned', 'confidence_score', and 'is_ambiguous'.
    """

    # Initialize OpenAI client if not provided
    if client is None:
        client = get_openai_client()

    start_time = time.time()
    try:
        # Call OpenAI chat completion with function calling to clean the text
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw_text},
            ],
            temperature=0,
            tools=[{"type": "function", "function": FUNCTION_SPEC}],
            tool_choice={
                "type": "function",
                "function": {"name": "clean_cpt_description"},
            },
        )

        # Extract the function call arguments from the model's response
        tool_call = response.choices[0].message.tool_calls[0]
        raw_args = tool_call.function.arguments
        parsed_args = json.loads(raw_args)
        cleaned = parsed_args["cleaned"]
        confidence_score = parsed_args["confidence_score"]
        is_ambiguous = parsed_args["is_ambiguous"]

        # Retrieve token usage info for cost calculation and logging
        prompt_tokens = int(getattr(response.usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(response.usage, "completion_tokens", 0) or 0)
        total_tokens = prompt_tokens + completion_tokens
        cost_total = compute_cost(prompt_tokens, completion_tokens)

        parsed_output = cleaned.strip()
        model_used = OPENAI_MODEL
        timestamp = datetime.now(PH_TZ).isoformat()
        runtime_ms = round((time.time() - start_time) * 1000, 2)

        # Log successful usage data
        log_usage_row(
            timestamp,
            raw_text,
            parsed_output,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            round(cost_total, 6),
            model_used,
            runtime_ms,
            True,
            "",
            log_name="parsing_agent",
        )

        return {
            "cleaned": parsed_output,
            "confidence_score": confidence_score,
            "is_ambiguous": is_ambiguous,
        }

    except (OpenAIError, KeyError, IndexError, json.JSONDecodeError) as e:
        # Handle known exceptions related to OpenAI API and parsing
        timestamp = datetime.now(PH_TZ).isoformat()
        model_used = OPENAI_MODEL
        runtime_ms = round((time.time() - start_time) * 1000, 2)
        # Log failure with error message
        log_usage_row(
            timestamp,
            raw_text,
            "",
            "",
            "",
            "",
            "",
            model_used,
            runtime_ms,
            False,
            True,
            str(e),
            log_name="parsing_agent",
        )
        return {
            "cleaned": raw_text.strip(),
            "confidence_score": 0.0,
            "is_ambiguous": False,
            "error": str(e),
        }
    except Exception as e:
        # Handle any unexpected exceptions
        timestamp = datetime.now(PH_TZ).isoformat()
        model_used = OPENAI_MODEL
        runtime_ms = round((time.time() - start_time) * 1000, 2)
        # Log failure with error message
        log_usage_row(
            timestamp,
            raw_text,
            "",
            "",
            "",
            "",
            "",
            model_used,
            runtime_ms,
            False,
            True,
            str(e),
            log_name="parsing_agent",
        )
        raise RuntimeError(f"Unexpected error in parsing: {e}")
