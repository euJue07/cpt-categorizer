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
from dataclasses import dataclass

PH_TZ = timezone(timedelta(hours=8))


@dataclass
class ParsingResult:
    cleaned: str
    confidence_score: float
    is_ambiguous: bool
    error: Optional[str] = None


# System prompt guiding the model on how to clean medical CPT descriptions
SYSTEM_PROMPT = """You are a medical text cleaning assistant for healthcare procedure codes (CPTs).
Your job is to clean messy, inconsistent procedure descriptions while preserving all potentially relevant medical or billing information.

Follow these strict rules:
1. Do NOT remove any medically or operationally meaningful detail, such as:
   - anatomical sites, age qualifiers (e.g., adult, pediatric)
   - imaging views (e.g., PA, lateral), sizes (e.g., 11x14), units
   - cost/billing terms (e.g., net of PHIC, inclusive of PF), discounts
   - device or procedure-specific phrases (e.g., with contrast, laparoscopic)
2. Expand known abbreviations where safe (e.g., PT → prothrombin time), but do not guess.
3. Remove only:
   - unnecessary quotes, trailing version codes, internal IDs
   - filler prefixes like "Code 1012", "v2", or hospital-specific artifacts
4. Preserve parentheses if the contents are relevant (e.g., "(adult)", "(with PF)")
5. Normalize spacing and punctuation only when safe to do so.

Return the minimally cleaned but detail-preserving phrase using the function `clean_cpt_description`.
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


# Helper function to log parsing results in a consistent way
def log_parsing_result(
    raw_text: str,
    parsed_output: str,
    prompt_tokens: int,
    completion_tokens: int,
    success: bool,
    is_error: bool,
    start_time: float,
    error_message: str = "",
):
    timestamp = datetime.now(PH_TZ).isoformat()
    total_tokens = prompt_tokens + completion_tokens
    cost_total = compute_cost(prompt_tokens, completion_tokens)
    runtime_ms = round((time.time() - start_time) * 1000, 2)

    log_usage_row(
        timestamp=timestamp,
        raw_text=raw_text,
        parsed_output=parsed_output,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=round(cost_total, 6),
        model=OPENAI_MODEL,
        runtime_ms=runtime_ms,
        success=success,
        is_error=is_error,
        error_message=error_message,
        log_name="parsing_agent",
    )


# Main parsing function that uses OpenAI's function calling to clean CPT descriptions
def parse(raw_text: str, client: Optional[openai.OpenAI] = None) -> ParsingResult:
    """
    Clean and normalize a raw CPT description using OpenAI function calling.

    Args:
        raw_text (str): The unprocessed CPT description.
        client (Optional[openai.OpenAI]): Optionally provide an OpenAI client.

    Returns:
        ParsingResult: Parsed result containing the cleaned description, confidence score, and ambiguity flag.
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

        parsed_output = cleaned.strip()

        # Log successful usage data
        log_parsing_result(
            raw_text=raw_text,
            parsed_output=parsed_output,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            success=True,
            is_error=False,
            start_time=start_time,
        )

        return ParsingResult(
            cleaned=parsed_output,
            confidence_score=confidence_score,
            is_ambiguous=is_ambiguous,
        )

    except (OpenAIError, KeyError, IndexError, json.JSONDecodeError) as e:
        # Handle known exceptions related to OpenAI API and parsing
        log_parsing_result(
            raw_text=raw_text,
            parsed_output="",
            prompt_tokens=0,
            completion_tokens=0,
            success=False,
            is_error=True,
            start_time=start_time,
            error_message=str(e),
        )
        return ParsingResult(
            cleaned=raw_text.strip(),
            confidence_score=0.0,
            is_ambiguous=False,
            error=str(e),
        )
    except Exception as e:
        # Handle any unexpected exceptions
        log_parsing_result(
            raw_text=raw_text,
            parsed_output="",
            prompt_tokens=0,
            completion_tokens=0,
            success=False,
            is_error=True,
            start_time=start_time,
            error_message=str(e),
        )
        raise RuntimeError(f"Unexpected error in parsing: {e}")
