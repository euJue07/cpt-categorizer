import csv
from cpt_categorizer.config.directory import LOG_DIR
from cpt_categorizer.config.openai import GPT4O_COST_INPUT, GPT4O_COST_OUTPUT


def log_agent_usage(
    timestamp,
    raw_text,
    description,
    parsed_output,
    prompt_tokens,
    completion_tokens,
    total_tokens,
    model,
    runtime_ms,
    success,
    is_error,
    error_message,
    request_id=None,
):
    """
    Append a log row to the usage log CSV file.

    Args:
        timestamp (str): Timestamp of the call.
        raw_text (str): The original raw input.
        description (str): Short description of the call (e.g., "classify_sections", "classify_subsections", etc.).
        parsed_output (str): Cleaned output from the agent.
        prompt_tokens (int or str): Number of prompt tokens.
        completion_tokens (int or str): Number of completion tokens.
        total_tokens (int or str): Total token count.
        model (str): Model used (e.g., gpt-4o).
        runtime_ms (int or str): Runtime in milliseconds.
        success (bool): Whether the call was successful.
        is_error (bool): Whether the result was due to a processing error.
        error_message (str): Error message if applicable.
        request_id (str, optional): Identifier for the request.
    """
    cost_usd = round(
        int(prompt_tokens) * GPT4O_COST_INPUT
        + int(completion_tokens) * GPT4O_COST_OUTPUT,
        6,
    )
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "usage.csv"
    is_new = not log_path.exists()

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                [
                    "timestamp",
                    "raw_text",
                    "description",
                    "parsed_output",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "cost_usd",
                    "request_id",
                    "model",
                    "runtime_ms",
                    "success",
                    "is_error",
                    "error_message",
                ]
            )
        writer.writerow(
            [
                timestamp,
                raw_text,
                description,
                parsed_output,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cost_usd,
                request_id,
                model,
                runtime_ms,
                success,
                is_error,
                error_message,
            ]
        )
