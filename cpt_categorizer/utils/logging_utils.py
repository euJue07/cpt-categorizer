import csv
from pathlib import Path


def log_usage_row(
    timestamp,
    raw_text,
    parsed_output,
    prompt_tokens,
    completion_tokens,
    total_tokens,
    cost_usd,
    model,
    runtime_ms,
    success,
    is_error,
    error_message,
    log_name,
):
    """
    Append a usage log row to a specified agent log CSV file.

    Args:
        timestamp (str): Timestamp of the call.
        raw_text (str): The original raw input.
        parsed_output (str): Cleaned output from the agent.
        prompt_tokens (int or str): Number of prompt tokens.
        completion_tokens (int or str): Number of completion tokens.
        total_tokens (int or str): Total token count.
        cost_usd (float or str): Estimated cost in USD.
        model (str): Model used (e.g., gpt-4o).
        runtime_ms (int or str): Runtime in milliseconds.
        success (bool): Whether the call was successful.
        is_error (bool): Whether the result was due to a processing error.
        error_message (str): Error message if applicable.
        log_name (str): Name of the log file (without extension).
    """
    log_dir = Path("logs/usage")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{log_name}.csv"
    is_new = not log_path.exists()

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                [
                    "timestamp",
                    "raw_text",
                    "parsed_output",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "cost_usd",
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
                parsed_output,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cost_usd,
                model,
                runtime_ms,
                success,
                is_error,
                error_message,
            ]
        )
