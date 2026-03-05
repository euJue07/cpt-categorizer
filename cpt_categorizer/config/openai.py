import os

from dotenv import load_dotenv
import openai

# Load environment variables from a .env file if present
load_dotenv()

# === Configuration ===
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
GPT4O_COST_INPUT = 0.005 / 1000
GPT4O_COST_OUTPUT = 0.015 / 1000

# Model-aware pricing: input and output cost per token (USD)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.005 / 1000, 0.015 / 1000),
    "gpt-4o-mini": (0.00015 / 1000, 0.0006 / 1000),
}


def get_model_costs(model: str) -> tuple[float, float]:
    """
    Return (input_per_token, output_per_token) in USD for the given model id.
    Returns (0.0, 0.0) for empty/unknown model (e.g. cache or store hits).
    For unknown non-empty model ids, falls back to gpt-4o so cost is not silently zero.
    """
    if not (model and model.strip()):
        return (0.0, 0.0)
    return MODEL_PRICING.get(model.strip(), (GPT4O_COST_INPUT, GPT4O_COST_OUTPUT))


def get_openai_client() -> openai.Client:
    """
    Initialize and return an OpenAI client using configured settings.

    Returns:
        openai.Client: A configured OpenAI client object.
    """
    if not OPENAI_API_KEY:
        raise ValueError("Missing OPENAI_API_KEY in environment variables.")

    return openai.OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )
