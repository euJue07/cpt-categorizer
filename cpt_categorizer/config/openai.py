import os
from dotenv import load_dotenv
import openai

# Load environment variables from a .env file if present
load_dotenv()

# === Configuration ===
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")


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
