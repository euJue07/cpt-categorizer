from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SCHEMA_DIR = _PROJECT_ROOT / "schema"
LOG_DIR = _PROJECT_ROOT / "logs"
RAW_DIR = _PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = _PROJECT_ROOT / "data" / "interim"
SUGGESTIONS_PATH = INTERIM_DIR / "suggestions.json"
TAGGING_CACHE_PATH = INTERIM_DIR / "tagging_cache.json"
