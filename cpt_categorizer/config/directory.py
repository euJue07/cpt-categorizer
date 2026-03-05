from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SCHEMA_DIR = _PROJECT_ROOT / "schema"
LOG_DIR = _PROJECT_ROOT / "logs"
RAW_DIR = _PROJECT_ROOT / "data" / "raw"
