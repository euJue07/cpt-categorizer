# cpt-categorizer

[![CCDS Project Template](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)

An automated categorization of unstandardized Philippine CPT descriptions using the OpenAI API.

---

## Project organization

```
├── Makefile           <- Convenience commands (lint, format, lint-schema, etc.)
├── README.md          <- This file
├── pyproject.toml     <- Project config, build (flit), Black, Ruff
├── requirements.txt   <- Python dependencies (use pip install -e . for local package)
├── setup.cfg          <- Legacy linter config (see pyproject.toml for Ruff/Black)
├── pytest.ini         <- Pytest markers and config
│
├── data               <- Data directory (not in repo; add CSV to data/raw/)
│   ├── external       <- Third-party data
│   ├── interim        <- Intermediate transformed data
│   ├── processed      <- Final canonical datasets
│   └── raw            <- Original immutable data (e.g. CPT List.csv)
│
├── docs               <- Project documentation (MkDocs)
│   ├── mkdocs.yml
│   ├── workflow.txt
│   └── docs/          <- MkDocs source pages (index, getting-started, agent_roles)
│
├── logs               <- Pipeline output (CSVs); generated files are gitignored
│
├── schema             <- CPT taxonomy (single source of truth)
│   ├── dimensions.json
│   ├── sections.json
│   └── subsections.json
│
├── scripts            <- One-off and utility scripts
│   ├── lint_schema.py
│   └── debug_tagging.py
│
├── tests              <- Pytest tests
│   └── agents/
│       └── test_tagging.py
│
└── cpt_categorizer    <- Source package
    ├── __init__.py
    ├── pipeline.py
    ├── agents/
    │   └── tagging.py
    ├── config/
    │   ├── directory.py
    │   └── openai.py
    └── utils/
        └── logging.py
```

---

## Development setup

1. Create and activate a virtual environment (Python 3.10+).
2. Install dependencies and the package in editable mode:

   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. Copy `.env.example` to `.env` and set any required keys (e.g. OpenAI).

Data is not included in the repo. Place your source CSV (e.g. CPT List) in `data/raw/`. Run `make data` for instructions if needed.

## Run the pipeline

From the project root:

```bash
python -m cpt_categorizer.pipeline
```

You will be prompted to choose: process the first N rows (default 10) or run a **random trial** (5 samples, seed 42) for reproducible verification. Outputs go to `logs/category_result.csv` and `logs/dimension_result.csv`. To run the trial with no prompts: `python scripts/run_trial.py`. See [Getting started](docs/docs/getting-started.md) for more options.

---

## AI agent framework

This project uses a modular AI agent framework to classify unstandardized CPT phrases. For agent roles and current implementation status, see [Agent roles](docs/docs/agent_roles.md).

**Implemented:**

- **Tagging Agent**: Identifies sections, subsections, and candidate detail tags from CPT descriptions using `schema/sections.json`, `schema/subsections.json`, and `schema/dimensions.json`.

**Planned:** Parsing, Normalizer, Scoring, Schema Compliance, Logging & Feedback, Correction, Schema Evolution, Schema Version Tracker (see docs).

### Schema files

- `schema/sections.json` — Top-level sections and subsections
- `schema/subsections.json` — Subsection definitions and allowed dimensions per section
- `schema/dimensions.json` — Dimension definitions and allowed values for tagging/normalization

---

## Lint and format

- **Lint**: `make lint` (Ruff + Black check)
- **Format**: `make format` (Black)
- **Schema**: `make lint-schema` (validates dimensions.json: no duplicates, snake_case, sort)
- **Test**: `make test` (or `pytest tests/`)

---

## Final output (goal)

- Validated CPT record with Section, Subsection, Details, confidence scores, and schema version ID
- Export as CSV, JSON, or database row

---

## Status

In development. Schema and scoring logic are evolving with feedback.
