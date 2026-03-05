# Getting started

This page describes how to set up and run the CPT categorizer on a clean install.

## Prerequisites

- Python 3.10 or newer
- A virtual environment (recommended)

## Setup

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies and the package in editable mode:

   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. Copy `.env.example` to `.env` and set your OpenAI API key (and optionally `OPENAI_MODEL`):

   ```bash
   cp .env.example .env
   # Edit .env: set OPENAI_API_KEY=sk-... and optionally OPENAI_MODEL=gpt-4o
   ```

4. **Data:** The repo does not include source data. Create the data directory and add your CSV:

   - The directory structure is already in the repo (`data/raw/`, `data/interim/`, etc.). Place your source CSV in `data/raw/`.
   - Default input file expected by the pipeline: `data/raw/CPT Desc With Utilization.csv`. Your CSV should have a description column (e.g. `CPTDesc`) and optionally a key column (e.g. `CPTDescKey`).
   - If you use a different file name or path, pass it when calling the pipeline from code (see below).

## Run the pipeline

From the project root:

```bash
python -m cpt_categorizer.pipeline
```

You will be prompted to choose a run mode:

- **1) First N rows** — Process the first N rows (default 10). You will then be prompted for how many rows.
- **2) Random trial (5 samples, seed 42)** — Process 5 randomly sampled rows with a fixed random seed for reproducibility. Use this to verify the pipeline works without processing the whole dataset.

Outputs are written to `logs/category_result.csv` and `logs/dimension_result.csv`. Token and cost usage are logged to `logs/usage.csv`.

To run the trial with no prompts (e.g. for scripting):

```bash
python scripts/run_trial.py
```

To run the pipeline from Python with custom options:

```python
from pathlib import Path
from cpt_categorizer.pipeline import run_pipeline

run_pipeline(
    top_n=10,
    csv_path=Path("data/raw/your_file.csv"),  # optional
    debug=True,
    col_name_desc="CPTDesc",
    col_name_key="CPTDescKey",
)

# Or use random sampling for a reproducible trial:
run_pipeline(sample_n=5, random_seed=42, debug=True)
```

## Next steps

- See the [README](../../README.md) for project layout, lint/format, and schema validation.
- See [Agent roles](agent_roles.md) for the nine-agent architecture (taggers, suggestors, governors) and suggestion store.
