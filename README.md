# cpt-categorizer

[![CCDS Project Template](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)

An automated categorization of unstandardized Philippine CPT descriptions using the OpenAI API.

---

## 📁 Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Project documentation and visual references
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks (e.g., `1.0-jme-cpt-parsing-analysis.ipynb`)
│
├── pyproject.toml     <- Project configuration and tool integration for `cpt_categorizer`
├── requirements.txt   <- The requirements file for reproducing the analysis environment
├── setup.cfg          <- Configuration file for flake8 and other linters
│
├── references         <- Data dictionaries, schema snapshots, schema version logs, and manuals
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures for reporting
│
└── cpt_categorizer    <- Source code
    ├── __init__.py             <- Initializes the Python module
    ├── config.py               <- Stores shared configurations
    ├── dataset.py              <- Ingests or prepares datasets
    ├── features.py             <- Feature extraction and transformation
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Model inference logic
    │   └── train.py            <- Model training logic
    ├── plots.py                <- Visualization utilities
    └── agents
        ├── parsing.py          <- Logic for Parsing Agent
        ├── tagging.py          <- Tagging Agent implementation
        ├── normalization.py    <- Normalizer Agent logic
        ├── scoring.py          <- Scoring Agent
        ├── compliance.py       <- Schema Compliance Agent
        ├── logging.py          <- Logging & Feedback Agent
        ├── correction.py       <- Correction Agent
        ├── evolution.py        <- Schema Evolution Agent
        └── versioning.py       <- Schema Version Tracker
```

---

## 🧠 AI Agent Framework Overview

This project uses a modular AI agent framework to classify unstandardized CPT phrases. Each agent performs a specific role in the classification pipeline. For full descriptions and agent responsibilities, see [`docs/agent_roles.md`](docs/agent_roles.md).

**Key Agent Roles:**

* **Parsing Agent**: Cleans and prepares raw CPT phrases
* **Tagging Agent**: Identifies Sections, Subsections, and candidate detail tags
* **Normalizer Agent**: Standardizes and schema-aligns detail tags
* **Scoring Agent**: Assigns confidence levels to each tag and classification
* **Schema Compliance Agent**: Validates against current schema and formatting rules
* **Logging & Feedback Agent**: Captures failures and borderline cases
* **Correction Agent**: Applies manual corrections and stores retraining data
* **Schema Evolution Agent**: Proposes updates to schema based on real-world use
* **Schema Version Tracker**: Maintains snapshots and traceability for schema changes

### ⚙️ Key Schema Files

* `cpt_detail.json` — Valid detail dimensions and tag values
* `cpt_section_subsection.json` — Allowed sections and subsections
* `cpt_detail_rule.json` — Formatting, spelling, and normalization rules
* Schema snapshots and logs stored in `references/`

---

## 🔁 Agent Interaction Flow

```
1. Raw CPT Description → Parsing Agent
2. Parsing Agent → cleans → Tagging Agent
3. Tagging Agent → identifies tags → Normalizer Agent
4. Normalizer Agent → standardizes → Scoring Agent
5. Scoring Agent → scores → Schema Compliance Agent
6. Schema Compliance Agent:
   - If valid → Final Output Generator
   - If invalid → Logging & Feedback Agent
7. Logging & Feedback Agent → Correction Agent
8. Correction Agent → Schema Evolution Agent
9. Schema Evolution Agent → loops back to Schema Compliance Agent
```

---

## 📤 Final Output

* Validated and classified CPT record with:

  * Section, Subsection, Details
  * Confidence scores
  * Schema version ID
* Exported as CSV, JSON, or database row

---

## ✅ Status

This project is in development. Schema proposals, rule evolution, and scoring logic are actively evolving with production feedback.