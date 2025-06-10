# cpt-categorizer

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

An automated categorization of unstandardized Philippine CPT descriptions using OpenAI API

## Project Organization

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
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         cpt_categorizer and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── cpt_categorizer   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes cpt_categorizer a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

AI Agent Roles and Interaction Framework for CPT Classification

1. Parsing Agent

Role: Prepares clean and informative text by extracting meaningful content from raw CPT descriptions without assigning structure

Tasks:

* Expand known medical abbreviations (e.g., "PT" → "prothrombin time" or flag for disambiguation)
* Remove non-informative or administrative phrases (e.g., prefixes, filler codes, redundant suffixes)
* Correct improper capitalization while preserving medically relevant acronyms (e.g., MRI, HIV)
* Isolate medically relevant phrases (e.g., "with contrast", "2D echo")
* Clean up punctuation, quotes, and parenthetical content when not critical
* Return cleaned version for downstream tagging and normalization

Maintains:

* N/A (passes cleaned string forward only)

2. Tagging Agent

Role: Assign the most appropriate Section, Subsection, and detail fields based on extracted medical phrases — using your schema files as lookup references.

Tasks:

* Match cleaned phrases against Section and Subsection vocabulary (from `cpt_section_subsection.json`)
* Map concepts to detail dimensions (e.g., method, sample\_type, location, analyte) using `cpt_detail.json`
* Apply priority rules or heuristics when multiple matches are possible
* Defer ambiguous decisions to Scoring Agent
* Do not normalize values — rely on the Normalizer Agent
* Flag uncertain tags

Maintains:

* Reference files: `cpt_section_subsection.json`, `cpt_detail.json`

3. Normalizer Agent

Role: Transforms detail candidates from the Tagging Agent into schema-compliant and standardized values

Tasks:

* Convert phrases and tag values into standardized `dimension:value` form
* Apply formatting conventions (snake\_case, lowercase, strict spelling)
* Resolve synonyms and unify equivalents
* Ensure outputs exist under valid dimensions in `cpt_detail.json`
* Reject or defer non-matching terms to Schema Compliance Agent

Maintains:

* Reference file: `cpt_detail_rule.json`
* Uses `cpt_detail.json` for validation

4. Scoring Agent

Role: Evaluates the confidence of each tag and overall classification

Tasks:

* Assign confidence scores to Section, Subsection, and each detail dimension
* Flag low-confidence entries for human review
* Feed confidence data to Logging & Feedback Agent

Evaluates:

* `section_subsection.json`, `cpt_detail.json`, internal logs

Maintains:

* Confidence logs (not schema files)

5. Schema Compliance Agent

Role: Validates the final structured output to ensure schema compliance

Tasks:

* Verify Section, Subsection, and detail values exist in schema
* Confirm format rules (e.g., snake\_case, lowercase, spelling)
* Detect invalid or deprecated values
* Flag multiple-dimension values and invalid structures
* Pass flagged outputs to Logging & Feedback Agent
* Never suggest rule changes

Maintains:

* Reference files: `cpt_section_subsection.json`, `cpt_detail.json`, `cpt_detail_rule.json`

6. Logging & Feedback Agent

Role: Captures and organizes edge cases, compliance failures, and ambiguities

Tasks:

* Log all UNCAT or ambiguous classifications
* Store flagged entries (invalid values, structure issues, conflicts)
* Route cases to Correction Agent or Schema Evolution Agent
* Track frequency of unknown/borderline terms
* Maintain structured log format

Maintains:

* Feedback logs
* Log fields include: raw\_cpt, extracted\_phrases, failing\_dimensions, originating\_agent, confidence\_score, resolution\_status, log\_type

7. Schema Evolution Agent

Role: Learns from logs and corrections to propose schema updates

Tasks:

* Propose new Sections, Subsections, dimensions, or values
* Detect ambiguities or conflicts
* Suggest merges/splits/deprecations
* Generate updates with justification and frequency
* Forward proposals to governance
* Maintain version history

Maintains:

* Proposed updates to: `cpt_detail.json`, `cpt_section_subsection.json`
* Version history (linked to Schema Version Tracker)

8. Correction Agent

Role: Applies human corrections and feeds improvements back into system

Tasks:

* Accept manual corrections for Section, Subsection, detail fields
* Validate input for schema compliance
* Forward corrections to Logging & Feedback Agent and Schema Evolution Agent
* Store corrections in retraining set

Maintains:

* Correction log
* Retraining dataset

9. Schema Version Tracker

Function: Ensures traceability and reproducibility for schema-driven classification

Tasks:

* Generate schema version ID/hash on updates
* Store schema snapshots:

  * `cpt_detail.json`
  * `cpt_section_subsection.json`
  * `cpt_detail_rule.json`
* Tag each classification output with schema version
* Maintain a change log (what, when, who)
* Support rollback and compatibility

Maintains:

* Schema snapshots
* Version ID/hash
* Change log

Agent Interaction Flow (Descriptive Outline):

1. Raw CPT Description → sent to → Parsing Agent
2. Parsing Agent → cleans and sends to → Tagging Agent
3. Tagging Agent → identifies tags and sends to → Normalizer Agent
4. Normalizer Agent → standardizes tags and sends to → Scoring Agent
5. Scoring Agent → assigns confidence and sends to → Schema Compliance Agent
6. Schema Compliance Agent

   * If valid → proceeds to → Final Output Generator
   * If invalid → sends to → Logging & Feedback Agent
7. Logging & Feedback Agent → logs and routes to → Correction Agent
8. Correction Agent → applies human corrections and sends to → Schema Evolution Agent
9. Schema Evolution Agent → updates schema and loops back to → Schema Compliance Agent

Final Output Generator

Function:

* Aggregates validated classification: Section, Subsection, Details
* Attaches schema version, confidence, and explanation
* Exports data (e.g., CSV, JSON)
