# Agent roles

This project uses a modular AI agent framework to classify unstandardized CPT phrases. Each agent has a specific role in the classification pipeline.

## Current status

### Pipeline support agents

- **Parsing Agent** (`cpt_categorizer.agents.parsing`): Lightweight phrase cleanup before tagging.
- **Normalizer Agent** (`cpt_categorizer.agents.normalizer`): Normalize keys/values to snake_case and canonical shape.
- **Schema Compliance Agent** (`cpt_categorizer.agents.compliance`): Validate and repair tag payloads in balanced mode (recoverable normalization + warnings; invalid tags are dropped).

### Nine classification agents

All nine are implemented. Taggers assign labels from the schema and signal when a CPT description is "outside" the schema; suggestors propose new section/subsection/dimension and persist to the suggestion store; governors resolve pending suggestions using the schema and store before calling the LLM to reduce API usage.

| Role        | Agent                    | Module / responsibility |
|------------|--------------------------|--------------------------|
| **Tagging** | Section Tagging Agent    | `cpt_categorizer.agents.tagging` — Assign Section from schema; if no match, assign `others` (outside schema). |
| **Tagging** | Subsection Tagging Agent | Same module — Given Section, assign Subsection from schema; if no match, signal outside. |
| **Tagging** | Dimension Tagging Agent  | Same module — Assign dimension values (actual + proposed existing/new); signal when values are outside schema. |
| **Suggestor** | Section Suggestor       | `cpt_categorizer.agents.section_suggestor` — When tagging yields `others`, suggest new section; check store before LLM; persist to store. |
| **Suggestor** | Subsection Suggestor    | `cpt_categorizer.agents.subsection_suggestor` — When no subsection matches, suggest new subsection; same store pattern. |
| **Suggestor** | Dimension Suggestor     | `cpt_categorizer.agents.dimension_suggestor` — When proposed dimensions exist, suggest new value/dimension; same store pattern. |
| **Governor** | Section Governor        | `cpt_categorizer.agents.section_governor` — Resolve pending section suggestions: schema/store first; LLM only for novel; update store status. |
| **Governor** | Subsection Governor     | `cpt_categorizer.agents.subsection_governor` — Same for subsection suggestions. |
| **Governor** | Dimension Governor      | `cpt_categorizer.agents.dimension_governor` — Same for dimension key/value suggestions. |

The three taggers are implemented as `SectionTaggingAgent`, `SubsectionTaggingAgent`, and `DimensionTaggingAgent` in `cpt_categorizer.agents.tagging`; `TaggingAgent` orchestrates them.

## Suggestion store

- **Location:** `data/interim/suggestions.json` (from `cpt_categorizer.config.directory`: `SUGGESTIONS_PATH`).
- **Module:** `cpt_categorizer.suggestion_store` — `load`, `find_by_type`, `find_by_type_key`, `find_by_status`, `append`, `update_status`.
- **Record shape:** `type` (section | subsection | dimension), `suggested_key`, `suggested_description` or `suggested_values`, `context`, `status` (pending | accepted | rejected | duplicate), `source`, `created_at`, `id`.
- **Usage:** Suggestors query by (type, key, context) before calling the LLM and append new suggestions; governors read pending suggestions, resolve via schema/store or LLM, then call `update_status`. Accepted suggestions can be promoted into `schema/sections.json`, `schema/subsections.json`, or `schema/dimensions.json` in a later workflow.

## Intended flow

1. Raw CPT description → **Parsing Agent**.
2. **Tagging:** Section → Subsection → Dimension (orchestrated by `TaggingAgent`). On section `others` → Section Suggestor; on empty subsection → Subsection Suggestor; on proposed dimensions → Dimension Suggestor. Each suggestor checks the store first, then optionally calls the LLM and appends to the store.
3. **Normalizer Agent** → **Schema Compliance Agent** → output or warnings.
4. After processing all rows: Section, Subsection, and Dimension **Governors** resolve all pending suggestions (schema + store first; LLM only for novel suggestions), updating store status.

Schema files in `schema/` are loaded only through `cpt_categorizer/schema_contract.py`, which validates cross-file consistency and generates a schema version hash attached to outputs.

## Planned (later)

- **Scoring Agent:** Assign confidence levels to each tag and classification.
- **Logging & Feedback Agent:** Capture failures and borderline cases for review.
- **Correction Agent:** Apply manual corrections and store retraining data.
- **Schema Evolution Agent:** Propose schema updates based on real-world use.
- **Schema Version Tracker:** Maintain snapshots and traceability for schema changes.
