# Agent roles

This project uses a modular AI agent framework to classify unstandardized CPT phrases. Each agent has a specific role in the classification pipeline.

## Current status

**Implemented:**

- **Tagging Agent** (`cpt_categorizer.agents.tagging`): Identifies sections, subsections, and candidate detail tags for a given CPT description using the schema (sections, subsections, dimensions). Supports `classify_sections`, `classify_subsections`, `generate_tags`, and `tag_entry`.

**Planned (not yet implemented):**

- **Parsing Agent**: Clean and prepare raw CPT phrases before tagging.
- **Normalizer Agent**: Standardize and schema-align detail tags produced by the Tagging Agent.
- **Scoring Agent**: Assign confidence levels to each tag and classification.
- **Schema Compliance Agent**: Validate outputs against the current schema and formatting rules.
- **Logging & Feedback Agent**: Capture failures and borderline cases for review.
- **Correction Agent**: Apply manual corrections and store retraining data.
- **Schema Evolution Agent**: Propose schema updates based on real-world use.
- **Schema Version Tracker**: Maintain snapshots and traceability for schema changes.

## Intended flow

1. Raw CPT description → Parsing Agent  
2. Parsing Agent → Tagging Agent  
3. Tagging Agent → Normalizer Agent  
4. Normalizer Agent → Scoring Agent  
5. Scoring Agent → Schema Compliance Agent  
6. If valid → final output; if invalid → Logging & Feedback → Correction → Schema Evolution → back to Schema Compliance.

Schema files in `schema/` (sections.json, subsections.json, dimensions.json) are the single source of truth for the taxonomy used by the Tagging Agent and future agents.
