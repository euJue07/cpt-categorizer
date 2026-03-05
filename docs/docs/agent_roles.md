# Agent roles

This project uses a modular AI agent framework to classify unstandardized CPT phrases. Each agent has a specific role in the classification pipeline.

## Current status

**Implemented:**

- **Parsing Agent** (`cpt_categorizer.agents.parsing`): Performs lightweight phrase cleanup before tagging.
- **Tagging Agent** (`cpt_categorizer.agents.tagging`): Identifies sections, subsections, and candidate detail tags for a CPT description using schema-backed function contracts.
- **Normalizer Agent** (`cpt_categorizer.agents.normalizer`): Normalizes keys/values to snake_case and canonical shape.
- **Schema Compliance Agent** (`cpt_categorizer.agents.compliance`): Validates and repairs tag payloads in balanced mode (recoverable normalization + warnings; invalid tags are dropped).

**Planned (not yet implemented):**

- **Scoring Agent**: Assign confidence levels to each tag and classification.
- **Logging & Feedback Agent**: Capture failures and borderline cases for review.
- **Correction Agent**: Apply manual corrections and store retraining data.
- **Schema Evolution Agent**: Propose schema updates based on real-world use.
- **Schema Version Tracker**: Maintain snapshots and traceability for schema changes.

## Intended flow

1. Raw CPT description → Parsing Agent  
2. Parsing Agent → Tagging Agent  
3. Tagging Agent → Normalizer Agent  
4. Normalizer Agent → Schema Compliance Agent  
5. If valid → final output; if invalid/partial → warning logs and continue with safe fallbacks.

Schema files in `schema/` are loaded only through `cpt_categorizer/schema_contract.py`, which validates cross-file consistency and generates a schema version hash attached to outputs.
