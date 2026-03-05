# Schema — source of truth

The **`schema/`** directory is the single source of truth for the CPT taxonomy and dimensions. All pipeline code and agents load from these files.

Runtime loading is centralized via `cpt_categorizer/schema_contract.py`, which validates cross-file references and computes a schema version hash used in pipeline outputs/logs.

## Files

| File | Role |
|------|------|
| **sections.json** | Top-level taxonomy: section id, description, list of subsection ids (e.g. anesthesiology, dental, laboratory, imaging, procedures). |
| **subsections.json** | Nested by section → subsection: description and **dimensions** allowed for that subsection. |
| **dimensions.json** | Dimension definitions: each dimension has `description` and `values` (allowed enum). Used for tagging and normalization. |

## Conventions

When editing schema files:

- **Dimension values** in `dimensions.json` must be **deduplicated** (no repeated entries within a dimension’s `values` array).
- **Format**: All dimension values must be **snake_case** (lowercase with underscores).
- **Order**: Each dimension’s `values` array should be **alphabetically sorted** for stable diffs and to satisfy the schema linter.

Run `make lint-schema` to validate these rules.

## Schema versioning

The runtime computes a deterministic schema version hash from `sections.json`, `subsections.json`, and `dimensions.json` contents. This version is attached to generated category and dimension outputs so downstream consumers can trace which schema revision produced each row.
