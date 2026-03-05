# CPT categorizer: nine-agent architecture

**Last updated:** 2025-03-05

## Overview

Refactor the current single Tagging Agent into **9 specialized agents**: 3 taggers (Section, Subsection, Dimension), 3 suggestors (one per level), and 3 governors (one per level). Taggers assign labels from the schema and decide when a CPT description is "outside" the current schema; suggestors propose new section/subsection/dimension when missing and persist every suggestion to a **suggestion store**; governors decide whether a suggestion is acceptable, a duplicate, or rejected, using the schema and store **before** calling the LLM to reduce API usage.

See also: [Agent roles](docs/agent_roles.md), full plan in `.cursor/plans/`.

---

## Agent breakdown

| Role        | Agent                    | Responsibility |
|------------|--------------------------|----------------|
| **Tagging** | Section Tagging Agent    | Assign Section from `schema/sections.json`; if no match, assign `others` (outside schema). |
| **Tagging** | Subsection Tagging Agent | Given Section, assign Subsection from schema; if no match, signal outside. |
| **Tagging** | Dimension Tagging Agent  | Assign dimension values (actual + proposed existing/new); signal when values are outside schema. |
| **Suggestor** | Section Suggestor       | When tagging yields "outside" (e.g. `others`), suggest new section (key + description); **persist to suggestion store**; check store before LLM to avoid duplicate suggestions. |
| **Suggestor** | Subsection Suggestor    | When no subsection matches, suggest new subsection; persist to store; check store before LLM. |
| **Suggestor** | Dimension Suggestor     | When proposed `existing_dimensions` or `new_dimensions` exist, suggest new value or new dimension; persist to store; check store before LLM. |
| **Governor** | Section Governor        | For each pending suggestion: if key in schema → mark duplicate; if in store as accepted/rejected → reuse; **only call LLM for novel suggestions**; update store status. |
| **Governor** | Subsection Governor     | Same as Section Governor for subsection suggestions. |
| **Governor** | Dimension Governor      | Same as Section Governor for dimension key/value suggestions. |

---

## Suggestion store

- **Location:** `data/interim/suggestions.json` (or one file per type: `suggestions_sections.json`, etc.). Option B: SQLite at `data/interim/suggestions.db`.
- **Record shape (minimal):** `type` (section | subsection | dimension), `suggested_key` (snake_case), `suggested_description` or `suggested_values`, `context` (e.g. parent section for subsection), `status` (pending | accepted | rejected | duplicate), `source` (e.g. CPT description or batch id), `created_at`.
- **Usage:** Suggestors append after checking for existing same key; governors read and update `status`. Downstream can promote `accepted` into `schema/sections.json`, `schema/subsections.json`, `schema/dimensions.json`.

---

## API savings strategy

1. **Governors:** Load schema + suggestion store. For each pending suggestion: (1) key already in schema → set `duplicate`, no API; (2) same key in store with status accepted/rejected → reuse, no API; (3) only for **novel** suggestions → optional single LLM call to decide accept/reject, then update store. Optionally batch multiple novel suggestions in one LLM call.
2. **Suggestors:** Before suggesting, look up store by (type, normalized key, context). If a suggestion with same key exists (pending or accepted), return it and **skip LLM**. Only call LLM when no match; then append to store.
3. **Taggers:** Keep in-memory cache (as in current `tagging.py`); optionally persist cache to e.g. `data/interim/tagging_cache.json` for cross-run API savings.
4. **Persist tagging cache:** Save in-memory section/subsection cache to disk (e.g. `data/interim/tagging_cache.json`); load on agent init so reruns over the same CPT set avoid repeat API calls.
5. **Optional batching:** For large runs, optionally batch N CPT descriptions in one section-classification prompt; document as optional follow-up (no commitment to implement in current tasks).

---

## Token and cost logging

- **Audit trail:** `logs/usage.csv` is the single source of truth for every token and cost. Each row = one logical call (API call, cache hit, or store hit). Total spend = sum of `cost_usd`.
- **Log every path:** Every agent that can call the LLM must log: (1) successful API call with real usage, (2) failed call with 0 tokens when no response, (3) no-call paths (cache hit, store hit) with 0 tokens and a distinct `description` (e.g. `classify_sections_cache_hit`, `*_store_hit`).
- **Model-aware cost:** Cost must be computed from the model actually used (e.g. pricing table by model id: gpt-4o, gpt-4o-mini, etc.) so `cost_usd` is correct when `OPENAI_MODEL` changes.
- **Single logging helper:** All agents use the same helper (e.g. `log_agent_usage()`) that writes to `logs/usage.csv`; suggestors and governors must use it for every invocation and no-call path.

---

## High-level flow

1. **Tagging:** Section → Subsection → Dimension (sequential). Each step can signal "outside schema" or emit proposed values.
2. **Suggestors:** Invoked when a tagger signals outside or proposes new values; check store, then optionally call LLM and append to store.
3. **Governors:** Run on pending suggestions (batch or on-demand); resolve using schema + store first; LLM only for novel items; write status back to store.

---

## Implementation scope

- **In scope:** Nine agents (refactor tagging into three taggers; add three suggestors and three governors), suggestion store module, pipeline wiring so "outside" / "proposed" trigger suggestors and governors use store before LLM. Documentation and tests for store and agents.
- **Out of scope (for later):** Full schema promotion workflow (accepted → edit schema JSON), UI for reviewing suggestions, optional tagging cache persistence.

---

## Key files

| Action  | Path |
|---------|------|
| New     | `cpt_categorizer/suggestion_store.py` — load/save/query suggestions by type, key, status. |
| New     | `cpt_categorizer/agents/section_tagging.py`, `subsection_tagging.py`, `dimension_tagging.py` (or three classes in one module). |
| New     | `cpt_categorizer/agents/section_suggestor.py`, `subsection_suggestor.py`, `dimension_suggestor.py`. |
| New     | `cpt_categorizer/agents/section_governor.py`, `subsection_governor.py`, `dimension_governor.py`. |
| Change  | `cpt_categorizer/pipeline.py` — use three taggers; on "outside" / "proposed", call suggestors; run governors on pending suggestions. |
| Change  | `docs/docs/agent_roles.md` — document 9 agents and suggestion store. |

---

## Tasks

- [x] Add suggestion store module (`cpt_categorizer/suggestion_store.py`) and data path (e.g. `data/interim/suggestions.json`).
- [x] Split current tagging into Section / Subsection / Dimension tagging agents (or three classes in one module).
- [ ] Add Section Suggestor agent; persist suggestions to store; check store before LLM.
- [ ] Add Subsection Suggestor agent; same pattern.
- [ ] Add Dimension Suggestor agent; same pattern.
- [ ] Add Section Governor agent; resolve pending suggestions using schema + store first; LLM only for novel suggestions.
- [ ] Add Subsection Governor agent; same pattern.
- [ ] Add Dimension Governor agent; same pattern.
- [ ] Wire taggers, suggestors, and governors in pipeline (trigger suggestors on "outside" / "proposed"; run governors on pending suggestions).
- [ ] Update docs (e.g. `docs/docs/agent_roles.md`) with 9 agents and suggestion store.
- [ ] Add unit tests for store and each agent; integration test for pipeline + store + governor resolution.
- [ ] Persist tagging cache to `data/interim/tagging_cache.json`; load on init.
- [ ] Ensure every token/cost is logged: log cache and store hits with 0 tokens; use model-aware pricing in usage logger.

---

## Recent completions

- **2025-03-05** — Split tagging into SectionTaggingAgent, SubsectionTaggingAgent, DimensionTaggingAgent in tagging.py; TaggingAgent now orchestrates the three; tests patched for sub-agents; agents/__init__.py exports new classes.
- **2025-03-05** — Added suggestion_store.py with load/save/query/append/update_status; INTERIM_DIR and SUGGESTIONS_PATH in config; data/interim/.gitkeep; tests/test_suggestion_store.py (12 tests).
