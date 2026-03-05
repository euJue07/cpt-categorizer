"""Dimension Suggestor agent: suggest new dimension values or new dimension keys when proposed exist; check store before LLM."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Optional

import openai

from cpt_categorizer.agents.tagging import to_snake_case
from cpt_categorizer.config.directory import SUGGESTIONS_PATH
from cpt_categorizer.config.openai import OPENAI_MODEL
from cpt_categorizer.suggestion_store import append, find_by_type, find_by_type_key
from cpt_categorizer.utils.logging import log_agent_usage

DIMENSION_SUGGEST_PROMPT_TEMPLATE = """You are a medical classification expert. The following CPT procedure description led to a proposed new dimension key "{dimension_key}" with example values: {values_str}.

Existing dimension keys in the schema (do not reuse for the key name): {existing_dimension_keys}

CPT description:
{text_description}

Propose a short description for this new dimension (what it represents clinically or technically). Return only the description; the dimension key and values are already captured.
"""


def _context_for_dimension(
    section: str, subsection: str, dimension_key: str, text_description: str
) -> str:
    """Stable context for deduplication: same section + subsection + dimension key + description -> same context."""
    normalized = text_description.strip().lower()
    desc_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{section}:{subsection}:{dimension_key}:{desc_hash}"


def _log_usage(
    schema_version: str,
    text_description: str,
    parsed_output: Any,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    model: str,
    runtime_ms: int,
    success: bool,
    error_message: str,
    description: str,
) -> None:
    log_agent_usage(
        timestamp=datetime.now().isoformat(),
        raw_text=text_description,
        description=description,
        parsed_output=(
            json.dumps(parsed_output) if not isinstance(parsed_output, str) else parsed_output
        ),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        model=model,
        runtime_ms=runtime_ms,
        success=success,
        is_error=not success,
        error_message=error_message,
        schema_version=schema_version,
    )


def _extract_values(value_items: Any) -> list[str]:
    """Extract snake_case value strings from tagger output: list of {value, confidence}."""
    if not isinstance(value_items, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value_items:
        if not isinstance(item, dict):
            continue
        raw = str(item.get("value", "")).strip()
        val = to_snake_case(raw)
        if val and val not in seen:
            seen.add(val)
            out.append(val)
    return out


class DimensionSuggestorAgent:
    """Suggests new dimension values (existing dimension) or new dimension keys with description; checks store before LLM."""

    def __init__(
        self,
        dimension_schema: dict[str, Any],
        subsection_schema: dict[str, Any],
        store_path: Optional[Path] = None,
        client: Optional[openai.OpenAI] = None,
        schema_version: str = "",
    ):
        self.dimension_schema = dimension_schema
        self.subsection_schema = subsection_schema
        self.store_path = store_path if store_path is not None else SUGGESTIONS_PATH
        self.client = client or openai.OpenAI()
        self.schema_version = schema_version
        self._existing_dimension_keys = ", ".join(sorted(dimension_schema.keys()))

    def suggest_dimensions(
        self,
        section: str,
        subsection: str,
        text_description: str,
        proposed: dict[str, Any],
        source: str = "",
    ) -> list[dict[str, Any]]:
        """
        Suggest new dimension values or new dimensions for proposed existing_dimensions and new_dimensions.
        Checks store before LLM; returns existing suggestion if same context or same key+parent in store.
        For existing_dimensions no LLM is used; for new_dimensions one LLM call per key gets suggested_description.
        """
        if not section or not section.strip():
            return []
        if not subsection or not subsection.strip():
            return []
        if section.strip().lower() == "others" or subsection.strip().lower() == "others":
            return []
        if not text_description or not text_description.strip():
            return []
        existing_dims = proposed.get("existing_dimensions") if isinstance(proposed, dict) else {}
        new_dims = proposed.get("new_dimensions") if isinstance(proposed, dict) else {}
        if not isinstance(existing_dims, dict):
            existing_dims = {}
        if not isinstance(new_dims, dict):
            new_dims = {}
        # Collect (dimension_key, value_list, is_new_dimension)
        items: list[tuple[str, list[str], bool]] = []
        for dim_key, value_list in existing_dims.items():
            key = to_snake_case(str(dim_key))
            if not key:
                continue
            values = _extract_values(value_list)
            if values:
                items.append((key, values, False))
        for dim_key, value_list in new_dims.items():
            key = to_snake_case(str(dim_key))
            if not key:
                continue
            values = _extract_values(value_list)
            if values:
                items.append((key, values, True))
        if not items:
            return []

        results: list[dict[str, Any]] = []

        for dimension_key, suggested_values, is_new_dimension in items:
            dimension_suggestions = find_by_type(self.store_path, "dimension")
            context = _context_for_dimension(section, subsection, dimension_key, text_description)
            start = time.time()

            # 1. Check store by context: already suggested for this (section, subsection, dimension_key, description)?
            matching_by_context = [
                s
                for s in dimension_suggestions
                if s.get("context") == context
                and s.get("suggested_key") == dimension_key
                and s.get("status") in ("pending", "accepted")
            ]
            if matching_by_context:
                runtime_ms = int((time.time() - start) * 1000)
                _log_usage(
                    schema_version=self.schema_version,
                    text_description=text_description,
                    parsed_output=matching_by_context[0],
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    model="",
                    runtime_ms=runtime_ms,
                    success=True,
                    error_message="",
                    description="dimension_suggest_store_hit",
                )
                results.append(matching_by_context[0])
                continue

            # 2. Build record; for new_dimensions call LLM for suggested_description
            suggested_description: str = ""
            response = None
            success = True
            error_message = ""

            if is_new_dimension:
                try:
                    prompt = DIMENSION_SUGGEST_PROMPT_TEMPLATE.format(
                        dimension_key=dimension_key,
                        values_str=", ".join(suggested_values),
                        existing_dimension_keys=self._existing_dimension_keys,
                        text_description=text_description,
                    )
                    function_spec = {
                        "name": "suggest_dimension_description",
                        "description": "Return a short description for the new dimension.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "suggested_description": {
                                    "type": "string",
                                    "description": "Short description of what this dimension represents.",
                                },
                            },
                            "required": ["suggested_description"],
                        },
                    }
                    response = self.client.chat.completions.create(
                        model=OPENAI_MODEL,
                        temperature=0,
                        messages=[
                            {
                                "role": "system",
                                "content": "You suggest short descriptions for medical/dimension keys in JSON.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        functions=[function_spec],
                        function_call={"name": "suggest_dimension_description"},
                    )
                    parsed = json.loads(response.choices[0].message.function_call.arguments)
                    suggested_description = (
                        str(parsed.get("suggested_description", "")).strip() or ""
                    )
                except Exception as exc:
                    error_message = str(exc)
                    success = False
                    runtime_ms = int((time.time() - start) * 1000)
                    _log_usage(
                        schema_version=self.schema_version,
                        text_description=text_description,
                        parsed_output={},
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        model="",
                        runtime_ms=runtime_ms,
                        success=False,
                        error_message=error_message,
                        description="dimension_suggest",
                    )
                    continue

            # 3. Check store by key + parent: same key under same section/subsection already suggested (pending/accepted)?
            existing_by_key = find_by_type_key(
                self.store_path, "dimension", dimension_key, context=None
            )
            existing_reusable = [
                s
                for s in existing_by_key
                if s.get("parent_section") == section
                and s.get("parent_subsection") == subsection
                and s.get("status") in ("pending", "accepted")
            ]
            if existing_reusable:
                runtime_ms = int((time.time() - start) * 1000)
                _log_usage(
                    schema_version=self.schema_version,
                    text_description=text_description,
                    parsed_output=existing_reusable[0],
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    model="",
                    runtime_ms=runtime_ms,
                    success=True,
                    error_message="",
                    description="dimension_suggest_duplicate_key",
                )
                results.append(existing_reusable[0])
                continue

            # 4. Append new suggestion
            record: dict[str, Any] = {
                "type": "dimension",
                "suggested_key": dimension_key,
                "suggested_values": suggested_values,
                "context": context,
                "parent_section": section,
                "parent_subsection": subsection,
                "status": "pending",
                "source": source,
            }
            if is_new_dimension:
                record["suggested_description"] = suggested_description
            append(self.store_path, record)
            dimension_suggestions = find_by_type(self.store_path, "dimension")
            new_record = next(
                (
                    s
                    for s in dimension_suggestions
                    if s.get("context") == context and s.get("suggested_key") == dimension_key
                ),
                record,
            )
            results.append(new_record)

            runtime_ms = int((time.time() - start) * 1000)
            usage = getattr(response, "usage", None) if response else None
            prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
            completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
            total_tokens = getattr(usage, "total_tokens", 0) if usage else 0
            model = getattr(response, "model", "") if response else ""
            _log_usage(
                schema_version=self.schema_version,
                text_description=text_description,
                parsed_output=new_record,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                model=model,
                runtime_ms=runtime_ms,
                success=success,
                error_message=error_message,
                description="dimension_suggest",
            )

        return results
