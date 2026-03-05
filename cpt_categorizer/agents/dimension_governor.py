"""Dimension Governor agent: resolve pending dimension suggestions using schema + store first; LLM only for novel suggestions."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import openai

from cpt_categorizer.agents.tagging import to_snake_case
from cpt_categorizer.config.directory import SUGGESTIONS_PATH
from cpt_categorizer.config.openai import OPENAI_MODEL
from cpt_categorizer.suggestion_store import (
    find_by_type,
    find_by_type_key,
    update_status,
)
from cpt_categorizer.utils.logging import log_agent_usage

DIMENSION_GOVERNOR_PROMPT_TEMPLATE = """You are a medical taxonomy curator. A suggested dimension (or new values for an existing dimension) was proposed for the CPT categorizer schema.

Suggested dimension key: {suggested_key}
Suggested values: {suggested_values_str}
{suggested_description_line}
Existing dimension keys in the schema (for reference): {existing_dimension_keys}
{existing_values_for_key_line}

Decide whether to ACCEPT or REJECT this suggestion. Accept only if the proposed dimension or values are distinct, clinically meaningful, and would improve the taxonomy. Reject if they duplicate or overlap existing dimensions/values, or are too narrow/vague.
"""


def _log_usage(
    schema_version: str,
    raw_text: str,
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
        raw_text=raw_text,
        description=description,
        parsed_output=json.dumps(parsed_output) if not isinstance(parsed_output, str) else parsed_output,
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


def _normalize_values(value_list: Any) -> list[str]:
    """Return list of snake_case strings from suggestion record suggested_values."""
    if not isinstance(value_list, list):
        return []
    out: list[str] = []
    for v in value_list:
        raw = str(v).strip() if v is not None else ""
        norm = to_snake_case(raw) if raw else ""
        if norm and norm not in out:
            out.append(norm)
    return out


class DimensionGovernorAgent:
    """Resolves pending dimension suggestions: schema duplicate and store reuse first; LLM only for novel suggestions."""

    def __init__(
        self,
        dimension_schema: dict[str, Any],
        store_path: Optional[Path] = None,
        client: Optional[openai.OpenAI] = None,
        schema_version: str = "",
    ):
        self.dimension_schema = dimension_schema
        self.store_path = store_path if store_path is not None else SUGGESTIONS_PATH
        self.client = client or openai.OpenAI()
        self.schema_version = schema_version
        self._existing_dimension_keys_str = ", ".join(sorted(dimension_schema.keys()))

    def _existing_values_for_key(self, dim_key: str) -> str:
        """Return comma-separated values for the given dimension key."""
        entry = self.dimension_schema.get(dim_key)
        if not isinstance(entry, dict):
            return "(none)"
        values = entry.get("values")
        if not isinstance(values, list):
            return "(none)"
        return ", ".join(sorted(str(v) for v in values))

    def _is_duplicate(self, suggested_key: str, suggested_values: list[str]) -> bool:
        """
        True if suggestion is duplicate: key in schema and (no values, or all values already in schema for that key).
        """
        if not suggested_key or suggested_key not in self.dimension_schema:
            return False
        schema_values = self.dimension_schema.get(suggested_key, {})
        if not isinstance(schema_values, dict):
            return True
        allowed = set(schema_values.get("values") or [])
        if not suggested_values:
            return True
        return all(v in allowed for v in suggested_values)

    def resolve_pending_dimensions(self) -> list[dict[str, Any]]:
        """
        Resolve all pending dimension suggestions: mark duplicate if key+values in schema,
        reuse status if same key+parent_section+parent_subsection has accepted/rejected in store, else call LLM.
        Returns list of updated suggestion records (with status after resolution).
        """
        dimension_suggestions = find_by_type(self.store_path, "dimension")
        pending = [s for s in dimension_suggestions if s.get("status") == "pending"]
        results: list[dict[str, Any]] = []

        for s in pending:
            suggestion_id = s.get("id", "")
            suggested_key_raw = s.get("suggested_key", "")
            suggested_key = to_snake_case(suggested_key_raw) if suggested_key_raw else ""
            suggested_values = _normalize_values(s.get("suggested_values", []))
            suggested_description = s.get("suggested_description", "")
            parent_section = s.get("parent_section", "")
            parent_subsection = s.get("parent_subsection", "")
            start = time.time()
            runtime_ms = 0

            # 1. Key in schema and (empty values or all values already in schema) -> duplicate
            if self._is_duplicate(suggested_key, suggested_values):
                update_status(self.store_path, suggestion_id, "duplicate")
                runtime_ms = int((time.time() - start) * 1000)
                _log_usage(
                    schema_version=self.schema_version,
                    raw_text=suggestion_id or suggested_key,
                    parsed_output={
                        "suggested_key": suggested_key,
                        "suggested_values": suggested_values,
                        "parent_section": parent_section,
                        "parent_subsection": parent_subsection,
                        "status": "duplicate",
                    },
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    model="",
                    runtime_ms=runtime_ms,
                    success=True,
                    error_message="",
                    description="dimension_governor_duplicate",
                )
                results.append({**s, "status": "duplicate"})
                continue

            # 2. Same key + parent_section + parent_subsection has accepted/rejected in store -> reuse
            by_key = find_by_type_key(self.store_path, "dimension", suggested_key, context=None)
            reused = next(
                (
                    x
                    for x in by_key
                    if x.get("parent_section") == parent_section
                    and x.get("parent_subsection") == parent_subsection
                    and x.get("status") in ("accepted", "rejected")
                    and x.get("id") != suggestion_id
                ),
                None,
            )
            if reused:
                status = reused["status"]
                update_status(self.store_path, suggestion_id, status)
                runtime_ms = int((time.time() - start) * 1000)
                _log_usage(
                    schema_version=self.schema_version,
                    raw_text=suggestion_id or suggested_key,
                    parsed_output={
                        "suggested_key": suggested_key,
                        "parent_section": parent_section,
                        "parent_subsection": parent_subsection,
                        "status": status,
                    },
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    model="",
                    runtime_ms=runtime_ms,
                    success=True,
                    error_message="",
                    description="dimension_governor_store_reuse",
                )
                results.append({**s, "status": status})
                continue

            # 3. Novel -> LLM accept/reject
            suggested_values_str = ", ".join(suggested_values) if suggested_values else "(none)"
            suggested_description_line = (
                f"Suggested description (new dimension): {suggested_description}"
                if suggested_description
                else ""
            )
            existing_values_for_key_line = ""
            if suggested_key and suggested_key in self.dimension_schema:
                existing_values_for_key_line = (
                    f"Existing values for dimension '{suggested_key}': {self._existing_values_for_key(suggested_key)}"
                )
            prompt = DIMENSION_GOVERNOR_PROMPT_TEMPLATE.format(
                suggested_key=suggested_key,
                suggested_values_str=suggested_values_str,
                suggested_description_line=suggested_description_line,
                existing_dimension_keys=self._existing_dimension_keys_str,
                existing_values_for_key_line=existing_values_for_key_line,
            )
            function_spec = {
                "name": "govern_dimension",
                "description": "Accept or reject the suggested dimension or dimension values.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "accept": {
                            "type": "boolean",
                            "description": "True to accept the suggestion, false to reject.",
                        },
                    },
                    "required": ["accept"],
                },
            }

            try:
                response = self.client.chat.completions.create(
                    model=OPENAI_MODEL,
                    temperature=0,
                    messages=[
                        {
                            "role": "system",
                            "content": "You decide whether to accept or reject taxonomy suggestions.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    functions=[function_spec],
                    function_call={"name": "govern_dimension"},
                )
                parsed = json.loads(response.choices[0].message.function_call.arguments)
                accept = parsed.get("accept", False)
                status = "accepted" if accept else "rejected"
                update_status(self.store_path, suggestion_id, status)
                runtime_ms = int((time.time() - start) * 1000)
                prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
                completion_tokens = getattr(response.usage, "completion_tokens", 0)
                total_tokens = getattr(response.usage, "total_tokens", 0)
                model = getattr(response, "model", "") or OPENAI_MODEL
                _log_usage(
                    schema_version=self.schema_version,
                    raw_text=suggestion_id or suggested_key,
                    parsed_output={
                        "suggested_key": suggested_key,
                        "parent_section": parent_section,
                        "parent_subsection": parent_subsection,
                        "status": status,
                    },
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    model=model,
                    runtime_ms=runtime_ms,
                    success=True,
                    error_message="",
                    description="dimension_governor",
                )
                results.append({**s, "status": status})
            except Exception as exc:
                runtime_ms = int((time.time() - start) * 1000)
                _log_usage(
                    schema_version=self.schema_version,
                    raw_text=suggestion_id or suggested_key,
                    parsed_output={},
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    model="",
                    runtime_ms=runtime_ms,
                    success=False,
                    error_message=str(exc),
                    description="dimension_governor",
                )
                results.append({**s, "status": "pending"})

        return results
