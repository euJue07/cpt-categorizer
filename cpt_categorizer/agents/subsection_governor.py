"""Subsection Governor agent: resolve pending subsection suggestions using schema + store first; LLM only for novel suggestions."""

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

SUBSECTION_GOVERNOR_PROMPT_TEMPLATE = """You are a medical taxonomy curator. A suggested new subsection was proposed for the CPT categorizer schema under section "{parent_section}".

Suggested subsection key: {suggested_key}
Suggested description: {suggested_description}

Existing subsection keys under this section (for reference): {existing_keys}

Decide whether to ACCEPT or REJECT this suggestion. Accept only if the proposed subsection is distinct, clinically meaningful, and would improve the taxonomy under this section. Reject if it duplicates or overlaps existing subsections, or is too narrow/vague.
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


class SubsectionGovernorAgent:
    """Resolves pending subsection suggestions: schema duplicate and store reuse first; LLM only for novel suggestions."""

    def __init__(
        self,
        subsection_schema: dict[str, Any],
        store_path: Optional[Path] = None,
        client: Optional[openai.OpenAI] = None,
        schema_version: str = "",
    ):
        self.subsection_schema = subsection_schema
        self.store_path = store_path if store_path is not None else SUGGESTIONS_PATH
        self.client = client or openai.OpenAI()
        self.schema_version = schema_version

    def _existing_keys_for_section(self, parent_section: str) -> str:
        """Return comma-separated subsection keys for the given section."""
        keys = list(self.subsection_schema.get(parent_section, {}).keys())
        return ", ".join(sorted(keys)) if keys else "(none)"

    def resolve_pending_subsections(self) -> list[dict[str, Any]]:
        """
        Resolve all pending subsection suggestions: mark duplicate if key in schema for parent_section,
        reuse status if same key+parent_section has accepted/rejected in store, else call LLM.
        Returns list of updated suggestion records (with status after resolution).
        """
        subsection_suggestions = find_by_type(self.store_path, "subsection")
        pending = [s for s in subsection_suggestions if s.get("status") == "pending"]
        results: list[dict[str, Any]] = []

        for s in pending:
            suggestion_id = s.get("id", "")
            suggested_key_raw = s.get("suggested_key", "")
            suggested_key = to_snake_case(suggested_key_raw) if suggested_key_raw else ""
            suggested_description = s.get("suggested_description", "")
            parent_section = s.get("parent_section", "")
            start = time.time()
            runtime_ms = 0

            # 1. Key already in schema for this parent_section -> duplicate
            if parent_section and suggested_key and suggested_key in self.subsection_schema.get(parent_section, {}):
                update_status(self.store_path, suggestion_id, "duplicate")
                runtime_ms = int((time.time() - start) * 1000)
                _log_usage(
                    schema_version=self.schema_version,
                    raw_text=suggestion_id or suggested_key,
                    parsed_output={
                        "suggested_key": suggested_key,
                        "parent_section": parent_section,
                        "status": "duplicate",
                    },
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    model="",
                    runtime_ms=runtime_ms,
                    success=True,
                    error_message="",
                    description="subsection_governor_duplicate",
                )
                results.append({**s, "status": "duplicate"})
                continue

            # 2. Same key + parent_section has accepted/rejected in store -> reuse
            by_key = find_by_type_key(self.store_path, "subsection", suggested_key, context=None)
            reused = next(
                (
                    x
                    for x in by_key
                    if x.get("parent_section") == parent_section
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
                        "status": status,
                    },
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    model="",
                    runtime_ms=runtime_ms,
                    success=True,
                    error_message="",
                    description="subsection_governor_store_reuse",
                )
                results.append({**s, "status": status})
                continue

            # 3. Novel -> LLM accept/reject
            existing_keys_str = self._existing_keys_for_section(parent_section)
            prompt = SUBSECTION_GOVERNOR_PROMPT_TEMPLATE.format(
                parent_section=parent_section,
                suggested_key=suggested_key,
                suggested_description=suggested_description,
                existing_keys=existing_keys_str,
            )
            function_spec = {
                "name": "govern_subsection",
                "description": "Accept or reject the suggested subsection.",
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
                    function_call={"name": "govern_subsection"},
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
                        "status": status,
                    },
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    model=model,
                    runtime_ms=runtime_ms,
                    success=True,
                    error_message="",
                    description="subsection_governor",
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
                    description="subsection_governor",
                )
                results.append({**s, "status": "pending"})

        return results
