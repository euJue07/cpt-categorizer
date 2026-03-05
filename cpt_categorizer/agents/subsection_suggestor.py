"""Subsection Suggestor agent: suggest new subsection (key + description) when no subsection matches; check store before LLM."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import openai

from cpt_categorizer.agents.tagging import to_snake_case
from cpt_categorizer.config.directory import SUGGESTIONS_PATH
from cpt_categorizer.config.openai import OPENAI_MODEL
from cpt_categorizer.suggestion_store import append, find_by_type
from cpt_categorizer.utils.logging import log_agent_usage

SUBSECTION_SUGGEST_PROMPT_TEMPLATE = """You are a medical classification expert. Given the section "{section}", the following CPT procedure description did not match any existing subsection.

Existing subsection keys under this section (do not reuse): {existing_keys}

CPT description:
{text_description}

Propose exactly one new subsection key (snake_case, e.g. wound_care) and a short description for it. The key must not be one of the existing keys above.
"""


def _context_for_subsection(section: str, text_description: str) -> str:
    """Stable context for deduplication: same section + description -> same context."""
    normalized = text_description.strip().lower()
    desc_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{section}:{desc_hash}"


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


def _existing_subsection_keys(section_schema: dict[str, Any], subsection_schema: dict[str, Any], section: str) -> str:
    """Return comma-separated existing subsection keys for the given section."""
    keys = section_schema.get(section, {}).get("subsections") or list(subsection_schema.get(section, {}).keys())
    return ", ".join(sorted(keys)) if keys else "(none)"


class SubsectionSuggestorAgent:
    """Suggests a new subsection (key + description) when subsection tagging returns no match; checks store before LLM."""

    def __init__(
        self,
        section_schema: dict[str, Any],
        subsection_schema: dict[str, Any],
        store_path: Optional[Path] = None,
        client: Optional[openai.OpenAI] = None,
        schema_version: str = "",
    ):
        self.section_schema = section_schema
        self.subsection_schema = subsection_schema
        self.store_path = store_path if store_path is not None else SUGGESTIONS_PATH
        self.client = client or openai.OpenAI()
        self.schema_version = schema_version

    def suggest_subsection(
        self, section: str, text_description: str, source: str = ""
    ) -> Optional[dict[str, Any]]:
        """
        Suggest a new subsection for a CPT description that matched a section but no subsection.
        Checks store before LLM; returns existing suggestion if same context or same key (under same section) in store.
        """
        if not section or not section.strip():
            return None
        if section.strip().lower() == "others":
            return None
        if not text_description or not text_description.strip():
            return None

        context = _context_for_subsection(section, text_description)
        start = time.time()

        # 1. Check store by context: already suggested for this (section, description)?
        subsection_suggestions = find_by_type(self.store_path, "subsection")
        matching_by_context = [
            s
            for s in subsection_suggestions
            if s.get("context") == context and s.get("status") in ("pending", "accepted")
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
                description="subsection_suggest_store_hit",
            )
            return matching_by_context[0]

        # 2. Call LLM to get suggested_key and suggested_description
        existing_keys = _existing_subsection_keys(self.section_schema, self.subsection_schema, section)
        prompt = SUBSECTION_SUGGEST_PROMPT_TEMPLATE.format(
            section=section,
            existing_keys=existing_keys,
            text_description=text_description,
        )
        function_spec = {
            "name": "suggest_subsection",
            "description": "Return one new subsection key (snake_case) and short description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "suggested_key": {
                        "type": "string",
                        "description": "New subsection key in snake_case, not in existing keys for this section.",
                    },
                    "suggested_description": {
                        "type": "string",
                        "description": "Short description of the new subsection.",
                    },
                },
                "required": ["suggested_key", "suggested_description"],
            },
        }

        response = None
        success = False
        error_message = ""
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": "You suggest new medical subsection keys and descriptions in JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                functions=[function_spec],
                function_call={"name": "suggest_subsection"},
            )
            parsed = json.loads(response.choices[0].message.function_call.arguments)
            suggested_key_raw = parsed.get("suggested_key", "")
            suggested_description = parsed.get("suggested_description", "")
            suggested_key = to_snake_case(suggested_key_raw)
            if not suggested_key:
                suggested_key = to_snake_case("new_subsection")
            success = True
        except Exception as exc:
            error_message = str(exc)
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
                description="subsection_suggest",
            )
            return None

        # 3. Check store by key + parent_section: same key under same section already suggested (pending/accepted)?
        all_subsection_suggestions = find_by_type(self.store_path, "subsection")
        existing_reusable = [
            s
            for s in all_subsection_suggestions
            if s.get("suggested_key") == suggested_key
            and s.get("parent_section") == section
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
                description="subsection_suggest_duplicate_key",
            )
            return existing_reusable[0]

        # 4. Append new suggestion and log real usage
        record = {
            "type": "subsection",
            "suggested_key": suggested_key,
            "suggested_description": suggested_description,
            "context": context,
            "parent_section": section,
            "status": "pending",
            "source": source,
        }
        append(self.store_path, record)
        # Reload to get id and created_at
        subsection_suggestions = find_by_type(self.store_path, "subsection")
        new_record = next(
            (
                s
                for s in subsection_suggestions
                if s.get("context") == context and s.get("suggested_key") == suggested_key
            ),
            record,
        )

        runtime_ms = int((time.time() - start) * 1000)
        prompt_tokens = getattr(response.usage, "prompt_tokens", 0) if response else 0
        completion_tokens = getattr(response.usage, "completion_tokens", 0) if response else 0
        total_tokens = getattr(response.usage, "total_tokens", 0) if response else 0
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
            description="subsection_suggest",
        )
        return new_record
