from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Optional

import openai
from openai import APIConnectionError, APITimeoutError, RateLimitError

from cpt_categorizer.config.openai import OPENAI_MODEL
from cpt_categorizer.tagging_cache import TaggingCache
from cpt_categorizer.utils.logging import log_agent_usage

SECTION_PROMPT_TEMPLATE = """You are a medical classification expert for CPT tagging.

Given the CPT description below, identify the relevant top-level medical Sections based on the clinical domain and type of service.

Available Sections:
{sections_str}

Instructions:
- Select one or more applicable sections based on the nature of the procedure.
- Assign a confidence score (0-1) to each.
- If the procedure clearly does not match any section, assign "others".
- Prioritize precision over guessing.

Return a JSON object with a list of section-confidence pairs.
"""


DIMENSION_PROMPT_TEMPLATE = """You are a medical tagging assistant extracting structured dimension values from CPT descriptions.

Given:
- Section: {section}
- Subsection: {subsection}
- CPT Description: {text_description}
- Allowed Dimensions: {allowed_dimensions}
- Enum Values per Dimension:
{dimension_str}

Instructions:
1. For each allowed dimension, extract matching enum values under "actual".
2. If a value fits a known dimension but is not in its enum, place it under "proposed.existing_dimensions".
3. If a value introduces a new concept, place it under "proposed.new_dimensions" with a snake_case dimension key.
4. Every item must include "value" and "confidence", and confidence must be >= 0.5.
5. Do not include any fields outside the required schema.
"""


def empty_dimensions() -> dict[str, Any]:
    return {
        "actual": {},
        "proposed": {
            "existing_dimensions": {},
            "new_dimensions": {},
        },
    }


def to_snake_case(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        return ""
    if normalized[0].isdigit():
        normalized = f"v_{normalized}"
    return normalized


def _log_usage(
    schema_version: str,
    text_description: str,
    parsed_output: Any,
    response: Any,
    start: float,
    success: bool,
    error_message: str,
    description: str,
) -> None:
    runtime_ms = int((time.time() - start) * 1000)
    prompt_tokens = getattr(response.usage, "prompt_tokens", 0) if response else 0
    completion_tokens = getattr(response.usage, "completion_tokens", 0) if response else 0
    total_tokens = getattr(response.usage, "total_tokens", 0) if response else 0
    model = getattr(response, "model", "") if response else ""
    log_agent_usage(
        timestamp=datetime.now().isoformat(),
        raw_text=text_description,
        description=description,
        parsed_output=json.dumps(parsed_output),
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


class SectionTaggingAgent:
    def __init__(
        self,
        section_schema: dict[str, Any],
        client: Optional[openai.OpenAI] = None,
        schema_version: str = "",
        cache: Optional[TaggingCache] = None,
    ):
        self.client = client or openai.OpenAI()
        self.section_schema = section_schema
        self.schema_version = schema_version
        self.sections = list(self.section_schema.keys()) + ["others"]
        self.sections_str = "\n".join(
            f"- {section}: {self.section_schema.get(section, {}).get('description', '')}"
            for section in self.sections
        )
        self.section_prompt = SECTION_PROMPT_TEMPLATE.format(sections_str=self.sections_str)
        self._cache = cache
        self._cache_sections: dict[str, list[tuple[str, float]]] = (
            cache.sections if cache is not None else {}
        )

    def _call_openai_completion(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

    def classify_sections(
        self, text_description: str, confidence_threshold: float = 0.5
    ) -> list[tuple[str, float]]:
        normalized_text = text_description.strip().lower()
        if normalized_text in self._cache_sections:
            cached = self._cache_sections[normalized_text]
            log_agent_usage(
                timestamp=datetime.now().isoformat(),
                raw_text=text_description,
                description="classify_sections_cache_hit",
                parsed_output=json.dumps(cached),
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                model="",
                runtime_ms=0,
                success=True,
                is_error=False,
                error_message="",
                schema_version=self.schema_version,
            )
            return cached

        function_spec = {
            "name": "select_sections",
            "description": "Return relevant top-level medical sections with confidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "section": {"type": "string", "enum": self.sections},
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                            },
                            "required": ["section", "confidence"],
                        },
                    }
                },
                "required": ["sections"],
            },
        }

        start = time.time()
        response = None
        success = False
        error_message = ""
        result: list[tuple[str, float]] = []
        try:
            for attempt in range(3):
                try:
                    response = self._call_openai_completion(
                        model=OPENAI_MODEL,
                        temperature=0,
                        messages=[
                            {"role": "system", "content": self.section_prompt},
                            {"role": "user", "content": f"CPT description:\n{text_description}"},
                        ],
                        functions=[function_spec],
                        function_call={"name": "select_sections"},
                    )
                    break
                except (RateLimitError, APIConnectionError, APITimeoutError):
                    if attempt < 2:
                        time.sleep(2**attempt)
                        continue
                    raise

            parsed_args = json.loads(response.choices[0].message.function_call.arguments)
            parsed_sections = parsed_args.get("sections", [])
            for item in parsed_sections:
                section = item.get("section")
                confidence = float(item.get("confidence", 0.0))
                if section in self.sections and confidence >= confidence_threshold:
                    result.append((section, confidence))
            result = list(dict.fromkeys(result))
            self._cache_sections[normalized_text] = result
            if self._cache is not None:
                self._cache.persist()
            success = True
        except Exception as exc:
            error_message = str(exc)
            logging.getLogger(__name__).warning("Section classification failed: %s", exc)
        finally:
            _log_usage(
                schema_version=self.schema_version,
                text_description=text_description,
                parsed_output=result,
                response=response,
                start=start,
                success=success,
                error_message=error_message,
                description="classify_sections",
            )
        return result


class SubsectionTaggingAgent:
    def __init__(
        self,
        section_schema: dict[str, Any],
        subsection_schema: dict[str, Any],
        client: Optional[openai.OpenAI] = None,
        schema_version: str = "",
        cache: Optional[TaggingCache] = None,
    ):
        self.client = client or openai.OpenAI()
        self.section_schema = section_schema
        self.subsection_schema = subsection_schema
        self.schema_version = schema_version
        self._cache = cache
        self._cache_subsections: dict[tuple[str, str], list[tuple[str, float]]] = (
            {} if cache is not None else {}
        )

    def _call_openai_completion(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

    def _get_subsection_prompt(self, section: str) -> str:
        subsection_keys = self.section_schema.get(section, {}).get("subsections", [])
        subsection_lines = "\n".join(
            f"- {subsection}: "
            f"{self.subsection_schema.get(section, {}).get(subsection, {}).get('description', '')}"
            for subsection in subsection_keys
        )
        return f"""You are a medical classification expert for CPT tagging.

Given the CPT description below and its assigned section ({section}), choose the best matching subsection(s).

Available Subsections:
{subsection_lines}

Instructions:
- Select one or more applicable subsections.
- Assign a confidence score (0-1) to each.
- If no listed subsection matches, return an empty list.
"""

    def _get_subsection_function_specification(self, section: str) -> dict[str, Any]:
        enum_values = self.section_schema.get(section, {}).get("subsections", [])
        return {
            "name": f"select_subsections_for_{section}",
            "description": f"Return relevant subsections under {section} with confidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subsections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subsection": {"type": "string", "enum": enum_values},
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                            },
                            "required": ["subsection", "confidence"],
                        },
                    }
                },
                "required": ["subsections"],
            },
        }

    def classify_subsections(
        self, section: str, text_description: str, confidence_threshold: float = 0.5
    ) -> list[tuple[str, float]]:
        if section == "others":
            return [("others", 1.0)]

        normalized_text = text_description.strip().lower()
        if self._cache is not None:
            cache_key_str = f"{section}|{normalized_text}"
            if cache_key_str in self._cache.subsections:
                cached = self._cache.subsections[cache_key_str]
                log_agent_usage(
                    timestamp=datetime.now().isoformat(),
                    raw_text=text_description,
                    description="classify_subsections_cache_hit",
                    parsed_output=json.dumps(cached),
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    model="",
                    runtime_ms=0,
                    success=True,
                    is_error=False,
                    error_message="",
                    schema_version=self.schema_version,
                )
                return cached
        else:
            cache_key = (section, normalized_text)
            if cache_key in self._cache_subsections:
                cached = self._cache_subsections[cache_key]
                log_agent_usage(
                    timestamp=datetime.now().isoformat(),
                    raw_text=text_description,
                    description="classify_subsections_cache_hit",
                    parsed_output=json.dumps(cached),
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    model="",
                    runtime_ms=0,
                    success=True,
                    is_error=False,
                    error_message="",
                    schema_version=self.schema_version,
                )
                return cached

        subsection_prompt = self._get_subsection_prompt(section)
        function_spec = self._get_subsection_function_specification(section)

        start = time.time()
        response = None
        success = False
        error_message = ""
        result: list[tuple[str, float]] = []
        try:
            for attempt in range(3):
                try:
                    response = self._call_openai_completion(
                        model=OPENAI_MODEL,
                        temperature=0,
                        messages=[
                            {"role": "system", "content": subsection_prompt},
                            {"role": "user", "content": f"CPT description:\n{text_description}"},
                        ],
                        functions=[function_spec],
                        function_call={"name": function_spec["name"]},
                    )
                    break
                except (RateLimitError, APIConnectionError, APITimeoutError):
                    if attempt < 2:
                        time.sleep(2**attempt)
                        continue
                    raise
            parsed_args = json.loads(response.choices[0].message.function_call.arguments)
            valid_subsections = set(self.section_schema.get(section, {}).get("subsections", []))
            for item in parsed_args.get("subsections", []):
                subsection = item.get("subsection")
                confidence = float(item.get("confidence", 0.0))
                if subsection in valid_subsections and confidence >= confidence_threshold:
                    result.append((subsection, confidence))
            if self._cache is not None:
                self._cache.subsections[cache_key_str] = result
                self._cache.persist()
            else:
                self._cache_subsections[cache_key] = result
            success = True
        except Exception as exc:
            error_message = str(exc)
            logging.getLogger(__name__).warning("Subsection classification failed: %s", exc)
        finally:
            _log_usage(
                schema_version=self.schema_version,
                text_description=text_description,
                parsed_output=result,
                response=response,
                start=start,
                success=success,
                error_message=error_message,
                description=f"classify_subsections:{section}",
            )
        return result


class DimensionTaggingAgent:
    def __init__(
        self,
        subsection_schema: dict[str, Any],
        dimension_schema: dict[str, Any],
        client: Optional[openai.OpenAI] = None,
        schema_version: str = "",
    ):
        self.client = client or openai.OpenAI()
        self.subsection_schema = subsection_schema
        self.dimension_schema = dimension_schema
        self.schema_version = schema_version

    def _call_openai_completion(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

    def _normalize_items(self, items: Any, min_confidence: float) -> list[dict[str, float | str]]:
        if not isinstance(items, list):
            return []
        normalized: list[dict[str, float | str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            confidence = float(item.get("confidence", 0.0))
            if confidence < min_confidence:
                continue
            value = to_snake_case(str(item.get("value", "")))
            if not value:
                continue
            normalized.append({"value": value, "confidence": confidence})
        return normalized

    def _normalize_dimension_payload(
        self, payload: dict[str, Any], dimension_enum_map: dict[str, list[str]]
    ) -> dict[str, Any]:
        actual_in = payload.get("actual", {})
        proposed_in = payload.get("proposed", {})
        allowed_dimensions = set(dimension_enum_map.keys())

        actual_out: dict[str, list[dict[str, float | str]]] = {}
        if isinstance(actual_in, dict):
            for dimension, values in actual_in.items():
                dim = to_snake_case(str(dimension))
                if dim not in allowed_dimensions:
                    continue
                allowed_values = set(dimension_enum_map[dim])
                normalized_items = self._normalize_items(values, min_confidence=0.5)
                filtered_items = [
                    item for item in normalized_items if item["value"] in allowed_values
                ]
                if filtered_items:
                    actual_out[dim] = filtered_items

        existing_out: dict[str, list[dict[str, float | str]]] = {}
        existing_in = (
            proposed_in.get("existing_dimensions", {}) if isinstance(proposed_in, dict) else {}
        )
        if isinstance(existing_in, dict):
            for dimension, values in existing_in.items():
                dim = to_snake_case(str(dimension))
                if dim not in allowed_dimensions:
                    continue
                normalized_items = self._normalize_items(values, min_confidence=0.5)
                if normalized_items:
                    existing_out[dim] = normalized_items

        new_out: dict[str, list[dict[str, float | str]]] = {}
        new_in = proposed_in.get("new_dimensions", {}) if isinstance(proposed_in, dict) else {}
        if isinstance(new_in, dict):
            for dimension, values in new_in.items():
                dim = to_snake_case(str(dimension))
                if not dim:
                    continue
                normalized_items = self._normalize_items(values, min_confidence=0.5)
                if normalized_items:
                    new_out[dim] = normalized_items

        return {
            "actual": actual_out,
            "proposed": {
                "existing_dimensions": existing_out,
                "new_dimensions": new_out,
            },
        }

    def classify_dimensions(
        self, section: str, subsection: str, text_description: str
    ) -> dict[str, Any]:
        if section == "others" or subsection == "others":
            return empty_dimensions()

        allowed_dimensions = (
            self.subsection_schema.get(section, {}).get(subsection, {}).get("dimensions", [])
        )
        if not allowed_dimensions:
            return empty_dimensions()

        dimension_enum_map = {
            dim: self.dimension_schema.get(dim, {}).get("values", []) for dim in allowed_dimensions
        }
        prompt = DIMENSION_PROMPT_TEMPLATE.format(
            section=section,
            subsection=subsection,
            text_description=text_description,
            allowed_dimensions=", ".join(allowed_dimensions),
            dimension_str="\n".join(
                f"- {dim}: {', '.join(values)}" for dim, values in dimension_enum_map.items()
            ),
        )

        function_spec = {
            "name": "extract_dimensions",
            "description": "Extract actual and proposed dimensions from CPT description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "actual": {"type": "object"},
                    "proposed": {
                        "type": "object",
                        "properties": {
                            "existing_dimensions": {"type": "object"},
                            "new_dimensions": {"type": "object"},
                        },
                        "required": ["existing_dimensions", "new_dimensions"],
                    },
                },
                "required": ["actual", "proposed"],
            },
        }

        start = time.time()
        response = None
        success = False
        error_message = ""
        normalized_result = empty_dimensions()
        try:
            response = self._call_openai_completion(
                model=OPENAI_MODEL,
                temperature=0,
                messages=[{"role": "system", "content": prompt}],
                functions=[function_spec],
                function_call={"name": "extract_dimensions"},
            )
            parsed_args = json.loads(response.choices[0].message.function_call.arguments)
            normalized_result = self._normalize_dimension_payload(
                parsed_args, dimension_enum_map=dimension_enum_map
            )
            success = True
        except Exception as exc:
            error_message = str(exc)
            logging.getLogger(__name__).warning("Dimension extraction failed: %s", exc)
        finally:
            _log_usage(
                schema_version=self.schema_version,
                text_description=text_description,
                parsed_output=normalized_result,
                response=response,
                start=start,
                success=success,
                error_message=error_message,
                description=f"classify_dimensions:{section}/{subsection}",
            )
        return normalized_result


class TaggingAgent:
    def __init__(
        self,
        section_schema: dict[str, Any],
        subsection_schema: Optional[dict[str, Any]] = None,
        dimension_schema: Optional[dict[str, Any]] = None,
        client: Optional[openai.OpenAI] = None,
        schema_version: str = "",
        cache_path: Optional[Path] = None,
    ):
        subsection_schema = subsection_schema or {}
        dimension_schema = dimension_schema or {}
        cache: Optional[TaggingCache] = None
        if cache_path is not None:
            cache = TaggingCache(cache_path)
            cache.load()
        self._section_agent = SectionTaggingAgent(
            section_schema=section_schema,
            client=client,
            schema_version=schema_version,
            cache=cache,
        )
        self._subsection_agent = SubsectionTaggingAgent(
            section_schema=section_schema,
            subsection_schema=subsection_schema,
            client=client,
            schema_version=schema_version,
            cache=cache,
        )
        self._dimension_agent = DimensionTaggingAgent(
            subsection_schema=subsection_schema,
            dimension_schema=dimension_schema,
            client=client,
            schema_version=schema_version,
        )

    def classify_sections(
        self, text_description: str, confidence_threshold: float = 0.5
    ) -> list[tuple[str, float]]:
        return self._section_agent.classify_sections(
            text_description, confidence_threshold=confidence_threshold
        )

    def classify_subsections(
        self, section: str, text_description: str, confidence_threshold: float = 0.5
    ) -> list[tuple[str, float]]:
        return self._subsection_agent.classify_subsections(
            section, text_description, confidence_threshold=confidence_threshold
        )

    def classify_dimensions(
        self, section: str, subsection: str, text_description: str
    ) -> dict[str, Any]:
        return self._dimension_agent.classify_dimensions(section, subsection, text_description)

    def generate_tags(
        self, text_description: str, confidence_threshold: float = 0.5
    ) -> list[dict[str, Any]]:
        if not text_description or not text_description.strip():
            return []
        tags: list[dict[str, Any]] = []
        candidate_sections = self._section_agent.classify_sections(
            text_description, confidence_threshold=confidence_threshold
        )
        if not candidate_sections:
            return []

        for section, section_conf in candidate_sections:
            candidate_subsections = self._subsection_agent.classify_subsections(
                section, text_description, confidence_threshold=confidence_threshold
            )
            if not candidate_subsections:
                continue
            for subsection, subsection_conf in candidate_subsections:
                dimensions = self._dimension_agent.classify_dimensions(
                    section, subsection, text_description
                )
                actual_values = [
                    item["confidence"]
                    for values in dimensions.get("actual", {}).values()
                    for item in values
                    if isinstance(item, dict) and "confidence" in item
                ]
                max_dim_conf = max(actual_values) if actual_values else 0.0
                combined_conf = (
                    (section_conf + subsection_conf + max_dim_conf) / 3
                    if subsection != "others"
                    else (section_conf + subsection_conf) / 2
                )
                tags.append(
                    {
                        "section": section,
                        "subsection": subsection,
                        "confidence": float(combined_conf),
                        "dimensions": (
                            dimensions if subsection != "others" else empty_dimensions()
                        ),
                    }
                )
        return tags

    def tag_entry(self, text_description: str, code: str) -> dict[str, Any]:
        return {
            "code": code,
            "description": text_description,
            "tags": [
                {
                    "section": tag.get("section"),
                    "subsection": tag.get("subsection"),
                    "confidence": float(tag.get("confidence", 0.0)),
                    "dimensions": tag.get("dimensions", empty_dimensions()),
                }
                for tag in self.generate_tags(text_description)
            ],
        }
