from datetime import datetime
import json
import time
from typing import List, Optional, Tuple
import uuid

import openai
from openai import APIConnectionError, RateLimitError, APITimeoutError

from cpt_categorizer.config.openai import OPENAI_MODEL
from cpt_categorizer.utils.logging import log_agent_usage


SECTION_PROMPT_TEMPLATE = """You are a medical classification expert for CPT tagging.

Given the CPT description below, identify the relevant top-level medical Sections based on the clinical domain and type of service.

Available Sections:
{sections_str}

Instructions:
- Select one or more applicable sections based on the nature of the procedure.
- Assign a confidence score (0–1) to each.
- If the procedure clearly does not match any section, assign "others".
- Prioritize precision over guessing. Avoid selecting too many sections unless truly justified.

Return a JSON object with a list of section-confidence pairs.
"""

# Dimension extraction prompt template
DIMENSION_PROMPT_TEMPLATE = """You are a medical tagging assistant extracting structured dimension values from CPT descriptions.

Given:
- Section: {section}
- Subsection: {subsection}
- CPT Description: {text_description}
- Allowed Dimensions: {allowed_dimensions}
- Enum Values per Dimension:
{dimension_str}

Instructions:
1. For each allowed dimension, extract any matching values from the description. Use enum values if matched exactly, and place them under "actual".
2. If a value seems to match a known dimension but isn't in its enum, place it under "proposed → existing_dimensions", and make sure its key is in snake_case.
3. If a value relates to a new concept not in any known dimension, propose a dimension name in snake_case and place it under "proposed → new_dimensions".
4. Each value must include both "value" and "confidence", and confidence must be ≥ 0.5.
5. Do not include explanatory text or any fields other than the required schema.

Return a JSON object matching the expected structure.
"""

SECTION_FUNCTION_SPECIFICATION_TEMPLATE = {
    "name": "select_sections",
    "description": "Returns the most relevant top-level medical Sections for a CPT description, each with a confidence score.",
    "parameters": {
        "type": "object",
        "properties": {
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "section": {
                            "type": "string",
                            "enum": [],  # Filled dynamically
                            "description": "Top-level medical section name.",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence score for this section.",
                        },
                    },
                    "required": ["section", "confidence"],
                },
            }
        },
        "required": ["sections"],
    },
}


class TaggingAgent:
    """
    TaggingAgent class is responsible for classifying CPT descriptions into medical
    sections and subsections using OpenAI's GPT model. It supports section-level and
    subsection-level tagging with confidence scores, and is designed to integrate with
    structured classification schemas for healthcare services.
    """

    def _to_snake_case(self, name: str) -> str:
        import re

        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).replace(" ", "_").lower()

    def _normalize_proposed_value(self, s: str) -> str:
        import re

        s = s.strip().lower()
        s = re.sub(
            r"[^\w\s]", "_", s
        )  # Replace non-alphanumeric characters with underscores
        s = re.sub(r"\s+", "_", s)  # Replace spaces with underscores
        s = re.sub(r"_+", "_", s)  # Collapse multiple underscores
        return s.strip("_")

    def __init__(
        self,
        section_schema: dict,
        subsection_schema: Optional[dict] = None,
        dimension_schema: Optional[dict] = None,
        client: Optional[openai.OpenAI] = None,
    ):
        self.client = client or openai.OpenAI()

        # Assign schemas from parameters
        self.section_schema = section_schema
        self.subsection_schema = subsection_schema
        self.dimension_schema = dimension_schema

        # Load section, subsection, and detail schema (now unified)
        self.sections = list(self.section_schema.keys()) + ["others"]

        self.sections_str = "\n".join(
            f"- {section}: {self.section_schema.get(section, {}).get('description', '')}"
            for section in self.sections
        )

        self.section_prompt = SECTION_PROMPT_TEMPLATE.format(
            sections_str=self.sections_str
        )

        self.section_function_specification = (
            SECTION_FUNCTION_SPECIFICATION_TEMPLATE.copy()
        )
        # Ensure "others" is included in the enum for valid section names
        self.section_function_specification["parameters"]["properties"]["sections"][
            "items"
        ]["properties"]["section"]["enum"] = self.sections
        self._cache_sections = {}
        self._cache_subsections = {}
        self._cache_section_calls = {}
        self._cache_subsection_calls = {}

    def classify_sections(
        self,
        text_description: str,
        confidence_threshold: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        Classifies a CPT description into plausible top-level medical Sections using the OpenAI API.

        Args:
            text_description (str): The CPT description to classify.

        Returns:
            List[Tuple[str, float]]: A list of tuples, each containing a section name and its associated confidence score.
        """
        import logging

        normalized_text = text_description.strip().lower()
        if normalized_text in self._cache_sections:
            self._cache_section_calls[normalized_text] = (
                self._cache_section_calls.get(normalized_text, 0) + 1
            )
            return self._cache_sections[normalized_text]

        request_id = str(uuid.uuid4())
        success = False
        error_message = ""
        start = time.time()
        response = None
        try:
            for attempt in range(3):
                try:
                    response = self._call_openai_completion(
                        model=OPENAI_MODEL,
                        temperature=0,
                        messages=[
                            {"role": "system", "content": self.section_prompt},
                            {
                                "role": "user",
                                "content": f"CPT description:\n{text_description}",
                            },
                        ],
                        functions=[self.section_function_specification],
                        function_call={"name": "select_sections"},
                    )
                    break  # If successful, exit the retry loop
                except (RateLimitError, APIConnectionError, APITimeoutError):
                    if attempt < 2:
                        time.sleep(2**attempt)
                        continue
                    else:
                        raise
            try:
                parsed_args = json.loads(
                    response.choices[0].message.function_call.arguments
                )
                validated = [
                    (item["section"], item.get("confidence", 0.0))
                    for item in parsed_args.get("sections", [])
                    if item["section"] in self.sections
                    and item.get("confidence", 0.0) >= confidence_threshold
                ]
                # Deduplicate the list of section-confidence tuples
                seen = set()
                deduped = []
                for section, confidence in validated:
                    key = (section, confidence)
                    if key not in seen:
                        deduped.append((section, confidence))
                        seen.add(key)
                result = deduped
                self._cache_sections[normalized_text] = result
            except Exception:
                logging.getLogger(__name__).warning(
                    "Failed to parse OpenAI function_call arguments. request_id=%s\nRaw function_call: %s",
                    request_id,
                    response.choices[0].message.function_call,
                )
                result = []
            success = True
            error_message = ""
        except (
            APIConnectionError,
            RateLimitError,
            APITimeoutError,
        ) as e:
            result = []
            success = False
            error_message = str(e)
        finally:
            end = time.time()
            log_agent_usage(
                timestamp=datetime.now().isoformat(),
                raw_text=text_description,
                parsed_output=json.dumps(result),
                prompt_tokens=getattr(response.usage, "prompt_tokens", 0)
                if success and response
                else 0,
                completion_tokens=getattr(response.usage, "completion_tokens", 0)
                if success and response
                else 0,
                total_tokens=getattr(response.usage, "total_tokens", 0)
                if success and response
                else 0,
                model=getattr(response, "model", "") if success and response else "",
                runtime_ms=int((end - start) * 1000),
                success=success,
                is_error=not success,
                error_message=error_message,
                request_id=request_id,
                description="classify_sections",
            )
        return result

    def _get_subsection_prompt(self, section: str) -> str:
        subsection_keys = self.section_schema.get(section, {}).get("subsections", [])
        formatted = "\n".join(
            f"- {key}: {self.subsection_schema.get(section, {}).get(key, {}).get('description', '')}"
            for key in subsection_keys
        )
        return f"""You are a medical classification expert for CPT tagging.

Given the CPT description below and its assigned Section: {section}, identify the most appropriate Subsections based on clinical content.

Available Subsections:
{formatted}

Instructions:
- Select one or more applicable subsections based on the clinical details.
- Assign a confidence score (0–1) to each.
- If none apply, assign "others".
- Prioritize accuracy over coverage.

Return a JSON object containing a list of subsection-confidence pairs.
"""

    def _get_subsection_function_specification(self, section: str) -> dict:
        """
        Constructs the OpenAI function specification for valid subsection classifications under a given section.

        Args:
            section (str): The section name.

        Returns:
            dict: A dictionary representing the JSON schema for allowed subsections under the section.
        """
        subsections_raw = self.section_schema.get(section, {}).get("subsections", {})
        enum_values = (
            list(subsections_raw.keys())
            if isinstance(subsections_raw, dict)
            else subsections_raw
        )

        return {
            "name": f"select_subsections_for_{section.lower()}",
            "description": f"Returns the most relevant Subsections under the section '{section}' with confidence scores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subsections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subsection": {
                                    "type": "string",
                                    "enum": enum_values,
                                    "description": f"Subsection under {section}",
                                },
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    "description": "Confidence score for this subsection.",
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
        self,
        section: str,
        text_description: str,
        confidence_threshold: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        Classifies a CPT description into plausible subsections under a given section using the OpenAI API.

        Args:
            section (str): The parent section under which to classify.
            text_description (str): The CPT description to classify.

        Returns:
            List[Tuple[str, float]]: A list of tuples, each containing a subsection name and its associated confidence score.
        """
        import logging

        normalized_text = text_description.strip().lower()
        if section == "others":
            return [("others", 1.0)]
        key = (section, normalized_text)
        if key in self._cache_subsections:
            self._cache_subsection_calls[key] = (
                self._cache_subsection_calls.get(key, 0) + 1
            )
            return self._cache_subsections[key]

        request_id = str(uuid.uuid4())
        success = False
        error_message = ""
        subsection_prompt = self._get_subsection_prompt(section)
        subsection_function_specification = self._get_subsection_function_specification(
            section
        )
        start = time.time()
        response = None  # Initialize response to handle exceptions

        try:
            for attempt in range(3):
                try:
                    response = self._call_openai_completion(
                        model=OPENAI_MODEL,
                        temperature=0,
                        messages=[
                            {"role": "system", "content": subsection_prompt},
                            {
                                "role": "user",
                                "content": f"CPT description:\n{text_description}",
                            },
                        ],
                        functions=[subsection_function_specification],
                        function_call={
                            "name": subsection_function_specification["name"]
                        },
                    )
                    break  # If successful, exit the retry loop
                except (RateLimitError, APIConnectionError, APITimeoutError):
                    if attempt < 2:
                        time.sleep(2**attempt)
                        continue
                    else:
                        raise
            try:
                parsed_args = json.loads(
                    response.choices[0].message.function_call.arguments
                )
                valid_subsections = self.section_schema.get(section, {}).get(
                    "subsections", {}
                )
                valid_keys = (
                    list(valid_subsections.keys())
                    if isinstance(valid_subsections, dict)
                    else valid_subsections
                )
                result = [
                    (item["subsection"], item.get("confidence", 0.0))
                    for item in parsed_args.get("subsections", [])
                    if item["subsection"] in valid_keys
                    and item.get("confidence", 0.0) >= confidence_threshold
                ]
                self._cache_subsections[key] = result
            except Exception:
                logging.getLogger(__name__).warning(
                    "Failed to parse OpenAI function_call arguments. request_id=%s\nRaw function_call: %s",
                    request_id,
                    response.choices[0].message.function_call,
                )
                result = []
            success = True
            error_message = ""
        except (
            APIConnectionError,
            RateLimitError,
            APITimeoutError,
        ) as e:
            result = []
            success = False
            error_message = str(e)
        finally:
            end = time.time()
            log_agent_usage(
                timestamp=datetime.now().isoformat(),
                raw_text=text_description,
                parsed_output=json.dumps(result),
                prompt_tokens=getattr(response.usage, "prompt_tokens", 0)
                if success and response
                else 0,
                completion_tokens=getattr(response.usage, "completion_tokens", 0)
                if success and response
                else 0,
                total_tokens=getattr(response.usage, "total_tokens", 0)
                if success and response
                else 0,
                model=getattr(response, "model", "") if success and response else "",
                runtime_ms=int((end - start) * 1000),
                success=success,
                is_error=not success,
                error_message=error_message,
                request_id=request_id,
                description=f"classify_subsections:{section}",
            )
        return result

    def classify_dimensions(
        self,
        section: str,
        subsection: str,
        text_description: str,
    ) -> dict:
        """
        Extracts structured dimension details for a given CPT description under a specific section and subsection.

        Returns:
            dict: A structured output with actual and proposed dimensions as per specification.
        """
        import logging
        import time
        from datetime import datetime
        import uuid

        start = time.time()
        response = None
        actual_dimensions = None
        proposed_existing_dimensions = None
        proposed_new_dimensions = None
        e = None

        # If no dimension schema provided, return empty
        if not self.subsection_schema or not self.dimension_schema:
            return {
                "actual": {},
                "proposed": {
                    "existing_dimensions": {},
                    "new_dimensions": {},
                },
            }

        # Use the self.subsection_schema to fetch allowed dimensions for the current subsection.
        allowed_dimensions = (
            self.subsection_schema.get(section, {})
            .get(subsection, {})
            .get("dimensions", [])
        )
        if not allowed_dimensions:
            return {
                "actual": {},
                "proposed": {
                    "existing_dimensions": {},
                    "new_dimensions": {},
                },
            }

        # Format dimension values for prompt
        dimension_enum_map = {
            dim: self.dimension_schema.get(dim, {}).get("values", [])
            for dim in allowed_dimensions
        }

        prompt = DIMENSION_PROMPT_TEMPLATE.format(
            section=section,
            subsection=subsection,
            text_description=text_description,
            allowed_dimensions=", ".join(allowed_dimensions),
            dimension_str="\n".join(
                f"- {dim}: {', '.join(values)}"
                for dim, values in dimension_enum_map.items()
            ),
        )

        function_spec = {
            "name": "extract_dimensions",
            "description": "Extract dimension values from a CPT description and organize them into actual (matched), proposed existing, and proposed new dimensions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "actual": {
                        "type": "object",
                        "description": "Values that match known enums for each allowed dimension.",
                        "properties": {
                            dim: {
                                "type": "array",
                                "description": self.dimension_schema.get(dim, {}).get(
                                    "description", ""
                                ),
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "value": {
                                            "type": "string",
                                            "enum": enum_values,
                                        },
                                        "confidence": {
                                            "type": "number",
                                            "minimum": 0,
                                            "maximum": 1,
                                        },
                                    },
                                    "required": ["value", "confidence"],
                                },
                            }
                            for dim, enum_values in dimension_enum_map.items()
                        },
                        "required": [],
                    },
                    "proposed": {
                        "type": "object",
                        "description": "Values that either don't match the known enums or suggest new dimension names.",
                        "properties": {
                            "existing_dimensions": {
                                "type": "object",
                                "description": "Values that likely belong to known dimensions but aren't in the enum.",
                                "properties": {
                                    dim: {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "value": {"type": "string"},
                                                "confidence": {
                                                    "type": "number",
                                                    "minimum": 0,
                                                    "maximum": 1,
                                                },
                                            },
                                            "required": ["value", "confidence"],
                                        },
                                    }
                                    for dim in allowed_dimensions
                                },
                                "required": [],
                            },
                            "new_dimensions": {
                                "type": "object",
                                "description": "Suggested new dimensions inferred from leftover text.",
                                "additionalProperties": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "value": {"type": "string"},
                                            "confidence": {
                                                "type": "number",
                                                "minimum": 0,
                                                "maximum": 1,
                                            },
                                        },
                                        "required": ["value", "confidence"],
                                    },
                                },
                            },
                        },
                        "required": [],
                    },
                },
                "required": ["actual", "proposed"],
            },
        }

        try:
            response = self._call_openai_completion(
                model=OPENAI_MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": prompt},
                ],
                functions=[function_spec],
                function_call={"name": "extract_dimensions"},
            )
            parsed_args = json.loads(
                response.choices[0].message.function_call.arguments
            )
            # Parse according to new schema
            actual_dimensions = parsed_args.get("actual", {})
            proposed = parsed_args.get("proposed", {})
            proposed_existing_dimensions = proposed.get("existing_dimensions", {})
            proposed_new_dimensions = proposed.get("new_dimensions", {})
            # Normalize proposed keys and values
            proposed_existing_dimensions = {
                self._normalize_proposed_value(k): [
                    {
                        "value": self._normalize_proposed_value(v["value"]),
                        "confidence": v["confidence"],
                    }
                    for v in vals
                ]
                for k, vals in proposed_existing_dimensions.items()
            }
            proposed_new_dimensions = {
                self._normalize_proposed_value(k): [
                    {
                        "value": self._normalize_proposed_value(v["value"]),
                        "confidence": v["confidence"],
                    }
                    for v in vals
                ]
                for k, vals in proposed_new_dimensions.items()
            }
        except Exception as exc:
            e = exc
            logging.getLogger(__name__).warning(f"Dimension tagging failed: {e}")
            actual_dimensions = {}
            proposed_existing_dimensions = {}
            proposed_new_dimensions = {}
        finally:
            end = time.time()
            log_agent_usage(
                timestamp=datetime.now().isoformat(),
                raw_text=text_description,
                parsed_output=json.dumps(
                    {
                        "actual": actual_dimensions,
                        "proposed": {
                            "existing_dimensions": proposed_existing_dimensions,
                            "new_dimensions": proposed_new_dimensions,
                        },
                    }
                ),
                prompt_tokens=getattr(response.usage, "prompt_tokens", 0)
                if "response" in locals() and response
                else 0,
                completion_tokens=getattr(response.usage, "completion_tokens", 0)
                if "response" in locals() and response
                else 0,
                total_tokens=getattr(response.usage, "total_tokens", 0)
                if "response" in locals() and response
                else 0,
                model=getattr(response, "model", "")
                if "response" in locals() and response
                else "",
                runtime_ms=int((time.time() - start) * 1000),
                success="actual_dimensions" in locals(),
                is_error="actual_dimensions" not in locals(),
                error_message=str(e) if "e" in locals() else "",
                request_id=str(uuid.uuid4()),
                description=f"classify_dimensions:{section}/{subsection}",
            )
        return {
            "actual": actual_dimensions,
            "proposed": {
                "existing_dimensions": proposed_existing_dimensions,
                "new_dimensions": proposed_new_dimensions,
            },
        }

    def generate_tags(
        self, text_description: str, confidence_threshold: float = 0.5
    ) -> List[dict]:
        """
        Generates section and subsection tags for a CPT description.

        Args:
            text_description (str): The CPT description to classify.

        Returns:
            List[dict]: A list of dictionaries containing section, subsection, confidence, and structured dimension details.
        """
        if not text_description or not text_description.strip():
            return []
        tags = []
        candidate_sections = self.classify_sections(
            text_description, confidence_threshold=confidence_threshold
        )
        if not candidate_sections:
            return []
        for section, sec_conf in candidate_sections:
            candidate_subsections = self.classify_subsections(
                section, text_description, confidence_threshold=confidence_threshold
            )
            if not candidate_subsections:
                continue
            for subsection, sub_conf in candidate_subsections:
                # Compute max confidence from all dimension values
                max_dim_conf = 0
                actual_dimensions = {}
                if subsection != "others":
                    dimension_struct = self.classify_dimensions(
                        section, subsection, text_description
                    )
                    # Unpack as per revised schema: actual is a dict of dimensions
                    actual_dimensions = dimension_struct.get("actual", {})
                    # Gather all confidences from all dimension values
                    all_dimension_confs = []
                    for values in actual_dimensions.values():
                        all_dimension_confs.extend(
                            [
                                v["confidence"]
                                for v in values
                                if isinstance(v, dict) and "confidence" in v
                            ]
                        )
                    max_dim_conf = (
                        max(all_dimension_confs) if all_dimension_confs else 0
                    )
                # Average section, subsection, and max dimension confidence
                if subsection != "others":
                    combined_conf = (sec_conf + sub_conf + max_dim_conf) / 3
                else:
                    combined_conf = (sec_conf + sub_conf) / 2
                if subsection != "others":
                    proposed_existing = dimension_struct.get("proposed", {}).get(
                        "existing_dimensions", {}
                    )
                    proposed_new = dimension_struct.get("proposed", {}).get(
                        "new_dimensions", {}
                    )
                    tags.append(
                        {
                            "section": section,
                            "subsection": subsection,
                            "confidence": combined_conf,
                            "dimensions": {
                                "actual": actual_dimensions,
                                "proposed": {
                                    "existing_dimensions": proposed_existing,
                                    "new_dimensions": proposed_new,
                                },
                            },
                        }
                    )
                else:
                    # For "others", output dimensions as per updated schema
                    tags.append(
                        {
                            "section": section,
                            "subsection": subsection,
                            "confidence": combined_conf,
                            "dimensions": {
                                "actual": {},
                                "proposed": {
                                    "existing_dimensions": {},
                                    "new_dimensions": {},
                                },
                            },
                        }
                    )
        return tags

    def tag_entry(
        self,
        text_description: str,
        code: str,
    ) -> dict:
        """
        Classifies a CPT entry with both code and description, returning a structured tag object.

        Args:
            text_description (str): The CPT description.
            code (str): The CPT code.

        Returns:
            dict: A dictionary containing the code, description, and classification tags.
        """
        tags = []
        generated_tags = self.generate_tags(text_description)
        for tag in generated_tags:
            # For "others", fill all fields for consistency
            if tag.get("subsection") == "others":
                tags.append(
                    {
                        "section": tag.get("section"),
                        "subsection": tag.get("subsection"),
                        "confidence": tag.get("confidence"),
                        "dimensions": {},
                    }
                )
            else:
                tags.append(
                    {
                        "section": tag.get("section"),
                        "subsection": tag.get("subsection"),
                        "confidence": tag.get("confidence"),
                        "dimensions": tag.get("dimensions", {}),
                    }
                )
        return {
            "code": code,
            "description": text_description,
            "tags": tags,
        }

    def _call_openai_completion(self, **kwargs):
        """
        Wrapper for calling OpenAI's chat completion API.
        This should be replaced or mocked during testing.
        """
        return self.client.chat.completions.create(**kwargs)


# Add json for cache
# Add "others" for section and subsection
# Implement detail tagging
