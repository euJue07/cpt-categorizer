from datetime import datetime
import json
import time
from typing import List, Optional, Tuple
import uuid

import openai
from openai import APIConnectionError, RateLimitError, APITimeoutError

from cpt_categorizer.config.openai import OPENAI_MODEL
from cpt_categorizer.utils.logging import log_agent_usage


SECTION_PROMPT_TEMPLATE = """You are a medical classification expert specializing in CPT tagging.
Your task is to classify a CPT description into the most appropriate top-level Sections for medical services.
Use the following Section names as your guide. Base your judgment on the clinical domain and nature of the service:
{sections_str}
Return a list of plausible sections with confidence scores between 0 and 1.

Output a JSON array of objects with fields: 'section' and 'confidence'.
"""

SECTION_FUNCTION_SPECIFICATION_TEMPLATE = {
    "name": "select_sections",
    "description": "Classifies a text description into multiple Sections with confidence scores",
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
                            "enum": [],  # To be filled dynamically
                            "description": "Top-level Section",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence score for the classification",
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

    def __init__(
        self,
        service_group_schema: dict,
        service_group_dimension_schema: Optional[dict] = None,
        service_group_dimension_value_schema: Optional[dict] = None,
        client: Optional[openai.OpenAI] = None,
    ):
        self.client = client or openai.OpenAI()

        # Assign schemas from parameters
        self.service_group_schema = service_group_schema
        self.service_group_dimension_schema = service_group_dimension_schema
        self.service_group_dimension_value_schema = service_group_dimension_value_schema

        # Load section, subsection, and detail schema (now unified)
        self.sections = list(self.service_group_schema["sections"].keys())

        self.sections_str = "\n".join(
            f"- {section}: {self.service_group_schema['sections'][section].get('description', '')}"
            for section in self.sections
        )

        self.section_prompt = SECTION_PROMPT_TEMPLATE.format(
            sections_str=self.sections_str
        )

        self.section_function_specification = (
            SECTION_FUNCTION_SPECIFICATION_TEMPLATE.copy()
        )
        self.section_function_specification["parameters"]["properties"]["sections"][
            "items"
        ]["properties"]["section"]["enum"] = self.sections

    def classify_sections(
        self,
        text_description: str,
    ) -> List[Tuple[str, float]]:
        """
        Classifies a CPT description into plausible top-level medical Sections using the OpenAI API.

        Args:
            text_description (str): The CPT description to classify.

        Returns:
            List[Tuple[str, float]]: A list of tuples, each containing a section name and its associated confidence score.
        """
        import logging

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
                except (RateLimitError, APIConnectionError, APITimeoutError) as e:
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
                ]
                result = validated
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
                parsed_output="{}",
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
            )
        return result

    def _get_subsection_prompt(self, section: str) -> str:
        """
        Constructs a prompt to classify subsections for a given section.

        Args:
            section (str): The section under which to classify subsections.

        Returns:
            str: A formatted prompt string including the subsection options.
        """
        section_data = self.service_group_schema["sections"].get(section, {})
        subsection_definitions = section_data.get("subsections", {})
        if isinstance(subsection_definitions, dict):
            formatted = "\n".join(
                f"- {key}: {value.get('description', '')}"
                for key, value in subsection_definitions.items()
            )
        else:
            formatted = "\n".join(f"- {item}" for item in subsection_definitions)
        return f"""You are a medical classification expert specializing in CPT tagging.
Given a CPT description and its assigned Section: {section}, identify the most appropriate Subsections based on clinical content.

Choose from the following options:
{formatted}

Return a list of plausible subsections with confidence scores between 0 and 1.
Output a JSON array of objects with fields: 'subsection' and 'confidence'.
"""

    def _get_subsection_function_specification(self, section: str) -> dict:
        """
        Constructs the OpenAI function specification for valid subsection classifications under a given section.

        Args:
            section (str): The section name.

        Returns:
            dict: A dictionary representing the JSON schema for allowed subsections under the section.
        """
        subsections_raw = (
            self.service_group_schema["sections"]
            .get(section, {})
            .get("subsections", {})
        )
        enum_values = (
            list(subsections_raw.keys())
            if isinstance(subsections_raw, dict)
            else subsections_raw
        )

        return {
            "name": f"select_subsections_for_{section.lower()}",
            "description": f"Classifies text description into multiple subsections under {section} with confidence scores",
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
                                    "description": "Confidence score for the classification",
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
                except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                    if attempt < 2:
                        time.sleep(2**attempt)
                        continue
                    else:
                        raise
            try:
                parsed_args = json.loads(
                    response.choices[0].message.function_call.arguments
                )
                valid_subsections = (
                    self.service_group_schema["sections"]
                    .get(section, {})
                    .get("subsections", {})
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
                ]
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
                parsed_output="{}",
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
            )
        return result

    def generate_tags(self, text_description: str) -> List[dict]:
        """
        Generates section and subsection tags for a CPT description.

        Args:
            text_description (str): The CPT description to classify.

        Returns:
            List[dict]: A list of dictionaries containing section, subsection, confidence, and placeholder details.
        """
        tags = []
        candidate_sections = self.classify_sections(text_description)
        print("Sections:", candidate_sections)
        for section, sec_conf in candidate_sections:
            candidate_subsections = self.classify_subsections(section, text_description)
            print("Subsections for", section, ":", candidate_subsections)
            for subsection, sub_conf in candidate_subsections:
                combined_conf = min(sec_conf, sub_conf)
                tags.append(
                    {
                        "section": section,
                        "subsection": subsection,
                        "confidence": combined_conf,
                        "details": {},  # Placeholder for future detail tagging
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
        tags = self.generate_tags(text_description)
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
