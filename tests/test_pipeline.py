"""Tests for pipeline wiring: suggestors triggered on outside/proposed, governors on pending."""

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from cpt_categorizer.agents.compliance import SchemaComplianceAgent
from cpt_categorizer.agents.normalizer import NormalizerAgent
from cpt_categorizer.agents.parsing import ParsingAgent
from cpt_categorizer.pipeline import process_row
from cpt_categorizer.pipeline import run_pipeline
from cpt_categorizer.schema_contract import load_schema_contract


@pytest.fixture
def schema_contract():
    return load_schema_contract(use_cache=False)


@pytest.mark.generate_tags
def test_process_row_section_others_invokes_section_suggestor(schema_contract):
    """When section is 'others', section_suggestor.suggest_section is called."""
    section_suggestor = MagicMock()
    section_suggestor.suggest_section.return_value = None

    class StubTaggingAgent:
        def classify_sections(self, text_description: str, confidence_threshold: float = 0.5):
            return [("others", 0.8)]

        def classify_subsections(
            self, section: str, text_description: str, confidence_threshold: float = 0.5
        ):
            return [("others", 1.0)]

        def classify_dimensions(
            self, section: str, subsection: str, text_description: str
        ):
            return {
                "actual": {},
                "proposed": {"existing_dimensions": {}, "new_dimensions": {}},
            }

    category_rows = []
    dimension_rows = []
    process_row(
        row={"CPTDesc": "Unknown procedure", "CPTDescKey": "K1"},
        idx=0,
        parsing_agent=ParsingAgent(),
        tagging_agent=StubTaggingAgent(),
        normalizer_agent=NormalizerAgent(),
        compliance_agent=SchemaComplianceAgent(schema_contract=schema_contract, mode="balanced"),
        schema_version=schema_contract.version,
        col_name_desc="CPTDesc",
        col_name_key="CPTDescKey",
        category_rows=category_rows,
        dimension_rows=dimension_rows,
        section_suggestor=section_suggestor,
        subsection_suggestor=None,
        dimension_suggestor=None,
    )
    section_suggestor.suggest_section.assert_called_once()
    call_kw = section_suggestor.suggest_section.call_args
    assert call_kw[0][0] == "Unknown procedure"  # parsed (strip/collapse only)
    assert call_kw[1]["source"] == "0000"


@pytest.mark.generate_tags
def test_process_row_subsection_empty_invokes_subsection_suggestor(schema_contract):
    """When section is not 'others' but subsections are empty, subsection_suggestor is called."""
    subsection_suggestor = MagicMock()

    class StubTaggingAgent:
        def classify_sections(self, text_description: str, confidence_threshold: float = 0.5):
            return [("imaging", 0.9)]

        def classify_subsections(
            self, section: str, text_description: str, confidence_threshold: float = 0.5
        ):
            return []  # empty -> outside

        def classify_dimensions(
            self, section: str, subsection: str, text_description: str
        ):
            return {}

    category_rows = []
    dimension_rows = []
    process_row(
        row={"CPTDesc": "Rare imaging", "CPTDescKey": "K2"},
        idx=1,
        parsing_agent=ParsingAgent(),
        tagging_agent=StubTaggingAgent(),
        normalizer_agent=NormalizerAgent(),
        compliance_agent=SchemaComplianceAgent(schema_contract=schema_contract, mode="balanced"),
        schema_version=schema_contract.version,
        col_name_desc="CPTDesc",
        col_name_key="CPTDescKey",
        category_rows=category_rows,
        dimension_rows=dimension_rows,
        section_suggestor=None,
        subsection_suggestor=subsection_suggestor,
        dimension_suggestor=None,
    )
    subsection_suggestor.suggest_subsection.assert_called_once()
    call_kw = subsection_suggestor.suggest_subsection.call_args
    assert call_kw[0][0] == "imaging"
    assert call_kw[0][1] == "Rare imaging"  # parsed (strip/collapse only)
    assert call_kw[1]["source"] == "0001"
    assert len(category_rows) == 0  # no tags produced


@pytest.mark.generate_tags
def test_process_row_proposed_dimensions_invokes_dimension_suggestor(schema_contract):
    """When a tag has proposed existing_dimensions or new_dimensions, dimension_suggestor is called."""
    dimension_suggestor = MagicMock()

    class StubTaggingAgent:
        def classify_sections(self, text_description: str, confidence_threshold: float = 0.5):
            return [("imaging", 0.95)]

        def classify_subsections(
            self, section: str, text_description: str, confidence_threshold: float = 0.5
        ):
            return [("xray", 0.9)]

        def classify_dimensions(
            self, section: str, subsection: str, text_description: str
        ):
            return {
                "actual": {"contrast": [{"value": "with_contrast", "confidence": 0.9}]},
                "proposed": {
                    "existing_dimensions": {"contrast": [{"value": "new_value", "confidence": 0.8}]},
                    "new_dimensions": {},
                },
            }

    category_rows = []
    dimension_rows = []
    process_row(
        row={"CPTDesc": "X-ray with new value", "CPTDescKey": "K3"},
        idx=2,
        parsing_agent=ParsingAgent(),
        tagging_agent=StubTaggingAgent(),
        normalizer_agent=NormalizerAgent(),
        compliance_agent=SchemaComplianceAgent(schema_contract=schema_contract, mode="balanced"),
        schema_version=schema_contract.version,
        col_name_desc="CPTDesc",
        col_name_key="CPTDescKey",
        category_rows=category_rows,
        dimension_rows=dimension_rows,
        section_suggestor=None,
        subsection_suggestor=None,
        dimension_suggestor=dimension_suggestor,
    )
    dimension_suggestor.suggest_dimensions.assert_called_once()
    call = dimension_suggestor.suggest_dimensions.call_args
    assert call[0][0] == "imaging"
    assert call[0][1] == "xray"
    assert "existing_dimensions" in call[0][3]  # 4th positional arg is proposed


@pytest.mark.generate_tags
def test_run_pipeline_calls_governors_after_processing(tmp_path: Path):
    """run_pipeline invokes all three governors after the tagging loop."""
    pytest.importorskip("pandas")
    csv_path = tmp_path / "cpt.csv"
    csv_path.write_text("CPTDesc,CPTDescKey\nChest X-ray,X1\n")

    with patch(
        "cpt_categorizer.pipeline.SectionGovernorAgent"
    ) as mock_section_gov_cls, patch(
        "cpt_categorizer.pipeline.SubsectionGovernorAgent"
    ) as mock_subsection_gov_cls, patch(
        "cpt_categorizer.pipeline.DimensionGovernorAgent"
    ) as mock_dim_gov_cls:
        mock_section_gov = MagicMock()
        mock_subsection_gov = MagicMock()
        mock_dim_gov = MagicMock()
        mock_section_gov_cls.return_value = mock_section_gov
        mock_subsection_gov_cls.return_value = mock_subsection_gov
        mock_dim_gov_cls.return_value = mock_dim_gov

        run_pipeline(
            top_n=1,
            csv_path=csv_path,
            col_name_desc="CPTDesc",
            col_name_key="CPTDescKey",
        )

    mock_section_gov.resolve_pending_sections.assert_called_once()
    mock_subsection_gov.resolve_pending_subsections.assert_called_once()
    mock_dim_gov.resolve_pending_dimensions.assert_called_once()
