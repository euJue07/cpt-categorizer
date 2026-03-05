import os
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Optional

from cpt_categorizer.agents.compliance import SchemaComplianceAgent
from cpt_categorizer.agents.dimension_governor import DimensionGovernorAgent
from cpt_categorizer.agents.dimension_suggestor import DimensionSuggestorAgent
from cpt_categorizer.agents.normalizer import NormalizerAgent
from cpt_categorizer.agents.parsing import ParsingAgent
from cpt_categorizer.agents.section_governor import SectionGovernorAgent
from cpt_categorizer.agents.section_suggestor import SectionSuggestorAgent
from cpt_categorizer.agents.subsection_governor import SubsectionGovernorAgent
from cpt_categorizer.agents.subsection_suggestor import SubsectionSuggestorAgent
from cpt_categorizer.agents.tagging import TaggingAgent
from cpt_categorizer.agents.tagging import empty_dimensions
from cpt_categorizer.config.directory import LOG_DIR
from cpt_categorizer.config.directory import RAW_DIR
from cpt_categorizer.schema_contract import load_schema_contract


def ensure_csv_exists(path, columns):
    import pandas as pd

    if not path.exists():
        pd.DataFrame(columns=columns).to_csv(path, index=False)


def process_row(
    row,
    idx,
    parsing_agent,
    tagging_agent,
    normalizer_agent,
    compliance_agent,
    schema_version,
    col_name_desc,
    col_name_key,
    category_rows,
    dimension_rows,
    *,
    section_suggestor: Optional[SectionSuggestorAgent] = None,
    subsection_suggestor: Optional[SubsectionSuggestorAgent] = None,
    dimension_suggestor: Optional[DimensionSuggestorAgent] = None,
    confidence_threshold: float = 0.5,
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base_code = f"{idx:04d}"
    raw_desc = row[col_name_desc]
    parsed_desc = parsing_agent.parse(raw_desc)

    # Step-by-step tagging with suggestor hooks (replicates TaggingAgent.generate_tags logic)
    candidate_sections = tagging_agent.classify_sections(
        parsed_desc, confidence_threshold=confidence_threshold
    )
    if not candidate_sections:
        return

    tags: list[dict[str, Any]] = []
    for section, section_conf in candidate_sections:
        if section == "others":
            if section_suggestor is not None:
                section_suggestor.suggest_section(parsed_desc, source=base_code)
            candidate_subsections = tagging_agent.classify_subsections(
                section, parsed_desc, confidence_threshold=confidence_threshold
            )
            for subsection, subsection_conf in candidate_subsections:
                combined_conf = (section_conf + subsection_conf) / 2
                tags.append(
                    {
                        "section": section,
                        "subsection": subsection,
                        "confidence": float(combined_conf),
                        "dimensions": empty_dimensions(),
                    }
                )
            continue

        candidate_subsections = tagging_agent.classify_subsections(
            section, parsed_desc, confidence_threshold=confidence_threshold
        )
        if not candidate_subsections:
            if subsection_suggestor is not None:
                subsection_suggestor.suggest_subsection(
                    section, parsed_desc, source=base_code
                )
            continue

        for subsection, subsection_conf in candidate_subsections:
            dimensions = tagging_agent.classify_dimensions(
                section, subsection, parsed_desc
            )
            proposed = dimensions.get("proposed") or {}
            has_proposed = bool(
                proposed.get("existing_dimensions") or proposed.get("new_dimensions")
            )
            if dimension_suggestor is not None and has_proposed:
                dimension_suggestor.suggest_dimensions(
                    section,
                    subsection,
                    parsed_desc,
                    proposed,
                    source=base_code,
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
            dims_for_tag = (
                dimensions if subsection != "others" else empty_dimensions()
            )
            tags.append(
                {
                    "section": section,
                    "subsection": subsection,
                    "confidence": float(combined_conf),
                    "dimensions": dims_for_tag,
                }
            )

    def append_dim_rows(code, source, is_prop_dim, is_prop_val):
        for dim, values in source.items():
            for val in values:
                dimension_rows.append(
                    {
                        "Code": code,
                        "DescKey": row.get(col_name_key, None),
                        "Dimension": dim,
                        "Value": val["value"],
                        "Confidence": val["confidence"],
                        "IsProposedDimension": is_prop_dim,
                        "IsProposedValue": is_prop_val,
                        "SchemaVersion": schema_version,
                        "Timestamp": timestamp,
                    }
                )

    def append_category_row(code, desc, tag):
        category_rows.append(
            {
                "Code": code,
                "CPTDesc": desc,
                "DescKey": row.get(col_name_key, None),
                "Section": tag["section"],
                "Subsection": tag["subsection"],
                "Confidence": tag["confidence"],
                "SchemaVersion": schema_version,
                "Timestamp": timestamp,
            }
        )

    for i, tag in enumerate(tags):
        normalized_tag = normalizer_agent.normalize_tag(tag)
        compliant_tag, _warnings = compliance_agent.validate_tag(normalized_tag)
        if compliant_tag is None:
            continue

        code = f"{base_code}-{i + 1:02d}"
        append_category_row(code, raw_desc, compliant_tag)
        actual = compliant_tag.get("dimensions", {}).get("actual", {})
        proposed = compliant_tag.get("dimensions", {}).get("proposed", {})
        append_dim_rows(code, actual, False, False)
        append_dim_rows(code, proposed.get("existing_dimensions", {}), False, True)
        append_dim_rows(code, proposed.get("new_dimensions", {}), True, True)


def write_output_csvs(category_rows, dimension_rows):
    import pandas as pd

    category_file = LOG_DIR / "category_result.csv"
    dimension_file = LOG_DIR / "dimension_result.csv"
    pd.DataFrame(category_rows).to_csv(category_file, index=False)
    pd.DataFrame(dimension_rows).to_csv(dimension_file, index=False)


def run_pipeline(
    top_n=10,
    csv_path=RAW_DIR / "CPT Desc With Utilization.csv",
    debug=False,
    col_name_desc="CPTDesc",
    col_name_key="CPTDescKey",
    suggestions_path: Optional[Path] = None,
):
    import pandas as pd
    from tqdm import tqdm

    df = pd.read_csv(csv_path).dropna(subset=[col_name_desc])
    top_cpts = df.head(top_n)
    schema_contract = load_schema_contract()

    os.makedirs(LOG_DIR, exist_ok=True)
    category_path = LOG_DIR / "category_result.csv"
    dimension_path = LOG_DIR / "dimension_result.csv"
    ensure_csv_exists(
        category_path,
        [
            "Code",
            "CPTDesc",
            "DescKey",
            "Section",
            "Subsection",
            "Confidence",
            "SchemaVersion",
            "Timestamp",
        ],
    )
    ensure_csv_exists(
        dimension_path,
        [
            "Code",
            "DescKey",
            "Dimension",
            "Value",
            "Confidence",
            "IsProposedDimension",
            "IsProposedValue",
            "SchemaVersion",
            "Timestamp",
        ],
    )

    parsing_agent = ParsingAgent()
    tagging_agent = TaggingAgent(
        section_schema=schema_contract.sections,
        subsection_schema=schema_contract.subsections,
        dimension_schema=schema_contract.dimensions,
        schema_version=schema_contract.version,
    )
    normalizer_agent = NormalizerAgent()
    compliance_agent = SchemaComplianceAgent(schema_contract=schema_contract, mode="balanced")

    section_suggestor = SectionSuggestorAgent(
        section_schema=schema_contract.sections,
        schema_version=schema_contract.version,
        store_path=suggestions_path,
    )
    subsection_suggestor = SubsectionSuggestorAgent(
        section_schema=schema_contract.sections,
        subsection_schema=schema_contract.subsections,
        schema_version=schema_contract.version,
        store_path=suggestions_path,
    )
    dimension_suggestor = DimensionSuggestorAgent(
        dimension_schema=schema_contract.dimensions,
        subsection_schema=schema_contract.subsections,
        schema_version=schema_contract.version,
        store_path=suggestions_path,
    )
    section_governor = SectionGovernorAgent(
        section_schema=schema_contract.sections,
        schema_version=schema_contract.version,
        store_path=suggestions_path,
    )
    subsection_governor = SubsectionGovernorAgent(
        subsection_schema=schema_contract.subsections,
        schema_version=schema_contract.version,
        store_path=suggestions_path,
    )
    dimension_governor = DimensionGovernorAgent(
        dimension_schema=schema_contract.dimensions,
        schema_version=schema_contract.version,
        store_path=suggestions_path,
    )

    category_rows = []
    dimension_rows = []
    for idx, row in tqdm(top_cpts.iterrows(), total=len(top_cpts), desc="Tagging CPTs"):
        if debug:
            tqdm.write(f"\n=== Processing CPT {idx:04d} ===")
            tqdm.write(f"Description: {row[col_name_desc]}")
        process_row(
            row=row,
            idx=idx,
            parsing_agent=parsing_agent,
            tagging_agent=tagging_agent,
            normalizer_agent=normalizer_agent,
            compliance_agent=compliance_agent,
            schema_version=schema_contract.version,
            col_name_desc=col_name_desc,
            col_name_key=col_name_key,
            category_rows=category_rows,
            dimension_rows=dimension_rows,
            section_suggestor=section_suggestor,
            subsection_suggestor=subsection_suggestor,
            dimension_suggestor=dimension_suggestor,
        )

    section_governor.resolve_pending_sections()
    subsection_governor.resolve_pending_subsections()
    dimension_governor.resolve_pending_dimensions()

    write_output_csvs(category_rows, dimension_rows)


if __name__ == "__main__":
    top_n_input = input("How many top rows to process? [default: 10]: ").strip()
    run_pipeline(top_n=int(top_n_input) if top_n_input else 10, debug=True)
