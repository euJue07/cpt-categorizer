import os
from datetime import datetime

from cpt_categorizer.agents.compliance import SchemaComplianceAgent
from cpt_categorizer.agents.normalizer import NormalizerAgent
from cpt_categorizer.agents.parsing import ParsingAgent
from cpt_categorizer.agents.tagging import TaggingAgent
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
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base_code = f"{idx:04d}"
    raw_desc = row[col_name_desc]
    parsed_desc = parsing_agent.parse(raw_desc)
    result = tagging_agent.tag_entry(text_description=parsed_desc, code=base_code)

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

    for i, tag in enumerate(result["tags"]):
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
        )

    write_output_csvs(category_rows, dimension_rows)


if __name__ == "__main__":
    top_n_input = input("How many top rows to process? [default: 10]: ").strip()
    run_pipeline(top_n=int(top_n_input) if top_n_input else 10, debug=True)
