import os
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from cpt_categorizer.agents.tagging import TaggingAgent
from cpt_categorizer.config.directory import RAW_DIR
from cpt_categorizer.config.directory import LOG_DIR
from cpt_categorizer.config.directory import SCHEMA_DIR


with open(SCHEMA_DIR / "sections.json") as f:
    section_schema = json.load(f)
with open(SCHEMA_DIR / "subsections.json") as f:
    subsection_schema = json.load(f)
with open(SCHEMA_DIR / "dimensions.json") as f:
    dimension_schema = json.load(f)


# Helper to ensure CSV file exists with given columns
def ensure_csv_exists(path, columns):
    if not path.exists():
        pd.DataFrame(columns=columns).to_csv(path, index=False)


def process_row(
    row, idx, agent, col_name_desc, col_name_key, category_rows, dimension_rows
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base_code = f"{idx:04d}"
    desc = row[col_name_desc]
    result = agent.tag_entry(text_description=desc, code=base_code)

    def append_dim_rows(source, is_prop_dim, is_prop_val):
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
                "Timestamp": timestamp,
            }
        )

    for i, tag in enumerate(result["tags"]):
        code = f"{base_code}-{i + 1:02d}"
        append_category_row(code, desc, tag)
        actual = tag.get("dimensions", {}).get("actual", {})
        proposed = tag.get("dimensions", {}).get("proposed", {})
        append_dim_rows(actual, False, False)
        append_dim_rows(proposed.get("existing_dimensions", {}), False, True)
        append_dim_rows(proposed.get("new_dimensions", {}), True, True)


def write_output_csvs(category_rows, dimension_rows):
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
    df = pd.read_csv(csv_path).dropna(subset=[col_name_desc])
    top_cpts = df.head(top_n)

    os.makedirs(LOG_DIR, exist_ok=True)

    # Create empty CSVs if they don't exist yet
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
            "Timestamp",
        ],
    )

    agent = TaggingAgent(section_schema, subsection_schema, dimension_schema)
    category_rows = []
    dimension_rows = []

    for idx, row in tqdm(top_cpts.iterrows(), total=len(top_cpts), desc="Tagging CPTs"):
        if debug:
            tqdm.write(f"\n=== Processing CPT {idx:04d} ===")
            tqdm.write(f"Description: {row[col_name_desc]}")
        process_row(
            row, idx, agent, col_name_desc, col_name_key, category_rows, dimension_rows
        )

    write_output_csvs(category_rows, dimension_rows)


if __name__ == "__main__":
    top_n_input = input("How many top rows to process? [default: 10]: ").strip()
    run_pipeline(
        top_n=int(top_n_input) if top_n_input else 10,
        debug=True,
    )
