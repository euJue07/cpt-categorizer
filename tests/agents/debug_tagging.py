import json
import pandas as pd

from cpt_categorizer.agents.tagging import TaggingAgent
from cpt_categorizer.config.directory import RAW_DIR
from cpt_categorizer.config.directory import SCHEMA_DIR
import inspect
from cpt_categorizer.agents import tagging

print(inspect.getfile(tagging.TaggingAgent))
with open(SCHEMA_DIR / "service_group_schema.json") as f:
    service_group_schema = json.load(f)
with open(SCHEMA_DIR / "service_group_dimension_schema.json") as f:
    service_group_dimension_schema = json.load(f)
with open(SCHEMA_DIR / "service_group_dimension_value_schema.json") as f:
    service_group_dimension_value_schema = json.load(f)

shared_agent = TaggingAgent(
    service_group_schema=service_group_schema,
    service_group_dimension_schema=service_group_dimension_schema,
    service_group_dimension_value_schema=service_group_dimension_value_schema,
)


def run_tagging_agent_interactively(description, code="0000", agent=shared_agent):
    sections = agent.classify_sections(description)
    print("Step 1 - Sections:", sections)

    for section, _ in sections:
        subsections = agent.classify_subsections(section, description)
        print(f"Step 2 - Subsections for section '{section}':", subsections)

    tags = agent.generate_tags(description)
    print("Step 3 - Tags:", tags)

    result = agent.tag_entry(description, code)
    print("Step 4 - Final Tag Entry:\n", json.dumps(result, indent=2))


def run_from_csv_sample(n=5, pattern=None):
    path = RAW_DIR / "CPT Desc With Utilization.csv"
    df = pd.read_csv(path)

    if pattern:
        df = df[df["CPTDesc"].str.contains(pattern, case=False, na=False, regex=True)]

    sample = df.sample(n=min(n, len(df)), random_state=42)
    for i, row in sample.iterrows():
        code = str(row.get("CPTDescKey", f"{i:04}"))
        description = row["CPTDesc"]
        print(f"\n=== SAMPLE {i} ===")
        run_tagging_agent_interactively(description, code=code)


if __name__ == "__main__":
    print("Select mode:")
    print("1 - Type a CPT description manually")
    print("2 - Sample from CSV file")
    mode = input("Enter 1 or 2: ").strip()

    if mode == "1":
        description = input("Enter CPT description: ").strip()
        run_tagging_agent_interactively(description)
    elif mode == "2":
        try:
            n = int(input("How many rows to sample? (default 5): ").strip() or "5")
        except ValueError:
            n = 5
        pattern = (
            input(
                "Optional regex to filter descriptions (press enter to skip): "
            ).strip()
            or None
        )
        run_from_csv_sample(n=n, pattern=pattern)
    else:
        print("Invalid selection. Please enter 1 or 2.")
