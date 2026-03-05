#!/usr/bin/env python3
"""
Lint the canonical schema files under schema/.
Checks dimensions.json: no duplicate values, snake_case, alphabetical sort.
Exits with non-zero status on failure.
"""
import json
import re
import sys
from pathlib import Path

SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")


def main():
    repo_root = Path(__file__).resolve().parent.parent
    schema_dir = repo_root / "schema"
    dimensions_path = schema_dir / "dimensions.json"

    if not dimensions_path.exists():
        print(f"Error: {dimensions_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(dimensions_path) as f:
        dimensions = json.load(f)

    errors = []

    for dim_name, dim_spec in dimensions.items():
        values = dim_spec.get("values", [])
        if not isinstance(values, list):
            errors.append(f"{dim_name}: 'values' must be a list")
            continue

        # 1. No duplicates
        seen = set()
        for v in values:
            if v in seen:
                errors.append(f"{dim_name}: duplicate value '{v}'")
            seen.add(v)

        # 2. snake_case
        for v in values:
            if not isinstance(v, str):
                errors.append(f"{dim_name}: value must be string, got {type(v).__name__}: {v}")
            elif not SNAKE_CASE.match(v):
                errors.append(f"{dim_name}: value not snake_case: '{v}'")

        # 3. Alphabetically sorted
        sorted_values = sorted(values)
        if values != sorted_values:
            errors.append(
                f"{dim_name}: 'values' not alphabetically sorted (e.g. first wrong: expected "
                f"'{sorted_values[0]}' at index 0, got '{values[0]}')"
            )

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)

    print("Schema lint passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
