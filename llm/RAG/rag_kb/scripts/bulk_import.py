"""
bulk_import.py

Bulk import entries from a CSV or JSONL file into the knowledge base
"""

import json
import csv
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Add parent directory to path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas.expression_schema import ExpressionEntry
from schemas.proverb_schema import ProverbEntry
from pipeline.validate_entries import validate_file, print_validation_report


def bulk_import_from_json(
    source_file: str, target_type: str = "expression"
) -> tuple[int, int]:
    """
    Import entries from a JSON file.

    Args:
        source_file: Path to source JSON file (should be array of objects)
        target_type: 'expression' or 'proverb'

    Returns:
        (successful_imports, failed_imports)
    """

    target_file = (
        Path(__file__).parent.parent / "data" / f"{target_type}s.json"
    )

    # Validate source
    valid_entries, errors = validate_file(source_file)

    if errors:
        print(f"\n⚠ Source file has {len(errors)} errors:")
        print_validation_report(valid_entries, errors)

    # Load target
    with open(target_file, "r", encoding="utf-8") as f:
        target_data = json.load(f)

    # Get next ID
    prefix = "expr_" if target_type == "expression" else "prov_"
    max_id = max(
        [int(entry["id"].split("_")[1]) for entry in target_data if entry["id"].startswith(prefix)],
        default=0,
    )

    # Import entries
    imported = 0
    failed = 0

    for i, entry in enumerate(valid_entries):
        try:
            # Assign new ID if needed
            if "id" not in entry or entry["id"].startswith("temp_"):
                max_id += 1
                entry["id"] = f"{prefix}{max_id:03d}"

            # Update timestamp
            entry["last_updated"] = datetime.now().strftime("%Y-%m-%d")

            # Validate against schema
            if target_type == "expression":
                ExpressionEntry(**entry)
            else:
                ProverbEntry(**entry)

            target_data.append(entry)
            imported += 1

        except Exception as e:
            print(f"  ✗ Entry {i} failed: {e}")
            failed += 1

    # Save
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(target_data, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Import complete:")
    print(f"  Imported: {imported}")
    print(f"  Failed: {failed}")
    print(f"  Total in {target_type}s.json: {len(target_data)}")

    return imported, failed


def bulk_import_from_csv(
    csv_file: str, target_type: str = "expression"
) -> tuple[int, int]:
    """
    Import entries from CSV format.

    CSV should have headers matching the schema fields.

    Args:
        csv_file: Path to CSV file
        target_type: 'expression' or 'proverb'

    Returns:
        (successful_imports, failed_imports)
    """

    target_file = (
        Path(__file__).parent.parent / "data" / f"{target_type}s.json"
    )

    # Load target
    with open(target_file, "r", encoding="utf-8") as f:
        target_data = json.load(f)

    # Get next ID
    prefix = "expr_" if target_type == "expression" else "prov_"
    max_id = max(
        [int(entry["id"].split("_")[1]) for entry in target_data if entry["id"].startswith(prefix)],
        default=0,
    )

    imported = 0
    failed = 0

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            try:
                # Convert string booleans
                if "gender_sensitive" in row:
                    row["gender_sensitive"] = row["gender_sensitive"].lower() == "true"

                # Handle optional lists
                if "scripts" in row and isinstance(row["scripts"], str):
                    row["scripts"] = [s.strip() for s in row["scripts"].split(",")]

                max_id += 1
                row["id"] = f"{prefix}{max_id:03d}"
                row["type"] = target_type
                row["last_updated"] = datetime.now().strftime("%Y-%m-%d")

                # Validate
                if target_type == "expression":
                    ExpressionEntry(**row)
                else:
                    ProverbEntry(**row)

                target_data.append(row)
                imported += 1

            except Exception as e:
                print(f"  ✗ Row {i+1} failed: {e}")
                failed += 1

    # Save
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(target_data, f, ensure_ascii=False, indent=2)

    print(f"\n✓ CSV Import complete:")
    print(f"  Imported: {imported}")
    print(f"  Failed: {failed}")
    print(f"  Total in {target_type}s.json: {len(target_data)}")

    return imported, failed


if __name__ == "__main__":
    print("Bulk import utilities ready")
    print("\nUsage:")
    print("  from bulk_import import bulk_import_from_json, bulk_import_from_csv")
    print("  bulk_import_from_json('path/to/file.json', 'expression')")
    print("  bulk_import_from_csv('path/to/file.csv', 'proverb')")
