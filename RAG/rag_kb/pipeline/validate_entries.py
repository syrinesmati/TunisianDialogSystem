"""
validate_entries.py

Validates JSON entries against Pydantic schemas.
Bad data = bad AI. This is critical.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from schemas.expression_schema import ExpressionEntry
from schemas.proverb_schema import ProverbEntry


def validate_file(file_path: str) -> Tuple[List[dict], List[Tuple[str, str]]]:
    """
    Validates a JSON file of entries against appropriate schemas.

    Args:
        file_path: Path to the JSON file

    Returns:
        Tuple of (valid_entries, errors)
        - valid_entries: List of validated entries
        - errors: List of (entry_id, error_message) tuples
    """
    valid_entries = []
    errors = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        errors.append(("FILE", f"File not found: {file_path}"))
        return [], errors
    except json.JSONDecodeError as e:
        errors.append(("FILE", f"Invalid JSON: {e}"))
        return [], errors

    if not isinstance(data, list):
        errors.append(("FILE", "Root must be a JSON array"))
        return [], errors

    for entry in data:
        entry_id = entry.get("id", "UNKNOWN")
        entry_type = entry.get("type", "UNKNOWN")

        try:
            if entry_type == "expression":
                validated = ExpressionEntry(**entry)
            elif entry_type == "proverb":
                validated = ProverbEntry(**entry)
            else:
                errors.append(
                    (
                        entry_id,
                        f"Unknown type: {entry_type}. Must be 'expression' or 'proverb'",
                    )
                )
                continue

            valid_entries.append(validated.model_dump())

        except Exception as e:
            errors.append((entry_id, str(e)))

    return valid_entries, errors


def print_validation_report(valid_entries: List[dict], errors: List[Tuple[str, str]]):
    """
    Prints a readable validation report.

    Args:
        valid_entries: List of validated entries
        errors: List of validation errors
    """
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    print(f"\n✓ Valid entries: {len(valid_entries)}")

    if errors:
        print(f"\n✗ Errors: {len(errors)}")
        print("-" * 60)
        for entry_id, error_msg in errors:
            print(f"  [{entry_id}] {error_msg}")
    else:
        print("\n✓ No errors found!")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Example usage
    valid, errors = validate_file("data/expressions.json")
    print_validation_report(valid, errors)
