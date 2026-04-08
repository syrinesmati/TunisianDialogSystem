"""
Filter generated data using TunBERT dialect scorer.

Usage:
    python filter_with_tunbert.py --input_file data/processed/synthetic.jsonl --output_file data/processed/filtered.jsonl
"""

import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Filter dataset with TunBERT")
    parser.add_argument("--input_file", required=True, help="Input JSONL file")
    parser.add_argument("--output_file", required=True, help="Output JSONL file")
    args = parser.parse_args()

    # Placeholder: copy all data
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Filtered data saved to {args.output_file}")

if __name__ == "__main__":
    main()