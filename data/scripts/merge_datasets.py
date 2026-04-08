"""
Merge gold + synthetic into final training file.

Usage:
    python merge_datasets.py --gold_file data/raw/gold.jsonl --synthetic_file data/processed/filtered.jsonl --output_file data/processed/final.jsonl
"""

import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Merge datasets")
    parser.add_argument("--gold_file", required=True, help="Gold data JSONL file")
    parser.add_argument("--synthetic_file", required=True, help="Synthetic data JSONL file")
    parser.add_argument("--output_file", required=True, help="Output JSONL file")
    args = parser.parse_args()

    data = []

    with open(args.gold_file, 'r', encoding='utf-8') as f:
        data.extend([json.loads(line) for line in f])

    with open(args.synthetic_file, 'r', encoding='utf-8') as f:
        data.extend([json.loads(line) for line in f])

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Merged {len(data)} samples to {args.output_file}")

if __name__ == "__main__":
    main()