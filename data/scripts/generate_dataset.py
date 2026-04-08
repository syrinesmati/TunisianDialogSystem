"""
Generate synthetic Tunisian pairs via Claude API.

Usage:
    python generate_dataset.py --output_file data/processed/synthetic.jsonl --num_samples 1000
"""

import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset")
    parser.add_argument("--output_file", required=True, help="Output JSONL file")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples")
    args = parser.parse_args()

    # Placeholder: generate dummy data
    data = []
    for i in range(args.num_samples):
        data.append({
            "instruction": f"Instruction {i}",
            "response": f"Response {i}"
        })

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Generated {args.num_samples} samples to {args.output_file}")

if __name__ == "__main__":
    main()