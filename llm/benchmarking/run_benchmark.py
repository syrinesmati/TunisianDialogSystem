"""
Zero-shot eval of 5 candidate models.

Usage:
    python run_benchmark.py --models aya-expanse-8b labess-7b silma-9b llama-3-8b llama-3.2-1b --output results.json
"""

import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Run LLM benchmarks")
    parser.add_argument("--models", nargs="+", required=True, help="List of model names")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    results = {}
    for model in args.models:
        results[model] = {"score": 0.5}  # Placeholder

    with open(args.output, 'w') as f:
        json.dump(results, f)

    print(f"Benchmark results saved to {args.output}")

if __name__ == "__main__":
    main()