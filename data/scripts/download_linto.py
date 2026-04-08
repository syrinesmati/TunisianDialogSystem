"""
Download LinTO dataset from HuggingFace.

Usage:
    python download_linto.py --output_dir data/raw/linto
"""

import argparse
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Download LinTO dataset")
    parser.add_argument("--output_dir", default="data/raw/linto", help="Output directory")
    args = parser.parse_args()

    dataset = load_dataset("linagora/linto-dataset-audio-ar-tn-0.1-augmented")
    dataset.save_to_disk(args.output_dir)

    print(f"Dataset saved to {args.output_dir}")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    print(f"Validation samples: {len(dataset['validation'])}")

    # Calculate total hours
    total_hours = sum(ex['duration'] for ex in dataset['train']) / 3600
    print(f"Total hours: {total_hours:.2f}")

if __name__ == "__main__":
    main()