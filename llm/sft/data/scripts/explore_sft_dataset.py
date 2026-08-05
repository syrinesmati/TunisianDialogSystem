#!/usr/bin/env python3
"""
Script to explore the Tunisian question-response SFT training dataset
Dataset: https://huggingface.co/datasets/Syrinesmati/tunisian-question-response-dataset
"""

from datasets import load_dataset
import pandas as pd

# Load the dataset
print("Loading dataset from HuggingFace...")
dataset = load_dataset("Syrinesmati/tunisian-question-response-dataset")

print(f"\nDataset keys: {dataset.keys()}")
print(f"\nDataset structure:")
for split in dataset.keys():
    print(f"  {split}: {len(dataset[split])} samples")

# Display basic info about the first split
first_split = list(dataset.keys())[0]
print(f"\n{'='*60}")
print(f"Exploring '{first_split}' split")
print(f"{'='*60}")

print(f"\nDataset features:")
print(dataset[first_split].features)

print(f"\nFirst few samples:")
for i in range(min(3, len(dataset[first_split]))):
    print(f"\n--- Sample {i+1} ---")
    for key, value in dataset[first_split][i].items():
        print(f"{key}: {value}")

# Search for multiple terms
search_terms = ["فلفل أسود", "بيض", "بيضات"]

print(f"\n{'='*60}")
print("Searching for multiple terms")
print(f"{'='*60}")

for search_term in search_terms:
    count = 0
    matching_samples = []

    for split in dataset.keys():
        for idx, sample in enumerate(dataset[split]):
            # Search in all text fields
            sample_text = " ".join(str(v) for v in sample.values())
            if search_term in sample_text:
                count += 1
                if len(matching_samples) < 5:  # Store first 5 matches for display
                    matching_samples.append({
                        "split": split,
                        "index": idx,
                        "sample": sample
                    })

    print(f"\n--- Searching for '{search_term}' ---")
    print(f"Total rows containing '{search_term}': {count}")

    if matching_samples:
        print(f"First {min(5, len(matching_samples))} matching samples:")
        for i, match in enumerate(matching_samples):
            print(f"\n  Match {i+1} (from {match['split']}, index {match['index']}):")
            for key, value in match['sample'].items():
                # Highlight the search term
                if search_term in str(value):
                    print(f"    {key}: *** {value} ***")
                else:
                    preview = str(value)[:80] if len(str(value)) > 80 else str(value)
                    print(f"    {key}: {preview}")

# Convert to pandas for easier analysis
print(f"\n{'='*60}")
print("Dataset Statistics")
print(f"{'='*60}")

for split in dataset.keys():
    df = pd.DataFrame(dataset[split])
    print(f"\n{split} statistics:")
    print(df.info())
    print(f"\nDataframe shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())

print("\n✓ Exploration complete!")
