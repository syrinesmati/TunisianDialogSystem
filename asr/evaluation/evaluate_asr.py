"""
Evaluate ASR models on WER, CER, Arabic character ratio.

Usage:
    python evaluate_asr.py --predictions_file predictions.txt --references_file references.txt --output_file results.json
"""

import argparse
import json
from typing import List

import jiwer

def compute_wer(predictions: List[str], references: List[str]) -> float:
    """Compute Word Error Rate."""
    return jiwer.wer(references, predictions)

def compute_cer(predictions: List[str], references: List[str]) -> float:
    """Compute Character Error Rate."""
    return jiwer.cer(references, predictions)

def compute_arabic_ratio(text: str) -> float:
    """Compute fraction of characters that are Arabic."""
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    total_chars = len(text)
    return arabic_chars / total_chars if total_chars > 0 else 0

def main():
    parser = argparse.ArgumentParser(description="Evaluate ASR models")
    parser.add_argument("--predictions_file", required=True, help="File with predicted transcripts")
    parser.add_argument("--references_file", required=True, help="File with reference transcripts")
    parser.add_argument("--output_file", required=True, help="Output JSON file for results")
    args = parser.parse_args()

    with open(args.predictions_file, 'r', encoding='utf-8') as f:
        predictions = [line.strip() for line in f]

    with open(args.references_file, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f]

    wer = compute_wer(predictions, references)
    cer = compute_cer(predictions, references)
    arabic_ratios = [compute_arabic_ratio(pred) for pred in predictions]
    avg_arabic_ratio = sum(arabic_ratios) / len(arabic_ratios)

    results = {
        "WER": wer,
        "CER": cer,
        "Average Arabic Character Ratio": avg_arabic_ratio
    }

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()