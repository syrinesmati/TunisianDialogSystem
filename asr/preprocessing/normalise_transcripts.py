"""
Normalise transcripts by transliterating Latin-script tokens to Arabic.

This script loads LinTO transcripts, detects Latin tokens, applies transliteration,
and saves normalised transcripts in Arabic script only.

Usage:
    python normalise_transcripts.py --input_dir data/raw/linto --output_dir data/processed/linto --dry_run
"""

import argparse
import os
from typing import List, Dict

from transliteration_dict import transliteration_dict

def detect_latin_tokens(text: str) -> List[str]:
    """Detect tokens that contain Latin characters."""
    tokens = text.split()
    latin_tokens = [token for token in tokens if any(c.isascii() and c.isalpha() for c in token)]
    return latin_tokens

def transliterate_token(token: str) -> str:
    """Transliterate a Latin token to Arabic phonetic equivalent."""
    lower_token = token.lower()
    if lower_token in transliteration_dict:
        return transliteration_dict[lower_token]
    # Fallback: character-by-character phonetic mapping
    mapping = {
        'a': 'ا', 'b': 'ب', 'c': 'ك', 'd': 'د', 'e': 'ي', 'f': 'ف', 'g': 'ج', 'h': 'ه',
        'i': 'ي', 'j': 'ج', 'k': 'ك', 'l': 'ل', 'm': 'م', 'n': 'ن', 'o': 'و', 'p': 'ب',
        'q': 'ق', 'r': 'ر', 's': 'س', 't': 'ت', 'u': 'و', 'v': 'ف', 'w': 'و', 'x': 'كس',
        'y': 'ي', 'z': 'ز'
    }
    return ''.join(mapping.get(c.lower(), c) for c in token if c.isascii())

def normalise_transcript(text: str) -> str:
    """Normalise a transcript by transliterating Latin tokens."""
    tokens = text.split()
    normalised_tokens = []
    for token in tokens:
        if any(c.isascii() and c.isalpha() for c in token):
            normalised_tokens.append(transliterate_token(token))
        else:
            normalised_tokens.append(token)
    return ' '.join(normalised_tokens)

def main():
    parser = argparse.ArgumentParser(description="Normalise LinTO transcripts")
    parser.add_argument("--input_dir", required=True, help="Input directory with transcripts")
    parser.add_argument("--output_dir", required=True, help="Output directory for normalised transcripts")
    parser.add_argument("--dry_run", action="store_true", help="Preview changes without saving")
    args = parser.parse_args()

    # Assume transcripts are in a file, e.g., transcripts.txt
    input_file = os.path.join(args.input_dir, "transcripts.txt")
    output_file = os.path.join(args.output_dir, "transcripts_normalised.txt")

    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        transcripts = f.readlines()

    normalised = []
    for transcript in transcripts:
        norm = normalise_transcript(transcript.strip())
        normalised.append(norm)
        if args.dry_run:
            print(f"Original: {transcript.strip()}")
            print(f"Normalised: {norm}")
            print()

    if not args.dry_run:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(normalised))
        print(f"Normalised transcripts saved to {output_file}")

if __name__ == "__main__":
    main()