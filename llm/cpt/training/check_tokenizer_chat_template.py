#!/usr/bin/env python3
"""Check whether the tokenizer's chat template appends EOS to assistant turns.

Usage:
    python llm/cpt/training/check_tokenizer_chat_template.py --tokenizer <model-or-path>

Default tokenizer: CohereLabs/aya-expanse-8b
"""
from __future__ import annotations

import argparse
import sys

from transformers import AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Check tokenizer.chat_template and EOS behavior")
    parser.add_argument(
        "--tokenizer",
        default="CohereLabs/aya-expanse-8b",
        help="Tokenizer name or path (default: CohereLabs/aya-expanse-8b)",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    print("\n=== tokenizer.chat_template ===")
    chat_template = getattr(tokenizer, "chat_template", None)
    print(repr(chat_template))

    print("\n=== tokenizer.eos_token/info ===")
    print("eos_token:", repr(getattr(tokenizer, "eos_token", None)))
    try:
        print("eos_token_id:", tokenizer.eos_token_id)
    except Exception:
        print("eos_token_id: (not available)")
    
    print("\n=== tokenizer.pad_token/info ===")
    print("pad_token:", repr(getattr(tokenizer, "pad_token", None)))
    try:
        print("pad_token_id:", tokenizer.pad_token_id)
    except Exception:
        print("pad_token_id: (not available)")

    # Build a small conversation where assistant content lacks EOS
    messages = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am fine, thanks."},
    ]

    print("\n=== apply_chat_template (string output) ===")
    try:
        # Try to get a plain string rendering of the chat template
        rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, return_dict=False)
        print(rendered)

        # Check whether assistant content in the rendered string ends with eos token text
        eos_token = getattr(tokenizer, "eos_token", "") or ""
        assistant_marker = "I am fine, thanks."
        # Simple check: does the rendered string contain the assistant content followed by eos token text?
        contains_eos = assistant_marker + eos_token in rendered
        print("Assistant content followed by eos token text in rendered string:", contains_eos)
    except Exception as exc:
        print("apply_chat_template (string) failed:", exc)

    print("\n=== tokenized form (if available) ===")
    try:
        tokenized = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors=None)
        # tokenized might be a dict with 'input_ids' or a string/token list depending on implementation
        print("apply_chat_template returned (tokenize=True, return_dict=True): type=", type(tokenized))
        print(tokenized)
    except Exception as exc:
        print("apply_chat_template (tokenize) failed:", exc)

    print("\nCheck complete. If the rendered template contains the eos token text immediately after the assistant content, the tokenizer/apply_chat_template appends EOS for assistant turns.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
