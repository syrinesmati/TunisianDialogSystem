"""Quick sanity check for the trained adapter.

Usage:
    source /home/ala/myenv/bin/activate
    python llm/cpt/training/quick_test.py --output_dir outputs/checkpoints/aya-expanse-8b-cpt-tunisian
"""
from __future__ import annotations
import argparse
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="CohereLabs/aya-expanse-8b")
    p.add_argument("--output_dir", default="outputs/checkpoints/aya-expanse-8b-cpt-tunisian")
    p.add_argument("--prompts", nargs="*", default=["اليوم الطقس مزيان شنوة نجم نعمل؟"]) 
    p.add_argument("--max_new_tokens", type=int, default=120)
    return p.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model (will load on-device automatically)...")
    base = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto", trust_remote_code=True)

    print("Attaching adapter from:", args.output_dir)
    model = PeftModel.from_pretrained(base, args.output_dir)
    model.eval()

    for prompt in args.prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        print("\n=== Prompt ===")
        print(prompt)
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.7)
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        # trim prompt
        if decoded.startswith(prompt):
            decoded = decoded[len(prompt):].strip()
        print("\n--- Generation ---")
        print(decoded)


if __name__ == "__main__":
    main()
