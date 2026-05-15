"""Test script for the fine-tuned SFT model.

Loads the trained CPT+SFT adapter and runs inference on Tunisian prompts.

Usage:
    python llm/cpt/training/test_sft_model.py
    python llm/cpt/training/test_sft_model.py \\
        --model_dir ./outputs/checkpoints/aya-expanse-8b-tunisian-sft \\
        --prompt "شنوة آخر أخبارك؟"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    'أنت "التيجاني"، مساعد ذكاء اصطناعي تونسي 100%. '
    'جاوب بالتونسي الدارجة فقط، وبالطول المناسب للسؤال: كان يلزم قصّر، وكان يلزم فسّر أكثر. '
    'ممنوع الهلوسة أو الخروج على الموضوع.'
)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("test_sft")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    default_model_dir = repo_root / "outputs" / "checkpoints" / "aya-expanse-8b-tunisian-sft"

    parser = argparse.ArgumentParser(description="Test fine-tuned SFT model for Tunisian dialogue")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(default_model_dir),
        help="Path to the fine-tuned model/adapter directory",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="CohereLabs/aya-expanse-8b",
        help="Base model name for loading tokenizer and base model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="عالسلامة انت شكونك ؟",
        help="Tunisian prompt to test",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling instead of greedy decoding",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling (if --do_sample)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling threshold",
    )
    return parser.parse_args()


def load_model_and_tokenizer(base_model_name: str, model_dir: str, logger: logging.Logger):
    """Load tokenizer, base model, and fine-tuned adapter."""
    logger.info("Loading tokenizer from: %s", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    logger.info("Loading base model: %s", base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.config.use_cache = True

    logger.info("Loading fine-tuned adapter from: %s", model_dir)
    model = PeftModel.from_pretrained(
        model,
        model_dir,
        is_trainable=False,
    )
    model.eval()

    return tokenizer, model


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a response using the fine-tuned model."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def main() -> None:
    args = parse_args()
    logger = setup_logging()

    logger.info("Loading model and tokenizer...")
    tokenizer, model = load_model_and_tokenizer(args.base_model_name, args.model_dir, logger)

    logger.info("\n" + "=" * 80)
    logger.info("Testing fine-tuned SFT model")
    logger.info("=" * 80)
    logger.info("Prompt: %s", args.prompt)
    logger.info("-" * 80)

    response = generate_response(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    logger.info("Response:")
    print(response)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
