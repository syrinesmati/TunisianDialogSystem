"""Supervised fine-tuning script for the Tunisian CPT model.

This script continues from the trained CPT adapter and SFT-finetunes it on the
`Syrinesmati/tunisian-question-response-dataset` dataset to improve:

- Tunisian Arabic vocabulary and style
- question/answer behavior
- responses that stay on-topic and match the needed length
- hallucination resistance through concise assistant-only training

Usage examples:
    python llm/cpt/training/sft_train_question_response.py
    python llm/cpt/training/sft_train_question_response.py \
        --output_dir ./outputs/checkpoints/aya-expanse-8b-tunisian-sft

For long runs:
    tmux new -s tunisian_sft
     python llm/cpt/training/sft_train_question_response.py > sft.out 2>&1 &
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTTrainer, SFTConfig
from transformers.trainer_utils import get_last_checkpoint


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


SYSTEM_PROMPT = (
    'أنت "التيجاني"، مساعد ذكاء اصطناعي تونسي 100%. '
    'جاوب بالتونسي الدارج فقط، وبالطول المناسب للسؤال: كان يلزم قصّر، وكان يلزم فسّر أكثر. '
    'ممنوع الهلوسة أو الخروج على الموضوع.'
)


@dataclass
class SFTDefaults:
    base_model_name: str = "CohereLabs/aya-expanse-8b"
    cpt_adapter_dir: str = "outputs/checkpoints/aya-expanse-8b-cpt-tunisian"
    dataset_name: str = "Syrinesmati/tunisian-question-response-dataset"
    train_split: str = "train"
    eval_split: str = "test"
    output_subdir: str = "aya-expanse-8b-tunisian-sft"
    learning_rate: float = 1e-5
    num_train_epochs: float = 2.0
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 3
    max_seq_length: int = 1024
    seed: int = 42
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01


def parse_args() -> argparse.Namespace:
    defaults = SFTDefaults()
    repo_root = Path(__file__).resolve().parents[3]
    default_output_dir = repo_root / "outputs" / "checkpoints" / defaults.output_subdir

    parser = argparse.ArgumentParser(
        description="SFT fine-tuning for the Tunisian CPT model on question-response data"
    )
    parser.add_argument("--base_model_name", default=defaults.base_model_name)
    parser.add_argument("--cpt_adapter_dir", default=str((repo_root / defaults.cpt_adapter_dir).resolve()))
    parser.add_argument("--dataset_name", default=defaults.dataset_name)
    parser.add_argument("--train_split", default=defaults.train_split)
    parser.add_argument("--eval_split", default=defaults.eval_split)
    parser.add_argument("--output_dir", default=str(default_output_dir))
    parser.add_argument("--learning_rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--num_train_epochs", type=float, default=defaults.num_train_epochs)
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=defaults.per_device_train_batch_size,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=defaults.per_device_eval_batch_size,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=defaults.gradient_accumulation_steps,
    )
    parser.add_argument("--logging_steps", type=int, default=defaults.logging_steps)
    parser.add_argument("--save_steps", type=int, default=defaults.save_steps)
    parser.add_argument("--eval_steps", type=int, default=defaults.eval_steps)
    parser.add_argument("--save_total_limit", type=int, default=defaults.save_total_limit)
    parser.add_argument("--max_seq_length", type=int, default=defaults.max_seq_length)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--hf_token", default=None, help="Optional Hugging Face token")
    parser.add_argument("--force_restart", action="store_true", help="Ignore existing checkpoints")
    parser.add_argument("--warmup_ratio", type=float, default=defaults.warmup_ratio)
    parser.add_argument("--weight_decay", type=float, default=defaults.weight_decay)
    return parser.parse_args()


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tunisian_sft")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(output_dir / "sft_train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def resolve_output_dir(output_dir: str) -> Path:
    path = Path(output_dir).expanduser()
    if not path.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        path = (repo_root / path).resolve()
    return path


def maybe_login_to_hub(hf_token: Optional[str], logger: logging.Logger) -> None:
    if not hf_token:
        logger.info("No Hugging Face token provided; skipping login")
        return

    try:
        from huggingface_hub import login

        login(token=hf_token)
        logger.info("Hugging Face login succeeded")
    except Exception as exc:
        raise RuntimeError(f"Hugging Face login failed: {exc}") from exc


def detect_last_checkpoint(output_dir: Path, force_restart: bool, logger: logging.Logger) -> Optional[str]:
    if force_restart:
        logger.info("force_restart enabled; checkpoint resume disabled")
        return None

    if not output_dir.exists():
        return None

    checkpoint = get_last_checkpoint(str(output_dir))
    if checkpoint:
        logger.info("Detected checkpoint: %s", checkpoint)
    else:
        logger.info("No checkpoint found in %s", output_dir)
    return checkpoint


# get_compute_dtype removed: BitsAndBytes quantization is not used for full bf16 training


def load_processor_and_model(
    base_model_name: str,
    cpt_adapter_dir: str,
    logger: logging.Logger,
):
    logger.info("Loading tokenizer from base model: %s", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # Ensure the tokenizer's chat template includes a generation marker compatible
    # with TRL's SFTTrainer. If the tokenizer exposes a `chat_template` string,
    # append a `{% generation %}` marker when missing. This makes
    # `tokenizer.apply_chat_template(..., add_generation_prompt=True)` insert
    # the generation prompt that SFTTrainer expects for assistant-only loss.
    try:
        if hasattr(tokenizer, "chat_template") and isinstance(tokenizer.chat_template, str):
            if "{% generation %}" not in tokenizer.chat_template:
                tokenizer.chat_template = tokenizer.chat_template + "\n{% generation %}"
                logger.info("Patched tokenizer.chat_template to include {% generation %} marker")
    except Exception as exc:
        logger.warning("Could not patch tokenizer chat template: %s", exc)

    logger.info("Loading base model: %s with full bf16 precision (no quantization)", base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False

    adapter_path = Path(cpt_adapter_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"CPT adapter directory not found: {adapter_path}. Update --cpt_adapter_dir to your CPT output."
        )

    logger.info("Loading CPT adapter from: %s", adapter_path)
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        is_trainable=True,
        local_files_only=True,
    )

    # Ensure gradients flow to PEFT adapter parameters when using gradient checkpointing
    try:
        model.enable_input_require_grads()
    except Exception:
        # Older PEFT versions may not have this helper; ignore failure but log
        logger.info("Could not call enable_input_require_grads() on PeftModel; continuing")

    return tokenizer, model


def detect_text_fields(dataset: Dataset) -> tuple[str, str]:
    columns = set(dataset.column_names)

    question_candidates = ["instruction", "question", "prompt", "input", "query"]
    answer_candidates = ["response", "answer", "output", "completion", "reply"]

    question_field = next((name for name in question_candidates if name in columns), None)
    answer_field = next((name for name in answer_candidates if name in columns), None)

    if not question_field or not answer_field:
        raise ValueError(
            f"Could not detect question/answer fields from columns: {dataset.column_names}. "
            "Expected something like instruction/response or question/answer."
        )

    return question_field, answer_field


def format_conversational_dataset(dataset: Dataset, tokenizer: AutoTokenizer, logger: logging.Logger) -> Dataset:
    question_field, answer_field = detect_text_fields(dataset)
    logger.info("Using dataset fields: question=%s, answer=%s", question_field, answer_field)

    def _to_text(example: dict) -> dict:
        question = str(example[question_field]).strip()
        answer = str(example[answer_field]).strip()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

        # Use tokenizer.apply_chat_template to render the full conversation
        # as text including the generation prompt marker for TRL.
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            return_dict=False,
        )

        return {"text": text}

    remove_columns = list(dataset.column_names)
    return dataset.map(_to_text, remove_columns=remove_columns)


def load_dataset_splits(dataset_name: str, train_split: str, eval_split: str, logger: logging.Logger) -> DatasetDict:
    logger.info("Loading dataset: %s", dataset_name)

    try:
        train_ds = load_dataset(dataset_name, split=train_split)
    except Exception as exc:
        raise RuntimeError(f"Failed to load train split '{train_split}': {exc}") from exc

    try:
        eval_ds = load_dataset(dataset_name, split=eval_split)
    except Exception:
        logger.info("Eval split '%s' not found; creating a split from train data", eval_split)
        split = train_ds.train_test_split(test_size=0.05, seed=42)
        return DatasetDict(train=split["train"], test=split["test"])

    return DatasetDict(train=train_ds, test=eval_ds)


def get_bf16_support() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def check_trl_version_compat(logger: logging.Logger) -> None:
    """Verify trl version supports assistant_only_loss."""
    try:
        from importlib.metadata import version
        trl_version = version("trl")
        major, minor = map(int, trl_version.split(".")[:2])
        if major == 0 and minor < 8:
            logger.warning(
                "trl version %s detected. assistant_only_loss may not work in versions < 0.8.0. "
                "If you see full-sequence loss in logs, consider using DataCollatorForCompletionOnlyLM instead.",
                trl_version,
            )
        else:
            logger.info("trl version %s: assistant_only_loss is fully supported", trl_version)
    except Exception as exc:
        logger.warning("Could not check trl version: %s", exc)


def build_trainer(
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    args: argparse.Namespace,
    output_dir: Path,
):
    use_bf16 = get_bf16_support()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    # Build TRL SFT config and trainer to enable assistant-only loss via the
    # tokenizer chat template (contains generation marker). SFTTrainer expects
    # a dataset text field produced by `format_conversational_dataset`.
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        report_to=[],
        remove_unused_columns=False,
    )

    # Construct SFTTrainer using only supported kwargs for the installed
    # trl version. Some releases changed the constructor signature and may
    # not accept `tokenizer` as a keyword; inspect the init and pass a
    # compatible set of kwargs.
    try:
        from inspect import signature

        init_params = set(signature(SFTTrainer.__init__).parameters.keys())
    except Exception:
        init_params = set()

    sft_kwargs = {}
    if "model" in init_params:
        sft_kwargs["model"] = model
    if "args" in init_params:
        sft_kwargs["args"] = sft_config
    # older/newer versions may name the dataset args differently
    if "train_dataset" in init_params:
        sft_kwargs["train_dataset"] = train_dataset
    elif "dataset" in init_params:
        sft_kwargs["dataset"] = train_dataset
    if "eval_dataset" in init_params:
        sft_kwargs["eval_dataset"] = eval_dataset
    if "tokenizer" in init_params:
        sft_kwargs["tokenizer"] = tokenizer
    if "dataset_text_field" in init_params:
        sft_kwargs["dataset_text_field"] = "text"

    # Fallback: if no recognizable keywords found, try positional call
    if not sft_kwargs:
        try:
            trainer = SFTTrainer(model, sft_config, train_dataset, eval_dataset, tokenizer, "text")
            return trainer
        except TypeError as exc:
            raise TypeError(
                "Could not construct SFTTrainer with detected signature; please check your trl version"
            ) from exc

    trainer = SFTTrainer(**sft_kwargs)
    return trainer


def generate_preview(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
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

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



def main() -> None:
    args = parse_args()
    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    output_dir = resolve_output_dir(args.output_dir)
    logger = setup_logging(output_dir)

    logger.info("Starting SFT fine-tuning job")
    logger.info("Output directory: %s", output_dir)
    logger.info("Base model: %s", args.base_model_name)
    logger.info("CPT adapter: %s", args.cpt_adapter_dir)
    logger.info("Dataset: %s", args.dataset_name)
    logger.info("Torch version: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    set_seed(args.seed)
    maybe_login_to_hub(args.hf_token, logger)
    check_trl_version_compat(logger)

    logger.info(
        "Loading configuration: batch_size=%d, accumulation=%d, warmup=%.2f, weight_decay=%.4f",
        args.per_device_train_batch_size,
        args.gradient_accumulation_steps,
        args.warmup_ratio,
        args.weight_decay,
    )
    tokenizer, model = load_processor_and_model(
        base_model_name=args.base_model_name,
        cpt_adapter_dir=args.cpt_adapter_dir,
        logger=logger,
    )

    dataset_splits = load_dataset_splits(args.dataset_name, args.train_split, args.eval_split, logger)
    train_dataset = format_conversational_dataset(dataset_splits["train"], tokenizer, logger)
    eval_dataset = format_conversational_dataset(dataset_splits["test"], tokenizer, logger)

    logger.info("Train rows: %d", len(train_dataset))
    logger.info("Eval rows: %d", len(eval_dataset))
    logger.info("Sample train example text: %s", train_dataset[0]["text"])

    checkpoint = detect_last_checkpoint(output_dir, args.force_restart, logger)

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
        output_dir=output_dir,
    )

    logger.info("Training started")
    trainer.train(resume_from_checkpoint=checkpoint)
    logger.info("Training finished")

    logger.info("Saving model and tokenizer to %s", output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save training metrics to JSON file
    try:
        import json
        metrics_file = output_dir / "training_metrics.json"
        metrics = {
            "log_history": trainer.state.log_history,
            "best_metric": trainer.state.best_metric,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "global_step": trainer.state.global_step,
            "num_train_epochs": trainer.state.num_train_epochs,
        }
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved training metrics to %s", metrics_file)
    except Exception as exc:
        logger.warning("Failed to save training metrics: %s", exc)

    preview_prompt = "عسلامة، شنوة تنصحني نعمل كي نكون تعبان وبرشة؟"
    logger.info("Running preview generation on a Tunisian prompt")
    preview = generate_preview(trainer.model, tokenizer, preview_prompt)
    logger.info("Preview prompt: %s", preview_prompt)
    logger.info("Preview output: %s", preview)


if __name__ == "__main__":
    main()