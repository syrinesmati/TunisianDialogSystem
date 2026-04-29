"""Production-ready CPT training script for Aya Expanse 8B on Tunisian dialect.

Features:
- Self-contained environment setup (imports, config, dataset loading, model loading)
- Automatic checkpoint discovery and resume
- Frequent checkpointing with disk safety limits
- File + stdout logging for long-running remote jobs
- Works without VS Code or Jupyter state

Example usage:
    python llm/cpt/training/train.py
    python llm/cpt/training/train.py --output_dir ./outputs/checkpoints/aya-expanse-8b-cpt-tunisian
    python llm/cpt/training/train.py --force_restart

For long remote runs:
    tmux new -s cpt
    nohup python llm/cpt/training/train.py > train.out 2>&1 &
"""

from __future__ import annotations

import argparse
import inspect
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class CPTDefaults:
    model_name: str = "CohereLabs/aya-expanse-8b"
    dataset_name: str = "Syrinesmati/tunisian-dialect-corpus"
    dataset_split: str = "train"
    text_column: str = "text"
    max_seq_length: int = 1024
    eval_size: float = 0.01
    output_subdir: str = "aya-expanse-8b-cpt-tunisian"
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    warmup_ratio: float = 0.03
    logging_steps: int = 25
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 3
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    dataset_map_batch_size: int = 1000
    seed: int = 42


class CheckpointLoggerCallback(TrainerCallback):
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def on_train_begin(self, args, state, control, **kwargs):
        self.logger.info("Training started")

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        self.logger.info("Checkpoint saved: %s", checkpoint_dir)

    def on_train_end(self, args, state, control, **kwargs):
        self.logger.info("Training finished")


def parse_args() -> argparse.Namespace:
    defaults = CPTDefaults()
    repo_root = Path(__file__).resolve().parents[3]
    default_output_dir = repo_root / "outputs" / "checkpoints" / defaults.output_subdir

    parser = argparse.ArgumentParser(
        description="Continuous pretraining for Aya Expanse 8B on Tunisian dialect"
    )
    parser.add_argument("--model_name", default=defaults.model_name)
    parser.add_argument("--dataset_name", default=defaults.dataset_name)
    parser.add_argument("--dataset_split", default=defaults.dataset_split)
    parser.add_argument("--text_column", default=defaults.text_column)
    parser.add_argument("--max_seq_length", type=int, default=defaults.max_seq_length)
    parser.add_argument("--eval_size", type=float, default=defaults.eval_size)
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
    parser.add_argument("--warmup_ratio", type=float, default=defaults.warmup_ratio)
    parser.add_argument("--logging_steps", type=int, default=defaults.logging_steps)
    parser.add_argument("--save_steps", type=int, default=defaults.save_steps)
    parser.add_argument("--eval_steps", type=int, default=defaults.eval_steps)
    parser.add_argument("--save_total_limit", type=int, default=defaults.save_total_limit)
    parser.add_argument("--lora_r", type=int, default=defaults.lora_r)
    parser.add_argument("--lora_alpha", type=int, default=defaults.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=defaults.lora_dropout)
    parser.add_argument("--dataset_map_batch_size", type=int, default=defaults.dataset_map_batch_size)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--hf_token", default=None, help="Optional Hugging Face token")
    parser.add_argument("--force_restart", action="store_true", help="Ignore existing checkpoints")
    parser.add_argument("--use_4bit", action=argparse.BooleanOptionalAction, default=defaults.use_4bit)
    parser.add_argument(
        "--bnb_4bit_quant_type",
        default=defaults.bnb_4bit_quant_type,
    )
    parser.add_argument(
        "--bnb_4bit_compute_dtype",
        default=defaults.bnb_4bit_compute_dtype,
        choices=["bfloat16", "float16"],
    )
    parser.add_argument(
        "--bnb_4bit_use_double_quant",
        action=argparse.BooleanOptionalAction,
        default=defaults.bnb_4bit_use_double_quant,
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cpt_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
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


def get_compute_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float16


def build_model_and_tokenizer(
    model_name: str,
    use_4bit: bool,
    bnb_4bit_quant_type: str,
    bnb_4bit_compute_dtype: str,
    bnb_4bit_use_double_quant: bool,
    logger: logging.Logger,
):
    if use_4bit and not torch.cuda.is_available():
        logger.warning("CUDA is not available; disabling 4-bit quantization")
        use_4bit = False

    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_supported else torch.float16 if torch.cuda.is_available() else torch.float32
    quant_dtype = get_compute_dtype(bnb_4bit_compute_dtype)

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=quant_dtype,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        )

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype if not use_4bit else None,
        device_map="auto",
        trust_remote_code=True,
    )

    if hasattr(model, "config"):
        model.config.use_cache = False

    return model, tokenizer, compute_dtype, bnb_config, use_4bit


def load_and_prepare_dataset(
    dataset_name: str,
    dataset_split: str,
    text_column: str,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
    eval_size: float,
    map_batch_size: int,
    logger: logging.Logger,
):
    logger.info("Loading dataset: %s (%s)", dataset_name, dataset_split)
    dataset = load_dataset(dataset_name, split=dataset_split)
    logger.info("Raw dataset rows: %s", len(dataset))
    logger.info("Columns: %s", dataset.column_names)

    if text_column not in dataset.column_names:
        raise ValueError(
            f"Text column '{text_column}' not found. Available columns: {dataset.column_names}"
        )

    logger.info("Filtering empty rows in column '%s'", text_column)
    dataset = dataset.filter(lambda x: x[text_column] is not None and str(x[text_column]).strip() != "")
    logger.info("Rows after filtering: %s", len(dataset))

    def tokenize_fn(batch):
        return tokenizer(batch[text_column], truncation=False)

    logger.info("Tokenizing dataset")
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=map_batch_size,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples["input_ids"])
        total_length = (total_length // max_seq_length) * max_seq_length
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    logger.info("Packing text into fixed-length blocks of %s tokens", max_seq_length)
    lm_dataset = tokenized.map(
        group_texts,
        batched=True,
        batch_size=map_batch_size,
        desc="Packing into fixed blocks",
    )

    if len(lm_dataset) == 0:
        raise ValueError(
            "Packing produced an empty dataset. Reduce max_seq_length or provide more text."
        )

    logger.info("Packed dataset rows: %s", len(lm_dataset))
    split = lm_dataset.train_test_split(test_size=eval_size, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    logger.info("Train rows: %s", len(train_dataset))
    logger.info("Eval rows: %s", len(eval_dataset))
    return train_dataset, eval_dataset


def build_training_args(
    output_dir: Path,
    learning_rate: float,
    num_train_epochs: float,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    gradient_accumulation_steps: int,
    warmup_steps: int,
    logging_steps: int,
    save_steps: int,
    eval_steps: int,
    save_total_limit: int,
    use_4bit: bool,
):
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16_enabled = torch.cuda.is_available() and not bf16_supported

    kwargs = dict(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        save_strategy="steps",
        save_steps=save_steps,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        bf16=bf16_supported,
        fp16=fp16_enabled,
        gradient_checkpointing=True,
        report_to="none",
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        lr_scheduler_type="cosine",
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    signature = inspect.signature(TrainingArguments)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = "steps"

    if "logging_strategy" in signature.parameters:
        kwargs["logging_strategy"] = "steps"

    if "eval_steps" in signature.parameters:
        kwargs["eval_steps"] = eval_steps

    return TrainingArguments(**kwargs)


def build_lora_model(model, lora_r: int, lora_alpha: int, lora_dropout: float, use_4bit: bool, logger: logging.Logger):
    if use_4bit:
        logger.info("Preparing model for k-bit training")
        model = prepare_model_for_kbit_training(model)

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    logger.info("Attaching LoRA adapters")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    logger.info("Starting CPT training job")
    logger.info("Output directory: %s", output_dir)
    logger.info("Transformers version: %s", __import__("transformers").__version__)
    logger.info("PyTorch version: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    maybe_login_to_hub(args.hf_token, logger)

    model, tokenizer, compute_dtype, bnb_config, use_4bit = build_model_and_tokenizer(
        model_name=args.model_name,
        use_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        logger=logger,
    )

    train_dataset, eval_dataset = load_and_prepare_dataset(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        text_column=args.text_column,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        eval_size=args.eval_size,
        map_batch_size=args.dataset_map_batch_size,
        logger=logger,
    )

    model = build_lora_model(
        model=model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_4bit=use_4bit,
        logger=logger,
    )

    total_train_samples = len(train_dataset)
    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    total_steps = (total_train_samples / max(effective_batch_size, 1)) * args.num_train_epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    logger.info("Estimated total optimizer steps: %.2f", total_steps)
    logger.info("Warmup steps: %s", warmup_steps)

    training_args = build_training_args(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        use_4bit=use_4bit,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[CheckpointLoggerCallback(logger)],
    )

    last_checkpoint = detect_last_checkpoint(output_dir, args.force_restart, logger)
    if last_checkpoint:
        logger.info("Resuming training from checkpoint: %s", last_checkpoint)
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        logger.info("Starting training from scratch")
        trainer.train()

    logger.info("Saving final model and tokenizer")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Final model saved to %s", output_dir)
    logger.info("Training run completed successfully")


if __name__ == "__main__":
    main()
