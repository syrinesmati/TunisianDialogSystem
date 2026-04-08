"""
QLoRA fine-tuning of aya-expanse-8b via peft + trl.

Usage:
    python finetune_aya_qlora.py --config configs/qlora_config.yaml
"""

import argparse
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Fine-tune aya-expanse-8b with QLoRA")
    parser.add_argument("--config", required=True, help="Config YAML file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # Load dataset
    dataset = load_dataset("json", data_files="data/processed/final.jsonl")

    def format_instruction(example):
        return f"Instruction: {example['instruction']}\nResponse: {example['response']}"

    dataset = dataset.map(lambda x: {"text": format_instruction(x)})

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        args=TrainingArguments(**config),
        data_collator=DataCollatorForCompletionOnlyLM("Response: ", tokenizer=tokenizer),
    )

    trainer.train()

    # Save adapter
    model.save_pretrained(config["output_dir"])

if __name__ == "__main__":
    main()