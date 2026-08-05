"""
Fine-tune w2v-BERT 2.0 as speed-critical alternative.

Usage:
    python finetune_w2vbert.py --config configs/w2vbert_finetune_config.yaml
"""

import argparse
import yaml
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments
from datasets import load_dataset
from jiwer import wer

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = pred_logits.argmax(-1)
    label_ids = pred.label_ids

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(label_ids)

    wer_score = wer(label_str, pred_str)

    return {"wer": wer_score}

def main():
    parser = argparse.ArgumentParser(description="Fine-tune w2v-BERT")
    parser.add_argument("--config", required=True, help="Config YAML file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load model and processor
    model = Wav2Vec2ForCTC.from_pretrained("facebook/w2v-bert-2.0")
    processor = Wav2Vec2Processor.from_pretrained("facebook/w2v-bert-2.0")

    # Load dataset
    dataset = load_dataset("linagora/linto-dataset-audio-ar-tn-0.1-augmented")

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_values[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"])

    # Training arguments
    training_args = TrainingArguments(**config)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save model
    trainer.save_model()

if __name__ == "__main__":
    main()