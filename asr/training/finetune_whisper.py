"""
Fine-tune Whisper Large v3 on LinTO corpus.

Usage:
    python finetune_whisper.py --config configs/whisper_finetune_config.yaml
"""

import argparse
import yaml
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Trainer, TrainingArguments, DataCollatorSpeechSeq2SeqWithPadding
from datasets import load_dataset
from jiwer import wer

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer_score = wer(label_str, pred_str)

    return {"wer": wer_score}

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper")
    parser.add_argument("--config", required=True, help="Config YAML file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load model and processor
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Load dataset
    dataset = load_dataset("linagora/linto-dataset-audio-ar-tn-0.1-augmented")

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"])

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Training arguments
    training_args = TrainingArguments(**config)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save model
    trainer.save_model()

if __name__ == "__main__":
    main()