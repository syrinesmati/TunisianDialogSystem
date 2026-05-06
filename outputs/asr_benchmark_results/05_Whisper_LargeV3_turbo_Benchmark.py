#!/usr/bin/env python3
"""Standalone runner for the Whisper Tunisian Arabic ASR benchmark.

This script mirrors the notebook logic so it can be launched from tmux or
another long-running server session without losing progress when SSH drops.

Outputs:
- per-split CSV files in the configured benchmark output directory
- summary CSV and WER/CER chart
- full stdout/stderr transcript in a log file alongside the outputs
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Whisper large-v3 turbo ASR benchmark as a plain Python script."
    )
    parser.add_argument(
        "--dataset-root",
        default="/home/ala/dataset",
        help="Path to the dataset root that contains config.yaml and the split folders.",
    )
    parser.add_argument(
        "--output-root",
        default="/home/ala/TunisianDialogSystem/outputs/asr_benchmark_results",
        help="Root directory for benchmark outputs.",
    )
    parser.add_argument(
        "--model-key",
        default="whisper_large_v3_turbo",
        help="Model key in config.yaml.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints if available.",
    )
    return parser.parse_args()


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def install_console_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    print(f"Logging to: {log_path}")


def main() -> int:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    assert dataset_root.exists(), f"Missing dataset root: {dataset_root}"

    # Make sure the shared benchmark package is importable.
    if str(dataset_root) not in sys.path:
        sys.path.insert(0, str(dataset_root))

    # Keep tokenizer parallelism quiet in long-running tmux sessions.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from benchmark_utils import (
        audio_inspector,
        display_bulk_predictions,
        display_preview,
        display_summary,
        display_worst,
        get_device,
        load_config,
        plot_wer_cer,
        print_gpu_info,
        run_labelled_splits,
        run_pipeline_inference,
        run_unlabelled_splits,
        setup_output_dir,
        split_benchmark,
    )
    import torch
    from datasets import DatasetDict, load_from_disk
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    cfg = load_config(str(dataset_root / "config.yaml"))

    # Override Colab-only paths for local execution.
    cfg["paths"]["dataset"] = str(dataset_root)
    cfg["paths"]["output_root"] = str(output_root)

    # Mirror notebook settings.
    target_sr = cfg["evaluation"]["target_sr"]
    top_n_worst = cfg["evaluation"]["top_n_worst"]
    preview_rows = cfg["evaluation"]["preview_rows"]

    # The model-specific output directory is created before logging redirection so
    # the log file can live alongside the benchmark outputs.
    benchmark_output_dir = Path(setup_output_dir(cfg, args.model_key))
    log_path = benchmark_output_dir / f"{args.model_key}_run.log"
    install_console_logging(log_path)

    print("Dataset path:", cfg["paths"]["dataset"])
    print("Output root :", cfg["paths"]["output_root"])
    print("Model key   :", args.model_key)
    print("Resume      :", args.resume)

    device = get_device()
    print_gpu_info()
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"torch_dtype : {torch_dtype}")

    mcfg = cfg["models"][args.model_key]
    model_id = mcfg["model_id"]
    batch_size = mcfg["batch_size"]
    output_dir = setup_output_dir(cfg, args.model_key)

    print(f"Loading {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    # Ensure forced_decoder_ids is cleared; the pipeline sets it via generate_kwargs.
    model.generation_config.forced_decoder_ids = None
    model.to(device)
    model.eval()

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        chunk_length_s=mcfg["chunk_length_s"],
        stride_length_s=mcfg["stride_length_s"],
        generate_kwargs={
            "language": mcfg["language"],
            "task": mcfg["task"],
        },
    )
    print(f"✓ {model_id} loaded on {device} ({torch_dtype})")

    # Build a DatasetDict from available on-disk split folders only.
    split_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    benchmark = DatasetDict()
    for split_dir in split_dirs:
        try:
            benchmark[split_dir.name] = load_from_disk(str(split_dir))
        except Exception:
            pass

    print(f"Loaded splits: {list(benchmark.keys())}")
    labelled_splits, unlabelled_splits = split_benchmark(benchmark)

    def infer_fn(ds):
        return run_pipeline_inference(ds, pipe, batch_size, target_sr)

    all_result_dfs, summary_rows = run_labelled_splits(
        benchmark,
        labelled_splits,
        infer_fn,
        output_dir,
        args.model_key,
        preview_rows,
        top_n_worst,
        resume=args.resume,
    )

    # Notebook-style inspection helpers.
    display_preview(all_result_dfs, preview_rows)
    display_worst(all_result_dfs, top_n_worst)

    unlabelled_result_dfs = run_unlabelled_splits(
        benchmark,
        unlabelled_splits,
        infer_fn,
        output_dir,
        args.model_key,
        resume=args.resume,
    )
    audio_inspector(
        benchmark,
        unlabelled_result_dfs,
        unlabelled_splits,
        target_sr=target_sr,
    )
    display_bulk_predictions(unlabelled_result_dfs)

    summary_df = display_summary(
        summary_rows,
        output_dir,
        args.model_key,
        mcfg["model_id"],
    )
    if summary_df is not None:
        plot_wer_cer(summary_df, output_dir, args.model_key, mcfg["model_id"])

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())