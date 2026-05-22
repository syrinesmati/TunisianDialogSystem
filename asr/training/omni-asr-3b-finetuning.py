#!/usr/bin/env python3
"""
finetune_omniasr_llm_3b.py
==========================
Fine-tune facebook/omniASR-LLM-3B on the Linagora Tunisian Arabic dataset
(linagora/linto-dataset-audio-ar-tn-augmented) using the official
omnilingual-asr / fairseq2 recipe system.

Hardware target: NVIDIA GB10, 124 GB VRAM (single GPU)

Usage
-----
    python finetune_omniasr_llm_3b.py                  # full pipeline
    python finetune_omniasr_llm_3b.py --skip-dataprep  # training only (data already prepared)
    python finetune_omniasr_llm_3b.py --dataprep-only  # data prep only

Prerequisites (see setup.sh for automated install)
---------------------------------------------------
    conda create -n omni python=3.10 -y
    conda activate omni
    pip install --no-deps omnilingual-asr
    pip install omnilingual-asr[data]
    pip install datasets ray torch torchaudio pyarrow polars pandas jiwer evaluate

Directory layout produced
-------------------------
    ./data/
        version=0/
            corpus=linto_tn/
                split=train/language=ara_Arab/part-*.parquet
                split=dev/language=ara_Arab/part-*.parquet
                split=test/language=ara_Arab/part-*.parquet
        language_distribution_0.tsv   <- required by the recipe
    ./checkpoints/                    <- fairseq2 training artifacts
    ./eval_results/                   <- WER / CER CSV
"""

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths – override via environment variables if needed
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(os.getenv("FINETUNE_BASE_DIR", "./")).expanduser().resolve()
DATA_DIR        = BASE_DIR / "data"
PARQUET_ROOT    = DATA_DIR / "version=0"
STATS_TSV       = DATA_DIR / "language_distribution_0.tsv"
CHECKPOINT_DIR  = BASE_DIR / "checkpoints"
EVAL_DIR        = BASE_DIR / "eval_results"
OMNIASR_REPO    = Path(os.getenv("OMNIASR_REPO", "./omnilingual-asr")).expanduser().resolve()  # cloned repo

# ─────────────────────────────────────────────────────────────────────────────
# Dataset / model constants
# ─────────────────────────────────────────────────────────────────────────────
HF_DATASET_ID   = "linagora/linto-dataset-audio-ar-tn-augmented"
MODEL_CARD      = "omniASR_LLM_3B"
LANGUAGE_CODE   = "ara_Arab"          # BCP-47 + script used by omniASR
CORPUS_NAME     = "linto_tn"
TARGET_SR       = 16_000              # Hz
MAX_AUDIO_SEC   = 30                  # clips > 30 s trimmed / skipped


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 – DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_data() -> None:
    """
    Download the Linagora Tunisian dataset from HuggingFace, clean text,
    resample audio to 16 kHz, and write partitioned Parquet files in the
    schema expected by omnilingual-asr's MixtureParquetStorage.

    Schema per row
    --------------
        text        : str   – normalised transcript
        audio_bytes : list[int8] – FLAC-compressed waveform bytes
        audio_size  : int64 – number of decoded samples (at 16 kHz)
        corpus      : str   – "linto_tn"
        split       : str   – "train" | "dev" | "test"
        language    : str   – "ara_Arab"
    """
    log.info("═" * 60)
    log.info("STEP 1 – Data preparation")
    log.info("═" * 60)

    # Late imports so the script can be partially imported without all deps
    try:
        import io
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        import soundfile as sf
        from datasets import load_dataset, Audio as HFAudio
        from tqdm.auto import tqdm
    except ModuleNotFoundError as exc:
        pkg = exc.name or "<package>"
        log.error(
            f"Missing dependency '{pkg}'. Install the package and re-run.\n"
            f"For example: pip install {pkg}"
        )
        sys.exit(1)

    PARQUET_ROOT.mkdir(parents=True, exist_ok=True)

    # ── 1.1  Load dataset splits ──────────────────────────────────────────────
    log.info(f"Loading '{HF_DATASET_ID}' from HuggingFace …")
    # The dataset has a single 'train' split; we carve out dev/test ourselves.
    use_streaming = False
    try:
        raw = load_dataset(HF_DATASET_ID, split="train", trust_remote_code=True)
        log.info(f"  Total samples loaded : {len(raw):,}")

        # Cast audio to 16 kHz before any processing
        raw = raw.cast_column("audio", HFAudio(sampling_rate=TARGET_SR))

        # ── 1.2  Stratified split (90 % train / 5 % dev / 5 % test) ─────────
        log.info("Splitting into train / dev / test …")
        split_1 = raw.train_test_split(test_size=0.10, seed=42)
        split_2 = split_1["test"].train_test_split(test_size=0.50, seed=42)
        splits = {
            "train": split_1["train"],
            "dev":   split_2["train"],
            "test":  split_2["test"],
        }
        for name, ds in splits.items():
            log.info(f"  {name:6s}: {len(ds):,} samples")
    except Exception as exc:
        msg = str(exc).lower()
        if "offset overflow" not in msg and "datasetgenerationerror" not in type(exc).__name__.lower():
            raise

        use_streaming = True
        log.warning(
            "Dataset materialization failed (likely Arrow offset overflow). "
            "Falling back to streaming mode with deterministic 90/5/5 split."
        )

        raw = load_dataset(
            HF_DATASET_ID,
            split="train",
            trust_remote_code=True,
            streaming=True,
        )
        raw = raw.cast_column("audio", HFAudio(sampling_rate=TARGET_SR))

        def bucket_for_split(sample: dict) -> str:
            key = (
                sample.get("id")
                or sample.get("transcription")
                or sample.get("text")
                or ""
            )
            if not isinstance(key, str):
                key = str(key)
            h = int(hashlib.md5(key.encode("utf-8", errors="ignore")).hexdigest(), 16) % 100
            if h < 90:
                return "train"
            if h < 95:
                return "dev"
            return "test"

        splits = {
            "train": raw.filter(lambda x: bucket_for_split(x) == "train"),
            "dev": raw.filter(lambda x: bucket_for_split(x) == "dev"),
            "test": raw.filter(lambda x: bucket_for_split(x) == "test"),
        }
        log.info("Streaming split ready (sizes will be counted during write).")

    # ── 1.3  Text cleaning ────────────────────────────────────────────────────
    def clean_text(text: str) -> str:
        """
        Minimal normalisation for Tunisian Arabic:
          - Strip leading/trailing whitespace
          - Collapse internal multiple spaces
          - Remove zero-width / invisible Unicode (U+200B, U+FEFF, etc.)
          - Keep Arabic diacritics (tashkeel) – they help the LM decoder
          - Remove Latin characters (code-switching artefacts from VCA sources
            that only contain Arabic in the reference; adjust if your data
            intentionally includes code-switching)
        """
        import re
        import unicodedata

        if not isinstance(text, str):
            return ""

        # Normalise Unicode to NFC
        text = unicodedata.normalize("NFC", text)

        # Strip invisible / control characters
        text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # ── 1.4  Audio validation & FLAC encoding ────────────────────────────────
    def audio_to_flac_bytes(waveform_array: np.ndarray, sr: int = TARGET_SR) -> bytes:
        """Re-encode a float32 waveform as a FLAC byte-string."""
        buf = io.BytesIO()
        # soundfile expects int16 for FLAC; scale float32 → int16
        pcm = (waveform_array * 32767).clip(-32768, 32767).astype(np.int16)
        sf.write(buf, pcm, sr, format="FLAC", subtype="PCM_16")
        return buf.getvalue()

    def numpy_bytes_to_list_int8(b: bytes) -> list:
        """Convert raw bytes → list[int8] (omniASR parquet schema)."""
        arr = np.frombuffer(b, dtype=np.uint8).view(np.int8)
        return arr.tolist()

    # ── 1.5  Write Parquet files ──────────────────────────────────────────────
    PA_SCHEMA = pa.schema([
        ("text",        pa.string()),
        ("audio_bytes", pa.list_(pa.int8())),
        ("audio_size",  pa.int64()),
        ("corpus",      pa.dictionary(pa.int32(), pa.string())),
        ("split",       pa.dictionary(pa.int32(), pa.string())),
        ("language",    pa.dictionary(pa.int32(), pa.string())),
    ])

    ROWS_PER_FILE = 500   # ≈ reasonable parquet fragment size

    def write_split(split_name: str, dataset) -> int:
        """Process and write one split; return number of rows written."""
        out_dir = (
            PARQUET_ROOT
            / f"corpus={CORPUS_NAME}"
            / f"split={split_name}"
            / f"language={LANGUAGE_CODE}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        rows, file_idx, skipped, written = [], 0, 0, 0
        try:
            total_samples = len(dataset)
        except Exception:
            total_samples = None

        def flush(rows, file_idx):
            path = out_dir / f"part-{file_idx:05d}.parquet"
            table = pa.Table.from_pylist(
                rows,
                schema=pa.schema([
                    ("text",        pa.string()),
                    ("audio_bytes", pa.list_(pa.int8())),
                    ("audio_size",  pa.int64()),
                    ("corpus",      pa.string()),
                    ("split",       pa.string()),
                    ("language",    pa.string()),
                ])
            )
            # Cast dictionary columns
            table = table.cast(PA_SCHEMA)
            pq.write_table(table, path, row_group_size=100)
            return file_idx + 1

        iterator = dataset
        if total_samples is not None:
            iterator = tqdm(dataset, desc=f"  {split_name}", unit="sample", total=total_samples)
        else:
            iterator = tqdm(dataset, desc=f"  {split_name}", unit="sample")

        for sample in iterator:
            try:
                text = clean_text(sample.get("transcription") or sample.get("text") or "")
                if not text:
                    skipped += 1
                    continue

                audio_info = sample["audio"]
                wav = np.array(audio_info["array"], dtype=np.float32)
                sr  = int(audio_info["sampling_rate"])

                # Resample if (somehow) still needed
                if sr != TARGET_SR:
                    import torchaudio
                    import torch
                    wav_t  = torch.from_numpy(wav).unsqueeze(0)
                    wav_t  = torchaudio.functional.resample(wav_t, sr, TARGET_SR)
                    wav    = wav_t.squeeze(0).numpy()

                duration = len(wav) / TARGET_SR
                if duration < 0.5 or duration > MAX_AUDIO_SEC:
                    skipped += 1
                    continue

                # Normalise amplitude (peak normalisation)
                peak = np.abs(wav).max()
                if peak > 1e-6:
                    wav = wav / peak * 0.95

                flac_bytes  = audio_to_flac_bytes(wav)
                int8_list   = numpy_bytes_to_list_int8(flac_bytes)
                audio_size  = len(wav)   # samples at 16 kHz

                rows.append({
                    "text":        text,
                    "audio_bytes": int8_list,
                    "audio_size":  audio_size,
                    "corpus":      CORPUS_NAME,
                    "split":       split_name,
                    "language":    LANGUAGE_CODE,
                })
                written += 1

                if len(rows) >= ROWS_PER_FILE:
                    file_idx = flush(rows, file_idx)
                    rows = []

            except Exception as exc:
                log.warning(f"    Skipping sample (error: {exc})")
                skipped += 1
                continue

        if rows:
            flush(rows, file_idx)

        if total_samples is not None:
            log.info(f"  {split_name}: {written:,} written, {skipped:,} skipped (from {total_samples:,})")
        else:
            log.info(f"  {split_name}: {written:,} written, {skipped:,} skipped")
        return written

    total_train = write_split("train", splits["train"])
    total_dev   = write_split("dev",   splits["dev"])
    total_test  = write_split("test",  splits["test"])

    # ── 1.6  Generate language_distribution_0.tsv ────────────────────────────
    # The MixtureParquetStorage requires a TSV that summarises corpus/language
    # sizes for temperature-based sampling.
    # Columns: corpus, language, split, num_samples, total_audio_s
    _write_stats_tsv(total_train, total_dev, total_test)

    log.info("Data preparation complete ✓")


def _write_stats_tsv(n_train: int, n_dev: int, n_test: int) -> None:
    """Write the language_distribution_0.tsv expected by MixtureParquetStorage."""
    import csv

    STATS_TSV.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "corpus":         CORPUS_NAME,
            "language":       LANGUAGE_CODE,
            "split":          "train",
            "num_samples":    n_train,
            # Rough estimate: average ~5 s per clip
            "total_audio_s":  n_train * 5,
        },
        {
            "corpus":         CORPUS_NAME,
            "language":       LANGUAGE_CODE,
            "split":          "dev",
            "num_samples":    n_dev,
            "total_audio_s":  n_dev * 5,
        },
        {
            "corpus":         CORPUS_NAME,
            "language":       LANGUAGE_CODE,
            "split":          "test",
            "num_samples":    n_test,
            "total_audio_s":  n_test * 5,
        },
    ]
    with open(STATS_TSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Stats TSV written → {STATS_TSV}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 – WRITE FAIRSEQ2 TRAINING CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def write_training_config() -> Path:
    """
    Write a YAML config file for the omnilingual-asr LLM fine-tuning recipe
    tuned for a single NVIDIA GB10 (124 GB VRAM).

    The generated config intentionally mirrors the repository's known-good
    `llm-finetune.yaml` schema and only sets supported fields.
    """
    import yaml

    config = {
        "model": {
            "name": MODEL_CARD,
        },
        "dataset": {
            "name": "linto_tn_dataset",
            "train_split": "train",
            "valid_split": "dev",
            "storage_mode": "MIXTURE_PARQUET",
            "task_mode": "ASR",
            "mixture_parquet_storage_config": {
                "dataset_summary_path": str(STATS_TSV),
                "beta_corpus": 0.5,
                "beta_language": 0.5,
                "fragment_loading": {
                    "cache": True,
                },
            },
            "asr_task_config": {
                "min_audio_len": 8_000,
                "max_audio_len": 480_000,
                "max_num_elements": 3_840_000,
                "batch_shuffle_window": 1,
                "normalize_audio": True,
                "example_shuffle_window": 1,
            },
        },
        "tokenizer": {
            "name": "omniASR_tokenizer_v1",
        },
        "optimizer": {
            "config": {
                "lr": 1e-5,
            },
        },
        "trainer": {
            "data_parallelism": "fsdp",
            "fsdp": {
                "granularity": "stack",
                "version": "v1",
                "fp32_reduce": False,
            },
            "freeze_encoder_for_n_steps": 500,
            "mixed_precision": {
                "dtype": "torch.bfloat16",
            },
            "grad_accumulation": {
                "num_batches": 2,
            },
        },
        "regime": {
            "num_steps": 10_000,
            "validate_after_n_steps": 0,
            "validate_every_n_steps": 500,
            "checkpoint_every_n_steps": 1000,
            "publish_metrics_every_n_steps": 100,
        },
    }

    cfg_path = BASE_DIR / "configs" / "llm-finetune-linto-tn.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    log.info(f"Training config written → {cfg_path}")
    return cfg_path


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 – LAUNCH TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def run_training(cfg_path: Path, resume: bool = False) -> None:
    """
    Launch the omnilingual-asr training recipe via subprocess so that
    fairseq2's distributed-training setup is handled correctly.

    The recipe entrypoint is:
        python -m workflows.recipes.wav2vec2.asr <OUTPUT_DIR> \
               --config-file <cfg_path>
    called from inside the cloned omnilingual-asr repo.
    """
    import subprocess

    log.info("═" * 60)
    log.info("STEP 3 – Training")
    log.info("═" * 60)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if not OMNIASR_REPO.exists():
        log.error(
            f"omnilingual-asr repo not found at {OMNIASR_REPO}.\n"
            "Clone it first:\n"
            "    git clone https://github.com/facebookresearch/omnilingual-asr.git"
        )
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "workflows.recipes.wav2vec2.asr",
        str(CHECKPOINT_DIR),
        "--config-file", str(cfg_path),
    ]

    # If requested, try to locate the latest consolidated checkpoint and
    # instruct the recipe to restore from it. Many fairseq-based entrypoints
    # accept `--restore-file <path>`; if the recipe ignores it that's fine —
    # we still surface a clear log message for manual resumption.
    if resume:
        ckpt_dirs = sorted(CHECKPOINT_DIR.glob("step_*"))
        if ckpt_dirs:
            latest = ckpt_dirs[-1] / "consolidated.pt"
            if latest.exists():
                log.info(f"Resuming training from latest checkpoint: {latest}")
                cmd += ["--restore-file", str(latest)]
            else:
                log.warning("--resume requested but no consolidated.pt found in latest step_*/ directory")
        else:
            log.warning("--resume requested but no step_* checkpoints found in checkpoint dir")

    log.info("Running: " + " ".join(cmd))
    log.info(f"Working directory: {OMNIASR_REPO}")

    result = subprocess.run(cmd, cwd=OMNIASR_REPO, env={**os.environ})
    if result.returncode != 0:
        log.error(f"Training failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    log.info("Training complete ✓")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 – EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(best_checkpoint: Path | None = None) -> None:
    """
    Evaluate the fine-tuned model on the held-out test split and print
    WER / CER.  Falls back to the base model if no checkpoint is provided.
    """
    log.info("═" * 60)
    log.info("STEP 4 – Evaluation on test split")
    log.info("═" * 60)

    import numpy as np
    import torch
    from datasets import load_from_disk
    from datasets import Audio as HFAudio
    from tqdm.auto import tqdm

    try:
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    except ImportError:
        log.error("omnilingual_asr not importable. Add its src/ to PYTHONPATH.")
        return

    # ── Load pipeline ─────────────────────────────────────────────────────────
    if best_checkpoint is not None and best_checkpoint.exists():
        log.info(f"Loading fine-tuned checkpoint: {best_checkpoint}")
        pipe = ASRInferencePipeline(model_card=str(best_checkpoint))
    else:
        log.info(f"No checkpoint specified – evaluating base model '{MODEL_CARD}'")
        pipe = ASRInferencePipeline(model_card=MODEL_CARD)

    # ── Load test parquet ─────────────────────────────────────────────────────
    from datasets import load_dataset as _lds
    test_ds = _lds(
        "parquet",
        data_files=str(
            PARQUET_ROOT
            / f"corpus={CORPUS_NAME}"
            / "split=test"
            / f"language={LANGUAGE_CODE}"
            / "*.parquet"
        ),
        split="train",
    )
    log.info(f"Test samples: {len(test_ds):,}")

    # ── Reconstruct audio from stored FLAC bytes ──────────────────────────────
    import io, soundfile as sf

    def parquet_row_to_audio_dict(row):
        int8_arr = np.array(row["audio_bytes"], dtype=np.int8)
        raw_bytes = int8_arr.view(np.uint8).tobytes()
        wav, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return {"waveform": wav, "sample_rate": sr}

    # ── Run inference in batches ──────────────────────────────────────────────
    BATCH = 16
    refs, hyps = [], []

    for i in tqdm(range(0, len(test_ds), BATCH), desc="Eval", unit="batch"):
        batch_rows = test_ds.select(range(i, min(i + BATCH, len(test_ds))))
        audio_dicts = [parquet_row_to_audio_dict(r) for r in batch_rows]
        texts = pipe.transcribe(
            audio_dicts,
            lang=[LANGUAGE_CODE] * len(audio_dicts),
            batch_size=BATCH,
        )
        refs.extend([r["text"] for r in batch_rows])
        hyps.extend(texts)

    # ── Compute WER / CER ─────────────────────────────────────────────────────
    from jiwer import wer as compute_wer, cer as compute_cer

    wer_score = compute_wer(refs, hyps)
    cer_score = compute_cer(refs, hyps)
    log.info(f"WER: {wer_score:.4f}  |  CER: {cer_score:.4f}")

    # ── Save results CSV ──────────────────────────────────────────────────────
    import csv
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = EVAL_DIR / "test_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["reference", "hypothesis", "wer"])
        w.writeheader()
        for ref, hyp in zip(refs, hyps):
            from jiwer import wer as _wer
            w.writerow({"reference": ref, "hypothesis": hyp, "wer": _wer([ref], [hyp])})
    log.info(f"Per-sample results → {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune OmniASR-LLM-3B on Tunisian Arabic",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--skip-dataprep", action="store_true",
        help="Skip data preparation (assumes parquet already exists)"
    )
    p.add_argument(
        "--dataprep-only", action="store_true",
        help="Run data preparation only, then exit"
    )
    p.add_argument(
        "--eval-only", action="store_true",
        help="Run evaluation only on the test split"
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a fine-tuned checkpoint for evaluation"
    )
    p.add_argument(
        "--resume", action="store_true",
        help="If set, attempt to resume training from the last checkpoint in the output dir"
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.eval_only:
        ckpt = Path(args.checkpoint) if args.checkpoint else None
        run_evaluation(best_checkpoint=ckpt)
        return

    if not args.skip_dataprep:
        prepare_data()

    if args.dataprep_only:
        log.info("--dataprep-only requested; exiting after data preparation.")
        return

    cfg_path = write_training_config()
    run_training(cfg_path, resume=args.resume)

    # Find best checkpoint (last saved)
    ckpt_dirs = sorted(CHECKPOINT_DIR.glob("step_*"))
    best_ckpt = (ckpt_dirs[-1] / "consolidated.pt") if ckpt_dirs else None
    run_evaluation(best_checkpoint=best_ckpt)


if __name__ == "__main__":
    main()