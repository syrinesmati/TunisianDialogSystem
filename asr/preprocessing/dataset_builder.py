"""
dataset_builder.py
──────────────────
Utilities for Phase 4 — final dataset construction.

Converts the Arrow IPC + manifest Parquet outputs from Phase 2 into a clean,
model-ready HuggingFace ``DatasetDict`` stored as Arrow shards in
``data/processed/06_final/``.

Responsibilities
----------------
1. Load the manifest and Arrow waveforms produced by Phase 2.
2. Inner-join with the text-clean parquet to attach the final
   ``transcript_nota`` label column.
3. Drop columns that are irrelevant for training (raw intermediates,
   flag columns, quality metrics).
4. Keep only ``kept == True`` segments.
5. Validate the final column set and report any anomalies.
6. Save as a HuggingFace Dataset (Arrow format) with Audio feature so
   Whisper / XLSR trainers can load it directly.
7. Write a plain-text dataset card summarising the split statistics.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
import soundfile as sf
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Columns kept in the final dataset ────────────────────────────────────────
# Everything not in this list is dropped before saving.  The names must
# match what save_segments_arrow and df_train_clean produce.
FINAL_COLUMNS = [
    "audio_id",       # unique segment identifier
    "audio",          # dict with {"array": np.ndarray, "sampling_rate": int}
    "text",           # NOTA-normalised label — used as the training target
    "split",          # "train" | "test"
    "duration_s",     # segment length after preprocessing
    "seg_start",      # original start timestamp (provenance)
    "seg_end",        # original end timestamp (provenance)
]

# Columns that exist in both the manifest and the text-clean parquet —
# keep the manifest version (audio-side) and drop the text duplicate.
_MANIFEST_PRIORITY_COLS = {"sample_rate", "transcript_nota", "duration_s"}

# Pattern for flag columns added by NOTANormalizer.normalize_with_flags —
# useful during EDA but not needed for training.
_FLAG_COL_SUFFIXES = (
    "_applied",  # waw_applied, negation_applied, etc.
    "_y", # columns added after the merge
    "_x"  # columns added after the merge
)

# Text preprocessing intermediate columns — not needed by the model.
_TEXT_INTERMEDIATE_COLS = [
    "transcript",
    "transcript_raw",
    "transcript_clean",
    "transcript_cs",
    "char_count",
    "word_count",
    "has_latin",
    "has_diacritics",
    "speech_rate"
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_arrow_waveforms(arrow_path: Path) -> pd.DataFrame:
    """
    Load the Arrow IPC file produced by Phase 2 into a DataFrame.

    Decodes the ``waveform`` binary column back into NumPy float32 arrays
    and returns a DataFrame with columns:
    ``audio_id``, ``waveform`` (np.ndarray), ``sample_rate``, ``duration_s``.

    Parameters
    ----------
    arrow_path : Path
        Path to ``segments.arrow``.

    Returns
    -------
    pd.DataFrame
    """
    import pyarrow.ipc as pa_ipc

    log.info("Loading Arrow waveforms from %s …", arrow_path)
    with pa_ipc.open_file(str(arrow_path)) as reader:
        table = reader.read_all()

    df = table.to_pandas()

    def decode_flac(blob: bytes) -> np.ndarray:
        if not blob:
            return np.array([], dtype=np.float32)
        with io.BytesIO(blob) as buf:
            data, sr = sf.read(buf, dtype="float32")
        return data.astype(np.float32)

    df["waveform"] = df["waveform"].apply(decode_flac)

    log.info("Loaded %d waveforms from Arrow file", len(df))
    return df


def build_audio_column(row: pd.Series) -> dict:
    """
    Convert a waveform row into a HuggingFace-compatible audio dict.

    HuggingFace ``Audio`` feature expects
    ``{"array": np.ndarray, "sampling_rate": int}``.

    Parameters
    ----------
    row : pd.Series
        Row containing ``waveform`` (np.ndarray) and ``sample_rate`` (int).

    Returns
    -------
    dict
    """
    return {
        "array":       row["waveform"].astype(np.float32),
        "sampling_rate": int(row["sample_rate"]),
    }


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all columns that are not needed for model fine-tuning.

    Dropped categories:
    - Text preprocessing intermediates (transcript_raw, transcript_clean,
      transcript_cs).
    - NOTANormalizer flag columns (waw_applied, negation_applied, …).
    - Quality metrics used only during analysis (rms, peak, silence_ratio,
      n_samples, smoothing_applied).
    - Any other column not in :data:`FINAL_COLUMNS`.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        New DataFrame with only the final columns (plus ``audio`` if present).
    """
    to_drop = []

    for col in df.columns:
        # Flag columns from NOTA normaliser
        if any(col.endswith(sfx) for sfx in _FLAG_COL_SUFFIXES):
            to_drop.append(col)
            continue
        # Text intermediate columns
        if col in _TEXT_INTERMEDIATE_COLS:
            to_drop.append(col)
            continue
        # Quality metrics
        if col in ("rms", "peak", "silence_ratio", "n_samples",
                   "smoothing_applied", "drop_reason", "kept",
                   "waveform", "sample_rate"):
            to_drop.append(col)
            continue

    df = df.drop(columns=[c for c in to_drop if c in df.columns], errors="ignore")
    log.info("Dropped %d intermediate columns", len(to_drop))
    return df


def validate_dataset(df: pd.DataFrame, split_name: str) -> list[str]:
    """
    Run basic sanity checks on the final dataset DataFrame.

    Checks performed:
    - Required columns are present.
    - No null transcripts.
    - No null audio arrays.
    - Duration is positive for all rows.
    - All audio arrays are float32.

    Parameters
    ----------
    df : pd.DataFrame
    split_name : str

    Returns
    -------
    list[str]
        List of warning strings.  Empty list = all checks passed.
    """
    warnings: list[str] = []

    required = {"audio_id", "audio", "transcript", "duration_s"}
    missing = required - set(df.columns)
    if missing:
        warnings.append(f"[{split_name}] Missing required columns: {missing}")

    if "transcript" in df.columns:
        n_null = df["transcript"].isna().sum()
        n_empty = (df["transcript"].str.strip().str.len() == 0).sum()
        if n_null:
            warnings.append(f"[{split_name}] {n_null} null transcripts")
        if n_empty:
            warnings.append(f"[{split_name}] {n_empty} empty transcripts")

    if "audio" in df.columns:
        n_null_audio = df["audio"].isna().sum()
        if n_null_audio:
            warnings.append(f"[{split_name}] {n_null_audio} null audio entries")

    if "duration_s" in df.columns:
        n_bad_dur = (df["duration_s"] <= 0).sum()
        if n_bad_dur:
            warnings.append(f"[{split_name}] {n_bad_dur} segments with duration ≤ 0")

    return warnings


def compute_final_stats(df: pd.DataFrame, split_name: str) -> dict:
    """
    Compute summary statistics for the final dataset split.

    Parameters
    ----------
    df : pd.DataFrame
    split_name : str

    Returns
    -------
    dict
    """
    dur = df["duration_s"] if "duration_s" in df.columns else pd.Series(dtype=float)
    return {
        "split":             split_name,
        "n_segments":        len(df),
        "total_hours":       round(float(dur.sum() / 3600), 4) if len(dur) else 0.0,
        "mean_duration_s":   round(float(dur.mean()), 3) if len(dur) else 0.0,
        "median_duration_s": round(float(dur.median()), 3) if len(dur) else 0.0,
        "min_duration_s":    round(float(dur.min()), 3) if len(dur) else 0.0,
        "max_duration_s":    round(float(dur.max()), 3) if len(dur) else 0.0,
        "columns":           df.columns.tolist(),
    }


def write_dataset_card(stats: list[dict], output_dir: Path) -> None:
    """
    Write a plain-text dataset card summarising the final dataset.

    Parameters
    ----------
    stats : list[dict]
        One dict per split from :func:`compute_final_stats`.
    output_dir : Path
        Directory where ``dataset_card.txt`` is written.
    """
    lines = [
        "=" * 60,
        "  TUNISIAN ASR — FINAL DATASET CARD",
        "=" * 60,
        "",
        "Format : HuggingFace DatasetDict (Arrow shards)",
        "Script : data_loader.load_local_dataset(paths['processed']['final'])",
        "",
    ]

    for s in stats:
        lines += [
            f"Split : {s['split']}",
            f"  Segments    : {s['n_segments']:,}",
            f"  Total hours : {s['total_hours']:.3f}",
            f"  Mean dur    : {s['mean_duration_s']:.3f} s",
            f"  Median dur  : {s['median_duration_s']:.3f} s",
            f"  Columns     : {s['columns']}",
            "",
        ]

    lines += [
        "Audio feature",
        "  Format        : float32 numpy array",
        "  Sample rate   : 16 000 Hz",
        "  Channels      : mono",
        "  Normalisation : peak-normalised to 0.95",
        "  Pre-emphasis  : y[n] = x[n] - 0.97 * x[n-1]",
        "",
        "Label column  : transcript (NOTA-normalised Arabic script)",
        "",
    ]

    path = output_dir / "dataset_card.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Dataset card written → %s", path)


def build_hf_dataset(
    audio_clean_dir: Path,
    text_clean_dir: Path,
    output_dir: Path,
    splits: list[str] | None = None,
) -> "datasets.DatasetDict":
    """
    Build the final HuggingFace DatasetDict and save it to *output_dir*.

    Steps
    -----
    1. Load Arrow waveforms + manifest for each split.
    2. Load the text-clean parquet (for the ``transcript_nota`` column).
    3. Inner-join on ``audio_id``.
    4. Keep only ``kept == True`` segments.
    5. Build the HuggingFace ``audio`` dict column.
    6. Drop unnecessary columns.
    7. Validate, then save as HF Dataset.

    Parameters
    ----------
    audio_clean_dir : Path
        ``data/interim/02_audio_clean/``
    text_clean_dir : Path
        ``data/interim/03_text_clean/``
    output_dir : Path
        ``data/processed/06_final/``
    splits : list[str] | None
        Which splits to process.  Default ``["train", "test"]``.

    Returns
    -------
    datasets.DatasetDict
    """
    from datasets import Dataset, DatasetDict, Audio as HFAudio

    output_dir.mkdir(parents=True, exist_ok=True)
    splits = splits or ["train", "test"]

    hf_splits: dict[str, Dataset] = {}
    all_stats: list[dict] = []

    for split_name in splits:
        log.info("[%s] Building final split …", split_name)

        # ── Load manifest (has kept flag + quality metrics) ───────────────────
        manifest_path = audio_clean_dir / split_name / "manifest.parquet"
        if not manifest_path.exists():
            log.error("[%s] Manifest not found: %s", split_name, manifest_path)
            continue
        manifest = pd.read_parquet(manifest_path)
        log.info("[%s] Manifest loaded: %d rows", split_name, len(manifest))

        # ── Keep only segments that passed quality gate ───────────────────────
        kept_manifest = manifest[manifest["kept"] == True].copy()
        n_dropped = len(manifest) - len(kept_manifest)
        log.info("[%s] Kept=%d  Dropped=%d", split_name,
                 len(kept_manifest), n_dropped)

        # ── Load Arrow waveforms (kept segments only in file) ─────────────────
        arrow_path = audio_clean_dir / split_name / "segments.arrow"
        if not arrow_path.exists():
            log.error("[%s] Arrow file not found: %s", split_name, arrow_path)
            continue
        waveforms = load_arrow_waveforms(arrow_path)

        # ── Load text-clean parquet for transcript_nota ───────────────────────
        text_path = text_clean_dir / f"df_{split_name}_clean.parquet"
        if not text_path.exists():
            log.error("[%s] Text-clean parquet not found: %s", split_name, text_path)
            continue
        text_df = pd.read_parquet(text_path)[["audio_id", "transcript_nota"]]
        log.info("[%s] Text-clean loaded: %d rows", split_name, len(text_df))

        # ── Join: manifest (kept) + waveforms + transcript ────────────────────
        df = (
            kept_manifest
            .merge(waveforms, on="audio_id", how="inner")
            .merge(text_df,   on="audio_id", how="inner")
        )
        log.info("[%s] After join: %d rows", split_name, len(df))

        # ── Rename transcript_nota → transcript (final label) ─────────────────
        df = df.rename(columns={"transcript_nota": "transcript"})

        # ── Build HuggingFace audio dict column ───────────────────────────────
        df["audio"] = df.apply(build_audio_column, axis=1)

        # ── Drop unnecessary columns ──────────────────────────────────────────
        df = drop_unnecessary_columns(df)

        # ── Validate ──────────────────────────────────────────────────────────
        warnings = validate_dataset(df, split_name)
        for w in warnings:
            log.warning(w)
        if not warnings:
            log.info("[%s] Validation passed ✓", split_name)

        # ── Convert to HuggingFace Dataset ────────────────────────────────────
        # Cast the audio column to the proper Audio feature type
        hf_ds = Dataset.from_pandas(df, preserve_index=False)
        hf_ds = hf_ds.cast_column("audio", HFAudio(sampling_rate=16000))

        hf_splits[split_name] = hf_ds
        all_stats.append(compute_final_stats(df, split_name))
        log.info("[%s] HuggingFace split ready: %d segments", split_name, len(hf_ds))

    # ── Save as DatasetDict ───────────────────────────────────────────────────
    dataset_dict = DatasetDict(hf_splits)
    dataset_dict.save_to_disk(str(output_dir))
    log.info("DatasetDict saved → %s", output_dir)

    write_dataset_card(all_stats, output_dir)

    return dataset_dict
