"""
audio_preprocessing.py
───────────────────────
Audio preprocessing utilities for Phase 2 of the Tunisian ASR pipeline.

Responsibilities
----------------
1. **Segment extraction** — slice raw audio bytes from a HuggingFace recording
   using DataFrame ``seg_start`` / ``seg_end`` timestamps and persist each
   segment as a compressed Arrow/Parquet binary column or as individual WAV files.
2. **Gap smoothing (DataFrame level)** — apply the same gap-smoothing logic as
   ``audit_audio.smooth_segments`` but operating on a flat segment DataFrame
   (adjusting ``seg_start`` / ``seg_end`` in-place rather than on raw dicts).
3. **Waveform transforms** — resample to 16 kHz, cast to mono, peak-normalise,
   apply a pre-emphasis filter, and trim leading/trailing silence.
4. **Quality gate** — discard segments that are too short, too long, or whose
   RMS energy falls below a noise floor threshold.
5. **Statistics & visualisation** — duration distributions, sample-rate
   breakdown, RMS histogram, and before/after smoothing comparisons.

Design notes
------------
- All numeric thresholds live in ``configs/audio_config.yaml`` (new file) so
  they are never hardcoded here.
- The extraction loop is deliberately sequential to keep memory usage flat;
  pass ``num_proc > 1`` only if your environment supports forked workers.
- Output storage uses an **Arrow IPC** file (``segments.arrow``) alongside a
  companion Parquet manifest (``manifest.parquet``).  The Arrow file stores the
  raw waveform array for each segment; the manifest stores all metadata plus a
  boolean ``kept`` flag.  This layout lets downstream training code memory-map
  the waveforms efficiently without loading the whole corpus into RAM.
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Iterator
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import pandas as pd

_log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# § 1  GAP SMOOTHING  (DataFrame level)
# ─────────────────────────────────────────────────────────────────────────────
# Compute gaps directly from seg_start / seg_end columns, grouped by recording

def df_gap_stats(df: pd.DataFrame, SMOOTH_CFG: dict, label: str = "") -> dict:
    """Compute inter-segment gap statistics from a flat segment DataFrame."""
    rec_id = df["audio_id"].str.rsplit("_", n=1).str[0]
    gaps = []
    for rid, grp in df.groupby(rec_id):
        sorted_grp = grp.sort_values("seg_start")
        ends   = sorted_grp["seg_end"].values[:-1]
        starts = sorted_grp["seg_start"].values[1:]
        gaps.extend((starts - ends).tolist())
    arr = np.array(gaps)
    if len(arr) == 0:
        return {}
    sm = SMOOTH_CFG
    return {
        "n_gaps":           len(arr),
        "mean_gap_s":       round(float(arr.mean()), 4),
        "median_gap_s":     round(float(np.median(arr)), 4),
        "pct_snap_cand":    round(float((arr < sm["small_gap"]).mean()), 4),
        "pct_silence":      round(float((arr > sm["large_gap"]).mean()), 4),
        "max_gap_s":        round(float(arr.max()), 4),
        "sum_silence_s":    round(float(arr[arr > 0].sum()), 2),
        "n_gaps_gt_60s":    int((arr > 60).sum()),
    }


def plot_gap_distribution(df: pd.DataFrame, SMOOTH_CFG: dict, DISP_CFG: dict, label: str, save_path: str = None):
    """Plot full and zoomed inter-segment gap distributions from a DataFrame."""
    rec_id = df["audio_id"].str.rsplit("_", n=1).str[0]
    gaps = []
    for rid, grp in df.groupby(rec_id):
        sorted_grp = grp.sort_values("seg_start")
        ends   = sorted_grp["seg_end"].values[:-1]
        starts = sorted_grp["seg_start"].values[1:]
        gaps.extend((starts - ends).tolist())

    zoom_cap = DISP_CFG["gap_zoom_cap_s"]
    sm = SMOOTH_CFG

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(gaps, bins=80, color="#4C72B0", edgecolor="white", linewidth=0.4)
    axes[0].axvline(sm["small_gap"], color="#E74C3C", lw=1.2, ls="--",
                    label=f"snap < {sm['small_gap']}s")
    axes[0].axvline(sm["large_gap"], color="#F39C12", lw=1.2, ls="--",
                    label=f"silence > {sm['large_gap']}s")
    axes[0].set_title(f"Inter-segment gaps — {label} (all)")
    axes[0].set_xlabel("Gap (s)"); axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=8)

    zoomed = [g for g in gaps if 0 <= g <= zoom_cap]
    axes[1].hist(zoomed, bins=60, color="#DD8452", edgecolor="white", linewidth=0.4)
    axes[1].axvline(sm["small_gap"], color="#E74C3C", lw=1.2, ls="--")
    axes[1].axvline(sm["large_gap"], color="#F39C12", lw=1.2, ls="--")
    axes[1].set_title(f"Inter-segment gaps — {label} (0–{zoom_cap}s zoom)")
    axes[1].set_xlabel("Gap (s)"); axes[1].set_ylabel("Count")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    print(f"  Gaps < {sm['small_gap']}s (snap)    : {sum(g < sm['small_gap'] for g in gaps):,}")
    print(f"  Gaps > {sm['large_gap']}s (silence)  : {sum(g > sm['large_gap'] for g in gaps):,}")


def apply_smoothing_to_df(
    df: pd.DataFrame,
    *,
    small_gap: float = 0.2,
    large_gap: float = 1.0,
    max_extension: float = 0.3,
    fill_ratio: float = 0.5,
) -> pd.DataFrame:
    """
    Apply gap-smoothing to a flat segment DataFrame.

    The function groups rows by recording (derived from ``audio_id`` by
    stripping the trailing ``_<seg_idx>`` suffix) and adjusts each segment's
    ``seg_end`` so that gaps between consecutive segments are closed or
    partially filled.

    Rules (same thresholds as ``audit_audio.smooth_segments``)
    ----------------------------------------------------------
    gap < small_gap              → **snap**: ``seg_end = next.seg_start``
    gap > large_gap              → **extend**: ``seg_end += max_extension``
    small_gap ≤ gap ≤ large_gap  → **fill**: ``seg_end += gap * fill_ratio``

    Parameters
    ----------
    df : pd.DataFrame
        Segment DataFrame with columns ``audio_id``, ``seg_start``, ``seg_end``.
        All other columns are preserved unchanged.
    small_gap, large_gap, max_extension, fill_ratio : float
        Smoothing thresholds — read from ``configs/audio_config.yaml``
        (``audio.smoothing`` section) by the caller and forwarded here.

    Returns
    -------
    pd.DataFrame
        A **copy** of *df* with ``seg_end`` (and ``seg_duration`` if present)
        updated, plus a new boolean column ``smoothing_applied`` that flags
        every row whose ``seg_end`` was changed.
    """
    df = df.copy()

    # Derive recording ID by stripping the trailing _<n> segment index.
    # audio_id format: "<recording_id>_<seg_idx>"  e.g. "foo_bar_3"
    df["_rec_id"] = df["audio_id"].str.rsplit("_", n=1).str[0]

    original_ends = df["seg_end"].copy()

    for rec_id, grp in df.groupby("_rec_id", sort=False):
        # Sort by start time within each recording
        idx = grp.sort_values("seg_start").index.tolist()
        for pos, cur_idx in enumerate(idx[:-1]):
            next_idx = idx[pos + 1]
            gap = df.at[next_idx, "seg_start"] - df.at[cur_idx, "seg_end"]
            if gap < small_gap:
                # Snap: extend end to next segment's start
                df.at[cur_idx, "seg_end"] = df.at[next_idx, "seg_start"]
            elif gap > large_gap:
                # Large silence: small fixed extension only
                df.at[cur_idx, "seg_end"] += max_extension
            else:
                # Medium gap: proportional fill
                df.at[cur_idx, "seg_end"] += gap * fill_ratio

    df["smoothing_applied"] = df["seg_end"] != original_ends

    # Update seg_duration if it already exists
    if "seg_duration" in df.columns:
        df["seg_duration"] = df["seg_end"] - df["seg_start"]

    df = df.drop(columns=["_rec_id"])
    return df


def smoothing_summary(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    """
    Compute a before/after smoothing statistics summary.

    Parameters
    ----------
    df_before : pd.DataFrame
        Original segment DataFrame (before smoothing).
    df_after : pd.DataFrame
        Smoothed segment DataFrame.

    Returns
    -------
    dict
        Keys: ``n_smoothed``, ``pct_smoothed``, ``total_duration_before_h``,
        ``total_duration_after_h``, ``duration_gain_s``.
    """
    dur_before = (df_before["seg_end"] - df_before["seg_start"]).sum()
    dur_after  = (df_after["seg_end"]  - df_after["seg_start"]).sum()
    n_smoothed = int(df_after.get("smoothing_applied", pd.Series(dtype=bool)).sum())
    n_total    = len(df_after)
    return {
        "n_smoothed":             n_smoothed,
        "pct_smoothed":           round(n_smoothed / max(n_total, 1) * 100, 2),
        "total_duration_before_h": round(dur_before / 3600, 4),
        "total_duration_after_h":  round(dur_after  / 3600, 4),
        "duration_gain_s":         round(dur_after - dur_before, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# § 2  WAVEFORM TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────

def decode_audio_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Decode raw audio bytes (any format readable by soundfile) into a NumPy array.

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio bytes from a HuggingFace ``Audio(decode=False)`` column.

    Returns
    -------
    tuple[np.ndarray, int]
        ``(waveform, sample_rate)`` where *waveform* is float32, shape
        ``(n_samples,)`` for mono or ``(n_channels, n_samples)`` for multi-channel.
    """
    waveform, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    return waveform, sr


def slice_segment(
    waveform: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
) -> np.ndarray:
    """
    Extract a temporal slice ``[start_s, end_s)`` from *waveform*.

    Clamps indices to valid bounds so out-of-range timestamps never raise.

    Parameters
    ----------
    waveform : np.ndarray
        Float32 waveform array, shape ``(n_samples,)`` or ``(n_channels, n_samples)``.
    sr : int
        Sample rate of *waveform*.
    start_s, end_s : float
        Start and end times in seconds.

    Returns
    -------
    np.ndarray
        Sliced waveform with the same number of dimensions as the input.
    """
    n_samples = waveform.shape[-1] if waveform.ndim > 1 else len(waveform)
    s = max(0, int(start_s * sr))
    e = min(n_samples, int(end_s * sr))
    if waveform.ndim > 1:
        return waveform[s:e]
    return waveform[s:e]


def to_mono(waveform: np.ndarray) -> np.ndarray:
    """
    Convert a multi-channel waveform to mono by averaging channels.

    Parameters
    ----------
    waveform : np.ndarray
        Shape ``(n_samples,)`` (already mono) or ``(n_channels, n_samples)``.

    Returns
    -------
    np.ndarray
        1-D float32 array, shape ``(n_samples,)``.
    """
    if waveform.ndim == 1:
        return waveform
    # Average across channel axis (axis 0 for (n_ch, n_samp))
    return waveform.mean(axis=1).astype(np.float32)


def resample_audio(
    waveform: np.ndarray,
    orig_sr: int,
    target_sr: int = 16000,
) -> np.ndarray:
    """
    Resample *waveform* from *orig_sr* to *target_sr*.

    Uses ``librosa.resample`` with the ``soxr_hq`` backend when available
    (highest quality, installed via the ``soxr`` package).  Falls back to the
    default ``kaiser_best`` if soxr is not installed.

    Parameters
    ----------
    waveform : np.ndarray
        1-D float32 mono waveform.
    orig_sr : int
        Original sample rate (Hz).
    target_sr : int
        Target sample rate (Hz).  Default 16 000 Hz.

    Returns
    -------
    np.ndarray
        Resampled float32 waveform.
    """
    if orig_sr == target_sr:
        return waveform

    import librosa  # lazy import — not needed if audio already at target_sr
    try:
        return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr,
                                res_type="soxr_hq").astype(np.float32)
    except Exception:
        # soxr not available — fall back to kaiser_best
        return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr,
                                res_type="kaiser_best").astype(np.float32)


def peak_normalize(waveform: np.ndarray, target_level: float = 0.95) -> np.ndarray:
    """
    Peak-normalise *waveform* so the maximum absolute value equals *target_level*.

    Parameters
    ----------
    waveform : np.ndarray
        Float32 waveform.
    target_level : float
        Target peak amplitude (0–1).  Default 0.95 leaves a small headroom.

    Returns
    -------
    np.ndarray
        Normalised float32 waveform.  Returns *waveform* unchanged if the
        peak is zero (silent segment).
    """
    peak = np.max(np.abs(waveform))
    if peak == 0.0:
        return waveform
    return (waveform * (target_level / peak)).astype(np.float32)


def pre_emphasis(waveform: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """
    Apply a first-order pre-emphasis filter to *waveform*.

    Pre-emphasis amplifies high frequencies and partially compensates for the
    natural spectral roll-off of speech.  It is standard practice before
    computing mel-spectrogram features for ASR models.

    y[n] = x[n] - coef * x[n-1]

    Parameters
    ----------
    waveform : np.ndarray
        1-D float32 mono waveform.
    coef : float
        Pre-emphasis coefficient (typically 0.95–0.97).

    Returns
    -------
    np.ndarray
        Pre-emphasised float32 waveform.
    """
    return np.append(waveform[0], waveform[1:] - coef * waveform[:-1]).astype(np.float32)


def trim_silence(
    waveform: np.ndarray,
    sr: int,
    top_db: float = 30.0,
    frame_length: int = 512,
    hop_length: int = 128,
) -> np.ndarray:
    """
    Trim leading and trailing silence from *waveform* using librosa.

    Parameters
    ----------
    waveform : np.ndarray
        1-D float32 mono waveform.
    sr : int
        Sample rate (unused by librosa but kept for API consistency).
    top_db : float
        Threshold (dB below peak) below which frames are considered silent.
        Lower values = more aggressive trimming.  Default 30 dB.
    frame_length, hop_length : int
        STFT parameters for the energy computation.

    Returns
    -------
    np.ndarray
        Trimmed float32 waveform.  Returns the original if trimming would
        leave fewer than 100 samples.
    """
    import librosa
    trimmed, _ = librosa.effects.trim(
        waveform, top_db=top_db,
        frame_length=frame_length, hop_length=hop_length,
    )
    if len(trimmed) < 100:
        return waveform
    return trimmed.astype(np.float32)


def full_preprocess(
    waveform: np.ndarray,
    orig_sr: int,
    *,
    target_sr: int = 16000,
    do_mono: bool = True,
    do_resample: bool = True,
    do_trim: bool = True,
    do_normalize: bool = True,
    do_pre_emphasis: bool = True,
    normalize_level: float = 0.95,
    pre_emphasis_coef: float = 0.97,
    trim_top_db: float = 30.0,
) -> tuple[np.ndarray, int]:
    """
    Apply the complete preprocessing chain to a single waveform.

    Steps (in order, each individually toggleable)
    -----------------------------------------------
    1. ``do_mono``          — average channels → 1-D
    2. ``do_resample``      — resample to ``target_sr``
    3. ``do_trim``          — trim leading/trailing silence
    4. ``do_normalize``     — peak-normalise
    5. ``do_pre_emphasis``  — first-order high-frequency boost

    Parameters
    ----------
    waveform : np.ndarray
        Input waveform (any number of channels, any sample rate).
    orig_sr : int
        Sample rate of *waveform*.
    target_sr : int
        Desired output sample rate.  Default 16 000 Hz.
    do_mono, do_resample, do_trim, do_normalize, do_pre_emphasis : bool
        Toggle individual processing steps.
    normalize_level : float
        Forwarded to :func:`peak_normalize`.
    pre_emphasis_coef : float
        Forwarded to :func:`pre_emphasis`.
    trim_top_db : float
        Forwarded to :func:`trim_silence`.

    Returns
    -------
    tuple[np.ndarray, int]
        ``(processed_waveform, output_sr)`` where *output_sr* equals
        *target_sr* if ``do_resample`` is ``True``, else *orig_sr*.
    """
    out_sr = orig_sr

    if do_mono:
        waveform = to_mono(waveform)

    if do_resample and orig_sr != target_sr:
        waveform = resample_audio(waveform, orig_sr, target_sr)
        out_sr = target_sr

    if do_trim:
        waveform = trim_silence(waveform, out_sr, top_db=trim_top_db)

    if do_normalize:
        waveform = peak_normalize(waveform, target_level=normalize_level)

    if do_pre_emphasis:
        waveform = pre_emphasis(waveform, coef=pre_emphasis_coef)

    return waveform, out_sr


# ─────────────────────────────────────────────────────────────────────────────
# § 3  QUALITY GATE
# ─────────────────────────────────────────────────────────────────────────────

def compute_segment_quality(
    waveform: np.ndarray,
    sr: int,
) -> dict:
    """
    Compute quality metrics for a single preprocessed segment waveform.

    Metrics
    -------
    ``duration_s``     : duration in seconds
    ``rms``            : root-mean-square energy
    ``peak``           : peak absolute amplitude
    ``silence_ratio``  : fraction of samples < 0.01 absolute amplitude
    ``n_samples``      : number of samples

    Parameters
    ----------
    waveform : np.ndarray
        1-D float32 mono waveform at the final sample rate.
    sr : int
        Sample rate of *waveform*.

    Returns
    -------
    dict
    """
    n = len(waveform)
    rms = float(np.sqrt(np.mean(waveform ** 2))) if n > 0 else 0.0
    peak = float(np.max(np.abs(waveform))) if n > 0 else 0.0
    silence_ratio = float(np.sum(np.abs(waveform) < 0.01) / max(n, 1))
    return {
        "duration_s":    round(n / sr, 4) if sr > 0 else 0.0,
        "rms":           round(rms, 6),
        "peak":          round(peak, 6),
        "silence_ratio": round(silence_ratio, 4),
        "n_samples":     n,
    }


def passes_quality_gate(
    metrics: dict,
    *,
    min_duration_s: float = 0.5,
    max_duration_s: float = 20.0,
    min_rms: float = 0.001,
    max_silence_ratio: float = 0.90,
) -> tuple[bool, str]:
    """
    Decide whether a segment meets quality requirements for ASR training.

    Parameters
    ----------
    metrics : dict
        Output of :func:`compute_segment_quality`.
    min_duration_s : float
        Minimum segment length in seconds.  Shorter → too short.
    max_duration_s : float
        Maximum segment length in seconds.  Longer → too long (likely not
        properly segmented).
    min_rms : float
        Minimum RMS energy.  Lower → near-silent segment.
    max_silence_ratio : float
        Maximum fraction of near-zero samples.  Higher → too much silence.

    Returns
    -------
    tuple[bool, str]
        ``(passed, reason)`` where *reason* is ``"ok"`` or a short string
        describing the failure cause.
    """
    dur = metrics["duration_s"]
    if dur < min_duration_s:
        return False, "too_short"
    if dur > max_duration_s:
        return False, "too_long"
    if metrics["rms"] < min_rms:
        return False, "too_silent"
    if metrics["silence_ratio"] > max_silence_ratio:
        return False, "mostly_silent"
    return True, "ok"


# ─────────────────────────────────────────────────────────────────────────────
# § 4  SEGMENT EXTRACTION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def outputs_exist(output_dir: Path, split: str) -> tuple[bool, Path, Path]:
    """ Checks if the target files already exists"""
    output_dir = Path(output_dir)
    split_dir = output_dir / split
    arrow_path = split_dir / "segments.arrow"
    manifest_path = split_dir / "manifest.parquet"

    exists = arrow_path.exists() and manifest_path.exists()
    return exists, arrow_path, manifest_path


def iter_segments(
    hf_dataset,
    df: pd.DataFrame,
    *,
    preprocess_kwargs: dict | None = None,
    gate_kwargs: dict | None = None,
    split_name: str = "train",
) -> Iterator[dict]:
    """
    Iterate over every segment in *df*, decode and preprocess the audio, and
    yield a rich result dict for each segment.

    The function loads each **recording** once (not once per segment), which
    is critical for performance when a recording contains many segments.

    Parameters
    ----------
    hf_dataset : datasets.Dataset
        HuggingFace Dataset cast with ``Audio(decode=False)``.  Must contain
        an ``audio_id`` column that matches ``df["audio_id"]`` prefixes.
    df : pd.DataFrame
        Segment DataFrame (possibly smoothed).  Must have columns:
        ``audio_id``, ``seg_start``, ``seg_end``, plus any text columns.
    preprocess_kwargs : dict | None
        Forwarded as keyword arguments to :func:`full_preprocess`.  ``None``
        uses all defaults (16 kHz, mono, trim, normalise, pre-emphasis).
    gate_kwargs : dict | None
        Forwarded to :func:`passes_quality_gate`.  ``None`` uses defaults.
    split_name : str
        ``"train"`` or ``"test"`` — embedded in the yielded dict.

    Yields
    ------
    dict
        Keys:
        ``audio_id``, ``split``, ``seg_start``, ``seg_end``,
        ``waveform`` (np.ndarray float32), ``sample_rate`` (int),
        ``duration_s``, ``rms``, ``peak``, ``silence_ratio``,
        ``kept`` (bool), ``drop_reason`` (str),
        ``smoothing_applied`` (bool),
        plus any text columns present in *df*.
    """
    pp_kw   = preprocess_kwargs or {}
    gate_kw = gate_kwargs or {}

    # Build a fast lookup: recording_id → HF dataset index
    _log.info("[%s] Building recording-ID index …", split_name)
    rec_id_to_hf_idx: dict[str, int] = {}
    for hf_idx in range(len(hf_dataset)):
        rec_id = hf_dataset[hf_idx]["audio_id"]
        rec_id_to_hf_idx[rec_id] = hf_idx

    # Group segments by recording to load each recording once
    df = df.copy()
    df["_rec_id"] = df["audio_id"].str.rsplit("_", n=1).str[0]

    # Text columns to carry through (exclude audio-specific derived cols)
    _skip = {"audio_id", "seg_start", "seg_end", "seg_duration",
             "smoothing_applied", "_rec_id"}
    text_cols = [c for c in df.columns if c not in _skip]

    n_total   = len(df)
    n_kept    = 0
    n_dropped = 0

    try:
        from tqdm.auto import tqdm
        _tqdm = tqdm
    except ImportError:
        _tqdm = lambda x, **kw: x  # noqa: E731

    for rec_id, grp in _tqdm(
        df.groupby("_rec_id", sort=False),
        desc=f"Extracting [{split_name}]",
        total=df["_rec_id"].nunique(),
    ):
        # Load this recording's audio once
        hf_idx = rec_id_to_hf_idx.get(rec_id)
        if hf_idx is None:
            _log.warning("Recording '%s' not found in HF dataset — skipping.", rec_id)
            for _, row in grp.iterrows():
                yield {**{c: row.get(c) for c in text_cols},
                       "audio_id": row["audio_id"], "split": split_name,
                       "seg_start": row["seg_start"], "seg_end": row["seg_end"],
                       "waveform": np.array([], dtype=np.float32),
                       "sample_rate": 0, "duration_s": 0.0,
                       "rms": 0.0, "peak": 0.0, "silence_ratio": 1.0,
                       "kept": False, "drop_reason": "recording_not_found",
                       "smoothing_applied": bool(row.get("smoothing_applied", False))}
            continue

        audio_bytes = hf_dataset[hf_idx]["audio"]["bytes"]
        if not audio_bytes:
            _log.warning("Recording '%s' has empty audio bytes — skipping.", rec_id)
            continue

        try:
            recording_waveform, orig_sr = decode_audio_bytes(audio_bytes)
            recording_waveform = to_mono(recording_waveform)  # ← NEW LINE
        except Exception as exc:
            _log.error("Failed to decode audio for '%s': %s", rec_id, exc)
            continue

        for _, row in grp.sort_values("seg_start").iterrows():
            seg_wave = slice_segment(
                recording_waveform, orig_sr,
                float(row["seg_start"]), float(row["seg_end"]),
            )

            # Skip truly empty slices (e.g. timestamps beyond recording end)
            if seg_wave.size == 0:
                yield {**{c: row.get(c) for c in text_cols},
                       "audio_id": row["audio_id"], "split": split_name,
                       "seg_start": row["seg_start"], "seg_end": row["seg_end"],
                       "waveform": seg_wave, "sample_rate": 0,
                       "duration_s": 0.0, "rms": 0.0, "peak": 0.0,
                       "silence_ratio": 1.0, "kept": False,
                       "drop_reason": "empty_slice",
                       "smoothing_applied": bool(row.get("smoothing_applied", False))}
                n_dropped += 1
                continue

            try:
                proc_wave, out_sr = full_preprocess(seg_wave, orig_sr, **pp_kw)
            except Exception as exc:
                _log.warning("Preprocessing failed for '%s': %s", row["audio_id"], exc)
                proc_wave, out_sr = seg_wave, orig_sr  # keep raw on error

            metrics = compute_segment_quality(proc_wave, out_sr)
            passed, reason = passes_quality_gate(metrics, **gate_kw)

            if passed:
                n_kept += 1
            else:
                n_dropped += 1

            yield {
                **{c: row.get(c) for c in text_cols},
                "audio_id":         row["audio_id"],
                "split":            split_name,
                "seg_start":        float(row["seg_start"]),
                "seg_end":          float(row["seg_end"]),
                "waveform":         proc_wave,
                "sample_rate":      out_sr,
                "duration_s":       metrics["duration_s"],
                "rms":              metrics["rms"],
                "peak":             metrics["peak"],
                "silence_ratio":    metrics["silence_ratio"],
                "kept":             passed,
                "drop_reason":      reason,
                "smoothing_applied": bool(row.get("smoothing_applied", False)),
            }

    _log.info("[%s] Extraction complete — kept=%d  dropped=%d  total=%d",
              split_name, n_kept, n_dropped, n_total)


# ─────────────────────────────────────────────────────────────────────────────
# § 5  STORAGE — Arrow IPC + Parquet manifest
# ─────────────────────────────────────────────────────────────────────────────
def is_valid_audio(w, sr):
    if w is None:
        return False
    if sr is None or sr <= 0:
        return False
    if len(w) == 0:
        return False
    if w.ndim > 1 and w.shape[1] == 0:
        return False
    return True


def save_segments_arrow(
    segments: list[dict],
    output_dir: Path,
    split_name: str,
    *,
    kept_only: bool = True,
) -> tuple[Path, Path]:
    import pyarrow as pa
    import pyarrow.ipc as pa_ipc
    import io
    import soundfile as sf

    split_dir = Path(output_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    arrow_path    = split_dir / "segments.arrow"
    manifest_path = split_dir / "manifest.parquet"

    # ── Build manifest (unchanged) ────────────────────────────────────────────
    manifest_rows = []
    for seg in segments:
        row = {k: v for k, v in seg.items() if k != "waveform"}
        manifest_rows.append(row)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_parquet(manifest_path, index=False)
    _log.info("[%s] Manifest saved → %s  (%d rows)",
              split_name, manifest_path, len(manifest_df))

    # ── Helper: compress waveform to FLAC ─────────────────────────────────────
    def compress_waveform(waveform: np.ndarray, sr: int) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, waveform, sr, format="FLAC")
        return buf.getvalue()

    # ── Build Arrow IPC (compressed waveforms) ────────────────────────────────
    to_write = [
        s for s in segments
        if (s["kept"] if kept_only else True)
           and is_valid_audio(s["waveform"], s["sample_rate"])
    ]


    audio_ids = pa.array([s["audio_id"] for s in to_write], type=pa.string())

    waveforms = pa.array(
        [compress_waveform(s["waveform"], s["sample_rate"]) for s in to_write],
        type=pa.large_binary(),
    )

    sample_rates = pa.array([s["sample_rate"] for s in to_write], type=pa.int32())
    durations    = pa.array([s["duration_s"]  for s in to_write], type=pa.float32())

    table = pa.table({
        "audio_id":    audio_ids,
        "waveform":    waveforms,
        "sample_rate": sample_rates,
        "duration_s":  durations,
    })

    with pa_ipc.new_file(str(arrow_path), table.schema) as writer:
        writer.write_table(table)

    _log.info("[%s] Arrow IPC saved → %s  (%d segments, kept_only=%s)",
              split_name, arrow_path, len(to_write), kept_only)

    return arrow_path, manifest_path


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """
    Load a manifest Parquet file produced by :func:`save_segments_arrow`.

    Parameters
    ----------
    manifest_path : Path
        Path to the ``manifest.parquet`` file.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_parquet(manifest_path)


# ─────────────────────────────────────────────────────────────────────────────
# § 6  STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_corpus_stats(manifest_df: pd.DataFrame) -> dict:
    """
    Compute corpus-level statistics from a manifest DataFrame.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        Manifest produced by :func:`save_segments_arrow`.

    Returns
    -------
    dict
        Keys: ``n_total``, ``n_kept``, ``n_dropped``, ``pct_kept``,
        ``total_hours_kept``, ``mean_duration_s``, ``median_duration_s``,
        ``std_duration_s``, ``min_duration_s``, ``max_duration_s``,
        ``mean_rms``, ``drop_reasons`` (dict).
    """
    kept = manifest_df[manifest_df["kept"] == True]
    n_total  = len(manifest_df)
    n_kept   = len(kept)
    n_dropped = n_total - n_kept
    dur = kept["duration_s"] if "duration_s" in kept.columns else pd.Series(dtype=float)
    drop_reasons = (
        manifest_df[manifest_df["kept"] == False]["drop_reason"]
        .value_counts().to_dict()
        if "drop_reason" in manifest_df.columns else {}
    )
    return {
        "n_total":            n_total,
        "n_kept":             n_kept,
        "n_dropped":          n_dropped,
        "pct_kept":           round(n_kept / max(n_total, 1) * 100, 2),
        "total_hours_kept":   round(dur.sum() / 3600, 4) if len(dur) else 0.0,
        "mean_duration_s":    round(float(dur.mean()), 3) if len(dur) else 0.0,
        "median_duration_s":  round(float(dur.median()), 3) if len(dur) else 0.0,
        "std_duration_s":     round(float(dur.std()), 3) if len(dur) else 0.0,
        "min_duration_s":     round(float(dur.min()), 3) if len(dur) else 0.0,
        "max_duration_s":     round(float(dur.max()), 3) if len(dur) else 0.0,
        "mean_rms":           round(float(kept["rms"].mean()), 6) if "rms" in kept.columns and len(kept) else 0.0,
        "drop_reasons":       drop_reasons,
    }
