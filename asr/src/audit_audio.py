"""
audit_audio.py
──────────────
Audio-level audit utilities for the Tunisian ASR dataset.

Two usage modes
---------------
1. **HuggingFace dataset level** – functions that accept a HF ``Dataset``
   object (cast with ``Audio(decode=False)`` to avoid decoding gigabytes).
2. **Segment / DataFrame level** – functions that operate on the exploded
   segment DataFrame produced by :func:`build_segment_df`.

All numeric thresholds (gap boundaries, plot caps, sample counts) are read
from ``configs/audit_config.yaml`` via the ``audio`` section so that they
never need to be changed inside source files.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Sequence

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


# ── Low-level audio helpers ───────────────────────────────────────────────────

def load_audio_raw(sample: dict) -> tuple[np.ndarray, int, tuple]:
    """
    Decode a single HF sample's audio bytes into a NumPy array.

    Parameters
    ----------
    sample : dict
        A row from a HF dataset with an ``'audio'`` key whose ``'bytes'``
        field holds raw audio data (dataset cast with ``Audio(decode=False)``).

    Returns
    -------
    tuple[np.ndarray, int, tuple]
        ``(waveform, sample_rate, shape)`` where *waveform* is a float32
        NumPy array, *sample_rate* is in Hz, and *shape* is
        ``waveform.shape``.

    Raises
    ------
    ValueError
        If the ``'bytes'`` field is ``None``.
    """
    audio_bytes = sample["audio"]["bytes"]
    if audio_bytes is None:
        raise ValueError("No audio bytes found in sample.")
    speech, sr = sf.read(io.BytesIO(audio_bytes))
    return speech, sr, speech.shape


def extract_audio_slice(audio_bytes: bytes, start: float, end: float) -> bytes:
    """
    Extract a temporal slice from raw audio bytes and return it as WAV bytes.

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio bytes (any format readable by ``soundfile``).
    start : float
        Slice start time in seconds.
    end : float
        Slice end time in seconds.

    Returns
    -------
    bytes
        In-memory WAV bytes for the ``[start, end)`` window.
    """
    speech, sr = sf.read(io.BytesIO(audio_bytes))
    s = max(0, int(start * sr))
    e = min(len(speech), int(end * sr))
    chunk = speech[s:e]
    buf = io.BytesIO()
    sf.write(buf, chunk, sr, format="WAV")
    return buf.getvalue()


def compute_rms(audio: np.ndarray) -> float:
    """
    Compute the root-mean-square amplitude of *audio*.

    RMS is a simple proxy for signal energy — useful for detecting near-silent
    segments before applying noise augmentation.

    Parameters
    ----------
    audio : np.ndarray
        1-D (mono) or 2-D (channels × samples) float array.

    Returns
    -------
    float
        RMS value in the same unit as *audio* (typically normalised –1..1).
    """
    return float(np.sqrt(np.mean(audio ** 2)))


def compute_silence_ratio(audio: np.ndarray, threshold: float = 0.01) -> float:
    """
    Return the fraction of samples whose absolute amplitude is below *threshold*.

    Parameters
    ----------
    audio : np.ndarray
        Float waveform array.
    threshold : float
        Amplitude cut-off below which a sample is considered silent.
        Default ``0.01`` works for normalised (–1..1) audio.

    Returns
    -------
    float
        Value in ``[0.0, 1.0]``. A ratio near 1.0 indicates a mostly-silent
        recording.
    """
    return float(np.sum(np.abs(audio) < threshold) / len(audio))


# ── Segment gap smoothing ─────────────────────────────────────────────────────

def smooth_segments(
    segments: list[dict],
    *,
    small_gap: float,
    large_gap: float,
    max_extension: float,
    fill_ratio: float,
) -> list[dict]:
    """
    Close or partially fill the silence between consecutive segments.

    Rules
    -----
    - ``gap < small_gap``                 → **snap**: ``current.end = next.start``
    - ``gap > large_gap``                 → **extend**: ``current.end += max_extension``
    - ``small_gap <= gap <= large_gap``   → **partial fill**: ``current.end += gap * fill_ratio``

    The thresholds come from ``configs/audit_config.yaml`` (``audio.smoothing``
    section) and are forwarded as keyword arguments so the function stays
    testable without touching any config file.

    Parameters
    ----------
    segments : list[dict]
        Segment dicts, each with ``'start'`` and ``'end'`` float fields.
    small_gap : float
        Gap (s) below which a snap is applied.
    large_gap : float
        Gap (s) above which a fixed extension is applied.
    max_extension : float
        Seconds added when ``gap > large_gap``.
    fill_ratio : float
        Fraction of the gap added to ``end`` for medium gaps.

    Returns
    -------
    list[dict]
        New list of dicts with adjusted ``'end'`` values; originals untouched.
    """
    adjusted = []
    for i, seg in enumerate(segments):
        current = dict(seg)
        if i < len(segments) - 1:
            gap = segments[i + 1]["start"] - current["end"]
            if gap < small_gap:
                current["end"] = segments[i + 1]["start"]
            elif gap > large_gap:
                current["end"] += max_extension
            else:
                current["end"] += gap * fill_ratio
        adjusted.append(current)
    return adjusted


# ── Exhaustive DataFrame builder ──────────────────────────────────────────────

def build_segment_df(hf_dataset):
    """
    Explode a HF dataset into a flat, one-row-per-segment pandas DataFrame.

    Columns produced
    ----------------
    ``audio_id``         : ``"{rec_audio_id}_{seg_idx}"``
    ``seg_start``        : segment start time (float, seconds)
    ``seg_end``          : segment end time (float, seconds)
    ``transcript``       : cleaned transcript (``segments[].transcript``)
    ``transcript_raw``   : raw transcript (``segments[].transcript_raw``)
    + any recording-level metadata columns present in the dataset

    Parameters
    ----------
    hf_dataset : datasets.Dataset
        HuggingFace Dataset cast with ``Audio(decode=False)``.

    Returns
    -------
    tuple[pandas.DataFrame, int]
        ``(segment_df, n_skipped)`` where *n_skipped* is the count of
        recordings that had an empty segments list and were omitted.
    """
    import logging
    import pandas as pd
    from tqdm import tqdm

    _log = logging.getLogger(__name__)

    rows = []
    skip_keys = {"audio_id", "audio", "segments", "transcript"}
    n_skipped = 0

    for rec_idx, row in tqdm(
        enumerate(hf_dataset),
        total=len(hf_dataset),
        desc="Building segment DataFrame",
    ):
        extra_meta = {k: row[k] for k in row.keys() if k not in skip_keys}
        segments = sorted(row.get("segments", []), key=lambda x: x["start"])
        if not segments:
            _log.warning(
                "Recording %s (index %d) has no segments — skipping.",
                row.get("audio_id", "?"), rec_idx,
            )
            n_skipped += 1
            continue

        for seg_idx, seg in enumerate(segments):
            rows.append({
                "audio_id":       f"{row['audio_id']}_{seg_idx}",
                "seg_start":      seg.get("start", 0.0),
                "seg_end":        seg.get("end",   0.0),
                "transcript":     seg.get("transcript", ""),
                "transcript_raw": seg.get("transcript_raw", seg.get("transcript", "")),
                **extra_meta,
            })

    if n_skipped:
        _log.warning("build_segment_df: skipped %d recording(s) with no segments.", n_skipped)

    return pd.DataFrame(rows), n_skipped


# ── Recording-level duration extraction ──────────────────────────────────────

def get_recording_durations(
    dataset,
    *,
    decode: bool = False,
) -> tuple[np.ndarray, list[int]]:
    """
    Extract total audio duration for every recording in *dataset*.

    Parameters
    ----------
    dataset : datasets.Dataset
        HuggingFace Dataset (with or without ``Audio(decode=False)``).
    decode : bool
        If ``False`` (default), duration is inferred from the last segment's
        ``end`` time — fast, no waveform decoded.
        If ``True``, decodes the waveform and computes ``len(array) / sr``.

    Returns
    -------
    tuple[np.ndarray, list[int]]
        ``(durations_seconds, sample_rates)`` where *sample_rates* is
        all-zeros when ``decode=False``.
    """
    durations: list[float] = []
    sample_rates: list[int] = []

    if decode:
        for row in dataset:
            audio = row["audio"]
            durations.append(len(audio["array"]) / audio["sampling_rate"])
            sample_rates.append(audio["sampling_rate"])
    else:
        for row in dataset:
            segs = row.get("segments", [])
            durations.append(max((s["end"] for s in segs), default=0.0))
            sample_rates.append(0)

    return np.array(durations), sample_rates


def get_sample_rates(dataset, n_samples: int = 500) -> list[int]:
    """
    Sample *n_samples* recordings and decode each to read its true sample rate.

    A uniform stride is used so the sample spans the full dataset rather than
    clustering at the beginning.

    Parameters
    ----------
    dataset : datasets.Dataset
        HuggingFace Dataset with audio bytes available.
    n_samples : int
        Maximum number of recordings to decode. Defaults to ``500``.

    Returns
    -------
    list[int]
        Sample-rate values (Hz) collected from successfully decoded files.
    """
    rates: list[int] = []
    step = max(1, len(dataset) // n_samples)
    for i in range(0, len(dataset), step):
        try:
            ab = dataset[i]["audio"]["bytes"]
            if ab is not None:
                _, sr = sf.read(io.BytesIO(ab))
                rates.append(int(sr))
        except Exception as exc:
            import logging as _logging
            _logging.getLogger(__name__).debug(
                "Could not decode audio for sample index %d: %s", i, exc
            )
    return rates


# ── Segment-level stats ───────────────────────────────────────────────────────

def segment_duration_stats(df) -> dict:
    """
    Compute descriptive statistics for segment durations.

    Parameters
    ----------
    df : pandas.DataFrame
        Segment DataFrame with ``'seg_start'`` and ``'seg_end'`` columns.

    Returns
    -------
    dict
        Keys: ``n_segments``, ``total_hours``, ``mean_s``, ``median_s``,
        ``std_s``, ``min_s``, ``max_s``, ``pct_5``, ``pct_95``.
    """
    durations = df["seg_end"] - df["seg_start"]
    return {
        "n_segments":  len(durations),
        "total_hours": round(float(durations.sum() / 3600), 4),
        "mean_s":      round(float(durations.mean()),        3),
        "median_s":    round(float(durations.median()),      3),
        "std_s":       round(float(durations.std()),         3),
        "min_s":       round(float(durations.min()),         3),
        "max_s":       round(float(durations.max()),         3),
        "pct_5":       round(float(np.percentile(durations,  5)), 3),
        "pct_95":      round(float(np.percentile(durations, 95)), 3),
    }


def enrich_audio_features(df):
    """
    Add a ``'seg_duration'`` column (seconds) to a segment DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``'seg_start'`` and ``'seg_end'`` columns.

    Returns
    -------
    pandas.DataFrame
        Same DataFrame with ``'seg_duration'`` appended (in-place).
    """
    df["seg_duration"] = df["seg_end"] - df["seg_start"]
    return df


# ── Recording-level stats ─────────────────────────────────────────────────────

def recording_duration_stats(durations: np.ndarray) -> dict:
    """
    Return descriptive statistics for an array of recording durations.

    Parameters
    ----------
    durations : np.ndarray
        1-D array of recording durations in seconds.

    Returns
    -------
    dict
        Keys: ``n_recordings``, ``total_hours``, ``mean_s``, ``median_s``,
        ``std_s``, ``min_s``, ``max_s``.
    """
    return {
        "n_recordings": len(durations),
        "total_hours":  round(float(durations.sum() / 3600), 4),
        "mean_s":       round(float(durations.mean()),        3),
        "median_s":     round(float(np.median(durations)),    3),
        "std_s":        round(float(durations.std()),         3),
        "min_s":        round(float(durations.min()),         3),
        "max_s":        round(float(durations.max()),         3),
    }


# ── Segment gap analysis ──────────────────────────────────────────────────────

def compute_segment_gaps(segments: list[dict]) -> np.ndarray:
    """
    Return the inter-segment gap durations for a single recording.

    Parameters
    ----------
    segments : list[dict]
        Unsorted segment dicts, each with ``'start'`` and ``'end'`` floats.

    Returns
    -------
    np.ndarray
        1-D array of gap lengths in seconds (empty if fewer than 2 segments).
    """
    segs = sorted(segments, key=lambda x: x["start"])
    return np.array([
        segs[i + 1]["start"] - segs[i]["end"]
        for i in range(len(segs) - 1)
    ])


def compute_total_silence_time(dataset) -> float:
    """
    Sum all inter-segment silence across the entire dataset.

    Parameters
    ----------
    dataset : datasets.Dataset
        HuggingFace Dataset; each row must have a ``'segments'`` list.

    Returns
    -------
    float
        Total silence time in seconds.
    """
    total = 0.0
    for row in dataset:
        segs = row.get("segments", [])
        if len(segs) > 1:
            total += float(compute_segment_gaps(segs).sum())
    return total


def segment_gap_stats(dataset, n_samples: int = 500) -> dict:
    """
    Compute gap-distribution statistics by sampling *n_samples* recordings.

    Parameters
    ----------
    dataset : datasets.Dataset
        HuggingFace Dataset with a ``'segments'`` field per row.
    n_samples : int
        Number of recordings to sample (uniform stride). Defaults to ``500``.

    Returns
    -------
    dict
        Keys: ``n_gaps_sampled``, ``mean_gap_s``, ``median_gap_s``,
        ``pct_small_gap``, ``pct_large_gap``, ``max_gap_s``,
        ``sum_gap_s``, ``n_gaps > 60_s``.
    """
    all_gaps: list[float] = []
    step = max(1, len(dataset) // n_samples)

    for i in range(0, len(dataset), step):
        segs = dataset[i].get("segments", [])
        if len(segs) > 1:
            all_gaps.extend(compute_segment_gaps(segs).tolist())

    arr = np.array(all_gaps)
    if len(arr) == 0:
        return {}

    total_silence = compute_total_silence_time(dataset)
    return {
        "n_gaps_sampled": len(arr),
        "mean_gap_s":     round(float(arr.mean()),                        4),
        "median_gap_s":   round(float(np.median(arr)),                    4),
        "pct_small_gap":  round(float((arr < 0.2).mean()),                4),
        "pct_large_gap":  round(float((arr > 1.0).mean()),                4),
        "max_gap_s":      round(float(arr.max()),                         4),
        "sum_gap_s":      round(total_silence,                            4),
        "n_gaps > 60_s":  int(len([x for x in arr if x > 60])),
    }


# ── Visualisations ────────────────────────────────────────────────────────────

def visualize_gap_distribution(
    dataset,
    n_samples: int,
    figures_path: str,
    *,
    zoom_cap: float = 3.0,
    small_gap: float = 0.2,
    large_gap: float = 1.0,
) -> None:
    """
    Plot the full and zoomed inter-segment gap distributions.

    Saves ``segment_gap_distribution.png`` to *figures_path*.

    Parameters
    ----------
    dataset : datasets.Dataset
        HuggingFace Dataset.
    n_samples : int
        Number of recordings to sample for gap collection.
    figures_path : str
        Directory where the figure is saved.
    zoom_cap : float
        Upper bound (seconds) for the zoomed panel.  Default ``3.0``.
        Read from ``configs/audit_config.yaml`` (``audio.gap_zoom_cap``).
    small_gap : float
        Threshold below which gaps are classified as snap candidates.
        Should match ``audio.smoothing.small_gap`` in the config.
        Default ``0.2``.
    large_gap : float
        Threshold above which gaps are classified as silence regions.
        Should match ``audio.smoothing.large_gap`` in the config.
        Default ``1.0``.
    """
    from pathlib import Path as _Path

    sampled_gaps: list[float] = []
    step = max(1, len(dataset) // n_samples)

    for i in range(0, len(dataset), step):
        segs = sorted(dataset[i].get("segments", []), key=lambda x: x["start"])
        for j in range(len(segs) - 1):
            sampled_gaps.append(segs[j + 1]["start"] - segs[j]["end"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(sampled_gaps, bins=80, color="#4C72B0",
                 edgecolor="white", linewidth=0.4)
    axes[0].set_title("Inter-segment gap distribution (all)")
    axes[0].set_xlabel("Gap (s)")
    axes[0].set_ylabel("Count")

    zoomed = [g for g in sampled_gaps if 0 <= g <= zoom_cap]
    axes[1].hist(zoomed, bins=60, color="#DD8452",
                 edgecolor="white", linewidth=0.4)
    axes[1].set_title(f"Inter-segment gap distribution (0–{zoom_cap} s zoom)")
    axes[1].set_xlabel("Gap (s)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(
        str(_Path(figures_path) / "segment_gap_distribution.png"),
        bbox_inches="tight",
    )
    plt.show()

    print(f"  Gaps < {small_gap} s (snap candidates) : {sum(g < small_gap for g in sampled_gaps):,}")
    print(f"  Gaps > {large_gap} s (silence regions)  : {sum(g > large_gap for g in sampled_gaps):,}")


def visualize_sample_rate_pie(
    sr_counter_train,
    sr_counter_test,
    save_path: str | None = None,
) -> None:
    """
    Render a side-by-side pie chart of sample-rate distributions.

    Parameters
    ----------
    sr_counter_train : collections.Counter
        Sample-rate counts for the training split.
    sr_counter_test : collections.Counter
        Sample-rate counts for the test split.
    save_path : str | None
        If given, saves the figure to this path.
    """
    SR_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#9C27B0", "#FF9800"]

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    for ax, counter, title in [
        (axes[0], sr_counter_train, "Train"),
        (axes[1], sr_counter_test,  "Test"),
    ]:
        if not counter:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
            ax.set_title(title, fontsize=8)
            continue

        items  = counter.most_common()
        labels = [f"{sr:,} Hz\n({cnt})" for sr, cnt in items]
        values = [cnt for _, cnt in items]
        colors = SR_PALETTE[:len(values)]

        ax.pie(
            values,
            labels=labels,
            colors=colors,
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1},
            textprops={"fontsize": 8},
            autopct=lambda p: f"{p:.1f}%" if p > 5 else "",
        )
        ax.set_title(title, fontsize=8)

    plt.suptitle("Sample Rate Distribution", fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def visualize_cross_analysis(
    df_train,
    df_test,
    figures_path: str,
    max_points: int = 3000,
) -> None:
    """
    Four-panel cross-analysis scatter / histogram plot.

    Panels
    ------
    1. Speech rate (words/s) histogram — train vs test overlay.
    2. Duration vs. word count scatter (train sample).
    3. Duration vs. char count scatter (train sample).
    4. Word count vs. char count scatter (train sample).

    Parameters
    ----------
    df_train : pandas.DataFrame
        Enriched training segment DataFrame.
    df_test : pandas.DataFrame
        Enriched test segment DataFrame.
    figures_path : str
        Directory where ``cross_analysis.png`` is saved.
    max_points : int
        Maximum scatter-plot points drawn (random sample). Default ``3000``.
    """
    n_scatter  = min(max_points, len(df_train))
    sample_idx = np.random.choice(len(df_train), size=n_scatter, replace=False)

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df_train["speech_rate"].dropna().clip(upper=10), bins=60,
             color="#4C72B0", edgecolor="white", linewidth=0.4, label="train")
    ax1.hist(df_test["speech_rate"].dropna().clip(upper=10),  bins=60,
             color="#DD8452", alpha=0.6, edgecolor="white", linewidth=0.4, label="test")
    ax1.set_title("Speech rate (words/s)")
    ax1.set_xlabel("Words per second")
    ax1.set_ylabel("Segments")
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(df_train["seg_duration"].iloc[sample_idx],
                df_train["word_count"].iloc[sample_idx],
                alpha=0.15, s=6, color="#DD8452")
    ax2.set_title(f"Duration vs. word count (train, {n_scatter:,} pts)")
    ax2.set_xlabel("Segment duration (s)")
    ax2.set_ylabel("Word count")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(df_train["seg_duration"].iloc[sample_idx],
                df_train["char_count"].iloc[sample_idx],
                alpha=0.15, s=6, color="#55A868")
    ax3.set_title(f"Duration vs. char count (train, {n_scatter:,} pts)")
    ax3.set_xlabel("Segment duration (s)")
    ax3.set_ylabel("Char count")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(df_train["word_count"].iloc[sample_idx],
                df_train["char_count"].iloc[sample_idx],
                alpha=0.15, s=6, color="#C44E52")
    ax4.set_title(f"Word count vs. char count (train, {n_scatter:,} pts)")
    ax4.set_xlabel("Word count")
    ax4.set_ylabel("Char count")

    plt.suptitle("Cross-analysis: audio × text features", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(str(Path(figures_path) / "cross_analysis.png"), bbox_inches="tight")
    plt.show()
