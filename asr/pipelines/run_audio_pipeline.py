"""
run_audio_pipeline.py
──────────────────────
Non-interactive batch runner for Phase 2 — audio preprocessing.

Mirrors the logic of ``notebooks/02_audio_preprocessing.ipynb`` exactly:
  text-clean parquets  →  optional gap smoothing (updates seg_start/seg_end)
  →  segment extraction + waveform transforms  →  quality gate
  →  Arrow IPC (waveforms)  +  manifest.parquet  +  summary

If smoothing is enabled the updated parquets are written back to
``data/interim/03_text_clean/`` so downstream steps use the smoothed
boundaries — identical to the notebook's §4 save cell.

Usage
-----
    python pipelines/run_audio_pipeline.py
    python pipelines/run_audio_pipeline.py --split train --force
    python pipelines/run_audio_pipeline.py --no-smoothing
    python pipelines/run_audio_pipeline.py --dry-run

Arguments
---------
--force          Re-run even if output Arrow file exists.
--split          ``train`` | ``test`` | ``both``  (default: ``both``).
--no-smoothing   Override config — skip gap smoothing.
--dry-run        Process first 200 rows only.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.utils               import load_config, ensure_dir, print_dict, print_section
from src.audit_audio         import enrich_audio_features
from src.audio_preprocessing import (
    apply_smoothing_to_df, smoothing_summary,
    iter_segments, save_segments_arrow, load_manifest, compute_corpus_stats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Core processing — mirrors notebook §3-§6 ─────────────────────────────────

def process_split(
    hf_split,
    df: pd.DataFrame,
    *,
    split_name: str,
    smooth_cfg: dict,
    preprocess_kwargs: dict,
    gate_kwargs: dict,
    storage_cfg: dict,
    audio_out_dir: Path,
    text_clean_dir: Path,
    do_smoothing: bool,
    dry_run: bool,
) -> dict:
    """
    Run the full Phase 2 audio pipeline on one split.

    Steps mirror notebook §3-§8:
      enrich_audio_features  →  optional smoothing (+ parquet save-back)
      →  iter_segments + full_preprocess  →  quality gate
      →  save_segments_arrow  →  compute_corpus_stats
    """
    # Ensure seg_duration exists (notebook §3)
    df = enrich_audio_features(df)

    # ── Optional gap smoothing (notebook §4) ─────────────────────────────────
    if do_smoothing and smooth_cfg.get("enabled", True):
        log.info("[%s] Applying gap smoothing …", split_name)
        df_before = df.copy()
        df = apply_smoothing_to_df(
            df,
            small_gap=smooth_cfg["small_gap"],
            large_gap=smooth_cfg["large_gap"],
            max_extension=smooth_cfg["max_extension"],
            fill_ratio=smooth_cfg["fill_ratio"],
        )
        sm = smoothing_summary(df_before, df)
        log.info("[%s] Smoothed %d/%d segments (%.1f%%)  gain=%.2fs",
                 split_name, sm["n_smoothed"], len(df),
                 sm["pct_smoothed"], sm["duration_gain_s"])

        # Notebook §4: write updated parquet back to text_clean so the
        # smoothed boundaries propagate to all downstream steps.
        save_back = text_clean_dir / f"df_{split_name}_clean.parquet"
        df.to_parquet(save_back, index=False)
        log.info("[%s] Smoothed parquet saved → %s", split_name, save_back)
    else:
        df["smoothing_applied"] = False
        log.info("[%s] Gap smoothing skipped.", split_name)

    # ── Dry-run truncation ────────────────────────────────────────────────────
    if dry_run:
        df = df.head(200).copy()
        log.info("[%s] DRY RUN — 200 rows", split_name)

    # ── Segment extraction + preprocessing (notebook §6) ─────────────────────
    log.info("[%s] Starting extraction … (%d segments)", split_name, len(df))
    t0 = time.perf_counter()

    segments = list(iter_segments(
        hf_split, df,
        preprocess_kwargs=preprocess_kwargs,
        gate_kwargs=gate_kwargs,
        split_name=split_name,
    ))

    log.info("[%s] Extraction done in %.1f min", split_name,
             (time.perf_counter() - t0) / 60)

    # ── Save Arrow + manifest (notebook §6 last cells) ────────────────────────
    arrow_path, manifest_path = save_segments_arrow(
        segments, audio_out_dir, split_name,
        kept_only=storage_cfg.get("kept_only", True),
    )
    log.info("[%s] Arrow   → %s", split_name, arrow_path)
    log.info("[%s] Manifest → %s", split_name, manifest_path)

    # ── Stats (notebook §7) ───────────────────────────────────────────────────
    mdf   = load_manifest(manifest_path)
    stats = compute_corpus_stats(mdf)
    log.info("[%s] kept=%d (%.1f%%)  %.3f hours",
             split_name, stats["n_kept"], stats["pct_kept"],
             stats["total_hours_kept"])

    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 — audio preprocessing")
    parser.add_argument("--force",        action="store_true")
    parser.add_argument("--split",        default="both",
                        choices=["train", "test", "both"])
    parser.add_argument("--no-smoothing", action="store_true")
    parser.add_argument("--dry-run",      action="store_true")
    args = parser.parse_args()

    # ── Configs ───────────────────────────────────────────────────────────────
    paths     = load_config(PROJECT_ROOT / "configs" / "paths.yaml")
    audio_cfg = load_config(PROJECT_ROOT / "configs" / "audio_config.yaml")

    data_path      = PROJECT_ROOT / paths["data"]["raw"]
    text_clean_dir = PROJECT_ROOT / paths["interim"]["text_clean"]
    audio_out_dir  = PROJECT_ROOT / paths["interim"]["audio_clean"]
    ensure_dir(audio_out_dir)

    smooth_cfg = audio_cfg["smoothing"]
    pp_cfg     = audio_cfg["preprocessing"]
    gate_cfg   = audio_cfg["quality_gate"]
    store_cfg  = audio_cfg["storage"]

    preprocess_kwargs = {
        "target_sr":         pp_cfg["target_sr"],
        "do_mono":           pp_cfg["do_mono"],
        "do_resample":       pp_cfg["do_resample"],
        "do_trim":           pp_cfg["do_trim"],
        "do_normalize":      pp_cfg["do_normalize"],
        "do_pre_emphasis":   pp_cfg["do_pre_emphasis"],
        "normalize_level":   pp_cfg["normalize_level"],
        "pre_emphasis_coef": pp_cfg["pre_emphasis_coef"],
        "trim_top_db":       pp_cfg["trim_top_db"],
    }

    gate_kwargs = {
        "min_duration_s":    gate_cfg["min_duration_s"],
        "max_duration_s":    gate_cfg["max_duration_s"],
        "min_rms":           gate_cfg["min_rms"],
        "max_silence_ratio": gate_cfg["max_silence_ratio"],
    }

    do_smoothing = not args.no_smoothing

    # ── Load HF dataset once ──────────────────────────────────────────────────
    from src.data_loader import load_local_dataset
    from datasets import Audio as HFAudio

    log.info("Loading HuggingFace dataset from %s …", data_path)
    dataset = load_local_dataset(str(data_path))

    # ── Process each split ────────────────────────────────────────────────────
    splits = ["train", "test"] if args.split == "both" else [args.split]
    all_stats: dict[str, dict] = {}

    for split_name in splits:
        arrow_path = audio_out_dir / split_name / "segments.arrow"
        if arrow_path.exists() and not args.force:
            log.info("[%s] Arrow exists — skipping (--force to re-run)", split_name)
            continue

        df_path = text_clean_dir / f"df_{split_name}_clean.parquet"
        if not df_path.exists():
            log.error("[%s] Text-clean parquet not found: %s", split_name, df_path)
            log.error("     Run run_text_pipeline.py first.")
            continue

        df = pd.read_parquet(df_path)
        hf_split = dataset[split_name].cast_column("audio", HFAudio(decode=False))
        log.info("[%s] Loaded %d segments, %d recordings",
                 split_name, len(df), len(hf_split))

        stats = process_split(
            hf_split, df,
            split_name=split_name,
            smooth_cfg=smooth_cfg,
            preprocess_kwargs=preprocess_kwargs,
            gate_kwargs=gate_kwargs,
            storage_cfg=store_cfg,
            audio_out_dir=audio_out_dir,
            text_clean_dir=text_clean_dir,
            do_smoothing=do_smoothing,
            dry_run=args.dry_run,
        )
        all_stats[split_name] = stats

    # ── Summary ───────────────────────────────────────────────────────────────
    print_section("PHASE 2 — AUDIO PREPROCESSING COMPLETE")
    for split_name, s in all_stats.items():
        print_dict({
            "In              ": f"{s['n_total']:,}",
            "Kept            ": f"{s['n_kept']:,}  ({s['pct_kept']:.1f}%)",
            "Dropped         ": f"{s['n_dropped']:,}",
            "Total hours kept": f"{s['total_hours_kept']:.3f} h",
            "Mean duration   ": f"{s['mean_duration_s']:.3f} s",
            "Drop reasons    ": str(s["drop_reasons"]),
        }, title=f"Split: {split_name}")


if __name__ == "__main__":
    main()
