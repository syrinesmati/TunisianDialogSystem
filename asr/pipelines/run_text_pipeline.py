"""
run_text_pipeline.py
────────────────────
Non-interactive batch runner for Phase 3 — text preprocessing.

Mirrors the logic of ``notebooks/01_text_preprocessing.ipynb`` exactly:
  raw parquet  →  cleaning  →  code-switch  →  NOTA (normalize_with_flags)
  →  flag columns expansion  →  quality gate  →  parquet + JSONL drop log

Usage
-----
    python pipelines/run_text_pipeline.py
    python pipelines/run_text_pipeline.py --split train --force
    python pipelines/run_text_pipeline.py --cs-policy remove
    python pipelines/run_text_pipeline.py --dry-run

Arguments
---------
--force         Re-run even if output files already exist.
--split         ``train`` | ``test`` | ``both``  (default: ``both``).
--cs-policy     Override code-switch policy from config.
--dry-run       Process first 500 rows only.
"""

from __future__ import annotations

import argparse
import json
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
from src.cleaning            import clean_dataframe, load_transcript_corrections
from src.nota_normalizer     import NOTANormalizer
from src.code_switch_handler import CodeSwitchHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Core processing — mirrors notebook logic cell-by-cell ─────────────────────

def process_split(
    df: pd.DataFrame,
    *,
    split_name: str,
    cleaning_cfg: dict,
    corrections: list,
    cs_handler: CodeSwitchHandler,
    normalizer: NOTANormalizer,
    min_transcript_len: int,
    log_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full Phase 3 pipeline on one split.

    Produces the same output columns as the notebook:
      transcript_clean, transcript_cs, transcript_nota,
      waw_applied, negation_applied, taa_applied, alef_applied,
      hamza_applied, variant_applied, override_applied.

    Returns (kept_df, dropped_df).
    """
    df = df.copy()
    log.info("[%s] Input: %d rows", split_name, len(df))

    # ── Cleaning (cells 11–16) ────────────────────────────────────────────────
    t0 = time.perf_counter()
    df = clean_dataframe(
        df,
        col="transcript",
        keep_digits=cleaning_cfg["keep_digits"],
        corrections=corrections,
    )
    n_changed = (df["transcript"] != df["transcript_clean"]).sum()
    log.info("[%s] Cleaning: %.1fs  changed=%d", split_name,
             time.perf_counter() - t0, n_changed)

    # ── Code-switch handling (cell 21) ────────────────────────────────────────
    t0 = time.perf_counter()
    df["transcript_cs"] = cs_handler.process_series(df["transcript_clean"])
    n_cs = (df["transcript_clean"] != df["transcript_cs"]).sum()
    log.info("[%s] Code-switch: %.1fs  changed=%d  policy=%s",
             split_name, time.perf_counter() - t0, n_cs, cs_handler.policy)

    # ── NOTA normalisation with per-row flag columns (cell 28) ───────────────
    # Uses normalize_with_flags exactly as the notebook does, then expands
    # the returned flags dict into separate boolean columns via pd.json_normalize.
    t0 = time.perf_counter()

    results = df["transcript_cs"].apply(
        lambda t: pd.Series(normalizer.normalize_with_flags(str(t)))
    )
    # results has columns [0, 1] — index 0 = normalised text, 1 = flags dict
    df["transcript_nota"] = results[0]
    flags_df = pd.json_normalize(results[1])  # expands dict to named columns
    # Drop columns that already exist in df to avoid concat conflicts
    flags_df = flags_df[[c for c in flags_df.columns if c not in df.columns]]
    df = pd.concat([df.reset_index(drop=True), flags_df.reset_index(drop=True)], axis=1)

    n_nota = (df["transcript_cs"] != df["transcript_nota"]).sum()
    log.info("[%s] NOTA: %.1fs  changed=%d", split_name,
             time.perf_counter() - t0, n_nota)

    # ── Quality gate (cell 51) ────────────────────────────────────────────────
    mask    = df["transcript_nota"].str.strip().str.len() >= min_transcript_len
    kept    = df[mask].copy()
    dropped = df[~mask].copy()
    log.info("[%s] Quality gate: kept=%d  dropped=%d",
             split_name, len(kept), len(dropped))

    # ── Drop log (cell 52) ────────────────────────────────────────────────────
    # Append mode so re-running one split does not wipe logs from others.
    with open(log_path, "a", encoding="utf-8") as fh:
        for _, row in dropped.iterrows():
            fh.write(json.dumps({
                "split":           split_name,
                "audio_id":        row["audio_id"],
                "phase":           3,
                "reason":          "transcript_too_short",
                "transcript_nota": str(row["transcript_nota"]),
                "char_len":        len(str(row["transcript_nota"]).strip()),
            }, ensure_ascii=False) + "\n")

    return kept, dropped


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 — text preprocessing")
    parser.add_argument("--force",     action="store_true")
    parser.add_argument("--split",     default="both",
                        choices=["train", "test", "both"])
    parser.add_argument("--cs-policy", default=None,
                        choices=["arabize", "keep", "remove"])
    parser.add_argument("--dry-run",   action="store_true",
                        help="Process first 500 rows only")
    args = parser.parse_args()

    # ── Configs ───────────────────────────────────────────────────────────────
    paths   = load_config(PROJECT_ROOT / "configs" / "paths.yaml")
    pp_cfg  = load_config(PROJECT_ROOT / "configs" / "preprocessing_config.yaml")

    reports_dir      = PROJECT_ROOT / paths["outputs"]["reports"]
    text_out_dir     = PROJECT_ROOT / paths["interim"]["text_clean"]
    lexicons_dir     = PROJECT_ROOT / pp_cfg["nota"]["lexicons_dir"]
    corrections_path = PROJECT_ROOT / paths["data"]["transcript_corrections"]

    ensure_dir(text_out_dir)

    cleaning_cfg = pp_cfg["cleaning"]
    min_len      = cleaning_cfg["min_transcript_length"]
    cs_policy    = args.cs_policy or pp_cfg["code_switch"]["policy"]
    layer_flags  = {k: pp_cfg["nota"][k] for k in pp_cfg["nota"]
                    if k.startswith("apply_")}

    # ── Init shared objects ───────────────────────────────────────────────────
    corrections = load_transcript_corrections(corrections_path)
    log.info("Transcript corrections: %d rules", len(corrections))

    cs_handler = CodeSwitchHandler(
        lexicons_dir=str(lexicons_dir),
        policy=cs_policy,
        case_insensitive=pp_cfg["code_switch"]["case_insensitive"],
    )
    log.info("CodeSwitchHandler: policy=%s  table=%d entries",
             cs_policy, len(cs_handler.loanword_table))

    normalizer = NOTANormalizer(
        lexicons_dir=str(lexicons_dir),
        layer_flags=layer_flags,
    )
    log.info("NOTANormalizer: active_layers=%s",
             [k for k, v in layer_flags.items() if v])

    # ── Process each split ────────────────────────────────────────────────────
    splits = ["train", "test"] if args.split == "both" else [args.split]
    all_stats: dict[str, dict] = {}

    for split_name in splits:
        out_path = text_out_dir / f"df_{split_name}_clean.parquet"
        log_path = text_out_dir / f"dropped_{split_name}.jsonl"

        if out_path.exists() and not args.force:
            log.info("[%s] Output exists — skipping (--force to re-run)", split_name)
            continue

        in_path = reports_dir / f"df_{split_name}_segments.parquet"
        if not in_path.exists():
            log.error("[%s] Input not found: %s", split_name, in_path)
            continue

        df = pd.read_parquet(in_path)
        if args.dry_run:
            df = df.head(500).copy()
            log.info("[%s] DRY RUN — 500 rows", split_name)

        kept, dropped = process_split(
            df,
            split_name=split_name,
            cleaning_cfg=cleaning_cfg,
            corrections=corrections,
            cs_handler=cs_handler,
            normalizer=normalizer,
            min_transcript_len=min_len,
            log_path=log_path,
        )

        # Save (cell 54)
        kept.to_parquet(out_path, index=False)
        log.info("[%s] Saved: %s  (%d rows × %d cols)",
                 split_name, out_path, len(kept), kept.shape[1])

        all_stats[split_name] = {
            "in": len(df), "kept": len(kept), "dropped": len(dropped),
            "out": str(out_path), "log": str(log_path),
        }

    # ── Summary (cell 56) ─────────────────────────────────────────────────────
    print_section("PHASE 3 — TEXT PREPROCESSING COMPLETE")
    summary_lines = ["PHASE 3 — TEXT PREPROCESSING COMPLETE\n"]

    for split_name, s in all_stats.items():
        pct = s["dropped"] / s["in"] * 100 if s["in"] else 0
        print_dict({
            "In       ": f"{s['in']:,}",
            "Kept     ": f"{s['kept']:,}  ({100 - pct:.2f}%)",
            "Dropped  ": f"{s['dropped']:,}  ({pct:.2f}%)",
            "Parquet  ": s["out"],
            "Drop log ": s["log"],
        }, title=f"Split: {split_name}")
        summary_lines += [
            f"Split: {split_name}",
            f"  In:      {s['in']:,}",
            f"  Kept:    {s['kept']:,}",
            f"  Dropped: {s['dropped']:,}  ({pct:.2f}%)",
            f"  Out:     {s['out']}",
            f"  Log:     {s['log']}", "",
        ]

    summary_path = text_out_dir / "phase3_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    log.info("Summary → %s", summary_path)


if __name__ == "__main__":
    main()
