"""
run_dataset_builder.py
──────────────────────
Non-interactive batch runner for Phase 4 — final dataset construction.

Mirrors ``notebooks/03_dataset_builder.ipynb`` exactly:
  Arrow + manifest (Phase 2)  +  text-clean parquet (Phase 3)
  →  inner join  →  keep == True filter  →  drop intermediate columns
  →  validate  →  HuggingFace DatasetDict  →  data/processed/06_final/

Usage
-----
    python pipelines/run_dataset_builder.py
    python pipelines/run_dataset_builder.py --split train --force

Arguments
---------
--force     Re-run even if output directory exists.
--split     ``train`` | ``test`` | ``both``  (default: ``both``).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils           import load_config, ensure_dir, print_dict, print_section
from src.dataset_builder import build_hf_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 — final dataset construction")
    parser.add_argument("--force",  action="store_true")
    parser.add_argument("--split",  default="both",
                        choices=["train", "test", "both"])
    args = parser.parse_args()

    paths = load_config(PROJECT_ROOT / "configs" / "paths.yaml")

    audio_clean_dir = PROJECT_ROOT / paths["interim"]["audio_clean"]
    text_clean_dir  = PROJECT_ROOT / paths["interim"]["text_clean"]
    output_dir      = PROJECT_ROOT / paths["processed"]["final"]

    # Skip if output already exists and --force not given
    if output_dir.exists() and not args.force:
        log.info("Output directory exists — skipping (--force to re-run): %s",
                 output_dir)
        return

    ensure_dir(output_dir)

    splits = ["train", "test"] if args.split == "both" else [args.split]
    log.info("Building splits: %s", splits)

    dataset_dict = build_hf_dataset(
        audio_clean_dir=audio_clean_dir,
        text_clean_dir=text_clean_dir,
        output_dir=output_dir,
        splits=splits,
    )

    print_section("PHASE 4 — DATASET BUILDER COMPLETE")
    for split_name, ds in dataset_dict.items():
        print_dict({
            "Split      ": split_name,
            "Segments   ": f"{len(ds):,}",
            "Columns    ": str(ds.column_names),
            "Features   ": str(ds.features),
        }, title=f"Split: {split_name}")

    print(f"\n  Saved → {output_dir}")
    print(f"  Load  : from datasets import load_from_disk")
    print(f"         dataset = load_from_disk('{output_dir}')")


if __name__ == "__main__":
    main()
