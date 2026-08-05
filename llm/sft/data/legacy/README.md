# Legacy / superseded SFT data artifacts

Kept for reference, not part of the current pipeline.

- **`cleaned/cleaned.csv`** — an earlier snapshot of the cleaned SFT seed dataset,
  last touched 2026-05-07. It was superseded by the top-level
  `llm/sft/data/cleaned.csv`, which is the one actually read by
  `llm/sft/data/scripts/data_augmentation.ipynb` and written by
  `llm/sft/data/scripts/data_cleaning.ipynb`.
- **`augmentation_with_mapper.ipynb`** — an earlier/parallel variant of
  `llm/sft/data/scripts/data_augmentation.ipynb` that reads from `cleaned/cleaned.csv`
  above instead of the current `cleaned.csv`. Superseded by the notebook in
  `llm/sft/data/scripts/`.
