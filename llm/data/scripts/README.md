# Data Collection & Cleaning Scripts

This directory contains utilities for collecting, loading, and cleaning Tunisian dialect text from Kaggle and HuggingFace.

---

## 📚 Files Overview

### `data_loader.py`
Functions to load Tunisian dialect datasets from HuggingFace.

**Key functions:**
- `load_all_datasets()` — Load all available sources at once
- `load_dialect_of_tunisia_collection()` — Load HuggingFace Dialect collection (2M samples)
- `load_tunisian_msa_parallel_corpus()` — Load parallel corpus
- `load_linagora_tunisian_derja()` — Load Linagora's 12 dataset configs
- `print_dataset_info()` — Show available HuggingFace datasets

### `data_cleaning.py`
Production-quality text cleaning and analysis functions.

**Key functions:**
- `full_cleaning_pipeline()` — One-shot cleaning with all steps
- `clean_text()` — Single-sample cleaning
- `classify_text_type()` — Classify as 'arabic', 'french', 'mixed', 'other'
- `filter_by_script_type()` — Keep only specific script types
- `count_tokens()` — Tokenize and count with HF tokenizers
- `get_top_words()` — Extract vocabulary statistics
- `print_cleaning_report()` — Before/after comparison

### `upload_to_huggingface.py`
Upload processed datasets to HuggingFace Hub for sharing & version control.

**Key functions:**
- `upload_dataset()` — Upload parquet file to HF Hub with metadata
- Command-line interface for easy uploading

**Usage:**
```bash
python upload_to_huggingface.py \
  --parquet_file processed/tunisian_arabic_clean.parquet \
  --repo_name tunisian-dialect-corpus-cleaned \
  --username your_username
```

Or in Python:
```python
from upload_to_huggingface import upload_dataset
upload_dataset(
    parquet_file="processed/tunisian_arabic_clean.parquet",
    repo_name="tunisian-dialect-corpus-cleaned",
    username="your_username"
)
```

### `tunisian-continuous-learning-data-collection.ipynb`
Main workflow notebook for Stage 1 (Data Collection) from HuggingFace.

**Workflow:**
1. Load datasets from HuggingFace (auto-cached locally)
2. Apply comprehensive cleaning
3. Analyze text distribution & statistics
4. Filter by script type (Arabic-only recommended)
5. Count tokens for pre-training estimation
6. Save processed data as Parquet files

**Prerequisites:**
```bash
# Set HuggingFace cache location (recommended: external SSD)
$env:HF_HOME = "E:/huggingface-cache"

# Install dependencies
pip install datasets transformers pandas
```

**Usage:**
```bash
jupyter notebook tunisian-continuous-learning-data-collection.ipynb
```

---

## 🚀 Quick Start

### Step 0: Configure HuggingFace Cache (Recommended)
```python
import os
# Set to external SSD for faster access and free up laptop storage
os.environ['HF_HOME'] = 'E:/huggingface-cache'
```

### Step 1: Load Raw Data from HuggingFace
```python
from data_loader import load_all_datasets

df_raw = load_all_datasets(include_linagora=True)
# Takes 5-10 minutes on first run (downloads from HuggingFace)
# Subsequent runs use cached data (fast)
```

### Step 2: Clean
```python
from data_cleaning import full_cleaning_pipeline

df_clean = full_cleaning_pipeline(df_raw, remove_symbols=True)
# Removes emojis, special chars, duplicates, short texts
```

### Step 3: Filter
```python
from data_cleaning import filter_by_script_type

df_arabic = filter_by_script_type(df_clean, script_type="arabic")
# Keep only pure Arabic script (removes MSA, French-only)
```

### Step 4: Save
```python
df_arabic.to_parquet("../processed/tunisian_arabic_clean.parquet")
```

---

## 🧹 What Gets Cleaned?

The `full_cleaning_pipeline()` removes or normalizes:

| Item | Action | Reason |
|------|--------|--------|
| Emojis | Removed | Not part of dialect text |
| `[موسيقى]` | Removed | Common artifact in datasets |
| RT, AFP | Removed | Social media noise |
| `<s>[INST]` markers | Removed | Instruction format artifacts |
| Multiple spaces | Normalized | Consistency |
| Empty rows | Removed | No training value |
| Duplicates | Removed | Exact duplicates |
| Short text (<3 chars) | Removed | Noise |
| Non-Arabic symbols | Removed | Cleans non-dialect content |

**Does NOT remove:**
- Spelling variation (it IS the dialect)
- Code-switching (optional separate dataset)
- Punctuation (some kept for grammar learning)

---

## 📊 Text Type Classification

After cleaning, each sample is classified:

```python
df["type"] = df["text"].apply(classify_text_type)
```

**Types:**
- **`"arabic"`**: Pure Arabic script — RECOMMENDED for pre-training
- **`"mixed"`**: Arabic + Latin (code-switching) — Optional, separate dataset
- **`"french"`**: Latin-only — Removed (not dialect)
- **`"other"`**: Neither — Removed (noise)

---

## 💾 Output Format

### Saved as Parquet
```python
df.to_parquet("tunisian_arabic_clean.parquet")
```

**Why Parquet?**
- Fast (columnar format)
- Efficient compression (3-5x reduction)
- Preserves data types
- Works with Pandas, Polars, DuckDB

**Load it back:**
```python
df = pd.read_parquet("tunisian_arabic_clean.parquet")
```

---

## ⚙️ Configuration Options

### `full_cleaning_pipeline()` Parameters

```python
full_cleaning_pipeline(
    df,
    text_col="text",              # Column name with text
    min_char_length=3,            # Minimum characters
    remove_symbols=True,          # Remove non-Arabic chars
    analyze=True,                 # Add analysis columns
)
```

### `load_all_datasets()` Parameters

```python
load_all_datasets(
    include_linagora=True,        # Load all 12 Linagora configs
    linagora_configs=None,        # Or specify subset: ["Derja_tunsi", ...]
)
```

---

## 📈 Expected Output Statistics

For **2 million HuggingFace Dialect of Tunisia samples**:

| Metric | Expected |
|--------|----------|
| After deduplication | ~1.8M unique |
| After cleaning | ~1.6M valid |
| Pure Arabic only | ~1.4M |  
| Avg tokens/sample | ~40-50 |
| **Total tokens** | ~60-70M |
| **Parquet file size** | ~3-5 GB || **HuggingFace cache size** | ~10-15 GB |
---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'datasets'"
Install HuggingFace datasets:
```bash
pip install datasets
```

### "Connection timeout loading dataset from HuggingFace"
Datasets are cached locally after first download.

**Default cache location:**
```bash
~/.cache/huggingface/datasets/
```

**To use external SSD for cache:**
```powershell
$env:HF_HOME = "E:/huggingface-cache"
python your_script.py
```

Or in code:
```python
import os
os.environ['HF_HOME'] = 'E:/huggingface-cache'
```

### Out of memory on large datasets
Process in chunks:
```python
for chunk in pd.read_parquet("file.parquet", chunksize=100000):
    processed_chunk = full_cleaning_pipeline(chunk)
    # ... process chunk
```

### Tokenizer not found
Install transformers:
```bash
pip install transformers torch
```

---

## 📖 Guide Files

- **[README.md](README.md)** — This file
- **[HUGGINGFACE_SETUP.md](HUGGINGFACE_SETUP.md)** — Download configuration (cache settings)
- **[HUGGINGFACE_UPLOAD.md](HUGGINGFACE_UPLOAD.md)** — Upload cleaned data to HF Hub
- **[../DATASET_MANIFEST.md](../DATASET_MANIFEST.md)** — Data storage strategy for 2M HuggingFace samples
- **[../README.md](../README.md)** — LLM pipeline overview  
- **[../../docs/data_strategy.md](../../docs/data_strategy.md)** — Data collection recommendations
- **[HuggingFace Docs](https://huggingface.co/docs/datasets/)** — Datasets library documentation

---

## 💡 Tips

1. **Set HF cache first**: Point to external SSD before downloading
   ```powershell
   $env:HF_HOME = "E:/huggingface-cache"
   ```

2. **Test locally first**: Use 10k sample before processing all 2M
3. **Monitor memory**: Each 1M samples ~1-2 GB in RAM
4. **Keep metadata**: Always save processing timestamp & stats
5. **Version datasets**: Add date to filenames: `tunisian_arabic_clean_2024-04-13.parquet`
6. **Validate samples**: Spot-check cleaned data for quality
7. **Reuse cache**: After first download, datasets are cached and load instantly
8. **Share on HF Hub**: Upload processed datasets for team collaboration

---

## 🚀 Example Complete Workflow

```python
import os

# Step 1: Set cache to external SSD
os.environ['HF_HOME'] = 'E:/huggingface-cache'

# Step 2: Load datasets from HuggingFace
from data_loader import load_all_datasets
df_raw = load_all_datasets(include_linagora=True)
print(f"✅ Loaded {len(df_raw):,} samples")

# Step 3: Clean
from data_cleaning import full_cleaning_pipeline, filter_by_script_type
df_clean = full_cleaning_pipeline(df_raw)
df_arabic = filter_by_script_type(df_clean, "arabic")
print(f"✅ Cleaned to {len(df_arabic):,} samples")

# Step 4: Save locally
df_arabic.to_parquet("../processed/tunisian_arabic_clean.parquet")
print("✅ Saved to processed/")

# Step 5: Upload to HF Hub (for team sharing)
from upload_to_huggingface import upload_dataset
upload_dataset(
    parquet_file="../processed/tunisian_arabic_clean.parquet",
    repo_name="tunisian-dialect-corpus-cleaned",
    username="your_username"
)
print("✅ Uploaded to HuggingFace Hub!")

# Step 6: Load from HF Hub (anywhere, anytime)
import pandas as pd
df_from_hub = pd.read_parquet(
    "hf://datasets/your_username/tunisian-dialect-corpus-cleaned/tunisian_arabic_clean.parquet"
)
print(f"✅ Loaded from HF Hub: {len(df_from_hub):,} samples")
```

---

## Next Steps

After Stage 1 (Data Collection):

→ Go to `llm/pretraining/` for Stage 2 (Continued Pre-training) setup
