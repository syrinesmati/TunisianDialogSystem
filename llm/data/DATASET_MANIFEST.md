# Tunisian Dialect Dataset Manifest

## Overview
This directory manages all data for the Tunisian Dialogue System LLM training pipeline.

---

## Dataset Storage Strategy

Since your raw dataset is **~2 million samples from HuggingFace** (estimated 10-30 GB), we use a **hybrid approach**:

### 📁 Directory Structure

```
data/
├── raw/                      # Small samples & indices only
│   ├── hf_2m_metadata.json          # Index of HuggingFace dataset
│   ├── hf_2m_sample.parquet         # ~10k sample (dev/testing)
│   ├── sources.json                  # Data source registry
│   └── DATASET_MANIFEST.md          # This file
│
├── processed/                # Cleaned, ready for training
│   ├── tunisian_arabic_clean.parquet     # Primary dataset (Arabic-only)
│   ├── tunisian_mixed_clean.parquet      # With French code-switching
│   ├── metadata.json                      # Collection stats & timestamp
│   └── README.md                          # Processing notes
│
└── scripts/                  # Data utilities
    ├── data_loader.py                # Load datasets from Kaggle/HF
    ├── data_cleaning.py              # Cleaning functions
    ├── upload_to_huggingface.py      # Upload to HF Hub
    ├── tunisian-continuous...ipynb   # Collection workflow
    ├── HUGGINGFACE_SETUP.md          # Download configuration
    ├── HUGGINGFACE_UPLOAD.md         # Upload to HF Hub guide
    └── README.md                      # Usage instructions
```

---

## Data Sources

| Source | Type | Approx Size | Samples | Location |
|--------|------|-----------|---------|----------|
| **Dialect of Tunisia Collection** | Raw text | 10-15 GB | 2M | HuggingFace |
| **Linagora Tunisian Derja (12 configs)** | Multiple splits | 2-3 GB | 1-3M | HuggingFace |
| **Tunisian-MSA Parallel Corpus** | Parallel text | 1-2 GB | 1M | HuggingFace |
| **Tunisiya Corpus** | LGP text | ~300 MB | 1.18M | HuggingFace |
| **Total** | — | **10-30 GB** | **~5-6M** | All HuggingFace |

---

## Storage Recommendations

### Option 1: External SSD (Recommended for 2M samples)
```
External SSD: E:/tunisian-data/
├── raw/
│   ├── hf_2m_raw.parquet              # 10-15 GB (full HF dataset)
│   └── metadata.json                   # File index & stats
└── temp/
    └── [intermediate processing files]
```

**Why**: Fast access, keeps laptop storage clean, easily portable. Caches HF downloads.

### Option 2: Cloud Storage (OneDrive/Google Drive)
```
OneDrive/Bureau/PFA/tunisian-data/
├── hf_2m_raw.parquet                 # Downloaded from HF (auto-sync)
└── metadata.json
```

**Why**: Automatic backup, accessible from anywhere, collaborative. Set up HF cache to cloud folder.

### Option 3: HuggingFace Hub (Recommended for Sharing) ⭐
```
https://huggingface.co/datasets/your_username/tunisian-dialect-corpus-cleaned

Stored on HF:
├── tunisian_arabic_clean.parquet   # 3-5 GB processed data
├── metadata.json                    # Dataset statistics
└── README.md                        # Documentation
```

**Why**: 
- Version control & history
- Shareable with team & research community
- Automatic backups
- Integrated with Hugging Face ecosystem
- Easy collaboration
- Free storage

### Option 3: HuggingFace Datasets Hub
```
huggingface.co/datasets/[username]/tunisian-dialect-corpus
```

**Why**: Version control, shareable, integrated with transformers library.

---

## Usage Guide

### Load Data (From Notebook)

```python
import pandas as pd
from pathlib import Path

# Raw collection (downloaded from HuggingFace)
raw_path = Path("../../../path/to/external/ssd/hf_2m_raw.parquet")
df_raw = pd.read_parquet(raw_path)

# Or load directly from HuggingFace (cached)
from data_loader import load_all_datasets
df_raw = load_all_datasets(include_linagora=True)

# Processed data (local, cleaned)
processed_path = Path("../processed/tunisian_arabic_clean.parquet")
df_processed = pd.read_parquet(processed_path)
```

### Check Metadata

```python
import json

with open("raw/hf_2m_metadata.json") as f:
    metadata = json.load(f)
    print(f"Total samples: {metadata['total_samples']}")
    print(f"Downloaded: {metadata['download_date']}")
    print(f"Sources: {metadata['sources']}")
```

---

## File Formats

- **`.parquet`**: Columnar format, fast for Pandas, good compression
  - Use for: Processed datasets, training data
  
- **`.csv`**: Human-readable, good for Excel
  - Use for: Small samples, manual inspection
  
- **`.json`**: Metadata, configuration
  - Use for: Indices, manifests, statistics

---

## Uploading Processed Data to HuggingFace Hub

Once you have cleaned data, **upload to HuggingFace Hub** for:
✅ Version control  
✅ Shareable with team  
✅ Research reproducibility  
✅ Automatic backups  
✅ Community contributions  

**See [HUGGINGFACE_UPLOAD.md](scripts/HUGGINGFACE_UPLOAD.md) for complete guide.**

**Quick upload:**
```python
from scripts.upload_to_huggingface import upload_dataset

upload_dataset(
    parquet_file="processed/tunisian_arabic_clean.parquet",
    repo_name="tunisian-dialect-corpus-cleaned",
    username="your_username"
)
```

**Or via command line:**
```bash
python scripts/upload_to_huggingface.py \
  --parquet_file processed/tunisian_arabic_clean.parquet \
  --repo_name tunisian-dialect-corpus-cleaned \
  --username your_username
```

---

## Data Pipeline Status

| Stage | Status | Output | Size |
|-------|--------|--------|------
| 1️⃣ Collection | In Progress | `raw/hf_2m_*.parquet` | ~10-15 GB |
| 2️⃣ Cleaning | Ready | `processed/tunisian_arabic_clean.parquet` | 3-5 GB |
| 🤗 Upload to HF | Next | `huggingface.co/datasets/your_username/...` | Auto-versioned |
| 3️⃣ Pre-training | Next | `pretraining/checkpoint/*` | TBD |
| 4️⃣ RAG Setup | Planned | `rag/indexes/*` | 100 MB |
| 5️⃣ Synthetic Q/A | Planned | `synthetic_data/generated/*` | — |
| 6️⃣ SFT | Planned | `sft/checkpoint/*` | — |

---

## Important Notes

### Do NOT Commit Large Files to Git
Add to `.gitignore`:
```
# Data files
data/raw/*.parquet
data/raw/*.csv
data/processed/*.parquet

# HuggingFace cache
.cache/
huggingface/

# Training checkpoints
*/checkpoints/*
*.bin
*.safetensors
```

### Managing 2M Samples

- **Development**: Use a 10-50k sample subset locally
- **Full training**: Store on external SSD or cloud
- **Backup**: Keep metadata + important files in git, raw data externally

### Token Estimation

For **2M Arabic text samples** with avg ~50 words each:
- ~100M tokens (relatively small pre-training corpus)
- With multilingual model: 50-100 tokens per sample with special chars

This is still valuable for a low-resource dialect despite size.

---

## Next Steps

1. **Set HuggingFace cache** → Point to external SSD: `$env:HF_HOME = "E:/hf-cache"`
2. **Run collection notebook** → Downloads from HF (auto-cached)
3. **Run cleaning pipeline** → `full_cleaning_pipeline()` in notebook
4. **Check processed output** → `processed/tunisian_arabic_clean.parquet`
5. **Upload to HF Hub** → Make it shareable & version controlled
6. **Update this manifest** with actual paths and sizes
7. **Move to Stage 2** → Pre-training setup

---

*Last updated: [Timestamp from metadata.json]*
