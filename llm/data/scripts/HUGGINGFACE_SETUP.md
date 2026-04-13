# HuggingFace Setup for Large Dataset Downloads

## Quick Setup (5 minutes)

The 2M Tunisian dialect samples are hosted on HuggingFace. To efficiently download and cache them, follow this guide.

---

## ⚙️ Step 1: Configure HuggingFace Cache Location

By default, HuggingFace caches downloads to `~/.cache/huggingface/` (your user profile). For a 10-30 GB dataset, **move this to external storage**:

### On Windows PowerShell:

```powershell
# Temporarily (current session only)
$env:HF_HOME = "E:/huggingface-cache"

# Permanently (add to profile)
[Environment]::SetEnvironmentVariable("HF_HOME", "E:/huggingface-cache", "User")
```

### In Python Script:

```python
import os
os.environ['HF_HOME'] = 'E:/huggingface-cache'

# Then import and use
from data_loader import load_all_datasets
df = load_all_datasets()
```

### In Jupyter Notebook (First Cell):

```python
import os
os.environ['HF_HOME'] = 'E:/huggingface-cache'  # Or OneDrive path

# Run this BEFORE importing datasets
from data_loader import load_all_datasets
```

---

## 📥 Step 2: Download Datasets (First Time Only)

The first run will download from HuggingFace. **This may take 30 minutes to 1 hour** depending on internet speed.

```python
from data_loader import print_dataset_info, load_all_datasets

# Show what's available
print_dataset_info()

# Download & load all (cached after first run)
df = load_all_datasets(include_linagora=True)
print(f"Loaded {len(df):,} samples")
```

**Expected output:**
```
✅ Loaded Dialect of Tunisia collection: 2,000,000 rows
✅ Loaded Tunisian-MSA parallel corpus: 1,000,000 rows
✅ Loaded Linagora Tunisian Derja: 3,000,000 rows total
========================================================
TOTAL LOADED: 6,000,000 samples
```

---

## ⚡ Step 3: Subsequent Runs (Fast)

After first download, all data is cached locally. **Subsequent loads are instant**:

```python
import os
os.environ['HF_HOME'] = 'E:/huggingface-cache'
from data_loader import load_all_datasets

df = load_all_datasets()  # Loads from cache in seconds
```

---

## 🎯 Where Data Gets Cached

```
E:/huggingface-cache/
├── datasets/
│   ├── atakaboudi/Dialect_of_Tunisia-Work_Collection/
│   ├── tunis-ai/tunisian-msa-parallel-corpus/
│   └── linagora/Tunisian_Derja_Dataset/
├── version.txt
└── datasets.cfg
```

Each dataset folder contains downloaded parquet files, not re-downloaded if already present.

---

## 💾 Save Downloaded Data for Backup

Once downloaded, save a copy of the full dataset:

```python
import os
import shutil

os.environ['HF_HOME'] = 'E:/huggingface-cache'
from data_loader import load_all_datasets

# Load from cache
df_raw = load_all_datasets(include_linagora=True)

# Save for backup/archival
output_file = 'E:/tunisian-data/hf_2m_raw.parquet'
df_raw.to_parquet(output_file)
print(f"Saved to {output_file}")
```

---

## 🔍 Verify Datasets Available

Before downloading, check what's available:

```python
from data_loader import get_dataset_info

info = get_dataset_info()
for name, details in info.items():
    print(f"\n{name}")
    print(f"  Source: {details['source']}")
    print(f"  Samples: {details['estimated_samples']}")
```

---

## 🚨 Troubleshooting

### "Connection timed out"
HuggingFace might be temporarily unavailable. Retry after a few minutes.

### "Permission denied" on cache folder
Make sure the HF_HOME path is writable:
```powershell
$env:HF_HOME = "E:/huggingface-cache"
# Or check folder permissions
icacls E:\huggingface-cache /grant:r "$env:USERNAME`:F" /T
```

### "Not enough disk space"
Adjust cache location to external drive with more space:
```powershell
$env:HF_HOME = "D:/large_ssd/huggingface-cache"
```

### Datasets not found
Verify internet connection and HuggingFace API availability:
```python
from huggingface_hub import list_repo_files
list_repo_files("atakaboudi/Dialect_of_Tunisia-Work_Collection")
```

---

## 📊 Estimated Download Sizes

| Dataset | Size | Time (Fast Connection) |
|---------|------|------------------------|
| Dialect of Tunisia | 10-15 GB | 10-15 min |
| Linagora Derja (12 configs) | 2-3 GB | 3-5 min |
| Tunisian-MSA Parallel | 1-2 GB | 2-3 min |
| **Total** | **~15 GB** | **~20-30 min** |

On a slow connection, can take 1-2 hours.

---

## 🎯 Recommended Workflow

```python
# hf_setup.py
import os

# Set cache BEFORE any HF imports
os.environ['HF_HOME'] = 'E:/huggingface-cache'

# Then import
from data_loader import load_all_datasets
from data_cleaning import full_cleaning_pipeline

# Download & load (cached after first run)
print("Loading datasets from HuggingFace...")
df = load_all_datasets(include_linagora=True)
print(f"✅ Loaded {len(df):,} samples")

# Clean
print("\nCleaning data...")
df_clean = full_cleaning_pipeline(df, min_char_length=3)
print(f"✅ Cleaned to {len(df_clean):,} samples")

# Save processed
df_clean.to_parquet('data/processed/tunisian_clean.parquet')
print("✅ Saved to data/processed/")
```

---

## 📖 More Info

- [HuggingFace Datasets Documentation](https://huggingface.co/docs/datasets/)
- [HuggingFace Hub API](https://huggingface.co/docs/hub/security-tokens)
- [Dialect of Tunisia Dataset](https://huggingface.co/datasets/atakaboudi/Dialect_of_Tunisia-Work_Collection)
- [Linagora Tunisian Derja](https://huggingface.co/datasets/linagora/Tunisian_Derja_Dataset)
