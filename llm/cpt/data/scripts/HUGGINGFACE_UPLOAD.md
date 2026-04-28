# Uploading Processed Data to HuggingFace Hub

## Why Upload to HuggingFace?

✅ **Benefits:**
- **Version control**: Track all iterations of your processed dataset
- **Shareable**: Easy for team members to download
- **Reproducible**: Other researchers can use your clean data
- **Backed up**: Automatic versioning and history
- **Integrated**: Works seamlessly with the Hugging Face ecosystem
- **Citation**: Get a DOI for academic citations

---

## Step 1: Create a HuggingFace Account (Free)

1. Go to [huggingface.co](https://huggingface.co)
2. Sign up with email or GitHub
3. Create API token: Settings → Access Tokens → New token
4. Copy the token (you'll need it)

---

## Step 2: Authenticate Locally

### Option A: Via Command Line (Recommended)

```powershell
pip install huggingface-hub
huggingface-cli login
# Paste your token when prompted
```

### Option B: Via Environment Variable

```powershell
$env:HF_TOKEN = "your_token_here"
python your_upload_script.py
```

### Option C: In Python Script

```python
from huggingface_hub import login
login(token="your_token_here")
```

---

## Step 3: Create a Dataset Repository on HuggingFace

Visit: https://huggingface.co/new-dataset

**Example setup:**
- **Name**: `tunisian-dialect-corpus-cleaned`
- **Organization**: Your username or organization
- **Description**: "Cleaned Tunisian dialect text (2M+ samples) - Arabic script only, deduplicated, emoji/symbol removed"
- **License**: CC-BY-4.0 (allows academic use with attribution)
- **Make private** (optional): If sensitive, or public for community

---

## Step 4A: Upload via Python Script (Easiest)

Use the provided `upload_to_huggingface.py`:

```python
from upload_to_huggingface import upload_dataset

# Upload your processed dataset
upload_dataset(
    parquet_file="../processed/tunisian_arabic_clean.parquet",
    repo_name="tunisian-dialect-corpus-cleaned",
    username="your_username",  # From HF settings
    description="Cleaned Tunisian dialect corpus (Arabic-only, 2M samples)",
    make_private=False,  # Set to True if you want to keep it private
    token="your_hf_token"  # Or use login() first, then omit this
)
```

---

## Step 4B: Upload via HuggingFace Hub API (Full Control)

```python
from huggingface_hub import HfApi, login

# Authenticate
login()

api = HfApi()

# Create repository (if doesn't exist)
api.create_repo(
    repo_id="tunisian-dialect-corpus-cleaned",
    repo_type="dataset",
    exist_ok=True
)

# Upload file
api.upload_file(
    path_or_fileobj="../processed/tunisian_arabic_clean.parquet",
    path_in_repo="tunisian_arabic_clean.parquet",
    repo_id="tunisian-dialect-corpus-cleaned",
    repo_type="dataset"
)

print("✅ Dataset uploaded!")
```

---

## Step 4C: Upload via Web Interface

1. Go to your dataset repo on huggingface.co
2. Click "Files and Versions"
3. Click "Upload file" (top right)
4. Drag & drop your `.parquet` file
5. Add a commit message (e.g., "Add cleaned Arabic-only dataset")
6. Click Commit

---

## Step 5: Add Dataset Card (README)

Create `README.md` in your dataset repo:

```markdown
# Tunisian Dialect Corpus - Cleaned

High-quality, cleaned Tunisian Arabic text data.

## Dataset Details

- **Language**: Tunisian Arabic (Darija)
- **Size**: 2M+ samples (~12-15 GB raw, 3-5 GB Parquet)
- **Source**: Combined from:
  - Dialect of Tunisia collection
  - Linagora Tunisian Derja Dataset
  - Tunisian-MSA Parallel Corpus
  - Tunisiya Corpus

## Cleaning Applied

✅ Removed duplicates  
✅ Removed emojis and special symbols  
✅ Normalized whitespace  
✅ Filtered to Arabic script only  
✅ Removed texts <3 characters  
✅ Removed MSA and French-only content

## Usage

```python
import pandas as pd

# Load from HuggingFace Hub
df = pd.read_parquet("hf://datasets/your_username/tunisian-dialect-corpus-cleaned/tunisian_arabic_clean.parquet")

# Or from local cache
df = pd.read_parquet("tunisian_arabic_clean.parquet")
```

## Scripts Used

All cleaning done with [data_cleaning.py](../llm/data/scripts/data_cleaning.py) from the Tunisian Dialogue System project.

## License

CC-BY-4.0

## Citation

```bibtex
@dataset{
  title={Tunisian Dialect Corpus - Cleaned},
  author={Your Name},
  year={2024},
  url={https://huggingface.co/datasets/your_username/tunisian-dialect-corpus-cleaned}
}
```
```

---

## Step 6: Update Your Processing Pipeline

In your notebook, add final upload step:

```python
# Save to local cache
df_clean.to_parquet("../processed/tunisian_arabic_clean.parquet")

# Upload to HuggingFace
from huggingface_hub import HfApi, login

login()  # Authenticate

api = HfApi()
api.upload_file(
    path_or_fileobj="../processed/tunisian_arabic_clean.parquet",
    path_in_repo="tunisian_arabic_clean.parquet",
    repo_id="your_username/tunisian-dialect-corpus-cleaned",
    repo_type="dataset",
    commit_message="Add cleaned Arabic-only dataset (2M samples)"
)

print("✅ Uploaded to HuggingFace Hub!")
```

---

## Loading Processed Data (After Upload)

**For you or team members:**

```python
import pandas as pd

# Option 1: Load from HF Hub (auto-cached)
df = pd.read_parquet("hf://datasets/your_username/tunisian-dialect-corpus-cleaned/tunisian_arabic_clean.parquet")

# Option 2: Using HF datasets library
from datasets import load_dataset
dataset = load_dataset("your_username/tunisian-dialect-corpus-cleaned")
df = dataset['train'].to_pandas()
```

---

## Multiple Dataset Versions

Upload different versions:

```python
api.upload_file(
    path_or_fileobj="../processed/tunisian_arabic_clean.parquet",
    path_in_repo="versions/v1_arabic_only.parquet",  # Nested path
    repo_id="your_username/tunisian-dialect-corpus-cleaned",
    repo_type="dataset",
    commit_message="v1: Arabic-only, 2M samples, cleaned"
)

api.upload_file(
    path_or_fileobj="../processed/tunisian_mixed_clean.parquet",
    path_in_repo="versions/v1_mixed_code_switching.parquet",
    repo_id="your_username/tunisian-dialect-corpus-cleaned",
    repo_type="dataset",
    commit_message="v1: Arabic+French (code-switching), keep language diversity"
)
```

Then load specific version:
```python
df = pd.read_parquet("hf://datasets/your_username/tunisian-dialect-corpus-cleaned/versions/v1_arabic_only.parquet")
```

---

## Metadata File

Upload a `metadata.json` alongside your data:

```python
import json
from huggingface_hub import HfApi, login

metadata = {
    "dataset_name": "Tunisian Dialect Corpus - Cleaned",
    "version": "1.0",
    "total_samples": 1_600_000,
    "total_tokens": 60_000_000,
    "language": "ar-TN (Tunisian Arabic)",
    "script": "Arabic (U+0600-U+06FF)",
    "cleaned_at": "2024-04-13T00:00:00Z",
    "sources": [
        "atakaboudi/Dialect_of_Tunisia-Work_Collection",
        "linagora/Tunisian_Derja_Dataset",
        "tunis-ai/tunisian-msa-parallel-corpus",
    ],
    "cleaning_steps": [
        "deduplication",
        "emoji_removal",
        "symbol_removal",
        "whitespace_normalization",
        "min_length_filter",
        "script_type_filtering"
    ],
    "avg_tokens_per_sample": 37,
    "file_size_parquet_mb": 4500,
}

with open("metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# Upload metadata
login()
api = HfApi()
api.upload_file(
    path_or_fileobj="metadata.json",
    path_in_repo="metadata.json",
    repo_id="your_username/tunisian-dialect-corpus-cleaned",
    repo_type="dataset"
)
```

---

## Private vs Public Dataset

**Public** (default):
- Anyone can find and use your dataset
- Good for sharing with research community
- Helps low-resource dialect work

**Private**:
- Only you and people you add can access
- Good for proprietary data
- Can make public later

To keep private during initial upload:
```python
api.create_repo(
    repo_id="tunisian-dialect-corpus-cleaned",
    repo_type="dataset",
    private=True,  # Private by default
    exist_ok=True
)
```

---

## Next Steps

1. **Set up HF account** and get API token
2. **Authenticate locally**: `huggingface-cli login`
3. **Create dataset repo** on huggingface.co
4. **Upload processed data**:
   - Via Python script (easiest)
   - Via web interface (simplest)
5. **Add README** with documentation
6. **Share** the link with your team/community

---

## Example Complete Workflow

```python
# Final cell in notebook
import os
from pathlib import Path
from huggingface_hub import HfApi, login
import json

print("🚀 UPLOADING TO HUGGINGFACE HUB")
print("="*60)

# Authenticate
try:
    login()
    print("✅ Authenticated with HuggingFace")
except:
    print("⚠️  Could not authenticate. Run: huggingface-cli login")
    exit()

# Upload processed dataset
api = HfApi()
api.create_repo(
    repo_id="tunisian-dialect-corpus-cleaned",
    repo_type="dataset",
    exist_ok=True
)

# Upload main parquet file
api.upload_file(
    path_or_fileobj="../processed/tunisian_arabic_clean.parquet",
    path_in_repo="tunisian_arabic_clean.parquet",
    repo_id="tunisian-dialect-corpus-cleaned",
    repo_type="dataset",
    commit_message="Add cleaned Tunisian corpus (2M samples, Arabic-only)"
)
print("✅ Uploaded: tunisian_arabic_clean.parquet")

# Upload metadata
metadata = {
    "total_samples": len(df_clean),
    "cleaned_date": pd.Timestamp.now().isoformat(),
    "sources": ["HuggingFace Hub datasets"],
    "script_type": "Arabic only",
}
with open("metadata.json", "w") as f:
    json.dump(metadata, f)

api.upload_file(
    path_or_fileobj="metadata.json",
    path_in_repo="metadata.json",
    repo_id="tunisian-dialect-corpus-cleaned",
    repo_type="dataset"
)
print("✅ Uploaded: metadata.json")

print("\n🎉 UPLOAD COMPLETE!")
print("📍 Visit: https://huggingface.co/datasets/your_username/tunisian-dialect-corpus-cleaned")
print("="*60)
```

---

## Resources

- **[HuggingFace Hub Documentation](https://huggingface.co/docs/hub/)**
- **[HuggingFace Datasets Guide](https://huggingface.co/docs/datasets/)**
- **[Upload to Hub](https://huggingface.co/docs/hub/security-tokens)**
- **[Dataset Cards](https://huggingface.co/docs/hub/datasets-cards)**
