"""
Data loading utilities for Tunisian dialect datasets from HuggingFace.

All data sources are hosted on HuggingFace Hub. The `load_all_datasets()` function
downloads and caches them locally (you can point to external storage via HF_HOME env var).

Total available: ~6M samples, ~10-30 GB
- Dialect of Tunisia Collection: 2M samples
- Linagora Tunisian Derja: 12 configs, 1-3M samples
- Tunisian-MSA Parallel Corpus: 1M samples
- Tunisiya Corpus: 1.18M samples
"""

from datasets import load_dataset, concatenate_datasets
import pandas as pd
import json
from typing import List, Dict, Union, Optional


# ============================================================================
# INDIVIDUAL DATASET LOADERS
# ============================================================================

def load_dialect_of_tunisia_collection() -> pd.DataFrame:
    """
    Load 'Dialect of Tunisia - Work Collection' dataset.
    atakaboudi/Dialect_of_Tunisia-Work_Collection
    """
    try:
        ds = load_dataset("atakaboudi/Dialect_of_Tunisia-Work_Collection", "default")
        all_rows = []
        
        for row in ds['train']:
            data_field = row.get("data") or row.get("dataset") or row
            
            try:
                if isinstance(data_field, str):
                    data_field = json.loads(data_field)
                
                if isinstance(data_field, list):
                    all_rows.extend(data_field)
                elif isinstance(data_field, dict):
                    all_rows.append(data_field)
            except:
                continue
        
        df = pd.DataFrame(all_rows)
        if "text" not in df.columns:
            df.columns = ["text"]
        
        print(f"✅ Loaded Dialect of Tunisia collection: {len(df):,} rows")
        return df
    
    except Exception as e:
        print(f"❌ Failed to load Dialect of Tunisia: {e}")
        return pd.DataFrame()


def load_tunisian_msa_parallel_corpus() -> pd.DataFrame:
    """
    Load 'Tunisian-MSA Parallel Corpus' dataset.
    tunis-ai/tunisian-msa-parallel-corpus
    """
    try:
        ds = load_dataset("tunis-ai/tunisian-msa-parallel-corpus", "default")
        df = ds["part_0"].to_pandas()
        
        if "chunk_text" in df.columns:
            df = df[["chunk_text"]].rename(columns={"chunk_text": "text"})
        
        print(f"✅ Loaded Tunisian-MSA parallel corpus: {len(df):,} rows")
        return df
    
    except Exception as e:
        print(f"❌ Failed to load Tunisian-MSA corpus: {e}")
        return pd.DataFrame()


# ============================================================================
# LINAGORA TUNISIAN DERJA DATASET
# ============================================================================

LINAGORA_CONFIGS = [
    "Derja_tunsi",
    "MADAR_TunisianDialect",
    "QADI_TunisianDialect",
    "HkayetErwi",
    "Sentiment_Derja",
    "TA_Segmentation",
    "TSAC",
    "TuDiCOI",
    "TunSwitchTunisiaOnly",
    "TunisianSentimentAnalysis",
    "Tunisian_Dialectic_English_Derja",
    "Tweet_TN"
]


def extract_text_from_dataset(ds) -> pd.DataFrame:
    """
    Extract text from HuggingFace dataset with unknown column names.
    Tries common text column names.
    """
    merged = concatenate_datasets([ds[split] for split in ds.keys()])
    df = merged.to_pandas()
    
    possible_cols = [
        "text", "sentence", "chunk_text",
        "content", "tweet", "utterance",
        "translation", "src", "tgt"
    ]
    
    for col in possible_cols:
        if col in df.columns:
            return df[[col]].rename(columns={col: "text"})
    
    # Fallback: use first column
    return df.iloc[:, [0]].rename(columns={df.columns[0]: "text"})


def load_linagora_tunisian_derja(
    configs: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load all configs from Linagora Tunisian Derja Dataset.
    
    Args:
        configs: List of configs to load. If None, loads all available.
        verbose: Print status for each config.
    
    Returns:
        Concatenated dataframe from all loaded configs.
    """
    if configs is None:
        configs = LINAGORA_CONFIGS
    
    dfs = []
    
    for config in configs:
        try:
            ds = load_dataset("linagora/Tunisian_Derja_Dataset", config)
            df_temp = extract_text_from_dataset(ds)
            dfs.append(df_temp)
            if verbose:
                print(f"  ✅ {config}: {len(df_temp):,} rows")
        except Exception as e:
            if verbose:
                print(f"  ❌ {config}: {str(e)[:50]}")
    
    if dfs:
        result = pd.concat(dfs, ignore_index=True)
        print(f"✅ Loaded Linagora Tunisian Derja: {len(result):,} rows total")
        return result
    else:
        print(f"❌ No Linagora configs loaded successfully")
        return pd.DataFrame()


# ============================================================================
# COMBINED LOADERS
# ============================================================================

def load_all_datasets(
    include_linagora: bool = True,
    linagora_configs: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load ALL available Tunisian dialect datasets.
    
    Args:
        include_linagora: Whether to load Linagora configs
        linagora_configs: Specific Linagora configs to load
    
    Returns:
        Combined dataframe from all sources
    """
    print("\n" + "="*60)
    print("LOADING TUNISIAN DIALECT DATASETS")
    print("="*60 + "\n")
    
    dfs = []
    
    # Load main datasets
    df1 = load_dialect_of_tunisia_collection()
    if not df1.empty:
        dfs.append(df1)
    
    df2 = load_tunisian_msa_parallel_corpus()
    if not df2.empty:
        dfs.append(df2)
    
    # Load Linagora
    if include_linagora:
        print("\nLoading Linagora Tunisian Derja Dataset configs:")
        df_linagora = load_linagora_tunisian_derja(configs=linagora_configs, verbose=True)
        if not df_linagora.empty:
            dfs.append(df_linagora)
    
    # Combine
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        print(f"\n{'='*60}")
        print(f"TOTAL LOADED: {len(combined):,} samples")
        print(f"{'='*60}\n")
        return combined
    else:
        print("❌ No datasets loaded successfully")
        return pd.DataFrame()


