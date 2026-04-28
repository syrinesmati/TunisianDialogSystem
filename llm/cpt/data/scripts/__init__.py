"""
Tunisian Dialect Data Loading & Cleaning Pipeline

This package provides utilities for:
1. Loading datasets from HuggingFace (2M+ Tunisian samples)
2. Cleaning and normalizing text
3. Analyzing and classifying text
4. Preparing data for model pre-training

Quick start:
    from data_loader import load_all_datasets
    from data_cleaning import full_cleaning_pipeline
    
    df_raw = load_all_datasets()
    df_clean = full_cleaning_pipeline(df_raw)
"""

__version__ = "1.0.0"
__author__ = "Tunisian Dialogue System Team"

# Import main functions for convenience
try:
    from .data_loader import (
        load_all_datasets,
        load_dialect_of_tunisia_collection,
        load_tunisian_msa_parallel_corpus,
        load_linagora_tunisian_derja,
        print_dataset_info,
        get_dataset_info,
    )
except ImportError:
    pass

try:
    from .data_cleaning import (
        full_cleaning_pipeline,
        clean_text,
        classify_text_type,
        filter_by_script_type,
        count_tokens,
        get_top_words,
        print_cleaning_report,
    )
except ImportError:
    pass

__all__ = [
    # Data loading
    "load_all_datasets",
    "load_dialect_of_tunisia_collection",
    "load_tunisian_msa_parallel_corpus",
    "load_linagora_tunisian_derja",
    "print_dataset_info",
    "get_dataset_info",
    # Data cleaning
    "full_cleaning_pipeline",
    "clean_text",
    "classify_text_type",
    "filter_by_script_type",
    "count_tokens",
    "get_top_words",
    "print_cleaning_report",
]
