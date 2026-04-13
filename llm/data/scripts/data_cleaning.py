"""
Data cleaning utilities for Tunisian dialect text.
Handles normalization, deduplication, filtering, and validation.
"""

import pandas as pd
import re
import unicodedata
from collections import Counter
from typing import Callable


# ============================================================================
# EMOJI & SPECIAL CHARACTER CLEANING
# ============================================================================

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "]+",
    flags=re.UNICODE
)


def remove_emojis(text: str) -> str:
    """Remove all emoji characters from text."""
    return EMOJI_PATTERN.sub('', str(text))


def remove_special_patterns(text: str) -> str:
    """Remove common unwanted patterns: bracketed tags, audio markers, RT, AFP, etc."""
    text = str(text)
    patterns = [
        r"\[[^\]]+\]",          # remove anything inside square brackets, e.g. [موسيقى], [ضحك]
        r"\(\s*ضحك\s*\)",       # remove (ضحك)
        r"\(\s*تصفيق\s*\)",     # remove (تصفيق)
        r"\(\s*صوت\s*\)",       # remove (صوت)
        r"[♪♫]+",                  # remove music note symbols
        r"^RT\s+",               # remove RT at start
        r"\bAFP\b",              # remove AFP word
        r"<s>\[INST\]",          # remove instruction markers
        r"\[/INST\]",
        r"</s>",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize multiple spaces to single space and strip."""
    return re.sub(r'\s+', ' ', str(text)).strip()


def remove_non_arabic_symbols(text: str) -> str:
    """
    Remove symbols except Arabic letters, numbers, and spaces.
    Keeps Arabic script (U+0600-U+06FF) and common punctuation.
    """
    # Keep: Arabic letters, numbers, spaces, and basic punctuation
    text = re.sub(r'[^\w\s\u0600-\u06FF\.،؛:\-!؟()""\'`]', '', str(text))
    return text


def clean_text(text: str, remove_symbols: bool = True) -> str:
    """
    Comprehensive text cleaning pipeline.
    
    Args:
        text: Input text to clean
        remove_symbols: If True, removes non-Arabic symbols
    
    Returns:
        Cleaned text
    """
    text = str(text)
    text = remove_emojis(text)
    text = remove_special_patterns(text)
    if remove_symbols:
        text = remove_non_arabic_symbols(text)
    text = normalize_whitespace(text)
    return text


# ============================================================================
# TEXT ANALYSIS
# ============================================================================

def analyze_text_lengths(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Add character length and word count columns."""
    df = df.copy()
    df["char_length"] = df[text_col].astype(str).apply(len)
    df["word_count"] = df[text_col].astype(str).apply(lambda x: len(x.split()))
    return df


def has_french_letters(text: str) -> bool:
    """Check if text contains any Latin-script letters (including accented French chars)."""
    for ch in str(text):
        if not ch.isalpha():
            continue
        if "LATIN" in unicodedata.name(ch, ""):
            return True
    return False


def has_numbers(text: str) -> bool:
    """Check if text contains at least one numeric digit."""
    return bool(re.search(r"\d", str(text)))


def has_hashtag(text: str) -> bool:
    """Check if text contains a hashtag token like #topic."""
    return bool(re.search(r"#\w+", str(text)))


def has_excessive_repeated_chars(text: str, max_repeat: int = 2) -> bool:
    """Return True if any character repeats consecutively more than max_repeat times."""
    # max_repeat=2 means patterns like 'سسس' or 'aaaa' are flagged.
    pattern = rf"(.)\1{{{max_repeat},}}"
    return bool(re.search(pattern, str(text)))


def has_consecutive_repeated_word(text: str) -> bool:
    """Return True if the same word is repeated consecutively 2+ times (e.g., 'سلام سلام')."""
    return bool(re.search(r"\b(\w+)\s+\1\b", str(text), flags=re.IGNORECASE))


def remove_numeric_entries(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Remove rows that contain at least one numeric digit."""
    df = df.copy()
    return df[~df[text_col].astype(str).apply(has_numbers)]


def remove_hashtag_entries(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Remove rows that contain hashtags."""
    df = df.copy()
    return df[~df[text_col].astype(str).apply(has_hashtag)]


def remove_excessive_repeated_char_entries(
    df: pd.DataFrame,
    text_col: str = "text",
    max_repeat: int = 2,
) -> pd.DataFrame:
    """Remove rows containing excessive consecutive repeated characters."""
    df = df.copy()
    return df[~df[text_col].astype(str).apply(lambda x: has_excessive_repeated_chars(x, max_repeat=max_repeat))]


def remove_consecutive_repeated_word_entries(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Remove rows where a word appears consecutively at least twice."""
    df = df.copy()
    return df[~df[text_col].astype(str).apply(has_consecutive_repeated_word)]


def classify_text_type(text: str) -> str:
    """
    Classify text script type:
    - 'arabic': Arabic script only
    - 'french': Latin script only  
    - 'mixed': Both Arabic and Latin
    - 'other': Neither
    """
    text = str(text)
    has_arabic = bool(re.search(r'[\u0600-\u06FF]', text))
    has_latin = has_french_letters(text)
    
    if has_arabic and has_latin:
        return "mixed"
    elif has_arabic:
        return "arabic"
    elif has_latin:
        return "french"
    else:
        return "other"


# ============================================================================
# DATA CLEANING PIPELINE
# ============================================================================

def remove_empty_rows(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Remove rows with empty or whitespace-only text."""
    df = df.copy()
    df = df[df[text_col].astype(str).str.strip() != ""]
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    return df.drop_duplicates()


def _canonical_text_for_dedup(text: str) -> str:
    """Create a normalized key for near-duplicate detection."""
    text = str(text)
    # Remove Arabic diacritics + tatweel
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652\u0670\u0640]", "", text)
    # Normalize common Arabic letter variants
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    # Remove punctuation/symbols for semantic-equivalent matching
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_near_duplicates(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Remove near-duplicate rows based on canonicalized text form."""
    df = df.copy()
    canonical_col = "_canonical_text"
    df[canonical_col] = df[text_col].astype(str).apply(_canonical_text_for_dedup)
    df = df.drop_duplicates(subset=[canonical_col], keep="first")
    return df.drop(columns=[canonical_col])


def remove_short_texts(df: pd.DataFrame, min_length: int = 3, text_col: str = "text") -> pd.DataFrame:
    """Remove texts shorter than min_length characters."""
    df = df.copy()
    return df[df[text_col].astype(str).apply(len) >= min_length]


def remove_small_word_entries(df: pd.DataFrame, text_col: str = "text", min_words: int = 5) -> pd.DataFrame:
    """Remove rows where text has fewer than min_words words (default: keep 5+ words only)."""
    df = df.copy()
    word_counts = df[text_col].astype(str).apply(lambda x: len(x.split()))
    return df[word_counts >= min_words]


def keep_fully_arabic_entries(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Keep only rows classified as Arabic (drop french, mixed, and other)."""
    df = df.copy()
    text_types = df[text_col].astype(str).apply(classify_text_type)
    return df[text_types == "arabic"]


def standardize_text_column(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Rename any text column to 'text' and keep only that column."""
    df = df.copy()
    if text_col != "text":
        if text_col in df.columns:
            df = df[[text_col]].rename(columns={text_col: "text"})
    return df


def full_cleaning_pipeline(
    df: pd.DataFrame,
    text_col: str = "text",
    min_char_length: int = 3,
    remove_symbols: bool = True,
    analyze: bool = True
) -> pd.DataFrame:
    """
    Apply complete cleaning pipeline to dataframe.
    
    Args:
        df: Input dataframe
        text_col: Column name containing text
        min_char_length: Minimum text length after cleaning (default: 3)
        remove_symbols: Whether to remove non-Arabic symbols
        analyze: Whether to add analysis columns (char_length, word_count, type)
    
    Returns:
        Cleaned dataframe with only high-quality texts
    """
    df = df.copy()
    
    # Standardize column
    df = standardize_text_column(df, text_col)
    
    # Clean in stages so we can drop full rows that contain Latin letters.
    # 1) Remove emojis/patterns first.
    df["text"] = df["text"].astype(str)
    df["text"] = df["text"].apply(remove_emojis)
    df["text"] = df["text"].apply(remove_special_patterns)

    # 2) Remove full samples containing any Latin letters (instead of stripping letters).
    df = df[~df["text"].apply(has_french_letters)]

    # 2.1) Remove full samples containing hashtags.
    df = remove_hashtag_entries(df, text_col="text")

    # 3) Continue symbol cleanup and whitespace normalization.
    if remove_symbols:
        df["text"] = df["text"].apply(remove_non_arabic_symbols)
    df["text"] = df["text"].apply(normalize_whitespace)
    
    # Keep only fully Arabic entries (no french, no mixed, no other)
    df = keep_fully_arabic_entries(df)

    # Remove rows that contain numbers
    df = remove_numeric_entries(df)

    # Remove noisy rows with exaggerated repeated characters (e.g., عسسسلامة)
    df = remove_excessive_repeated_char_entries(df, max_repeat=2)

    # Remove rows with duplicated consecutive words (e.g., سلام سلام)
    df = remove_consecutive_repeated_word_entries(df)

    # Remove duplicates and empty
    df = remove_duplicates(df)
    df = remove_near_duplicates(df, text_col="text")
    df = remove_empty_rows(df)
    df = remove_short_texts(df, min_length=min_char_length)
    df = remove_small_word_entries(df, min_words=5)
    
    # Analyze
    if analyze:
        df = analyze_text_lengths(df)
        df["has_french"] = df["text"].apply(has_french_letters)
        df["type"] = df["text"].apply(classify_text_type)
    
    return df.reset_index(drop=True)


def filter_by_script_type(df: pd.DataFrame, script_type: str) -> pd.DataFrame:
    """
    Filter dataframe by text script type.
    
    Args:
        df: Dataframe with 'type' column
        script_type: One of 'arabic', 'french', 'mixed', 'other'
    
    Returns:
        Filtered dataframe
    """
    if "type" not in df.columns:
        raise ValueError("Dataframe must have 'type' column. Run full_cleaning_pipeline first.")
    return df[df["type"] == script_type].reset_index(drop=True)


# ============================================================================
# STATISTICS & REPORTING
# ============================================================================

def get_statistics(df: pd.DataFrame, text_col: str = "text") -> dict:
    """Generate cleaning statistics."""
    return {
        "total_rows": len(df),
        "unique_rows": len(df.drop_duplicates()),
        "empty_rows": (df[text_col].astype(str).str.strip() == "").sum(),
        "rows_with_numbers": df[text_col].astype(str).apply(has_numbers).sum(),
        "avg_char_length": df[text_col].astype(str).apply(len).mean(),
        "avg_word_count": df[text_col].astype(str).apply(lambda x: len(x.split())).mean(),
    }


def get_top_words(df: pd.DataFrame, text_col: str = "text", top_n: int = 100) -> list:
    """Get most frequent words in corpus."""
    all_text = " ".join(df[text_col].astype(str))
    words = all_text.split()
    return Counter(words).most_common(top_n)


def count_tokens(df: pd.DataFrame, tokenizer, text_col: str = "text") -> int:
    """Count total tokens after tokenization."""
    return sum(
        len(tokenizer.encode(text, add_special_tokens=False))
        for text in df[text_col]
    )


# ============================================================================
# REPORTING
# ============================================================================

def print_cleaning_report(df_before: pd.DataFrame, df_after: pd.DataFrame, text_col: str = "text"):
    """Print a before/after cleaning report."""
    print("\n" + "="*60)
    print("DATA CLEANING REPORT")
    print("="*60)
    print(f"Rows before:        {len(df_before):,}")
    print(f"Rows after:         {len(df_after):,}")
    print(f"Rows removed:       {len(df_before) - len(df_after):,}")
    print(f"Retention rate:     {100 * len(df_after) / len(df_before):.1f}%")
    
    if "char_length" in df_after.columns:
        print(f"\nAvg char length:    {df_after['char_length'].mean():.0f}")
        print(f"Avg word count:     {df_after['word_count'].mean():.0f}")
    
    if "type" in df_after.columns:
        print(f"\nText types:")
        for script_type, count in df_after["type"].value_counts().items():
            print(f"  {script_type}: {count:,} ({100*count/len(df_after):.1f}%)")
    
    print("="*60 + "\n")
