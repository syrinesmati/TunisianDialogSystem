"""
audit_text.py
─────────────
Text-level audit utilities for the Tunisian ASR dataset.

All functions operate on plain Python sequences of strings so they work
whether the caller passes ``train_dataset["transcript"]`` directly or a
list extracted from the exploded segments DataFrame.

Key responsibilities
--------------------
- Character / word statistics and vocabulary analysis.
- Arabic normalisation (light, audit-only — not the CODA normaliser).
- Code-switching detection and visualisation.
- Per-segment feature enrichment for downstream DataFrames.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ── Compiled patterns ─────────────────────────────────────────────────────────

ARABIC_DIACRITICS = re.compile(r"[ًٌٍَُِّْ]")
ARABIC_ALEF       = re.compile(r"[إأآ]")
LATIN_PATTERN     = re.compile(r"[A-Za-z]")
ARABIC_CHAR_RANGE = re.compile(r"[\u0600-\u06FF]")

# ── Character category sets ───────────────────────────────────────────────────

ARABIC_LETTERS  = set("ابتثجحخدذرزسشصضطظعغفقكلمنهويءأإآةىؤئ")
ARABIC_EXTENDED = set("ڨڤپ")           # Maghrebi / Tunisian-specific letters
ARABIC_DIAC_SET = set("ًٌٍَُِّْ")
ARABIC_PUNCT    = set("،؟؛«»")
DIGITS_AR       = set("٠١٢٣٤٥٦٧٨٩")
DIGITS_LATIN    = set("0123456789")
LATIN_LETTERS   = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

# ── Visualisation colour palette ──────────────────────────────────────────────

CATEGORY_COLORS: dict[str, str] = {
    "arabic_letter":    "#2979FF",
    "arabic_extended":  "#000000",
    "arabic_diacritic": "#FF3D00",
    "arabic_punct":     "#FF9100",
    "digit_arabic":     "#00C853",
    "digit_latin":      "#AEEA00",
    "latin_letter":     "#D500F9",
    "punct_other":      "#00E5FF",
    "whitespace":       "#9E9E9E",
    "other":            "#FFD600",
}


# ── Character classification ──────────────────────────────────────────────────

def classify_char(c: str) -> str:
    """
    Assign a single character to one of ten display categories.

    Categories (in priority order): ``arabic_letter``, ``arabic_extended``,
    ``arabic_diacritic``, ``arabic_punct``, ``digit_arabic``, ``digit_latin``,
    ``latin_letter``, ``whitespace``, ``punct_other``, ``other``.

    Parameters
    ----------
    c : str
        A single Unicode character.

    Returns
    -------
    str
        Category label string.
    """
    if c in ARABIC_LETTERS:   return "arabic_letter"
    if c in ARABIC_EXTENDED:  return "arabic_extended"
    if c in ARABIC_DIAC_SET:  return "arabic_diacritic"
    if c in ARABIC_PUNCT:     return "arabic_punct"
    if c in DIGITS_AR:        return "digit_arabic"
    if c in DIGITS_LATIN:     return "digit_latin"
    if c in LATIN_LETTERS:    return "latin_letter"
    if c in (" ", "\t"):      return "whitespace"
    cat = unicodedata.category(c)
    if cat.startswith("P"):   return "punct_other"
    if cat.startswith("Z"):   return "whitespace"
    return "other"


# ── Arabic normalisation (audit-only, light pass) ─────────────────────────────

def normalize_arabic(text: str) -> str:
    """
    Apply a light Arabic normalisation pass for audit/comparison purposes.

    Transformations applied
    -----------------------
    - Strip all Arabic diacritics (harakat).
    - Unify Alef variants (أ إ آ) → bare Alef (ا).
    - Alef Maqsura (ى) → Ya (ي).
    - Waw with Hamza (ؤ) → Waw (و).
    - Ya with Hamza (ئ) → Ya (ي).
    - Taa Marbuta (ة) → Ha (ه).

    Note: this is the audit-stage normaliser used to measure vocabulary
    reduction. The full CODA-compliant normaliser lives in
    ``src/cleaning.py`` and ``src/coda_normalizer.py``.

    Parameters
    ----------
    text : str
        Raw transcript string.

    Returns
    -------
    str
        Lightly normalised string.
    """
    text = ARABIC_DIACRITICS.sub("", text)
    text = ARABIC_ALEF.sub("ا", text)
    text = re.sub("ى",  "ي", text)
    text = re.sub("ؤ",  "و", text)
    text = re.sub("ئ",  "ي", text)
    text = re.sub("ة",  "ه", text)
    return text


# ── Basic statistics ──────────────────────────────────────────────────────────

def get_basic_text_stats(texts: Sequence[str]) -> dict:
    """
    Compute character- and word-level statistics for a list of transcripts.

    Metrics returned
    ----------------
    ``n_samples``, ``total_chars``, ``avg_chars``, ``median_chars``,
    ``max_chars``, ``total_words``, ``avg_words``, ``median_words``,
    ``max_words``, ``empty_count``, ``diacritic_count``,
    ``diacritic_ratio``, ``has_latin_pct``, ``arabic_words``,
    ``arabic_words_pct``, ``latin_words``, ``latin_words_pct``.

    Parameters
    ----------
    texts : Sequence[str]
        Transcript strings (train or test split).

    Returns
    -------
    dict
        Flat mapping of metric name → value.
    """
    char_lens = np.array([len(t) for t in texts])
    word_lens = np.array([len(t.split()) for t in texts])

    diacritic_count = sum(len(ARABIC_DIACRITICS.findall(t)) for t in texts)
    total_chars     = int(char_lens.sum())

    has_latin_mask = np.array([bool(LATIN_PATTERN.search(t)) for t in texts])
    has_latin_pct  = round(float(has_latin_mask.mean()) * 100, 2)

    all_words    = [w for t in texts for w in t.split()]
    latin_words  = sum(1 for w in all_words if LATIN_PATTERN.search(w))
    arabic_words = sum(1 for w in all_words if ARABIC_CHAR_RANGE.search(w))
    total_words  = len(all_words)

    return {
        "n_samples":        len(texts),
        "total_chars":      total_chars,
        "avg_chars":        round(float(char_lens.mean()),        2),
        "median_chars":     round(float(np.median(char_lens)),    2),
        "max_chars":        int(char_lens.max()),
        "total_words":      total_words,
        "avg_words":        round(float(word_lens.mean()),        2),
        "median_words":     round(float(np.median(word_lens)),    2),
        "max_words":        int(word_lens.max()),
        "empty_count":      int((char_lens == 0).sum()),
        "diacritic_count":  diacritic_count,
        "diacritic_ratio":  round(diacritic_count / max(total_chars, 1), 4),
        "has_latin_pct":    has_latin_pct,
        "arabic_words":     arabic_words,
        "arabic_words_pct": round(arabic_words  / max(total_words, 1) * 100, 2),
        "latin_words":      latin_words,
        "latin_words_pct":  round(latin_words   / max(total_words, 1) * 100, 2),
    }


# ── Vocabulary ────────────────────────────────────────────────────────────────

def get_char_vocab(texts: Sequence[str]) -> tuple[set, Counter]:
    """
    Build the character-level vocabulary for a list of transcripts.

    Parameters
    ----------
    texts : Sequence[str]
        Transcript strings.

    Returns
    -------
    tuple[set, Counter]
        ``(char_set, char_counter)`` where *char_set* contains every unique
        character and *char_counter* maps each character to its total count.
    """
    counter: Counter = Counter()
    for t in texts:
        counter.update(t)
    return set(counter), counter


def get_word_vocab(texts: Sequence[str]) -> Counter:
    """
    Build the word-level vocabulary (unigram counts) for a list of transcripts.

    Parameters
    ----------
    texts : Sequence[str]
        Transcript strings.

    Returns
    -------
    Counter
        Mapping of word → occurrence count across all *texts*.
    """
    counter: Counter = Counter()
    for t in texts:
        counter.update(t.split())
    return counter


def vocab_comparison(texts: Sequence[str]) -> dict:
    """
    Measure vocabulary reduction from light Arabic normalisation.

    Parameters
    ----------
    texts : Sequence[str]
        Raw transcript strings.

    Returns
    -------
    dict
        Keys: ``raw_char_vocab_size``, ``normalized_char_vocab_size``,
        ``reduction``.
    """
    raw_vocab, _  = get_char_vocab(texts)
    norm_texts    = [normalize_arabic(t) for t in texts]
    norm_vocab, _ = get_char_vocab(norm_texts)
    return {
        "raw_char_vocab_size":        len(raw_vocab),
        "normalized_char_vocab_size": len(norm_vocab),
        "reduction":                  len(raw_vocab) - len(norm_vocab),
    }


def categorize_vocab(char_set: set) -> dict[str, list[str]]:
    """
    Group the full character vocabulary into labelled category buckets.

    Parameters
    ----------
    char_set : set
        Set of unique characters (typically from :func:`get_char_vocab`).

    Returns
    -------
    dict[str, list[str]]
        ``{category_name: sorted_char_list}`` for each of the ten categories
        defined by :func:`classify_char`.
    """
    buckets: dict[str, list] = defaultdict(list)
    for c in char_set:
        buckets[classify_char(c)].append(c)
    return dict(buckets)


# ── Code-switching ─────────────────────────────────────────────────────────────

def code_switch_ratio(texts: Sequence[str]) -> float:
    """
    Compute the fraction of *segments* that contain at least one Latin character.

    Parameters
    ----------
    texts : Sequence[str]
        Transcript strings.

    Returns
    -------
    float
        Value in ``[0.0, 1.0]``.
    """
    switched = sum(1 for t in texts if LATIN_PATTERN.search(t))
    return switched / len(texts) if texts else 0.0


def code_switch_word_ratio(texts: Sequence[str]) -> float:
    """
    Compute the fraction of *words* (not segments) containing Latin characters.

    Parameters
    ----------
    texts : Sequence[str]
        Transcript strings.

    Returns
    -------
    float
        Value in ``[0.0, 1.0]``.
    """
    all_words   = [w for t in texts for w in t.split()]
    latin_words = sum(1 for w in all_words if LATIN_PATTERN.search(w))
    return latin_words / len(all_words) if all_words else 0.0


# ── Per-segment feature enrichment ────────────────────────────────────────────

def enrich_text_features(df):
    """
    Add text-derived feature columns to a segment-level DataFrame.

    Columns added
    -------------
    ``char_count``             : number of characters in ``transcript``
    ``word_count``             : number of whitespace-split words
    ``has_latin``              : bool — contains at least one Latin character
    ``has_diacritics``         : bool — contains at least one Arabic diacritic

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a ``'transcript'`` string column.

    Returns
    -------
    pandas.DataFrame
        Same DataFrame with new columns appended (in-place).
    """
    col = "transcript"
    df["char_count"]            = df[col].str.len()
    df["word_count"]            = df[col].str.split().str.len()
    df["has_latin"]             = df[col].str.contains(r"[A-Za-z]", regex=True)
    df["has_diacritics"]        = df[col].str.contains(ARABIC_DIACRITICS.pattern, regex=True)
    return df


# ── Visualisations ────────────────────────────────────────────────────────────

def visualize_characters(
    all_chars_sorted: list[str],
    categories: dict[str, list[str]],
    figures_path: str,
    *,
    cols: int = 20,
) -> None:
    """
    Render a grid of all unique characters colour-coded by category.

    Saves ``char_inventory.png`` to *figures_path*.

    Parameters
    ----------
    all_chars_sorted : list[str]
        Characters ordered by frequency (most common first).
    categories : dict[str, list[str]]
        Category → character list mapping from :func:`categorize_vocab`.
    figures_path : str
        Output directory.
    cols : int
        Number of columns in the character grid. Default ``20``.
    """
    n_chars = len(all_chars_sorted)
    rows    = (n_chars + cols - 1) // cols

    fig, ax = plt.subplots(figsize=(cols * 0.7, rows * 0.85))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.axis("off")
    ax.set_facecolor("#F5F5F5")
    fig.patch.set_facecolor("#F5F5F5")

    for idx, c in enumerate(all_chars_sorted):
        col_pos = idx % cols
        row_pos = rows - 1 - (idx // cols)
        color   = CATEGORY_COLORS.get(classify_char(c), "#9E9E9E")
        ax.text(col_pos + 0.5, row_pos + 0.5, c,
                ha="center", va="center", fontsize=13,
                color=color, fontfamily="DejaVu Sans")

    legend_handles = [
        mpatches.Patch(color=CATEGORY_COLORS[cat],
                       label=f"{cat} ({len(chars)})")
        for cat, chars in sorted(categories.items())
        if cat in CATEGORY_COLORS
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              bbox_to_anchor=(1.25, 1.0), fontsize=8, framealpha=0.9)

    ax.set_title(
        f"All {n_chars} unique characters in training transcripts\n"
        "(sorted by frequency left→right top→bottom; coloured by category)",
        fontsize=11, pad=10,
    )
    plt.tight_layout()
    from pathlib import Path as _Path
    plt.savefig(str(_Path(figures_path) / "char_inventory.png"), bbox_inches="tight")
    plt.show()


def visualize_word_pie_chart(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    save_path: str | None = None,
) -> None:
    """
    Render side-by-side pie charts of Arabic / Latin / other word fractions.

    Parameters
    ----------
    train_texts : Sequence[str]
        Training split transcripts.
    test_texts : Sequence[str]
        Test split transcripts.
    save_path : str | None
        If provided, saves the figure to this path.
    """
    def _word_type_counts(texts: Sequence[str]) -> tuple[int, int, int]:
        ar, lat, other = 0, 0, 0
        for t in texts:
            for w in t.split():
                if ARABIC_CHAR_RANGE.search(w):
                    ar += 1
                elif LATIN_LETTERS.issuperset(set(w.lower().replace(" ", ""))):
                    lat += 1
                else:
                    other += 1
        return ar, lat, other

    ar_tr, lat_tr, oth_tr = _word_type_counts(train_texts)
    ar_te, lat_te, oth_te = _word_type_counts(test_texts)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    for ax, (ar, lat, oth), title in [
        (axes[0], (ar_tr, lat_tr, oth_tr), "Train"),
        (axes[1], (ar_te, lat_te, oth_te), "Test"),
    ]:
        total  = ar + lat + oth
        values = [ar, lat, oth]
        labels = [
            f"Arabic\n{ar  / total * 100:.1f}%",
            f"Latin\n{lat  / total * 100:.1f}%",
            f"Other\n{oth  / total * 100:.1f}%",
        ]
        ax.pie(
            values,
            labels=labels,
            colors=["#2979FF", "#FF1744", "#9E9E9E"],
            startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1},
            textprops={"fontsize": 6},
        )
        ax.set_title(title, fontsize=8)

    plt.suptitle("Word Type Distribution", fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
