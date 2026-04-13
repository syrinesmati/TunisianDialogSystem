"""
cleaning.py
───────────
Deterministic text-cleaning rules for Phase 3 of the ASR pipeline.

This module handles everything that does **not** require a lexicon lookup:
Unicode normalisation, diacritic removal, punctuation stripping,
special-character removal, digit handling, and basic whitespace collapsing.

All regex patterns are kept as module-level compiled constants so they remain
easy to audit and change.

Separation of concerns
-----------------------
- **cleaning.py** (this file) — stateless, regex-driven, no external data.
- **nota_normalizer.py**       — stateful, lookup-table-driven NOTA rules.
- **code_switch_handler.py**   — language-ID + Arabization of loanwords.

Custom / dataset-specific corrections
--------------------------------------
Transcript-level patches discovered during EDA (e.g. specific garbled
strings found in the LinTO-TN dataset) are **not** hardcoded here.  They
live in ``data/lexicons/transcript_corrections.tsv`` and are loaded once by
:func:`load_transcript_corrections`.  This keeps source code free of
data-dependent hacks and makes patches auditable without touching Python.
"""

from __future__ import annotations

import csv
import logging
import re
import unicodedata
from pathlib import Path

log = logging.getLogger(__name__)

# ── Compiled patterns ─────────────────────────────────────────────────────────

# Arabic diacritics (harakat + sukun + shadda + tanwin + tatweel)
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")

# Alef variants → bare Alef.
# آ (U+0622, Alef Madda) is intentionally included: NOTA does not preserve
# Alef Madda at the cleaning stage; the NOTA normaliser handles it later via
# the override/variant tables for words that require the classical form
# (e.g. قرآن).  Remove آ from this pattern
_ALEF_VARIANTS = re.compile(r"[إأٱ]")

# Characters to strip entirely
_PUNCTUATION    = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~،؟؛«»\u2014\u2013…]")
_URLS           = re.compile(r"https?://\S+|www\.\S+")
_HASHTAGS       = re.compile(r"#\w+")
_MENTIONS       = re.compile(r"@\w+")
_EXTRA_SPACES   = re.compile(r" {2,}")

# Digit normalisation — Arabic-Indic → Western Arabic
_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Systematic character-level substitutions that are universal across the
# dataset (dialect characters, French accented letters, stray Unicode).
# Digit lexicalisation (e.g. "60" → "ستين") is intentionally NOT here —
# it belongs in the NOTA normaliser or a dedicated number-expansion module
# because the mapping is language/dialect-specific.
_CHAR_SUBSTITUTIONS: dict[str, str] = {
    # Maghrebi/Tunisian dialect letter
    "ڤ": "ق",
    # French accented letters → unaccented ASCII equivalents
    "à": "a",
    "â": "a",
    "ç": "c",
    "è": "e",
    "é": "e",
    "ê": "e",
    "î": "i",
    "ù": "u",
    "û": "u",
    "ô": "o",
    # Stray / exotic Unicode artifacts
    "ɑ": "",    # Latin alpha (IPA) — not a valid transcript character
    "ـ": "",    # Arabic tatweel
    "ʝ": "",    # IPA voiced palatal fricative — noise
    "\ufeff": "",    # BOM
    "\u200f": "",    # Right-to-left mark
}


# ── Transcript corrections loader ─────────────────────────────────────────────

def load_transcript_corrections(path: str | Path) -> list[tuple[str, str]]:
    """
    Load dataset-specific transcript patch rules from a TSV file.

    The TSV has two columns: ``bad_string`` and ``corrected_string``.
    Lines starting with ``#`` are treated as comments.  Rules are returned
    in file order and applied as verbatim substring replacements.

    Parameters
    ----------
    path : str | Path
        Path to ``transcript_corrections.tsv`` (or equivalent).

    Returns
    -------
    list[tuple[str, str]]
        ``[(bad, good), ...]`` in file order.  Returns an empty list if the
        file does not exist (non-fatal).
    """
    p = Path(path)
    if not p.exists():
        log.debug("Transcript corrections file not found: %s  (skipping)", p)
        return []

    rules: list[tuple[str, str]] = []
    with open(p, encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for lineno, row in enumerate(reader, start=1):
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 2:
                log.warning(
                    "%s line %d: expected 2 tab-separated columns, got %d — skipped.",
                    p.name, lineno, len(row),
                )
                continue
            rules.append((row[0], row[1]))

    log.debug("Loaded %d transcript correction rules from %s", len(rules), p)
    return rules


# ── Public API ────────────────────────────────────────────────────────────────

def remove_diacritics(text: str) -> str:
    """
    Strip all Arabic diacritics (harakat, sukun, shadda, tanwin, tatweel).

    ASR models trained on Tunisian dialect data should operate on
    undiacritised text, so this step is applied universally.

    Parameters
    ----------
    text : str
        Raw transcript string.

    Returns
    -------
    str
        Text with all diacritic codepoints removed.
    """
    return _DIACRITICS.sub("", text)


def normalize_unicode(text: str) -> str:
    """
    Apply Unicode NFC normalisation and collapse Alef variants.

    Steps
    -----
    1. NFC compose (resolves combining sequences).
    2. Map أ إ آ ٱ → bare Alef ا.

    Parameters
    ----------
    text : str
        Input string (may contain combining characters).

    Returns
    -------
    str
        NFC-normalised string with unified Alef.
    """
    text = unicodedata.normalize("NFC", text)
    text = _ALEF_VARIANTS.sub("ا", text)
    return text


def remove_punctuation(text: str) -> str:
    """
    Remove Latin and Arabic punctuation marks.

    Kept: whitespace, Arabic letters, Latin letters (for code-switching),
    digits.  Stripped: all punctuation listed in ``_PUNCTUATION``.

    Parameters
    ----------
    text : str
        Transcript string.

    Returns
    -------
    str
        String with punctuation removed.
    """
    return _PUNCTUATION.sub("", text)


def remove_urls_and_noise(text: str) -> str:
    """
    Strip URLs, hashtags, and @mentions from a transcript.

    These appear in transcripts sourced from social-media or forum content
    and are meaningless for speech recognition.

    Parameters
    ----------
    text : str
        Transcript string.

    Returns
    -------
    str
        Cleaned string with URLs, hashtags, and mentions removed.
    """
    text = _URLS.sub("", text)
    text = _HASHTAGS.sub("", text)
    text = _MENTIONS.sub("", text)
    return text


def normalize_digits(text: str, *, keep_digits: bool = True) -> str:
    """
    Normalise digit characters.

    If *keep_digits* is ``True`` (default), Arabic-Indic digits (٠–٩) are
    converted to their Western Arabic equivalents (0–9) and retained.
    If ``False``, all digit characters are removed entirely.

    The decision between keeping or removing digits should be made once per
    project and recorded in ``configs/preprocessing_config.yaml``.

    Parameters
    ----------
    text : str
        Transcript string.
    keep_digits : bool
        Whether to keep digits (as Western Arabic) or remove them.

    Returns
    -------
    str
        String with normalised (or removed) digits.
    """
    text = text.translate(_ARABIC_INDIC_DIGITS)
    if not keep_digits:
        text = re.sub(r"[0-9]", "", text)
    return text


def collapse_whitespace(text: str) -> str:
    """
    Collapse runs of whitespace into a single space and strip leading/trailing space.

    Parameters
    ----------
    text : str
        Transcript string (possibly with multiple spaces after other cleaning).

    Returns
    -------
    str
        Stripped, single-space-separated string.
    """
    return _EXTRA_SPACES.sub(" ", text).strip()


def apply_char_substitutions(text: str) -> str:
    """
    Apply systematic character-level substitutions (dialect chars, French
    accents, stray Unicode artifacts).

    These rules are universal — they apply to every transcript regardless of
    content.  Dataset-specific **transcript-level** corrections (e.g. fixing a
    specific garbled string) are handled separately by
    :func:`apply_transcript_corrections`.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    for src, tgt in _CHAR_SUBSTITUTIONS.items():
        text = text.replace(src, tgt)
    return text


def apply_transcript_corrections(
    text: str,
    corrections: list[tuple[str, str]],
) -> str:
    """
    Apply dataset-specific transcript patch rules loaded from a TSV file.

    Each rule is a ``(bad_string, corrected_string)`` pair applied as a
    verbatim substring replacement in list order.

    Parameters
    ----------
    text : str
        Transcript string, ideally after the main cleaning steps.
    corrections : list[tuple[str, str]]
        Rules loaded by :func:`load_transcript_corrections`.

    Returns
    -------
    str
    """
    for bad, good in corrections:
        text = text.replace(bad, good)
    return text


def clean_transcript(
    text: str,
    *,
    keep_digits: bool = True,
    corrections: list[tuple[str, str]] | None = None,
) -> str:
    """
    Apply the full deterministic cleaning pipeline to a single transcript.

    Pipeline order
    --------------
    1. :func:`remove_urls_and_noise`
    2. :func:`normalize_unicode`
    3. :func:`remove_diacritics`
    4. :func:`remove_punctuation`
    5. :func:`normalize_digits`
    6. :func:`apply_char_substitutions`
    7. :func:`apply_transcript_corrections`  (only if *corrections* is provided)
    8. :func:`collapse_whitespace`           (applied after every mutating step)

    Parameters
    ----------
    text : str
        Raw transcript string.
    keep_digits : bool
        Forwarded to :func:`normalize_digits`.  Default ``True``.
    corrections : list[tuple[str, str]] | None
        Dataset-specific patch rules from :func:`load_transcript_corrections`.
        Pass ``None`` (default) to skip this step.

    Returns
    -------
    str
        Fully cleaned transcript.
    """
    text = remove_urls_and_noise(text)
    text = normalize_unicode(text)
    text = remove_diacritics(text)
    text = remove_punctuation(text)
    text = normalize_digits(text, keep_digits=keep_digits)
    text = apply_char_substitutions(text)
    text = collapse_whitespace(text)

    if corrections:
        text = apply_transcript_corrections(text, corrections)
        text = collapse_whitespace(text)

    return text


def clean_dataframe(
    df,
    col: str = "transcript",
    *,
    keep_digits: bool = True,
    corrections: list[tuple[str, str]] | None = None,
):
    """
    Apply :func:`clean_transcript` to every row of a DataFrame column.

    Adds a ``'transcript_clean'`` column alongside the original so the raw
    value is preserved for provenance.

    .. note::
        This function operates on a **copy** of *df* so the caller's original
        DataFrame is never mutated.

    Parameters
    ----------
    df : pandas.DataFrame
        Segment DataFrame containing *col*.
    col : str
        Name of the transcript column.  Default ``'transcript'``.
    keep_digits : bool
        Forwarded to :func:`clean_transcript`.
    corrections : list[tuple[str, str]] | None
        Dataset-specific patch rules.  Pass the result of
        :func:`load_transcript_corrections` to activate transcript-level
        fixes.  ``None`` (default) skips the step.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with ``'transcript_clean'`` column added.
    """
    df = df.copy()
    df["transcript_clean"] = df[col].apply(
        lambda t: clean_transcript(
            str(t),
            keep_digits=keep_digits,
            corrections=corrections,
        )
    )
    return df
