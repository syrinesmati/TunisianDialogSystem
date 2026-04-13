"""
code_switch_handler.py
──────────────────────
Code-switching detection and loanword Arabization for Tunisian Arabic.

Background
----------
Tunisian Arabic is characterised by heavy code-switching, primarily with
French but also Italian, Turkish, Spanish, Berber, and English. Two
categories exist (per NOTA / CODA-TUN guidelines):

1. **Integrated loanwords** — phonologically absorbed into TUN morphology
   (e.g. بوسطة 'post office', ماكينة 'machine'). These have a canonical
   Arabic-script NOTA form and should be looked up + replaced.

2. **Genuine code-switches** — words inserted in their source language
   without phonological integration (e.g. "ça va", "désolé", "rendez-vous").
   NOTA and CODA-TUN both permit retaining these in Latin script, but for
   ASR purposes we implement a configurable policy:
   - ``'arabize'``  — look up in table and replace; else transliterate.
   - ``'keep'``     — keep as-is (Latin characters remain in vocab).
   - ``'remove'``   — strip the word from the transcript.

Design
------
The lookup table (``loanword_arabization.tsv``) maps Latin forms to their
NOTA Arabic forms. It is the primary resource; langdetect / heuristics are
used only to flag words NOT in the table.

All functions are stateless except for the :class:`CodeSwitchHandler` class
which loads the lookup table once at construction time.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from .utils import sort_tsv_file
from pathlib import Path
from typing import Literal

# Optional dependency — reserved for a future language-detection pass that
# flags words not found in the Arabization table for manual review.
# Currently unused; import is kept here as a reminder of the planned feature.
# To enable: implement _detect_language(word) using langdetect.detect() and
# wire it into detect_code_switches() to populate a 'detected_lang' field.
try:
    from langdetect import detect as _langdetect_detect, LangDetectException
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False

# ── Patterns ──────────────────────────────────────────────────────────────────

_LATIN_WORD     = re.compile(r"\b[A-Za-zÀ-ÿ''\-]{2,}\b")
_ARABIC_CHAR    = re.compile(r"[\u0600-\u06FF]")
_LATIN_ONLY     = re.compile(r"^[A-Za-zÀ-ÿ''\-]+$")

# Common French function words / particles that appear in TUN code-switching.
# Used in detect_code_switches() to add an 'is_french_stopword' flag so
# downstream analysis can distinguish grammatical filler words (le, la, pas…)
# from content-bearing loanwords (voiture, ordinateur…).
_FRENCH_STOPWORDS = frozenset({
    "le", "la", "les", "de", "du", "des", "un", "une",
    "et", "ou", "mais", "donc", "or", "ni", "car",
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
    "en", "au", "aux", "par", "pour", "sur", "sous", "dans",
    "avec", "sans", "ça", "va", "merci", "non", "oui",
    "très", "bien", "mal", "pas", "plus", "moins",
})

# ── Public class ──────────────────────────────────────────────────────────────

PolicyType = Literal["arabize", "keep", "remove"]


class CodeSwitchHandler:
    """
    Detect and handle code-switched words in Tunisian Arabic transcripts.

    Parameters
    ----------
    lexicons_dir : str | Path
        Directory containing ``loanword_arabization.tsv``.
    policy : {'arabize', 'keep', 'remove'}
        What to do with Latin-script words:
        - ``'arabize'`` (default) — replace with NOTA Arabic form from table;
          words not in table are kept as-is.
        - ``'keep'``    — leave all Latin words unchanged.
        - ``'remove'``  — delete all Latin-script tokens from the transcript.
    case_insensitive : bool
        If ``True``, look up words in the table ignoring case. Default ``True``.

    Attributes
    ----------
    loanword_table : dict[str, str]
        ``{latin_form_lower: arabic_nota_form}`` loaded from TSV.
    policy : str
    """

    def __init__(
        self,
        lexicons_dir: str | Path,
        policy: PolicyType = "arabize",
        case_insensitive: bool = True,
    ) -> None:
        self.policy          = policy
        self.case_insensitive = case_insensitive
        self.loanword_table  = self._load_table(
            Path(lexicons_dir) / "loanword_arabization.tsv",
            lower=case_insensitive,
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def process(self, text: str) -> str:
        """
        Apply the code-switch policy to *text*, word by word.

        .. note::
            **Precondition:** *text* must already have passed through
            :func:`cleaning.collapse_whitespace`.  This method splits on
            whitespace and re-joins with a single space, so multi-space runs
            or leading/trailing whitespace in the input would be silently
            collapsed — rely on the caller's cleaning step rather than
            discovering this later.

        Parameters
        ----------
        text : str
            Pre-cleaned transcript (diacritics removed, unicode NFC,
            whitespace already collapsed).

        Returns
        -------
        str
            Transcript with code-switched words handled per policy.
        """
        if self.policy == "keep":
            return text

        words  = text.split()
        result = []
        for word in words:
            if self._is_latin_word(word):
                replaced = self._handle_word(word)
                if replaced:
                    result.append(replaced)
                # if replaced is None → remove
            else:
                result.append(word)
        return " ".join(result)

    def process_series(self, series):
        """
        Apply :meth:`process` element-wise to a pandas Series.

        Parameters
        ----------
        series : pandas.Series

        Returns
        -------
        pandas.Series
        """
        return series.apply(lambda t: self.process(str(t)) if t == t else t)

    def detect_code_switches(self, text: str) -> list[dict]:
        """
        Return metadata about every Latin-script word in *text*.

        Parameters
        ----------
        text : str
            Transcript string.

        Returns
        -------
        list[dict]
            Each entry contains:

            ``word``              — the original Latin-script token.
            ``position``         — zero-based token index in the transcript.
            ``in_table``         — whether the word has a known Arabization.
            ``arabic_form``      — the Arabized form, or ``""`` if not in table.
            ``is_french_stopword`` — ``True`` if the lowercased word is a
                                     common French function word (see
                                     ``_FRENCH_STOPWORDS``).  Useful for
                                     distinguishing grammatical fillers from
                                     content-bearing loanwords.
        """
        results = []
        for i, word in enumerate(text.split()):
            if self._is_latin_word(word):
                key    = word.lower() if self.case_insensitive else word
                arabic = self.loanword_table.get(key)
                results.append({
                    "word":               word,
                    "position":           i,
                    "in_table":           arabic is not None,
                    "arabic_form":        arabic or "",
                    "is_french_stopword": key in _FRENCH_STOPWORDS,
                })
        return results

    def get_unknown_latin_words(self, texts, top_k: int = 100) -> list[tuple[str, int]]:
        """
        Return the most frequent Latin words NOT in the Arabization table.

        Useful for discovering which loanwords to add to the TSV next.

        Parameters
        ----------
        texts : Sequence[str]
            Iterable of transcript strings.
        top_k : int
            Return at most *top_k* entries.

        Returns
        -------
        list[tuple[str, int]]
            ``[(word, count), ...]`` sorted descending by count.
        """
        from collections import Counter
        counter: Counter = Counter()
        for t in texts:
            for w in t.split():
                if self._is_latin_word(w):
                    key = w.lower() if self.case_insensitive else w
                    if key not in self.loanword_table:
                        counter[key] += 1
        return counter.most_common(top_k)

    # ── Private helpers ────────────────────────────────────────────────────

    def _handle_word(self, word: str) -> str | None:
        """
        Apply the configured policy to a single Latin-script word.

        Returns the replacement string, or ``None`` if the word should be
        removed.
        """
        if self.policy == "remove":
            return None

        # policy == "arabize"
        key = word.lower() if self.case_insensitive else word
        arabic = self.loanword_table.get(key)
        if arabic:
            return arabic
        # Not in table — keep the word as-is (do not invent transliterations)
        return word

    @staticmethod
    def _is_latin_word(word: str) -> bool:
        """Return True if *word* consists entirely of Latin-script characters."""
        # Allow apostrophes and hyphens (French contractions, compound words)
        stripped = re.sub(r"['\-]", "", word)
        return bool(stripped) and _LATIN_ONLY.match(stripped) is not None

    @staticmethod
    def _load_table(path: Path, lower: bool = True) -> dict[str, str]:
        """
        Load the loanword Arabization TSV.

        Expected columns: ``latin_form  TAB  arabic_nota_form  [TAB  origin]``

        Parameters
        ----------
        path : Path
        lower : bool
            Lowercase the Latin key for case-insensitive lookup.

        Returns
        -------
        dict[str, str]
        """
        if not path.exists():
            return {}
        # Ensure TSV is sorted before loading (one-time cleanup)
        sort_tsv_file(path)
        out: dict[str, str] = {}
        with open(path, encoding="utf-8") as fh:
            for row in csv.reader(fh, delimiter="\t"):
                if len(row) >= 2 and not row[0].startswith("#"):
                    key = row[0].strip().lower() if lower else row[0].strip()
                    out[key] = row[1].strip()
        return out


