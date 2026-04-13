"""
nota_normalizer.py
──────────────────
NOTA/CODA*-compliant orthographic normaliser for Tunisian Arabic.

Layer overview (applied in priority order: highest first)
---------------------------------------------------------
Layer 10 — canonical overrides  (``coda_word_list.tsv``)
    Hard-coded full-word substitutions that short-circuit all other layers.
Layer 9  — variant collapse      (``variant_collapse_map.tsv``)
    Many-to-one mapping of known orthographic variants.
Layer 7  — hamza drop            (regex)
    Word-initial أ / إ → bare Alef ا.
Layer 6  — waw plurality         (``waw_exceptions.tsv`` + regex)
    Word-final و → وا for plural forms (with exceptions).
Layer 5  — taa marbuta           (``taa_marbuta_ha_list.tsv``)
    Lexicon-driven ه → ة restoration.
Layer 4  — alef maqsura          (``alef_maqsura_list.tsv`` + regex)
    Word-final ى → ي (with exceptions where ى is etymologically correct).
Layer 8  — negation separation   (regex, sentence-level)
    Agglutinated negation circumfix: ماقالش → ما قال ش.
    Applied before word-level layers so the split tokens are visible to them.

NOTE: layers run highest-priority-first (10 → 4).  Layer 10 short-circuits
all subsequent layers for the matched word.

Public API
----------
- :meth:`NOTANormalizer.normalize`                   — text only (backward-compatible).
- :meth:`NOTANormalizer.normalize_with_flags`        — (text, flags_dict).
- :meth:`NOTANormalizer.normalize_with_log`          — (text, change_log).
- :meth:`NOTANormalizer.normalize_series`            — pandas Series, text only.
- :meth:`NOTANormalizer.normalize_series_with_flags` — pandas Series of (text, flags).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from .utils import sort_tsv_file


# ── Pre-compiled patterns ─────────────────────────────────────────────────────

_FINAL_ALEF_MAQSURA = re.compile(r"ى(?=\s|$)")
_INITIAL_HAMZA      = re.compile(r"(?<!\S)[أإ]")

# Tunisian Arabic negation circumfix  ما … ش  (Layer 8).
#
# Pattern breakdown:
#   (ما)          — the pre-verbal negation particle
#   ([^\s]{2,})   — the verb stem (≥2 chars so that ماش "not going / gone"
#                   is NOT split into ما + ش; the stem must be at least one
#                   character beyond the circumfix)
#   (ش)           — the post-verbal negation clitic
#   (?=\s|$)      — positive lookahead: the ش must be word-final (followed by
#                   whitespace or end-of-string) to avoid matching inside
#                   longer words that happen to end in ش
_NEG_AGGLUTINATED = re.compile(r"(ما)([^\s]{2,})(ش)(?=\s|$)")


# ── Public class ──────────────────────────────────────────────────────────────

class NOTANormalizer:
    """
    NOTA/CODA*-compliant orthographic normaliser for Tunisian Arabic.

    Parameters
    ----------
    lexicons_dir : str | Path
        Directory containing the TSV lexicon files.
    layer_flags : dict | None
        Override individual layer booleans.  Unknown keys are ignored with a
        warning.  Defaults to all layers enabled.

    Attributes
    ----------
    _flags : dict[str, bool]
        Merged layer-enable flags (defaults + overrides).
    """

    _DEFAULT_FLAGS: dict[str, bool] = {
        "apply_alef_maqsura":   True,
        "apply_taa_marbuta":    True,
        "apply_waw_plurality":  True,
        "apply_hamza_drop":     True,
        "apply_negation_split": True,
        "apply_variant_map":    True,
        "apply_overrides":      True,
    }

    def __init__(self, lexicons_dir, layer_flags=None):
        d = Path(lexicons_dir)

        self.gaf_exceptions        = self._load_set(d / "gaf_exceptions.tsv")
        self.variant_map           = self._load_map(d / "variant_collapse_map.tsv")
        self.canonical_overrides   = self._load_map(d / "coda_word_list.tsv")
        self.taa_marbuta_ha        = self._load_map(d / "taa_marbuta_ha_list.tsv")
        self.alef_maqsura_preserve = self._load_set(d / "alef_maqsura_list.tsv")
        self.waw_exceptions        = self._load_set(d / "waw_exceptions.tsv")

        self._flags = {**self._DEFAULT_FLAGS, **(layer_flags or {})}

    # ── Core API ──────────────────────────────────────────────────────────────

    def normalize(self, text: str) -> str:
        """
        Normalize *text* and return the normalised string only.

        Backward-compatible entry point — does not return flags or a log.

        Parameters
        ----------
        text : str

        Returns
        -------
        str
        """
        if not isinstance(text, str) or not text.strip():
            return text

        if self._flags["apply_negation_split"]:
            text = self._separate_negation(text)

        words = text.split()
        words = [self._apply_word_layers(w)[0] for w in words]
        return " ".join(words)
    def normalize_with_flags(self, text: str) -> tuple[str, dict[str, bool]]:
        """
        Normalize *text* and return ``(normalised_text, flags_dict)``.

        Each flag in the returned dict is ``True`` if the corresponding layer
        made at least one change to the input.

        Parameters
        ----------
        text : str

        Returns
        -------
        tuple[str, dict[str, bool]]
        """
        if not isinstance(text, str) or not text.strip():
            return text, self._empty_flags()

        line_flags = self._empty_flags()

        # Layer 8 — negation (sentence-level)
        if self._flags["apply_negation_split"]:
            new_text = self._separate_negation(text)
            if new_text != text:
                line_flags["negation_applied"] = True
            text = new_text

        words = text.split()
        new_words = []
        for w in words:
            nw, word_flags, _ = self._apply_word_layers(w)
            new_words.append(nw)
            for k in line_flags:
                if word_flags.get(k, False):
                    line_flags[k] = True

        return " ".join(new_words), line_flags

    def normalize_with_log(self, text: str) -> tuple[str, list[dict]]:
        """
        Normalize *text* and return a structured, per-change log.

        Each change entry contains ``layer`` (int), ``rule`` (str),
        ``before`` (str), and ``after`` (str).  This is the recommended
        method for debugging and auditing normalisation decisions.

        Parameters
        ----------
        text : str

        Returns
        -------
        tuple[str, list[dict]]
            ``(normalised_text, changes)``
        """
        log: list[dict] = []

        if not isinstance(text, str) or not text.strip():
            return text, log

        # Layer 8 — negation (sentence-level)
        if self._flags["apply_negation_split"]:
            after_neg = self._separate_negation(text)
            if after_neg != text:
                log.append({
                    "layer":  8,
                    "rule":   "negation_separation",
                    "before": text,
                    "after":  after_neg,
                })
            text = after_neg

        # Word-level layers
        words = text.split()
        result = []
        for w in words:
            nw, word_flags, word_log = self._apply_word_layers(w, record_log=True)
            result.append(nw)
            log.extend(word_log)

        return " ".join(result), log

    # ── Pandas helpers ────────────────────────────────────────────────────────

    def normalize_series(self, series):
        """
        Apply :meth:`normalize` element-wise to a pandas Series.

        Parameters
        ----------
        series : pandas.Series

        Returns
        -------
        pandas.Series
        """
        return series.apply(lambda t: self.normalize(str(t)) if t == t else t)

    def normalize_series_with_flags(self, series):
        """
        Apply :meth:`normalize_with_flags` element-wise to a pandas Series.

        Parameters
        ----------
        series : pandas.Series

        Returns
        -------
        pandas.Series
            Each element is a ``(normalised_text, flags_dict)`` tuple.
        """
        return series.apply(
            lambda t: self.normalize_with_flags(str(t)) if t == t
            else (t, self._empty_flags())
        )

    # ── Layer 8 — negation separation (sentence-level) ────────────────────────

    def _separate_negation(self, text: str) -> str:
        """Split agglutinated negation circumfix: ماقالش → ما قال ش."""
        return _NEG_AGGLUTINATED.sub(r"\1 \2 \3", text)

    # ── Unified word-level layer runner ───────────────────────────────────────

    def _apply_word_layers(
        self,
        word: str,
        *,
        record_log: bool = False,
    ) -> tuple[str, dict[str, bool], list[dict]]:
        """
        Apply all word-level normalisation layers (10 → 4) to a single token.

        This is the **single authoritative implementation** of the layer
        sequence.  :meth:`normalize`, :meth:`normalize_with_flags`, and
        :meth:`normalize_with_log` all call this method so that adding a new
        layer requires editing exactly one place.

        Parameters
        ----------
        word : str
            A single whitespace-delimited token.
        record_log : bool
            If ``True``, populate the returned log list with per-change dicts.
            When ``False`` (default) the list is always empty, saving the
            overhead of building log entries during production runs.

        Returns
        -------
        tuple[str, dict[str, bool], list[dict]]
            ``(normalised_word, flags_dict, change_log)``
            *change_log* is empty when *record_log* is ``False``.
        """
        flags: dict[str, bool] = self._empty_flags()
        change_log: list[dict] = []

        def _record(layer: int, rule: str, before: str, after: str) -> None:
            if record_log:
                change_log.append(
                    {"layer": layer, "rule": rule, "before": before, "after": after}
                )

        # ── Layer 10: canonical overrides ─────────────────────────────────────
        # Short-circuit: if a hard canonical form exists, return immediately
        # without running any other layer.
        if self._flags["apply_overrides"] and word in self.canonical_overrides:
            after = self.canonical_overrides[word]
            flags["override_applied"] = True
            _record(10, "canonical_override", word, after)
            return after, flags, change_log

        # ── Layer 9: variant collapse ──────────────────────────────────────────
        if self._flags["apply_variant_map"] and word in self.variant_map:
            after = self.variant_map[word]
            flags["variant_applied"] = True
            _record(9, "variant_map", word, after)
            word = after

        # ── Layer 7: initial hamza drop ───────────────────────────────────────
        if self._flags["apply_hamza_drop"]:
            after = _INITIAL_HAMZA.sub("ا", word)
            if after != word:
                flags["hamza_applied"] = True
                _record(7, "hamza_drop", word, after)
                word = after

        # ── Layer 6: waw plurality ────────────────────────────────────────────
        if self._flags["apply_waw_plurality"]:
            if (
                word not in self.waw_exceptions
                and len(word) >= 3
                and word.endswith("و")
                and not word.endswith("وا")
            ):
                after = word + "ا"
                flags["waw_applied"] = True
                _record(6, "waw_plurality", word, after)
                word = after

        # ── Layer 5: taa marbuta restoration ──────────────────────────────────
        if self._flags["apply_taa_marbuta"] and word in self.taa_marbuta_ha:
            after = self.taa_marbuta_ha[word]
            flags["taa_applied"] = True
            _record(5, "taa_marbuta", word, after)
            word = after

        # ── Layer 4: alef maqsura ─────────────────────────────────────────────
        if self._flags["apply_alef_maqsura"] and word not in self.alef_maqsura_preserve:
            after = _FINAL_ALEF_MAQSURA.sub("ي", word)
            if after != word:
                flags["alef_applied"] = True
                _record(4, "alef_maqsura", word, after)
                word = after

        return word, flags, change_log

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _empty_flags() -> dict[str, bool]:
        return {
            "alef_applied":      False,
            "taa_applied":       False,
            "waw_applied":       False,
            "hamza_applied":     False,
            "variant_applied":   False,
            "override_applied":  False,
            "negation_applied":  False,
        }

    @staticmethod
    def _load_set(path: Path) -> set[str]:
        if not path.exists():
            return set()
        sort_tsv_file(path)
        out: set[str] = set()
        with open(path, encoding="utf-8") as fh:
            for row in csv.reader(fh, delimiter="\t"):
                if row and not row[0].startswith("#"):
                    out.add(row[0].strip())
        return out

    @staticmethod
    def _load_map(path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        sort_tsv_file(path)
        out: dict[str, str] = {}
        with open(path, encoding="utf-8") as fh:
            for row in csv.reader(fh, delimiter="\t"):
                if len(row) >= 2 and not row[0].startswith("#"):
                    out[row[0].strip()] = row[1].strip()
        return out
