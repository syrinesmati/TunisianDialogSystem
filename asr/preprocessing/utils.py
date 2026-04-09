"""
utils.py
────────
Shared helpers: config loading, path management, display formatting,
DataFrame inspection, and corpus analysis utilities.

All functions are pure / side-effect-free except for :func:`ensure_dir`
(creates directories), the ``print_*`` family (writes to stdout), and
:func:`sort_tsv_file` (rewrites a TSV file in-place).
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict:
    """
    Parse a YAML file and return its contents as a plain Python dict.

    Parameters
    ----------
    path : str | Path
        Path to the ``.yaml`` / ``.yml`` config file.

    Returns
    -------
    dict
        Parsed YAML contents. Nested keys are returned as nested dicts.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ── Filesystem ─────────────────────────────────────────────────────────────────

def ensure_dir(path: str | Path) -> Path:
    """
    Create *path* (including any missing parents) if it does not yet exist.

    Parameters
    ----------
    path : str | Path
        Target directory path.

    Returns
    -------
    Path
        The resolved ``Path`` object (useful for chaining).
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Display helpers ────────────────────────────────────────────────────────────

def print_dict(d: dict[str, Any], title: str = "") -> None:
    """
    Pretty-print a flat key→value stats dictionary to stdout.

    Parameters
    ----------
    d : dict
        Flat mapping of metric names to values.
    title : str, optional
        Optional section heading printed above the key-value pairs.
    """
    if title:
        print(f"\n{'─' * 50}")
        print(f"  {title}")
        print(f"{'─' * 50}")
    for k, v in d.items():
        print(f"  {k:<35} {v}")


def print_section(title: str) -> None:
    """
    Print a bold section separator with *title* to stdout.

    Parameters
    ----------
    title : str
        Label displayed inside the separator box.
    """
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


# ── DataFrame helpers ──────────────────────────────────────────────────────────

def df_overview(df) -> None:
    """
    Print a concise structural summary of a pandas DataFrame to stdout.

    Columns reported: dtype, null count, null percentage, and unique-value count.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to inspect.
    """
    import pandas as pd  # lazy import — avoids a hard dependency at module load time

    stats = pd.DataFrame({
        "dtype":   df.dtypes,
        "nulls":   df.isnull().sum(),
        "null_%":  (df.isnull().mean() * 100).round(2),
        "nunique": df.nunique(),
    })
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} cols\n")
    print(stats.to_string())


# ── TSV helpers ────────────────────────────────────────────────────────────────

def sort_tsv_file(path: Path, sort_col: int = 0, comment_prefix: str = "#") -> None:
    """
    Sort a TSV file alphabetically by a specified column and rewrite it in-place.

    Preserves comment lines at the top, ignores malformed rows, deduplicates
    rows (first occurrence wins), and sorts case-insensitively.

    .. warning::
        This function rewrites *path* on disk.  Do **not** call it from
        concurrent processes pointing at the same file — use a lock or run it
        as a one-time data-preparation step instead.

    Parameters
    ----------
    path : Path
        Path to the TSV file.
    sort_col : int, default=0
        Column index to sort by.
    comment_prefix : str, default="#"
        Lines starting with this prefix are treated as comments and preserved
        at the top of the output, in their original order.
    """
    if not path.exists():
        return

    comments: list[list[str]] = []
    rows: list[list[str]] = []

    with open(path, encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row[0].startswith(comment_prefix):
                comments.append(row)
            elif len(row) > sort_col:
                rows.append(row)

    # Deduplicate rows (keep first occurrence)
    seen: set[tuple] = set()
    unique_rows: list[list[str]] = []
    for row in rows:
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)

    unique_rows.sort(key=lambda r: r[sort_col].lower())

    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        for c in comments:
            writer.writerow(c)
        for r in unique_rows:
            writer.writerow(r)


# ── Corpus analysis helpers ────────────────────────────────────────────────────

def extract_words_from_pattern(
    column,
    patt: str,
    limit: int | None = None,
) -> list[str]:
    """
    Find all words matching *patt* across a text column and print a frequency
    summary.

    This is a general corpus-exploration helper used in EDA notebooks and the
    NOTA normaliser development workflow.  It was previously located in
    ``nota_normalizer.py`` but belongs here because it has no dependency on the
    normaliser itself.

    Parameters
    ----------
    column : Iterable[str]
        An iterable of transcript strings (e.g. a pandas Series or a list).
    patt : str
        Regular-expression pattern.  Every whitespace-delimited token whose
        text matches this pattern (via :func:`re.search`) is collected.
    limit : int | None
        If given, return at most *limit* of the most frequent matching words.
        ``None`` returns all matching words sorted by frequency.

    Side effects
    ------------
    Prints to stdout: total unique matching words and the top-*limit* list.
    """
    pattern = re.compile(patt)

    all_words = [
        word
        for text in column
        if isinstance(text, str)
        for word in text.split()
        if pattern.search(word)
    ]

    word_counts: Counter = Counter(all_words)
    sorted_counts = word_counts.most_common()

    print(f"Total unique words matching pattern: {len(word_counts)}")

    n = len(sorted_counts) if (limit is None or limit > len(sorted_counts)) else limit
    top_words = [word for word, _ in sorted_counts[:n]]
    print("Top words:", " ".join(top_words))
