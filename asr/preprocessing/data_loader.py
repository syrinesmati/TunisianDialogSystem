"""
data_loader.py
──────────────
HuggingFace dataset I/O helpers.

Responsibilities
----------------
- Download a named HF dataset and persist it to disk (one-time operation).
- Reload a previously saved dataset from disk for all downstream notebooks.

Design note: keeping I/O in one place means every notebook imports from here
instead of calling `load_dataset` / `load_from_disk` inline, so the dataset
path and name are changed in exactly one location.
"""

from __future__ import annotations

from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk


def download_and_save(dataset_name: str, save_path: str | Path) -> DatasetDict:
    """
    Download a HuggingFace dataset by its hub name and persist it to disk.

    This is a one-time bootstrap step. The saved dataset can later be loaded
    without a network connection via :func:`load_local_dataset`.

    Parameters
    ----------
    dataset_name : str
        The HuggingFace Hub identifier, e.g. ``"linagora/linto-dataset-audio-ar-tn"``.
    save_path : str | Path
        Directory where the dataset will be saved (created if absent).

    Returns
    -------
    DatasetDict
        The downloaded dataset, already saved to *save_path*.
    """
    dataset = load_dataset(dataset_name)
    dataset.save_to_disk(str(save_path))
    return dataset


def load_local_dataset(path: str | Path) -> DatasetDict:
    """
    Load a HuggingFace dataset that was previously saved to disk.

    Parameters
    ----------
    path : str | Path
        Directory produced by a prior :func:`download_and_save` call
        (or ``dataset.save_to_disk``).

    Returns
    -------
    DatasetDict
        The loaded dataset with all original splits intact.
    """
    return load_from_disk(str(path))
