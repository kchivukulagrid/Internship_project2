"""Dataset loading utilities."""

from __future__ import annotations

from datasets import load_from_disk


def load_conll2003_local(path: str = "data/raw"):
    """Load the locally cached CoNLL2003 `DatasetDict`."""
    return load_from_disk(path)
