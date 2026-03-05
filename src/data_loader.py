"""Dataset loading utilities."""

from __future__ import annotations

from datasets import DatasetDict
from datasets import load_from_disk


# ---------------------------
# Public Loader API
# ---------------------------
def load_conll2003_local(path: str = "data/raw") -> DatasetDict:
    """Load the locally cached CoNLL2003 `DatasetDict`."""
    return load_from_disk(path)
