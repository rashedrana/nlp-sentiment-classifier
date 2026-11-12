"""Data loading and text cleaning utilities.

This module holds the small helpers that both the training script and the
prediction script share, so the text is always cleaned the same way.
"""

import re
from pathlib import Path

import pandas as pd

# Project root = one folder above this src/ package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "reviews.csv"


def clean_text(text: str) -> str:
    """Clean a raw review before it is fed to the model.

    Steps:
      1. Lowercase everything ("Great" and "great" become the same word).
      2. Remove anything that is not a letter, digit, or simple punctuation
         that can carry sentiment ("!" and "?" are kept, then removed by
         the vectorizer anyway — the important part is dropping symbols,
         URLs and other noise).
      3. Collapse extra whitespace.

    The TF-IDF vectorizer does its own tokenizing afterwards, so this
    function only needs to do light, safe normalisation.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # drop URLs
    text = re.sub(r"[^a-z0-9\s!?']", " ", text)          # keep basic chars
    text = re.sub(r"\s+", " ", text)                     # collapse spaces
    return text.strip()


def load_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the reviews CSV and return a tidy DataFrame.

    The CSV must have two columns: ``review`` (the text) and ``label``
    ("positive" or "negative"). The returned DataFrame has the reviews
    already cleaned and any empty rows removed.
    """
    df = pd.read_csv(path)
    expected = {"review", "label"}
    if not expected.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns {sorted(expected)}, got {list(df.columns)}"
        )

    df = df.dropna(subset=["review", "label"])
    df["review"] = df["review"].astype(str).apply(clean_text)
    df = df[df["review"].str.len() > 0]
    return df.reset_index(drop=True)
