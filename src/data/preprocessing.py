"""
MentalScope — Mental Health Text Classification
Data preprocessing pipeline.

Handles cleaning of Reddit posts for mental health classification.
Preserves emotionally relevant text while removing noise.
"""

import re
import string
import unicodedata
from typing import Optional
import pandas as pd
import numpy as np


# ── Label mappings ──────────────────────────────────────────────────────────

LABEL2ID = {
    "Normal": 0,
    "Depression": 1,
    "Anxiety": 2,
    "Bipolar": 3,
    "Stress": 4,
    "Personality Disorder": 5,
    "Suicidal": 6,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

NUM_LABELS = len(LABEL2ID)


# ── Text cleaning ────────────────────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """Normalize unicode to ASCII-compatible form."""
    return unicodedata.normalize("NFKC", text)


def remove_urls(text: str) -> str:
    """Remove URLs (http/https/www)."""
    return re.sub(r"http\S+|www\.\S+", " ", text)


def remove_reddit_markup(text: str) -> str:
    """Remove Reddit-specific formatting: /r/, /u/, [deleted], etc."""
    text = re.sub(r"/r/\w+", " ", text)
    text = re.sub(r"/u/\w+", " ", text)
    text = re.sub(r"\[deleted\]|\[removed\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"&amp;|&gt;|&lt;|&nbsp;", " ", text)  # HTML entities
    return text


def remove_excessive_punctuation(text: str) -> str:
    """
    Reduce repeated punctuation (e.g. '!!!!!!' -> '!').
    We preserve single instances — they carry emotional signal.
    """
    text = re.sub(r"([!?.]){2,}", r"\1", text)
    text = re.sub(r"[.]{2,}", "...", text)  # Preserve ellipsis
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines to single space."""
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(
    text: str,
    remove_urls: bool = True,
    remove_reddit_markup: bool = True,
    lowercase: bool = False,  # NOTE: False by default — BERT has cased variants
    min_length: int = 10,
) -> Optional[str]:
    """
    Full cleaning pipeline for a single text.

    Args:
        text: Raw Reddit post text.
        remove_urls: Strip URLs.
        remove_reddit_markup: Strip Reddit-specific markup.
        lowercase: Lowercase text. Keep False for cased transformers.
        min_length: Minimum character length after cleaning.

    Returns:
        Cleaned text string, or None if text is too short.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return None

    text = normalize_unicode(text)

    if remove_urls:
        text = globals()["remove_urls"](text)

    if remove_reddit_markup:
        text = globals()["remove_reddit_markup"](text)

    text = remove_excessive_punctuation(text)
    text = normalize_whitespace(text)

    if lowercase:
        text = text.lower()

    if len(text) < min_length:
        return None

    return text


# ── DataFrame-level processing ───────────────────────────────────────────────

def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "status",
    label_map: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Clean and normalize an entire DataFrame.

    Args:
        df: Raw dataframe with text and label columns.
        text_col: Column containing post text.
        label_col: Column containing mental health label.
        label_map: Custom label-to-int mapping. Uses LABEL2ID if None.

    Returns:
        Cleaned DataFrame with columns: ['text', 'label', 'label_name']
    """
    if label_map is None:
        label_map = LABEL2ID

    df = df.copy()
    df = df.rename(columns={text_col: "text", label_col: "label_name"})

    # Drop rows with missing text or label
    df = df.dropna(subset=["text", "label_name"])

    # Clean text
    df["text"] = df["text"].apply(clean_text)
    df = df.dropna(subset=["text"])  # Remove rows where cleaning returned None

    # Normalize label names (strip whitespace, title case)
    df["label_name"] = df["label_name"].str.strip()

    # Keep only known labels
    known_labels = set(label_map.keys())
    df = df[df["label_name"].isin(known_labels)]

    # Map labels to integers
    df["label"] = df["label_name"].map(label_map)

    # Final column selection and reset index
    df = df[["text", "label", "label_name"]].reset_index(drop=True)

    print(f"[Preprocessing] {len(df)} samples after cleaning.")
    print(f"[Preprocessing] Class distribution:")
    print(df["label_name"].value_counts())

    return df


def compute_class_weights(df: pd.DataFrame, label_col: str = "label") -> np.ndarray:
    """
    Compute inverse-frequency class weights for Weighted Cross-Entropy.

    Returns:
        numpy array of shape (num_classes,) with weights ordered by label id.
    """
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.array(sorted(df[label_col].unique()))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=df[label_col].values,
    )
    print(f"[Class Weights] {dict(zip(classes, weights.round(4)))}")
    return weights.astype(np.float32)
