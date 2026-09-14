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
    "Suicidal": 5,
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
    We preserve single instances and ellipses ('...') — they carry emotional signal.
    """
    # First normalize ellipsis (2 or more dots -> standard '...')
    text = re.sub(r"\.{2,}", "...", text)
    # Reduce repeated exclamation and question marks
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines to single space."""
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(
    text: str,
    strip_urls: bool = True,
    strip_reddit_markup: bool = True,
    lowercase: bool = False,  # NOTE: False by default — BERT has cased variants
    min_length: int = 10,
    **kwargs,
) -> Optional[str]:
    """
    Full cleaning pipeline for a single text.

    Args:
        text: Raw Reddit post text.
        strip_urls: Strip URLs.
        strip_reddit_markup: Strip Reddit-specific markup.
        lowercase: Lowercase text. Keep False for cased transformers.
        min_length: Minimum character length after cleaning.

    Returns:
        Cleaned text string, or None if text is too short.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return None

    # Backward compatibility with remove_urls / remove_reddit_markup kwargs
    if "remove_urls" in kwargs:
        strip_urls = kwargs["remove_urls"]
    if "remove_reddit_markup" in kwargs:
        strip_reddit_markup = kwargs["remove_reddit_markup"]

    text = normalize_unicode(text)

    if strip_urls:
        text = remove_urls(text)

    if strip_reddit_markup:
        text = remove_reddit_markup(text)

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

    # Smart column resolution: handle varying column names across raw vs processed data
    if text_col not in df.columns:
        for candidate in ["text", "statement", "post", "body"]:
            if candidate in df.columns:
                text_col = candidate
                break

    if label_col not in df.columns:
        for candidate in ["label_name", "status", "label", "category"]:
            if candidate in df.columns:
                label_col = candidate
                break

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Required columns not found in DataFrame. Available columns: {list(df.columns)}. "
            f"Expected text column '{text_col}' and label column '{label_col}'."
        )

    # If df already has 'text' or 'label_name', rename safely without collisions
    if text_col != "text":
        df = df.rename(columns={text_col: "text"})
    if label_col != "label_name":
        df = df.rename(columns={label_col: "label_name"})

    # Drop rows with missing text or label
    df = df.dropna(subset=["text", "label_name"])

    # Clean text
    df["text"] = df["text"].apply(clean_text)
    df = df.dropna(subset=["text"])  # Remove rows where cleaning returned None

    # Normalize label names (strip whitespace, title case)
    df["label_name"] = df["label_name"].astype(str).str.strip().str.title()

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
    unique_classes = np.array(sorted(df[label_col].unique()))
    weights_computed = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=df[label_col].values,
    )

    # Ensure weights array length matches NUM_LABELS to avoid dimension mismatch
    num_classes = max(NUM_LABELS, int(unique_classes.max()) + 1)
    weights = np.ones(num_classes, dtype=np.float32)
    for cls_idx, w in zip(unique_classes, weights_computed):
        weights[int(cls_idx)] = w

    print(f"[Class Weights] {dict(zip(unique_classes, weights_computed.round(4)))}")
    return weights.astype(np.float32)
