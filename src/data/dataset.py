"""
MentalScope — Mental Health Text Classification
PyTorch Dataset classes for HuggingFace Trainer compatibility.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import PreTrainedTokenizerBase


class MentalHealthDataset(Dataset):
    """
    PyTorch Dataset wrapping tokenized mental health text samples.

    Compatible with HuggingFace Trainer — returns dicts with
    'input_ids', 'attention_mask', 'token_type_ids' (if applicable),
    and 'labels'.

    Args:
        texts: List of raw/cleaned post texts.
        labels: List of integer class labels.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum token sequence length (default 512).
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ):
        assert len(texts) == len(labels), "texts and labels must be the same length"

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Pre-tokenize all texts for faster DataLoader iteration
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def build_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
) -> tuple[MentalHealthDataset, MentalHealthDataset, MentalHealthDataset]:
    """
    Build train/val/test PyTorch Datasets from preprocessed DataFrames.

    Args:
        train_df, val_df, test_df: DataFrames with 'text' and 'label' columns.
        tokenizer: HuggingFace tokenizer (must match the backbone model).
        max_length: Token sequence length cap.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    train_dataset = MentalHealthDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    val_dataset = MentalHealthDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    test_dataset = MentalHealthDataset(
        texts=test_df["text"].tolist(),
        labels=test_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )

    print(f"[Dataset] Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset
