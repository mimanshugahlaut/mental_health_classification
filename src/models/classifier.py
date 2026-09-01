"""
MentalScope — Mental Health Text Classification
Model loading and classifier construction.

Handles:
  - Loading transformer backbone from HuggingFace Hub
  - Wrapping with LoRA adapters (PEFT) for parameter-efficient training
  - Full fine-tuning mode (all parameters unfrozen)
  - Printing trainable parameter counts
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig

from src.data.preprocessing import LABEL2ID, ID2LABEL, NUM_LABELS
from src.models.lora_config import get_lora_config


# ── Supported model registry ──────────────────────────────────────────────────

MODEL_REGISTRY = {
    # General-purpose baselines
    "bert":          "bert-base-uncased",
    "roberta":       "roberta-base",
    "distilbert":    "distilbert-base-uncased",

    # Domain-specific (pre-trained on mental health text)
    "mental-bert":    "mental/mental-bert-base-uncased",
    "mental-roberta": "mental/mental-roberta-base",
}


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.

    Returns:
        Tuple of (total_params, trainable_params).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_parameter_summary(model: nn.Module, model_name: str = "") -> None:
    """Print formatted parameter count summary for paper reporting."""
    total, trainable = count_parameters(model)
    pct = 100.0 * trainable / total if total > 0 else 0
    print(
        f"\n{'='*55}\n"
        f"  Model: {model_name}\n"
        f"  Total params:     {total:>12,}\n"
        f"  Trainable params: {trainable:>12,}  ({pct:.2f}%)\n"
        f"  Frozen params:    {total - trainable:>12,}\n"
        f"{'='*55}\n"
    )


def load_model(
    model_key: str,
    use_lora: bool = False,
    lora_rank: int = 8,
    num_labels: int = NUM_LABELS,
    dropout: float = 0.1,
) -> Tuple[nn.Module, object]:
    """
    Load a pre-trained transformer backbone and configure it for classification.

    Args:
        model_key: Key from MODEL_REGISTRY (e.g. 'mental-roberta') or
                   a full HuggingFace Hub model ID.
        use_lora: If True, apply LoRA adapters and freeze base model.
                  If False, full fine-tuning (all params unfrozen).
        lora_rank: LoRA rank r. Only used when use_lora=True.
        num_labels: Number of output classes.
        dropout: Classifier dropout probability.

    Returns:
        Tuple of (model, tokenizer).

    Example:
        >>> model, tokenizer = load_model("mental-roberta", use_lora=True, lora_rank=8)
    """
    # Resolve model ID
    model_id = MODEL_REGISTRY.get(model_key, model_key)
    print(f"\n[Model] Loading: {model_id}  (use_lora={use_lora})")

    # Detect architecture type for LoRA target module selection
    arch_type = _get_arch_type(model_id)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # Load base model with classification head
    # HuggingFace's AutoModelForSequenceClassification adds a linear head
    # on top of the pooled [CLS] representation automatically.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        ignore_mismatched_sizes=True,  # Safe for domain models with different vocab
    )

    if use_lora:
        # Build LoRA config and wrap model
        lora_cfg = get_lora_config(arch_type, rank=lora_rank)
        model = get_peft_model(model, lora_cfg)
        mode_str = f"LoRA (rank={lora_rank})"
    else:
        mode_str = "Full Fine-Tuning"

    print_parameter_summary(model, model_name=f"{model_id} — {mode_str}")
    return model, tokenizer


def _get_arch_type(model_id: str) -> str:
    """Infer architecture type from model ID string for LoRA target modules."""
    model_id_lower = model_id.lower()
    if "roberta" in model_id_lower:
        return "roberta"
    elif "distilbert" in model_id_lower:
        return "distilbert"
    else:
        return "bert"  # Default covers bert-base and mental-bert
