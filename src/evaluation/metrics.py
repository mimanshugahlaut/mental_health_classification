"""
MentalScope — Mental Health Text Classification
Evaluation metrics for HuggingFace Trainer and standalone use.

Computes:
  - Accuracy
  - Macro F1, Weighted F1
  - Per-class F1 (for paper's per-class table)
  - Matthews Correlation Coefficient (MCC)
  - Confusion Matrix
  - Classification Report (text)
"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)
from src.data.preprocessing import ID2LABEL


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute metrics for HuggingFace Trainer's `compute_metrics` argument.

    This function is passed directly to MentalHealthTrainer and called
    at every evaluation step. Trainer passes an EvalPrediction object
    with `.predictions` (logits) and `.label_ids` (ground truth labels).

    Returns:
        Dict with keys: 'eval_accuracy', 'eval_macro_f1', 'eval_weighted_f1', 'eval_mcc'
        The 'eval_macro_f1' key is used as the model selection criterion.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(labels, predictions)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mcc": mcc,
    }


def full_evaluation_report(
    model,
    dataset,
    trainer,
    split_name: str = "test",
) -> Dict[str, Any]:
    """
    Run full evaluation on a dataset split and return all metrics + artifacts.

    This is called ONCE at the end on the held-out test set only.
    Do not call this on validation set during development.

    Args:
        model: Trained model (best checkpoint loaded by Trainer).
        dataset: PyTorch Dataset (test split).
        trainer: Fitted MentalHealthTrainer instance.
        split_name: Name of the split for logging ('test' or 'swmh').

    Returns:
        Dict with keys:
          - 'predictions': np.array of predicted class indices
          - 'labels': np.array of ground truth class indices
          - 'accuracy': float
          - 'macro_f1': float
          - 'weighted_f1': float
          - 'mcc': float
          - 'per_class_f1': dict mapping class name -> F1
          - 'confusion_matrix': np.array (num_classes x num_classes)
          - 'classification_report': str
    """
    print(f"\n[Evaluation] Running full evaluation on {split_name} set...")

    # Get predictions from Trainer (handles batching, device placement, etc.)
    pred_output = trainer.predict(dataset)
    logits = pred_output.predictions
    labels = pred_output.label_ids

    predictions = np.argmax(logits, axis=-1)

    # Per-class F1 scores
    per_class_f1_arr = f1_score(labels, predictions, average=None, zero_division=0)
    per_class_f1 = {ID2LABEL[i]: round(float(f), 4) for i, f in enumerate(per_class_f1_arr)}

    # Full metrics
    accuracy = round(accuracy_score(labels, predictions), 4)
    macro_f1 = round(f1_score(labels, predictions, average="macro", zero_division=0), 4)
    weighted_f1 = round(f1_score(labels, predictions, average="weighted", zero_division=0), 4)
    mcc = round(matthews_corrcoef(labels, predictions), 4)

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)

    # Classification report (text)
    target_names = [ID2LABEL[i] for i in range(len(ID2LABEL))]
    report = classification_report(labels, predictions, target_names=target_names, zero_division=0)

    print(f"\n{'='*60}")
    print(f"  [{split_name.upper()} SET RESULTS]")
    print(f"  Accuracy:      {accuracy:.4f}")
    print(f"  Macro F1:      {macro_f1:.4f}   ← Primary metric")
    print(f"  Weighted F1:   {weighted_f1:.4f}")
    print(f"  MCC:           {mcc:.4f}")
    print(f"\n  Per-class F1:")
    for cls_name, f1 in per_class_f1.items():
        print(f"    {cls_name:<22}: {f1:.4f}")
    print(f"\n{report}")
    print(f"{'='*60}\n")

    return {
        "predictions": predictions,
        "labels": labels,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mcc": mcc,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }
