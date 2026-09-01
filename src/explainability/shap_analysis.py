"""
MentalScope — Mental Health Text Classification
SHAP-based explainability analysis.

Computes SHAP values for transformer models to identify which words/tokens
contribute most to each mental health classification decision.

Used to generate:
  - Global feature importance plots (per class, per model)
  - Instance-level explanation for paper examples
  - Mental health linguistic marker analysis

NOTE: SHAP with transformers is memory-intensive.
      Run on CPU with small sample sets (100-200 samples).
"""

from __future__ import annotations

from typing import List, Optional, Dict
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from transformers import pipeline

from src.data.preprocessing import ID2LABEL


def build_shap_explainer(
    model,
    tokenizer,
    device: str = "cpu",
    max_length: int = 128,
) -> shap.Explainer:
    """
    Build a SHAP Partition explainer for a transformer sequence classifier.

    Uses `shap.Explainer` with HuggingFace pipeline as the prediction function.
    PartitionExplainer handles tokenized text naturally.

    Args:
        model: Trained classification model (best checkpoint).
        tokenizer: Corresponding tokenizer.
        device: 'cpu' (recommended for SHAP — avoids VRAM issues) or 'cuda'.
        max_length: Max token length for SHAP explanations (shorter = faster).

    Returns:
        SHAP Explainer object ready for `.shap_values()` calls.
    """
    # Move model to device
    model = model.to(device)
    model.eval()

    # Build HuggingFace pipeline (SHAP works best with pipeline interface)
    cls_pipeline = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
        top_k=None,       # Return all class scores
        truncation=True,
        max_length=max_length,
        padding=True,
    )

    # Masker: handles subword tokenization for SHAP
    masker = shap.maskers.Text(tokenizer=tokenizer)

    explainer = shap.Explainer(
        model=cls_pipeline,
        masker=masker,
        output_names=list(ID2LABEL.values()),
    )

    return explainer


def compute_shap_values(
    explainer: shap.Explainer,
    texts: List[str],
    batch_size: int = 4,
) -> shap.Explanation:
    """
    Compute SHAP values for a list of input texts.

    Args:
        explainer: SHAP Explainer (from build_shap_explainer).
        texts: List of text samples to explain.
        batch_size: SHAP batch size. Keep low (4-8) to avoid OOM on CPU.

    Returns:
        shap.Explanation object containing values, base_values, and data.
    """
    print(f"[SHAP] Computing SHAP values for {len(texts)} samples...")
    shap_values = explainer(texts, batch_size=batch_size)
    print("[SHAP] Done.")
    return shap_values


def plot_global_importance(
    shap_values: shap.Explanation,
    class_name: str,
    top_n: int = 20,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot top-N tokens by mean absolute SHAP value for a specific class.
    This is your paper's Figure 4 (or equivalent).

    Args:
        shap_values: SHAP Explanation from compute_shap_values.
        class_name: Class label string (e.g., 'Depression').
        top_n: Number of top features to show.
        save_path: If provided, saves figure as PNG to this path.
    """
    # Get class index
    class_idx = {v: k for k, v in ID2LABEL.items()}.get(class_name, 0)

    # Extract SHAP values for this class: (num_samples, num_tokens)
    class_shap = shap_values[:, :, class_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(class_shap, max_display=top_n, ax=ax, show=False)
    ax.set_title(f"Top {top_n} SHAP Features — {class_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SHAP] Saved global importance plot: {save_path}")

    plt.show()


def plot_instance_explanation(
    shap_values: shap.Explanation,
    sample_idx: int,
    class_name: str,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot token-level SHAP contributions for a single text sample.
    This generates your paper's qualitative examples section.

    Args:
        shap_values: SHAP Explanation from compute_shap_values.
        sample_idx: Index of the sample in the explanations.
        class_name: Class to explain contributions for.
        save_path: Optional path to save the figure.
    """
    class_idx = {v: k for k, v in ID2LABEL.items()}.get(class_name, 0)
    sample_explanation = shap_values[sample_idx, :, class_idx]

    shap.plots.text(sample_explanation, display=False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SHAP] Saved instance explanation: {save_path}")
    plt.show()


def extract_top_markers(
    shap_values: shap.Explanation,
    class_name: str,
    top_n: int = 30,
) -> Dict[str, float]:
    """
    Extract top linguistic markers (tokens) for a class by mean |SHAP|.

    This generates the data for your paper's Table 4
    "Top Linguistic Markers per Mental Health Category".

    Args:
        shap_values: SHAP Explanation.
        class_name: Target class.
        top_n: Number of top tokens to return.

    Returns:
        Dict mapping token string -> mean absolute SHAP value,
        sorted descending.
    """
    class_idx = {v: k for k, v in ID2LABEL.items()}.get(class_name, 0)

    # Aggregate tokens across all samples
    token_shap: Dict[str, List[float]] = {}

    for i in range(len(shap_values)):
        sample = shap_values[i]
        tokens = sample.data          # Token strings
        values = sample.values        # (num_tokens, num_classes)
        class_values = values[:, class_idx]

        for token, val in zip(tokens, class_values):
            token = token.strip().lower()
            if token and token not in {"[cls]", "[sep]", "[pad]", "<s>", "</s>"}:
                token_shap.setdefault(token, []).append(abs(val))

    # Mean |SHAP| per token
    mean_shap = {tok: float(np.mean(vals)) for tok, vals in token_shap.items()}
    sorted_markers = dict(sorted(mean_shap.items(), key=lambda x: x[1], reverse=True)[:top_n])

    print(f"\n[SHAP] Top {top_n} markers for '{class_name}':")
    for tok, score in sorted_markers.items():
        print(f"  {tok:<20}: {score:.4f}")

    return sorted_markers
