"""
MentalScope — Mental Health Text Classification
LoRA configuration factory.

Provides pre-set LoRA configurations for each backbone model,
plus a flexible factory function for custom configurations.
"""

from __future__ import annotations

from peft import LoraConfig, TaskType


# ── LoRA rank presets ─────────────────────────────────────────────────────────

LORA_RANK_PRESETS = {
    "r4":  {"r": 4,  "lora_alpha": 8,  "description": "Minimal — fastest, fewest params"},
    "r8":  {"r": 8,  "lora_alpha": 16, "description": "Default — good balance"},
    "r16": {"r": 16, "lora_alpha": 32, "description": "Rich — closer to full FT capacity"},
}


# ── Target modules per architecture ──────────────────────────────────────────

# Which attention projection matrices to adapt.
# BERT-family: query + value (standard choice from original LoRA paper)
# We deliberately avoid adapting key/output to keep trainable param count low.
TARGET_MODULES = {
    "bert":         ["query", "value"],
    "roberta":      ["query", "value"],
    "distilbert":   ["q_lin", "v_lin"],   # Different naming in DistilBERT
    "default":      ["query", "value"],   # Fallback for unknown architectures
}


def get_lora_config(
    model_type: str = "bert",
    rank: int = 8,
    lora_alpha: Optional[int] = None,
    lora_dropout: float = 0.1,
    bias: str = "none",
) -> LoraConfig:
    """
    Build a LoraConfig for a given backbone architecture and rank.

    Args:
        model_type: Architecture type — 'bert', 'roberta', or 'distilbert'.
                    Also works for MentalBERT and MentalRoBERTa (same architecture).
        rank: LoRA rank r. Higher = more expressive but more parameters.
              Ablation values: 4, 8, 16. Default: 8.
        lora_alpha: Scaling factor. If None, defaults to 2 * rank (standard).
        lora_dropout: Dropout on LoRA layers. Default: 0.1.
        bias: Bias training mode. 'none' = don't update biases (recommended).

    Returns:
        Configured peft.LoraConfig instance.

    Example:
        >>> cfg = get_lora_config("roberta", rank=8)
        >>> model = get_peft_model(base_model, cfg)
    """
    if lora_alpha is None:
        lora_alpha = rank * 2  # Standard scaling from LoRA paper

    target_modules = TARGET_MODULES.get(model_type.lower(), TARGET_MODULES["default"])

    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        inference_mode=False,
    )

    # Human-readable summary
    # Approximate trainable params: 2 * rank * d_model * num_layers * len(target_modules)
    # For BERT-base: 2 * 8 * 768 * 12 * 2 ≈ 2.36M — compare to 109M total
    approx_params = 2 * rank * 768 * 12 * len(target_modules)
    print(
        f"[LoRA] rank={rank}, alpha={lora_alpha}, "
        f"modules={target_modules}, "
        f"approx trainable params ~{approx_params/1e6:.2f}M"
    )

    return config


# ── Optional: add None to handle the typing annotation above ─────────────────
from typing import Optional  # noqa: E402 (put here to avoid circular at top)
