"""
MentalScope — Mental Health Text Classification
Custom HuggingFace Trainer with pluggable loss function.

Overrides `compute_loss()` to inject our custom Focal Loss / Weighted CE
while retaining all other Trainer functionality (checkpointing,
evaluation, logging, mixed-precision, etc.).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.modeling_outputs import SequenceClassifierOutput

from src.training.losses import get_loss_fn


class MentalHealthTrainer(Trainer):
    """
    HuggingFace Trainer subclass that injects a custom loss function.

    Only `compute_loss` is overridden — everything else (optimizer,
    scheduler, gradient clipping, evaluation, logging) is inherited
    from the base Trainer class.

    Args:
        loss_type: 'ce' | 'weighted_ce' | 'focal' | 'focal_smooth'
        class_weights: Tensor of class weights (shape: num_classes).
        gamma: Focal Loss gamma parameter.
        smoothing: Label smoothing epsilon.
        *args, **kwargs: Passed to base Trainer.
    """

    def __init__(
        self,
        loss_type: str = "focal",
        class_weights: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        smoothing: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_fn = get_loss_fn(
            loss_type=loss_type,
            class_weights=class_weights,
            gamma=gamma,
            smoothing=smoothing,
        )
        print(f"[Trainer] Loss function: {loss_type} | gamma={gamma} | smoothing={smoothing}")

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict,
        return_outputs: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, SequenceClassifierOutput]]:
        """
        Override Trainer.compute_loss to use our custom loss function.

        HuggingFace passes `inputs` as a dict with keys matching the model's
        forward() signature. We pop 'labels' out (to avoid HF computing its
        own built-in CE loss) and compute ours instead.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        logits = outputs.logits  # (batch_size, num_labels)
        loss = self.loss_fn(logits, labels)

        # Put labels back so other Trainer internals work correctly
        inputs["labels"] = labels

        return (loss, outputs) if return_outputs else loss


def build_training_args(
    output_dir: str,
    num_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    eval_steps: int = 200,
    save_steps: int = 200,
    fp16: bool = True,
    seed: int = 42,
    run_name: Optional[str] = None,
) -> TrainingArguments:
    """
    Build TrainingArguments with best-practice defaults for Colab T4.

    Key design decisions:
    - fp16=True for T4 GPU (saves ~40% VRAM, 30% training time)
    - eval_strategy='steps' + load_best_model_at_end=True enables early stopping
    - gradient_accumulation_steps=2 simulates batch_size=32 on limited VRAM
    - save_total_limit=2 prevents Colab disk from filling up

    Args:
        output_dir: Directory to save checkpoints.
        num_epochs: Total training epochs. 5 works well for BERT-class models.
        batch_size: Per-device train/eval batch size. 16 fits T4 for BERT-base.
        learning_rate: AdamW learning rate. 2e-5 is standard for BERT fine-tuning.
        weight_decay: L2 regularization coefficient.
        warmup_ratio: Fraction of total steps used for LR warmup.
        eval_steps: Evaluate every N steps.
        save_steps: Save checkpoint every N steps.
        fp16: Use mixed-precision (float16). Keep True on T4.
        seed: Random seed for reproducibility.
        run_name: WandB / MLflow run name.
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,  # Larger eval batch (no grad)
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",             # Cosine decay (better than linear for longer runs)
        gradient_accumulation_steps=2,          # Effective batch = batch_size * 2 = 32
        fp16=fp16,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        load_best_model_at_end=True,            # Required for EarlyStoppingCallback
        metric_for_best_model="eval_macro_f1",  # Use macro F1, not accuracy
        greater_is_better=True,
        save_total_limit=2,                     # Keep only best + last checkpoint
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        report_to="wandb" if run_name else "none",
        run_name=run_name,
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )
