"""
MentalScope — Mental Health Text Classification
Main experiment runner script.

Run a single experiment from a YAML config file:
    python scripts/run_experiment.py --config configs/lora_mentalbert.yaml

Designed for Colab compatibility:
  - Saves checkpoint every N steps (resume after disconnect)
  - Logs to wandb (optional) or local JSON
  - Saves full evaluation report as JSON for later aggregation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
import pandas as pd

# Add project root to path (for Colab where you run from notebooks/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import preprocess_dataframe, compute_class_weights
from src.data.dataset import build_datasets
from src.models.classifier import load_model
from src.training.trainer import MentalHealthTrainer, build_training_args
from src.evaluation.metrics import compute_metrics, full_evaluation_report


def load_config(config_path: str) -> dict:
    """Load YAML experiment config."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"\n[Config] Loaded: {config_path}")
    print(json.dumps(cfg, indent=2))
    return cfg


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[Seed] Random seed set to {seed}")


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    # ── 1. Load and preprocess data ──────────────────────────────────────────
    print("\n[Step 1] Loading dataset...")
    data_path = cfg["data"]["path"]
    df = pd.read_csv(data_path)
    df = preprocess_dataframe(
        df,
        text_col=cfg["data"].get("text_col", "text"),
        label_col=cfg["data"].get("label_col", "status"),
    )

    # ── 2. Split data (stratified) ───────────────────────────────────────────
    print("\n[Step 2] Splitting dataset (70/15/15)...")
    seed = cfg.get("seed", 42)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=seed
    )
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ── 3. Load model + tokenizer ────────────────────────────────────────────
    print(f"\n[Step 3] Loading model: {cfg['model']['key']}...")
    model, tokenizer = load_model(
        model_key=cfg["model"]["key"],
        use_lora=cfg["model"].get("use_lora", False),
        lora_rank=cfg["model"].get("lora_rank", 8),
        dropout=cfg["model"].get("dropout", 0.1),
    )

    # ── 4. Build datasets ────────────────────────────────────────────────────
    print("\n[Step 4] Tokenizing datasets...")
    max_length = cfg.get("max_length", 256)  # 256 is sweet spot for Reddit posts
    train_dataset, val_dataset, test_dataset = build_datasets(
        train_df, val_df, test_df, tokenizer, max_length=max_length
    )

    # ── 5. Compute class weights ─────────────────────────────────────────────
    class_weights = compute_class_weights(train_df)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    # ── 6. Build TrainingArguments ───────────────────────────────────────────
    experiment_name = cfg.get("experiment_name", "experiment")
    output_dir = f"checkpoints/{experiment_name}"

    training_args = build_training_args(
        output_dir=output_dir,
        num_epochs=cfg["training"].get("epochs", 5),
        batch_size=cfg["training"].get("batch_size", 16),
        learning_rate=cfg["training"].get("learning_rate", 2e-5),
        weight_decay=cfg["training"].get("weight_decay", 0.01),
        warmup_ratio=cfg["training"].get("warmup_ratio", 0.1),
        fp16=torch.cuda.is_available(),
        seed=seed,
        run_name=experiment_name if cfg.get("use_wandb", False) else None,
    )

    # ── 7. Build and run Trainer ─────────────────────────────────────────────
    print("\n[Step 7] Initializing trainer...")
    from transformers import EarlyStoppingCallback

    trainer = MentalHealthTrainer(
        loss_type=cfg["training"].get("loss", "focal"),
        class_weights=class_weights_tensor,
        gamma=cfg["training"].get("focal_gamma", 2.0),
        smoothing=cfg["training"].get("label_smoothing", 0.1),
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print(f"\n[Step 7] Starting training — {experiment_name}...")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    print(f"\n[Training] Completed in {training_time/60:.1f} minutes")

    # ── 8. Final test evaluation ─────────────────────────────────────────────
    print("\n[Step 8] Final evaluation on test set...")
    results = full_evaluation_report(model, test_dataset, trainer, split_name="test")
    results["experiment_name"] = experiment_name
    results["training_minutes"] = round(training_time / 60, 1)
    results["config"] = cfg

    # ── 9. Save results ──────────────────────────────────────────────────────
    results_dir = Path("reports/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{experiment_name}_results.json"

    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in results.items()
        if k not in {"predictions", "labels"}  # Skip large arrays
    }
    results_serializable["classification_report"] = results["classification_report"]

    with open(results_path, "w") as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\n[Results] Saved to: {results_path}")
    print(f"\n{'='*50}")
    print(f"  EXPERIMENT COMPLETE: {experiment_name}")
    print(f"  Macro F1:  {results['macro_f1']:.4f}")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  MCC:       {results['mcc']:.4f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single MentalScope experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML experiment config (e.g., configs/lora_mentalbert.yaml)",
    )
    args = parser.parse_args()
    main(args.config)
