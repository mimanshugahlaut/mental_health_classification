"""
MentalScope — Traditional ML Baselines
Trains and evaluates Logistic Regression, SVM, and Random Forest
with TF-IDF features as traditional ML baselines.

Run:
    python scripts/run_baselines.py --data data/processed/mental_health_cleaned.csv

These results populate the first rows of your paper's Table 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.preprocessing import preprocess_dataframe, ID2LABEL

SEED = 42


def evaluate_model(name: str, clf, X_test, y_test) -> dict:
    """Evaluate a fitted classifier and print results."""
    y_pred = clf.predict(X_test)

    acc = round(accuracy_score(y_test, y_pred), 4)
    macro_f1 = round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)
    weighted_f1 = round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4)
    mcc = round(matthews_corrcoef(y_test, y_pred), 4)

    target_names = [ID2LABEL[i] for i in range(len(ID2LABEL))]
    report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print(f"  MCC:         {mcc:.4f}")
    print(f"\n{report}")
    print(f"{'='*55}")

    return {
        "model": name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mcc": mcc,
        "classification_report": report,
        "trainable_params": "N/A",
    }


def main(data_path: str) -> None:
    print("[Baselines] Loading and preprocessing data...")
    df = pd.read_csv(data_path)
    df = preprocess_dataframe(df)

    # Split
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED)

    X_train = train_df["text"].tolist()
    y_train = train_df["label"].tolist()
    X_test = test_df["text"].tolist()
    y_test = test_df["label"].tolist()

    print(f"[Baselines] Train: {len(X_train)} | Test: {len(X_test)}")

    # TF-IDF vectorizer (shared across all models)
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),   # Unigrams + bigrams
        max_features=50000,
        min_df=2,
        sublinear_tf=True,    # Use log(tf) instead of raw tf
    )

    all_results = []

    # ── Logistic Regression ──────────────────────────────────────────────────
    print("\n[1/3] Training Logistic Regression...")
    lr_clf = Pipeline([
        ("tfidf", tfidf),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )),
    ])
    lr_clf.fit(X_train, y_train)
    all_results.append(evaluate_model("Logistic Regression + TF-IDF", lr_clf, X_test, y_test))

    # ── Linear SVM ───────────────────────────────────────────────────────────
    print("\n[2/3] Training Linear SVM...")
    svm_clf = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), max_features=50000, min_df=2, sublinear_tf=True
        )),
        ("clf", LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=SEED,
            max_iter=2000,
        )),
    ])
    svm_clf.fit(X_train, y_train)
    all_results.append(evaluate_model("Linear SVM + TF-IDF", svm_clf, X_test, y_test))

    # ── Random Forest ────────────────────────────────────────────────────────
    print("\n[3/3] Training Random Forest...")
    rf_clf = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), max_features=30000, min_df=2, sublinear_tf=True
        )),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )),
    ])
    rf_clf.fit(X_train, y_train)
    all_results.append(evaluate_model("Random Forest + TF-IDF", rf_clf, X_test, y_test))

    # ── Save results ──────────────────────────────────────────────────────────
    output_dir = Path("reports/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline_results.json"

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[Baselines] Results saved to: {output_path}")

    # Print comparison table
    print("\n" + "="*60)
    print("  BASELINE COMPARISON SUMMARY")
    print("="*60)
    print(f"  {'Model':<30} {'Accuracy':>10} {'Macro F1':>10} {'MCC':>8}")
    print("  " + "-"*56)
    for r in all_results:
        print(f"  {r['model']:<30} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f} {r['mcc']:>8.4f}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate traditional ML baselines")
    parser.add_argument("--data", type=str, default="data/processed/mental_health_cleaned.csv")
    args = parser.parse_args()
    main(args.data)
