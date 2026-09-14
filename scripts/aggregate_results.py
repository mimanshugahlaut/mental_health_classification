"""
MentalScope — Experiment results aggregator.
Reads all individual experiment JSON files and produces a unified
comparison table (markdown + CSV) for the paper.

Run after all experiments complete:
    python scripts/aggregate_results.py
"""

import json
from pathlib import Path
import pandas as pd


def main():
    results_dir = Path("reports/results")
    result_files = list(results_dir.glob("*.json"))

    if not result_files:
        print("[Aggregate] No result files found in reports/results/")
        return

    rows = []
    for f in sorted(result_files):
        with open(f) as fp:
            data = json.load(fp)

        # Handle both baseline (list) and experiment (dict) formats
        if isinstance(data, list):
            for item in data:
                rows.append({
                    "Experiment": item.get("model", f.stem),
                    "Accuracy": item.get("accuracy", "-"),
                    "Macro F1": item.get("macro_f1", "-"),
                    "Weighted F1": item.get("weighted_f1", "-"),
                    "MCC": item.get("mcc", "-"),
                    "Trainable Params": item.get("trainable_params", "-"),
                    "Training Time (min)": "-",
                })
        else:
            params_str = "-"
            if "trainable_params" in data and isinstance(data["trainable_params"], (int, float)):
                pct = data.get("trainable_params_pct", "")
                pct_str = f" ({pct}%)" if pct != "" else ""
                params_str = f"{data['trainable_params']:,}{pct_str}"
            elif "trainable_params" in data:
                params_str = str(data["trainable_params"])

            rows.append({
                "Experiment": data.get("experiment_name", f.stem),
                "Accuracy": data.get("accuracy", "-"),
                "Macro F1": data.get("macro_f1", "-"),
                "Weighted F1": data.get("weighted_f1", "-"),
                "MCC": data.get("mcc", "-"),
                "Trainable Params": params_str,
                "Training Time (min)": data.get("training_minutes", "-"),
            })

    df = pd.DataFrame(rows)
    sort_col = pd.to_numeric(df["Macro F1"], errors="coerce")
    df = df.iloc[sort_col.fillna(-1).argsort()[::-1]].reset_index(drop=True)

    # Save CSV
    csv_path = Path("reports/results_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Aggregate] Saved CSV: {csv_path}")

    # Save Markdown table
    md_path = Path("reports/results_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# MentalScope — Experiment Results Summary\n\n")
        try:
            f.write(df.to_markdown(index=False))
        except (ImportError, Exception):
            # Fallback when tabulate is not installed
            headers = list(df.columns)
            header_line = "| " + " | ".join(headers) + " |"
            separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
            rows_lines = [
                "| " + " | ".join(str(val) for val in row) + " |"
                for row in df.itertuples(index=False)
            ]
            f.write("\n".join([header_line, separator_line] + rows_lines))
        f.write("\n\n*Primary metric: Macro F1 (macro-averaged F1 over all 6 classes)*\n")
    print(f"[Aggregate] Saved markdown: {md_path}")

    # Print to console
    print("\n" + "="*80)
    print("  RESULTS SUMMARY (sorted by Macro F1)")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)


if __name__ == "__main__":
    main()
