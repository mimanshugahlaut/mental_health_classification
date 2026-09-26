# MentalScope: Parameter-Efficient Multi-Class Mental Health Text Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Research Project** — Work by Shreyansh Gupta, Mimanshu Gahlaut, and Ranjan Walia, AIT-CSE, Chandigarh University.
> Accompanying paper: *"Parameter-Efficient Domain Adaptation with Explainability Analysis for Multi-Class Mental Health Classification on Social Media"*

---

## Overview

MentalScope is a rigorous empirical study comparing **LoRA (Parameter-Efficient Fine-Tuning)** against **full fine-tuning** on both general-purpose and domain-specific transformer models for 6-class mental health text classification on Reddit data.

**Novel Contributions:**
1. First systematic LoRA vs. full fine-tuning comparison on domain-specific mental health transformers (MentalBERT, MentalRoBERTa)
2. Class-imbalance-aware training with Focal Loss ablation study
3. SHAP-based explainability analysis of learned mental health linguistic markers

**Classes:** `Depression` · `Anxiety` · `Bipolar` · `Stress` · `Suicidal` · `Normal`

---

## Results Summary

| Model | Accuracy | Macro F1 | Trainable Params | Training Time |
|-------|----------|----------|------------------|---------------|
| SVM + TF-IDF (baseline) | 77.74% | 75.24% | - | - |
| BERT-base Full FT | 83.68% | 83.22% | 109.5M (100%) | 33.2 min |
| MentalBERT Full FT + Focal | **83.36%** | **83.85%** | 109.5M (100%) | 38.0 min |
| **MentalRoBERTa + LoRA + Focal** | 82.32% | 82.41% | **~0.89M (0.71%)**| **13.3 min** |

*Detailed results can be found in `MentalHealth_Classification/results/all_experiments_comparison.csv`*

---

## Project Structure

```text
mental_health_classification/
├── MentalHealth_Classification/
│   ├── analysis/              # Comparison plots & SHAP explainability figures
│   │   ├── plots/             # Accuracy, F1, MCC metric charts
│   │   └── shap_figures/      # Per-class SHAP global importance plots
│   └── results/               # Experiment metrics (CSV, JSON) & model comparison plot
├── app/
│   └── app.py                 # Gradio interactive web demo
├── configs/
│   ├── base_config.yaml       # Shared training & model hyperparameters
│   ├── full_ft_*.yaml         # Full fine-tuning configurations (BERT, MentalBERT)
│   └── lora_*.yaml            # LoRA fine-tuning configurations (BERT, MentalBERT, MentalRoBERTa)
├── data/                      # Local dataset folder (downloaded from Kaggle, gitignored)
│   ├── raw/                   # Raw sentiment-analysis CSV
│   └── processed/             # Cleaned & processed CSV dataset
├── reports/
│   └── results/               # Baseline results (classical ML)
├── scripts/
│   ├── prepare_data.py        # Data cleaning and preprocessing pipeline
│   ├── run_baselines.py       # Classical baseline runner (Linear SVM, Logistic Regression, RF)
│   ├── run_experiment.py      # Transformer training and LoRA experiment runner
│   └── aggregate_results.py   # Aggregates evaluation results across all runs
├── src/
│   ├── data/                  # PyTorch Dataset classes and preprocessing pipelines
│   ├── models/                # Classification model architecture and LoRA adapters
│   ├── training/              # Custom Trainer, Focal Loss, and Label Smoothing
│   ├── evaluation/            # Metrics computation (Macro F1, per-class F1, MCC)
│   └── explainability/        # SHAP explainer implementation
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Environment Setup

```bash
# Clone the repo
git clone https://github.com/mimanshugahlaut/mental_health_classification
cd mental_health_classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset & Preprocessing

The dataset is hosted on Kaggle:
- **Kaggle Dataset**: [Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)

Download the dataset, place the raw CSV at `data/raw/mental_health.csv`, and run the preprocessing pipeline:
```bash
python scripts/prepare_data.py
```

### 3. Run a Single Experiment

```bash
# Example: MentalBERT + LoRA + Focal Loss
python scripts/run_experiment.py --config configs/lora_mentalbert.yaml
```

### 4. Run Full Experiment Matrix

```bash
python scripts/run_all_experiments.py
```

---

## Google Colab

All notebooks in `notebooks/` are Colab-ready. Start with:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

`notebooks/03_training.ipynb` contains the full training pipeline optimized for T4 GPU.

---

## Gradio Demo

```bash
cd app
pip install -r requirements_app.txt
python app.py
```

---

## Citation

```bibtex
@article{gupta2026mentalscope,
  title={Parameter-Efficient Domain Adaptation with Explainability Analysis 
         for Multi-Class Mental Health Classification on Social Media},
  author={Gupta, Shreyansh and Gahlaut, Mimanshu and Walia, Ranjan},
  year={2026}
}
```

---

## Ethical Note

This project uses publicly available Reddit data for research purposes only. The models produced are **not** intended for clinical diagnosis. Mental health classification from text is a research tool, not a medical device. If you or someone you know needs help, please contact a mental health professional.

---

## Acknowledgments

Built on [MentalBERT](https://huggingface.co/mental/mental-bert-base-uncased) (Ji et al., 2022), [HuggingFace Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), and [SHAP](https://github.com/slundberg/shap).
