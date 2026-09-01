# MentalScope: Parameter-Efficient Multi-Class Mental Health Text Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Research Project** — Solo work by [Your Name], B.E. CSE (AI&ML), Chandigarh University.
> Accompanying paper: *"Parameter-Efficient Domain Adaptation with Explainability Analysis for Multi-Class Mental Health Classification on Social Media"*

---

## Overview

MentalScope is a rigorous empirical study comparing **LoRA (Parameter-Efficient Fine-Tuning)** against **full fine-tuning** on both general-purpose and domain-specific transformer models for 7-class mental health text classification on Reddit data.

**Novel Contributions:**
1. First systematic LoRA vs. full fine-tuning comparison on domain-specific mental health transformers (MentalBERT, MentalRoBERTa)
2. Class-imbalance-aware training with Focal Loss ablation study
3. SHAP-based explainability analysis of learned mental health linguistic markers

**Classes:** `Depression` · `Anxiety` · `Bipolar` · `Stress` · `Personality Disorder` · `Suicidal` · `Normal`

---

## Results Summary

| Model | Accuracy | Macro F1 | Trainable Params |
|-------|----------|----------|-----------------|
| SVM + TF-IDF (baseline) | - | - | - |
| BERT-base Full FT | - | - | 109M |
| MentalBERT Full FT | - | - | 109M |
| **MentalRoBERTa + LoRA + Focal Loss** | **-** | **-** | **~0.6M** |

*Results to be filled after experiments. See `reports/results_summary.md`*

---

## Project Structure

```
mental health/
├── data/
│   ├── raw/                   # Downloaded datasets (gitignored)
│   ├── processed/             # Cleaned, split datasets
│   └── README.md              # Dataset documentation
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py         # PyTorch Dataset classes
│   │   ├── preprocessing.py   # Text cleaning pipeline
│   │   └── dataloader.py      # DataLoader factory
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifier.py      # Classification head + model wrapper
│   │   └── lora_config.py     # LoRA configuration factory
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py         # HuggingFace Trainer setup
│   │   ├── losses.py          # Focal Loss + Label Smoothing
│   │   └── callbacks.py       # Custom training callbacks
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py         # Macro F1, per-class F1, MCC, confusion matrix
│   └── explainability/
│       ├── __init__.py
│       ├── shap_analysis.py   # SHAP value computation
│       └── attention_viz.py   # Attention visualization
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb # Data cleaning walkthrough
│   ├── 03_training.ipynb      # Training experiments (Colab-ready)
│   ├── 04_evaluation.ipynb    # Results analysis + plots
│   └── 05_explainability.ipynb # SHAP + attention visualization
├── scripts/
│   ├── download_data.py       # Dataset download utility
│   ├── run_experiment.py      # Single experiment runner
│   └── run_all_experiments.py # Full experiment matrix runner
├── configs/
│   ├── base_config.yaml       # Shared hyperparameters
│   ├── full_ft_bert.yaml      # BERT full fine-tuning config
│   ├── lora_bert.yaml         # BERT + LoRA config
│   ├── full_ft_mentalbert.yaml
│   ├── lora_mentalbert.yaml
│   ├── full_ft_roberta.yaml
│   ├── lora_roberta.yaml
│   ├── full_ft_mentalroberta.yaml
│   └── lora_mentalroberta.yaml
├── reports/
│   ├── results_summary.md     # Final results table
│   ├── figures/               # All paper figures (PNG/PDF)
│   └── paper/                 # LaTeX paper source
├── app/
│   ├── app.py                 # Gradio demo app
│   └── requirements_app.txt
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quick Start

### 1. Environment Setup

```bash
# Clone the repo
git clone https://github.com/[your-username]/mental-health-classification
cd mental-health-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

```bash
# Download from Kaggle (requires kaggle API key)
python scripts/download_data.py
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
@article{[yourname]2026mentalscope,
  title={Parameter-Efficient Domain Adaptation with Explainability Analysis 
         for Multi-Class Mental Health Classification on Social Media},
  author={[Your Full Name]},
  year={2026}
}
```

---

## Ethical Note

This project uses publicly available Reddit data for research purposes only. The models produced are **not** intended for clinical diagnosis. Mental health classification from text is a research tool, not a medical device. If you or someone you know needs help, please contact a mental health professional.

---

## Acknowledgments

Built on [MentalBERT](https://huggingface.co/mental/mental-bert-base-uncased) (Ji et al., 2022), [HuggingFace Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), and [SHAP](https://github.com/slundberg/shap).
