"""
MentalScope — Data Preparation Script
Reads the raw dataset, runs the cleaning pipeline, and saves the processed version.

Run:
    python scripts/prepare_data.py
"""

import os
from pathlib import Path
import pandas as pd
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.preprocessing import preprocess_dataframe

def main():
    raw_path = Path("data/raw/mental_health.csv")
    processed_dir = Path("data/processed")
    processed_path = processed_dir / "mental_health_cleaned.csv"

    if not raw_path.exists():
        print(f"❌ Error: Raw data file not found at {raw_path}")
        print("Please download the dataset from Kaggle and place it there.")
        print("Dataset: https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health")
        print("Rename the downloaded CSV to 'mental_health.csv'")
        return

    print(f"Reading raw data from {raw_path}...")
    df = pd.read_csv(raw_path)
    
    print("Running preprocessing pipeline...")
    # The dataset uses 'statement' for text and 'status' for labels
    df_clean = preprocess_dataframe(df, text_col="statement", label_col="status")
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(processed_path, index=False)
    
    print(f"✅ Successfully saved processed data to {processed_path}")
    print(f"Total samples: {len(df_clean)}")

if __name__ == "__main__":
    main()
