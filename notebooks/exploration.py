"""Day 1: E-Commerce Post-Dispatch RTO Risk Predictor - Exploratory Data Analysis & Schema Inspection

This script runs the Day 1 exploration on the raw orders dataset and processed splits.
"""

import io
import pandas as pd
import numpy as np


def main():
    print("=" * 80)
    print("DAY 1: DATASET EXPLORATION & TARGET VERIFICATION")
    print("=" * 80)

    # 1. Load raw orders
    df = pd.read_csv("data/raw/orders.csv")
    print(f"\n1. Loaded data/raw/orders.csv: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # 2. Shape and columns
    print("\n2. Columns inventory:")
    for idx, col in enumerate(df.columns, 1):
        print(f"  {idx:2d}. {col} ({df[col].dtype})")

    # 3. Head preview
    print("\n3. First 3 rows preview:")
    print(df.head(3).T)

    # 4. Info, Missing values, Duplicates
    print("\n4. Missing values & Duplicates check:")
    nulls = df.isnull().sum()
    print(f"  Total missing values across all columns: {nulls.sum()}")
    print(f"  Duplicate rows count: {df.duplicated().sum()}")

    # 5. Categorical column breakdown
    print("\n5. Categorical / Object columns summary:")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        print(f"  - {col}: {df[col].nunique():,} unique values. Top 3: {dict(df[col].value_counts().head(3))}")

    # 6. Target Variable Verification
    print("\n6. Target Variable ('rto') Verification:")
    counts = df["rto"].value_counts()
    pcts = df["rto"].value_counts(normalize=True) * 100
    for label, count, pct, meaning in zip(
        counts.index, counts.values, pcts.values,
        ["Delivered Successfully (Class 0)", "Returned to Origin (RTO - Class 1)"]
    ):
        print(f"  Label {label} ({meaning}): {count:,} orders ({pct:.2f}%)")

    # 7. Pre-dispatch Driver Breakdown
    print("\n7. Key Driver RTO Breakdown:")
    print("\n  [Payment Mode]:")
    print(df.groupby("payment_mode")["rto"].agg(["count", "mean"]).rename(columns={"mean": "rto_rate"}))

    print("\n  [Pincode Tier]:")
    print(df.groupby("pincode_tier")["rto"].agg(["count", "mean"]).rename(columns={"mean": "rto_rate"}))

    print("\n  [Category]:")
    print(df.groupby("category")["rto"].agg(["count", "mean"]).rename(columns={"mean": "rto_rate"}))

    print("\n  [Festive Period]:")
    print(df.groupby("is_festive")["rto"].agg(["count", "mean"]).rename(columns={"mean": "rto_rate"}))

    # 8. Chronological Splits
    train = pd.read_parquet("data/processed/train.parquet")
    val = pd.read_parquet("data/processed/val.parquet")
    test = pd.read_parquet("data/processed/test.parquet")

    print("\n8. Chronological Splits (No Future Leakage):")
    print(f"  Train : {train.shape[0]:,} rows x {train.shape[1]} cols | {train['order_ts'].min().strftime('%Y-%m-%d')} to {train['order_ts'].max().strftime('%Y-%m-%d')} | Base RTO: {train['rto'].mean():.2%}")
    print(f"  Val   : {val.shape[0]:,} rows x {val.shape[1]} cols | {val['order_ts'].min().strftime('%Y-%m-%d')} to {val['order_ts'].max().strftime('%Y-%m-%d')} | Base RTO: {val['rto'].mean():.2%}")
    print(f"  Test  : {test.shape[0]:,} rows x {test.shape[1]} cols | {test['order_ts'].min().strftime('%Y-%m-%d')} to {test['order_ts'].max().strftime('%Y-%m-%d')} | Base RTO: {test['rto'].mean():.2%}")
    print("\nDay 1 Dataset Exploration Complete.")


if __name__ == "__main__":
    main()
