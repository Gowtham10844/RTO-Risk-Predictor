"""src/train_model.py

Day 2: Data Preprocessing Pipeline & Baseline Logistic Regression Model.

This module builds a leakage-safe scikit-learn ColumnTransformer and Pipeline,
fits strictly on the chronological training split, and generates predictions
and probability estimates on the validation split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Feature Definitions (Leakage-Safe Pre-Dispatch Features)
# ---------------------------------------------------------------------------

CATEGORICAL_FEATURES: List[str] = [
    "payment_mode",
    "category",
    "pincode_tier",
    "is_festive",
    "is_alternate_address",
]

NUMERICAL_FEATURES: List[str] = [
    "order_value",
    "discount_pct",
    "delivery_days_est",
    "account_age_days",
    "order_velocity_24h",
]

TARGET_COLUMN: str = "rto"

# Explicitly excluded columns (PII, IDs, latents, post-dispatch data)
EXCLUDED_COLUMNS: List[str] = [
    "order_id",
    "order_ts",  # Used strictly for chronological ordering / splitting
    "customer_id",
    "customer_name",
    "phone",
    "address_line",
    "pincode",
    "city",
    "district",
    "state",
    "order_value_band",
    "signup_date",
    "is_cod",  # Redundant with payment_mode
    "_pincode_rto_prior",
    "_reliability_z",
    "_address_quality",
    "_p_rto_true",
]


def load_data(
    data_dir: str | Path = "data",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load chronological train, validation, and test splits.

    Prefers pre-generated parquet files in data/processed/. If not present,
    falls back to loading data/raw/orders.csv and performing a strict
    chronological 70/10/20 split sorted by order_ts.
    """
    data_path = Path(data_dir)
    train_parquet = data_path / "processed" / "train.parquet"
    val_parquet = data_path / "processed" / "val.parquet"
    test_parquet = data_path / "processed" / "test.parquet"

    if train_parquet.exists() and val_parquet.exists() and test_parquet.exists():
        train = pd.read_parquet(train_parquet)
        val = pd.read_parquet(val_parquet)
        test = pd.read_parquet(test_parquet)
        return train, val, test

    raw_csv = data_path / "raw" / "orders.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"Dataset not found at {train_parquet} or {raw_csv}."
        )

    df = pd.read_csv(raw_csv)
    df["order_ts"] = pd.to_datetime(df["order_ts"])
    df = df.sort_values("order_ts").reset_index(drop=True)

    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.10)

    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train : n_train + n_val].copy()
    test = df.iloc[n_train + n_val :].copy()

    return train, val, test


def build_preprocessor(
    categorical_cols: List[str] = CATEGORICAL_FEATURES,
    numerical_cols: List[str] = NUMERICAL_FEATURES,
) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer.

    - Categorical/Boolean features: SimpleImputer(most_frequent) + OneHotEncoder(ignore unknown)
    - Continuous Numerical features: SimpleImputer(median) + StandardScaler()
    """
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, categorical_cols),
            ("num", num_pipeline, numerical_cols),
        ]
    )
    return preprocessor


def build_baseline_pipeline(
    preprocessor: ColumnTransformer = None,
    random_state: int = 42,
) -> Pipeline:
    """Wrap preprocessor and Logistic Regression inside a scikit-learn Pipeline."""
    if preprocessor is None:
        preprocessor = build_preprocessor()

    classifier = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


def train_and_validate(
    pipeline: Pipeline,
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: List[str] = CATEGORICAL_FEATURES + NUMERICAL_FEATURES,
    target_col: str = TARGET_COLUMN,
) -> Dict[str, Any]:
    """Fit pipeline strictly on training split and evaluate on validation split.

    Guarantees that fit() is never called on validation or test data,
    preventing any form of data leakage.
    """
    X_train = train[feature_cols]
    y_train = train[target_col]

    X_val = val[feature_cols]
    y_val = val[target_col]

    # Fit strictly on training split
    pipeline.fit(X_train, y_train)

    # Predict discrete classes and continuous probabilities on validation split
    val_preds = pipeline.predict(X_val)
    val_probs = pipeline.predict_proba(X_val)[:, 1]

    # Compute metrics
    acc = accuracy_score(y_val, val_preds)
    prec = precision_score(y_val, val_preds, zero_division=0)
    rec = recall_score(y_val, val_preds)
    f1 = f1_score(y_val, val_preds)
    roc_auc = roc_auc_score(y_val, val_probs)
    cm = confusion_matrix(y_val, val_preds)

    return {
        "pipeline": pipeline,
        "predictions": val_preds,
        "probabilities": val_probs,
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
        },
        "confusion_matrix": cm,
        "X_val": X_val,
        "y_val": y_val,
    }


def inspect_model_weights(pipeline: Pipeline) -> pd.DataFrame:
    """Extract and sort learned Logistic Regression coefficients."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    df_coefs = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
        }
    ).sort_values("coefficient", ascending=False)
    return df_coefs.reset_index(drop=True)


def main():
    print("=" * 80)
    print("DAY 2: DATA PREPROCESSING & BASELINE LOGISTIC REGRESSION MODEL")
    print("=" * 80)

    # 1. Load data
    train, val, test = load_data()
    print(f"\n1. Data Loaded:")
    print(f"   Train split : {train.shape[0]:,} rows x {train.shape[1]} cols")
    print(f"   Val split   : {val.shape[0]:,} rows x {val.shape[1]} cols")
    print(f"   Test split  : {test.shape[0]:,} rows x {test.shape[1]} cols (HELD OUT)")

    # 2. Features
    print(f"\n2. Selected Features ({len(CATEGORICAL_FEATURES) + len(NUMERICAL_FEATURES)} total):")
    print(f"   Categorical ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")
    print(f"   Numerical   ({len(NUMERICAL_FEATURES)}): {NUMERICAL_FEATURES}")
    print(f"   Target: '{TARGET_COLUMN}'")

    # 3. Build & Train Pipeline
    pipeline = build_baseline_pipeline()
    print("\n3. Fitting Pipeline strictly on (X_train, y_train)...")
    results = train_and_validate(pipeline, train, val)
    metrics = results["metrics"]

    print("\n4. Validation Evaluation Metrics (Default threshold = 0.50):")
    print(f"   Accuracy : {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"   Precision: {metrics['precision']:.4f} ({metrics['precision'] * 100:.2f}%)")
    print(f"   Recall   : {metrics['recall']:.4f} ({metrics['recall'] * 100:.2f}%)")
    print(f"   F1-Score : {metrics['f1']:.4f}")
    print(f"   ROC-AUC  : {metrics['roc_auc']:.4f}")

    print("\n   Confusion Matrix [TN, FP; FN, TP]:")
    cm = results["confusion_matrix"]
    print(f"   [[TN={cm[0, 0]:>4}, FP={cm[0, 1]:>4}],")
    print(f"    [FN={cm[1, 0]:>4}, TP={cm[1, 1]:>4}]]")

    # 5. Inspect Coefficients
    print("\n5. Top 5 Drivers Increasing RTO Risk (Positive Coefficients):")
    coef_df = inspect_model_weights(pipeline)
    print(coef_df.head(5).to_string(index=False))

    print("\n   Top 5 Drivers Decreasing RTO Risk (Negative Coefficients):")
    print(coef_df.tail(5).to_string(index=False))

    # 6. Sample Validation Probabilities
    y_val = results["y_val"].values
    y_pred = results["predictions"]
    y_prob = results["probabilities"]
    print("\n6. Sample Validation Predictions (First 5 orders):")
    for i in range(5):
        print(
            f"   Order {i+1}: True={y_val[i]} | Pred={y_pred[i]} | Prob(RTO)={y_prob[i]:.4f} ({y_prob[i]*100:.1f}%)"
        )

    print("\nDay 2 Implementation Complete.")


if __name__ == "__main__":
    main()
