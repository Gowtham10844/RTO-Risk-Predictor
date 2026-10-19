"""src/train_model.py

Pre-Dispatch RTO Risk Predictor: Preprocessing, Baseline & Model Comparison.

This module builds a leakage-safe scikit-learn ColumnTransformer and Pipeline,
fits strictly on the chronological training split, evaluates multiple model
architectures (Logistic Regression unweighted/balanced and Decision Tree),
and conducts probability threshold sweeps on the validation split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Dict, Any, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
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


def build_model_pipeline(
    model_type: str = "logistic_balanced",
    preprocessor: ColumnTransformer = None,
    random_state: int = 42,
) -> Pipeline:
    """Wrap preprocessor and selected classifier inside a scikit-learn Pipeline.

    Supported model_type options:
    - 'logistic_unweighted': Baseline LogisticRegression(max_iter=1000)
    - 'logistic_balanced': Balanced LogisticRegression(class_weight='balanced')
    - 'decision_tree_balanced': DecisionTreeClassifier(max_depth=5, class_weight='balanced')
    """
    if preprocessor is None:
        preprocessor = build_preprocessor()

    if model_type == "logistic_unweighted":
        classifier = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
        )
    elif model_type == "logistic_balanced":
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
            solver="lbfgs",
        )
    elif model_type == "decision_tree_balanced":
        classifier = DecisionTreeClassifier(
            max_depth=5,
            class_weight="balanced",
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


def build_baseline_pipeline(
    preprocessor: ColumnTransformer = None,
    random_state: int = 42,
) -> Pipeline:
    """Convenience alias for unweighted baseline Logistic Regression pipeline."""
    return build_model_pipeline(
        model_type="logistic_unweighted",
        preprocessor=preprocessor,
        random_state=random_state,
    )


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

    # Compute metrics at default threshold 0.50
    acc = accuracy_score(y_val, val_preds)
    prec = precision_score(y_val, val_preds, zero_division=0)
    rec = recall_score(y_val, val_preds)
    f1 = f1_score(y_val, val_preds)
    roc_auc = roc_auc_score(y_val, val_probs)
    cm = confusion_matrix(y_val, val_preds)
    report = classification_report(
        y_val, val_preds, target_names=["Delivered (0)", "RTO (1)"], digits=4
    )

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
        "classification_report": report,
        "X_val": X_val,
        "y_val": y_val,
    }


def evaluate_threshold_sweep(
    probabilities: np.ndarray,
    y_val: np.ndarray,
    thresholds: List[float] = [0.30, 0.40, 0.50],
) -> pd.DataFrame:
    """Evaluate precision, recall, F1, and confusion matrix across probability thresholds."""
    rows = []
    for t in thresholds:
        preds = (probabilities >= t).astype(int)
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds)
        f1 = f1_score(y_val, preds)
        cm = confusion_matrix(y_val, preds)
        tn, fp, fn, tp = cm.ravel()
        rows.append(
            {
                "threshold": t,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return pd.DataFrame(rows)


def run_experiments(
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> Dict[str, Any]:
    """Train and compare all 3 model experiments on validation set."""
    experiment_configs = {
        "Logistic Regression (Unweighted)": "logistic_unweighted",
        "Logistic Regression (Balanced)": "logistic_balanced",
        "Decision Tree (Balanced, depth=5)": "decision_tree_balanced",
    }

    results = {}
    sweep_tables = {}

    for name, mtype in experiment_configs.items():
        pipe = build_model_pipeline(model_type=mtype)
        res = train_and_validate(pipe, train, val)
        results[name] = res

        # Threshold sweep
        sweep_df = evaluate_threshold_sweep(
            res["probabilities"], res["y_val"].values, [0.30, 0.40, 0.50]
        )
        sweep_tables[name] = sweep_df

    return {"models": results, "threshold_sweeps": sweep_tables}


def inspect_model_weights(pipeline: Pipeline) -> pd.DataFrame:
    """Extract and sort learned Logistic Regression coefficients."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    if not hasattr(classifier, "coef_"):
        raise AttributeError("Classifier does not have linear coefficients.")

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


def save_winning_model(
    train: pd.DataFrame = None,
    output_path: str | Path = "models/rto_model.joblib",
) -> Pipeline:
    """Fit the winning Balanced Logistic Regression pipeline and serialize to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if train is None:
        train, _, _ = load_data()

    pipeline = build_model_pipeline(model_type="logistic_balanced")
    features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X_train = train[features]
    y_train = train[TARGET_COLUMN]

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, output_path)
    print(f"Serialized winning pipeline to: {output_path}")
    return pipeline


def main():
    print("=" * 80)
    print("MODEL COMPARISON, CLASS IMBALANCE & THRESHOLD TUNING")
    print("=" * 80)

    # 1. Load data
    train, val, test = load_data()
    print(f"\n1. Chronological Splits Loaded:")
    print(f"   Train set : {train.shape[0]:,} orders (Jan 2024 - Jan 2025)")
    print(f"   Val set   : {val.shape[0]:,} orders (Jan 2025 - Mar 2025)")
    print(f"   Test set  : {test.shape[0]:,} orders (HELD OUT)")

    # 2. Run Experiments
    print("\n2. Training & Evaluating 3 Model Experiments...")
    exp_output = run_experiments(train, val)
    models = exp_output["models"]
    sweeps = exp_output["threshold_sweeps"]

    # 3. Model Comparison Table (at default threshold 0.50)
    print("\n3. Validation Metrics Comparison (at threshold = 0.50):")
    comp_rows = []
    for name, res in models.items():
        m = res["metrics"]
        comp_rows.append(
            {
                "Model": name,
                "Accuracy": f"{m['accuracy']:.4f}",
                "Precision": f"{m['precision']:.4f}",
                "Recall": f"{m['recall']:.4f}",
                "F1-Score": f"{m['f1']:.4f}",
                "ROC-AUC": f"{m['roc_auc']:.4f}",
            }
        )
    print(pd.DataFrame(comp_rows).to_string(index=False))

    # 4. Detailed Classification Reports
    print("\n4. Classification Reports & Confusion Matrices:")
    for name, res in models.items():
        print(f"\n--- {name} ---")
        cm = res["confusion_matrix"]
        print(f"Confusion Matrix [TN={cm[0,0]}, FP={cm[0,1]}; FN={cm[1,0]}, TP={cm[1,1]}]")
        print(res["classification_report"])

    # 5. Threshold Sweeps (0.30, 0.40, 0.50)
    print("\n5. Probability Threshold Sweep Comparison:")
    for name, sw_df in sweeps.items():
        print(f"\n[{name}]:")
        display_df = sw_df.copy()
        for c in ["accuracy", "precision", "recall", "f1_score"]:
            display_df[c] = display_df[c].apply(lambda x: f"{x:.4f}")
        print(display_df.to_string(index=False))

    # 6. Save Winning Model Pipeline
    print("\n6. Serializing Winning Model Pipeline (models/rto_model.joblib)...")
    save_winning_model(train)

    print("\nModel Evaluation & Serialization Complete.")


if __name__ == "__main__":
    main()
