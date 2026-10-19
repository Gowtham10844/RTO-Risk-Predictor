"""src/evaluate_test.py

Final held-out test evaluation of the serialized RTO risk model.
Evaluates models/rto_model.joblib on data/processed/test.parquet.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from src.train_model import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN


def run_test_evaluation(
    model_path: str | Path = "models/rto_model.joblib",
    test_path: str | Path = "data/processed/test.parquet",
    val_path: str | Path = "data/processed/val.parquet",
) -> dict:
    """Run one-shot evaluation on the held-out test split."""
    model = joblib.load(model_path)
    test = pd.read_parquet(test_path)
    val = pd.read_parquet(val_path)

    features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X_test, y_test = test[features], test[TARGET_COLUMN]
    X_val, y_val = val[features], val[TARGET_COLUMN]

    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = model.predict(X_test)

    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = model.predict(X_val)

    # Compute metrics
    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    roc_auc = roc_auc_score(y_test, test_probs)
    cm = confusion_matrix(y_test, test_preds)
    report = classification_report(
        y_test, test_preds, target_names=["Delivered (0)", "RTO (1)"], digits=4
    )

    print("=" * 80)
    print("FINAL HELD-OUT TEST EVALUATION REPORT")
    print("=" * 80)
    print(f"\nTest Split Size: {len(test):,} orders (March 16, 2025 - June 30, 2025)")
    print(f"Test RTO Rate  : {y_test.mean():.2%} ({y_test.sum():,} actual returns)")
    print(f"\nFinal Test Metrics (Threshold = 0.50):")
    print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"  Recall   : {rec:.4f} ({rec*100:.2f}%)")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")

    print(f"\nTest Confusion Matrix [TN, FP; FN, TP]:")
    print(f"  [[TN={cm[0,0]:>4}, FP={cm[0,1]:>4}],")
    print(f"   [FN={cm[1,0]:>4}, TP={cm[1,1]:>4}]]")

    print(f"\nDetailed Classification Report:")
    print(report)

    # Generalization comparison
    val_auc = roc_auc_score(y_val, val_probs)
    val_rec = recall_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds)

    print("Generalization Check (Validation vs Held-out Test):")
    print(f"  ROC-AUC  : Val={val_auc:.4f}  | Test={roc_auc:.4f}  (Delta={roc_auc - val_auc:+.4f})")
    print(f"  Recall   : Val={val_rec:.4f}  | Test={rec:.4f}  (Delta={rec - val_rec:+.4f})")
    print(f"  F1-Score : Val={val_f1:.4f}  | Test={f1:.4f}  (Delta={f1 - val_f1:+.4f})")

    # Financial impact calculation
    tp, fp, fn, tn = cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0]
    unmitigated_loss = (tp + fn) * 200.0
    model_loss = (fn * 200.0) + (fp * 20.0)
    net_savings = unmitigated_loss - model_loss

    print(f"\nBusiness Financial Impact (on 10,000 test orders):")
    print(f"  Unmitigated RTO Freight Loss : ₹{unmitigated_loss:,.2f}")
    print(f"  Model Operating Loss         : ₹{model_loss:,.2f}")
    print(f"  Net Merchant Freight Savings : ₹{net_savings:,.2f} ({net_savings/unmitigated_loss:.1%} reduction)")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "net_savings": net_savings,
    }


if __name__ == "__main__":
    run_test_evaluation()
