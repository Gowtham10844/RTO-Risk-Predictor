"""src/predict.py

Inference and prediction helper module for the RTO Risk Predictor.
Loads the serialized winning pipeline and generates calibrated RTO probabilities,
risk classifications, and merchant action recommendations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Union, List
import joblib
import numpy as np
import pandas as pd

DEFAULT_MODEL_PATH = Path("models/rto_model.joblib")

# ---------------------------------------------------------------------------
# Required Model Input Features
# ---------------------------------------------------------------------------
REQUIRED_FEATURES: List[str] = [
    "payment_mode",
    "category",
    "pincode_tier",
    "is_festive",
    "is_alternate_address",
    "order_value",
    "discount_pct",
    "delivery_days_est",
    "account_age_days",
    "order_velocity_24h",
]

_CACHED_MODEL = None


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    """Load and cache the serialized scikit-learn pipeline."""
    global _CACHED_MODEL
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{path}'. Please run src/train_model.py first."
        )
    if _CACHED_MODEL is None:
        _CACHED_MODEL = joblib.load(path)
    return _CACHED_MODEL


def get_risk_assessment(probability: float, threshold: float = 0.50) -> Dict[str, str]:
    """Map calibrated RTO probability into operational risk tiers and merchant actions.

    - Low Risk:    P < 0.35  (Frictionless auto-dispatch)
    - Medium Risk: 0.35 <= P < 0.50 (Automated WhatsApp/SMS verification)
    - High Risk:   P >= 0.50 (Mandatory deposit or convert to prepaid)
    """
    if probability < 0.35:
        return {
            "risk_level": "Low",
            "risk_badge": "🟢 Low Risk",
            "badge_color": "green",
            "recommended_action": "Auto-dispatch COD order. High delivery confidence.",
        }
    elif probability < 0.50:
        return {
            "risk_level": "Medium",
            "risk_badge": "🟡 Medium Risk",
            "badge_color": "orange",
            "recommended_action": "Send automated WhatsApp/SMS address confirmation link before dispatch.",
        }
    else:
        return {
            "risk_level": "High",
            "risk_badge": "🔴 High Risk",
            "badge_color": "red",
            "recommended_action": "Flag for verification: Request ₹50–₹100 advance deposit or convert to Prepaid.",
        }


def explain_risk_factors(order_data: Dict[str, Any]) -> List[str]:
    """Provide rule-based qualitative reasons explaining why an order is risky."""
    factors = []
    pm = str(order_data.get("payment_mode", "")).upper()
    cat = str(order_data.get("category", "")).lower()
    tier = str(order_data.get("pincode_tier", "")).lower()
    disc = float(order_data.get("discount_pct", 0.0))
    del_days = float(order_data.get("delivery_days_est", 4.0))
    age = float(order_data.get("account_age_days", 100.0))
    festive = bool(order_data.get("is_festive", False))
    alt_addr = bool(order_data.get("is_alternate_address", False))
    velocity = int(order_data.get("order_velocity_24h", 0))

    if pm == "COD":
        factors.append("Payment Mode is Cash on Delivery (COD carries ~18x higher baseline RTO rate than Prepaid).")
    else:
        factors.append("Prepaid payment mode significantly reduces RTO risk (-87% odds).")

    if cat == "fashion":
        factors.append("Product category is Fashion (highest return vertical due to sizing and fit mismatch).")
    elif cat == "electronics":
        factors.append("Electronics category typically has lower return rates (~6%).")

    if tier in ["tier_3", "tier_2"]:
        factors.append(f"Destination is in {tier.replace('_', ' ').title()} (lower address density and courier coverage).")

    if disc >= 0.30:
        factors.append(f"Deep discount applied ({disc*100:.0f}%), which often triggers impulsive, low-commitment purchases.")

    if del_days >= 6.0:
        factors.append(f"Estimated transit time is {del_days:.1f} days (longer delivery windows cool buyer enthusiasm).")

    if age <= 7.0:
        factors.append(f"Brand-new user account ({age:.0f} days old; higher baseline fraud/cancellation risk).")

    if festive:
        factors.append("Order placed during Festive season (higher seasonal returns and logistics congestion).")

    if alt_addr:
        factors.append("Shipping to an alternate/unverified address.")

    if velocity >= 1:
        factors.append(f"Customer placed {velocity} other order(s) in the past 24 hours (burst ordering).")

    return factors


def predict_order(
    order_data: Dict[str, Any],
    model_path: str | Path = DEFAULT_MODEL_PATH,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Generate RTO probability and risk assessment for a single order dictionary.

    Parameters
    ----------
    order_data : dict containing all required features:
        - payment_mode ('COD' or 'PREPAID')
        - category ('fashion', 'home_kitchen', 'electronics', 'accessories', 'beauty', 'footwear')
        - pincode_tier ('metro', 'tier_1', 'tier_2', 'tier_3')
        - is_festive (bool)
        - is_alternate_address (bool)
        - order_value (float, in INR)
        - discount_pct (float, 0.0 to 1.0)
        - delivery_days_est (float, in days)
        - account_age_days (float, in days)
        - order_velocity_24h (int, count)
    """
    model = load_model(model_path)

    # Convert to single-row DataFrame ensuring exact column presence and types
    row_dict = {}
    for col in REQUIRED_FEATURES:
        if col not in order_data:
            raise KeyError(f"Missing required feature '{col}' in input order data.")
        row_dict[col] = [order_data[col]]

    input_df = pd.DataFrame(row_dict)

    # Coerce boolean and numerical types
    input_df["is_festive"] = input_df["is_festive"].astype(bool)
    input_df["is_alternate_address"] = input_df["is_alternate_address"].astype(bool)
    input_df["order_value"] = input_df["order_value"].astype(float)
    input_df["discount_pct"] = input_df["discount_pct"].astype(float)
    input_df["delivery_days_est"] = input_df["delivery_days_est"].astype(float)
    input_df["account_age_days"] = input_df["account_age_days"].astype(float)
    input_df["order_velocity_24h"] = input_df["order_velocity_24h"].astype(int)

    # Model inference
    probabilities = model.predict_proba(input_df)[0]
    p_rto = float(probabilities[1])
    is_rto = bool(p_rto >= threshold)

    assessment = get_risk_assessment(p_rto, threshold=threshold)
    reasons = explain_risk_factors(order_data)

    return {
        "rto_probability": p_rto,
        "rto_probability_pct": round(p_rto * 100, 2),
        "is_rto_risk": is_rto,
        "risk_level": assessment["risk_level"],
        "risk_badge": assessment["risk_badge"],
        "badge_color": assessment["badge_color"],
        "recommended_action": assessment["recommended_action"],
        "risk_factors": reasons,
    }


if __name__ == "__main__":
    print("Testing src/predict.py across test cases...")

    # Case 1: High Risk (COD Fashion in Tier 3 with deep discount)
    high_risk_order = {
        "payment_mode": "COD",
        "category": "fashion",
        "pincode_tier": "tier_3",
        "is_festive": True,
        "is_alternate_address": True,
        "order_value": 899.0,
        "discount_pct": 0.45,
        "delivery_days_est": 7.5,
        "account_age_days": 3.0,
        "order_velocity_24h": 1,
    }

    # Case 2: Low Risk (Prepaid Electronics in Metro)
    low_risk_order = {
        "payment_mode": "PREPAID",
        "category": "electronics",
        "pincode_tier": "metro",
        "is_festive": False,
        "is_alternate_address": False,
        "order_value": 4500.0,
        "discount_pct": 0.05,
        "delivery_days_est": 2.5,
        "account_age_days": 250.0,
        "order_velocity_24h": 0,
    }

    res_high = predict_order(high_risk_order)
    print("\n[High Risk Scenario]:")
    print(f"P(RTO) = {res_high['rto_probability_pct']}% | {res_high['risk_badge']}")
    print(f"Action: {res_high['recommended_action']}")
    print("Factors:", res_high["risk_factors"])

    res_low = predict_order(low_risk_order)
    print("\n[Low Risk Scenario]:")
    print(f"P(RTO) = {res_low['rto_probability_pct']}% | {res_low['risk_badge']}")
    print(f"Action: {res_low['recommended_action']}")
    print("Factors:", res_low["risk_factors"])

    print("\nsrc/predict.py verification complete!")
