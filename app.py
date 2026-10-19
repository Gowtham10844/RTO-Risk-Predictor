"""app.py

E-Commerce Post-Dispatch RTO Risk Predictor — Streamlit Web Application.

Provides an interactive dashboard for merchants and risk managers to evaluate
Return-to-Origin (RTO) risk before parcels are dispatched to logistics hubs.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from src.predict import predict_order, load_model, get_risk_assessment, explain_risk_factors

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RTO Risk Predictor | E-Commerce",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #2ea44f;
    }
    .badge-low {
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-med {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header & Navigation
# ---------------------------------------------------------------------------
st.title("📦 E-Commerce Post-Dispatch RTO Risk Predictor")
st.markdown(
    "Predict the probability that a placed order will be **Returned to Origin (RTO)** before handover to courier partners, using an explainable **Balanced Logistic Regression** pipeline."
)

tab1, tab2, tab3 = st.tabs(["🎯 Single Order Risk Scorer", "📊 Model & Economics Benchmark", "📁 Batch Validation Sample"])

# ---------------------------------------------------------------------------
# Tab 1: Single Order Risk Scorer
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Checkout Parameters & Order Risk Evaluation")

    # Preset Profiles for Quick Testing
    preset = st.selectbox(
        "⚡ Quick Presets (Click to Auto-fill Scenarios):",
        [
            "Custom Input",
            "High-Risk Scenario (COD Fashion, Tier-3, Deep Discount)",
            "Low-Risk Scenario (Prepaid Electronics, Metro, Loyal Customer)",
            "Borderline Medium-Risk (COD Home & Kitchen, Tier-2, Festive)",
        ],
    )

    # Set default values based on preset
    if preset == "High-Risk Scenario (COD Fashion, Tier-3, Deep Discount)":
        d_payment_mode = "COD"
        d_category = "fashion"
        d_tier = "tier_3"
        d_order_value = 1199.0
        d_discount = 45
        d_delivery_days = 7.0
        d_account_age = 5.0
        d_velocity = 1
        d_festive = True
        d_alt_addr = True
    elif preset == "Low-Risk Scenario (Prepaid Electronics, Metro, Loyal Customer)":
        d_payment_mode = "PREPAID"
        d_category = "electronics"
        d_tier = "metro"
        d_order_value = 3999.0
        d_discount = 5
        d_delivery_days = 2.5
        d_account_age = 320.0
        d_velocity = 0
        d_festive = False
        d_alt_addr = False
    elif preset == "Borderline Medium-Risk (COD Home & Kitchen, Tier-2, Festive)":
        d_payment_mode = "COD"
        d_category = "home_kitchen"
        d_tier = "tier_2"
        d_order_value = 1450.0
        d_discount = 20
        d_delivery_days = 5.0
        d_account_age = 45.0
        d_velocity = 0
        d_festive = True
        d_alt_addr = False
    else:
        d_payment_mode = "COD"
        d_category = "fashion"
        d_tier = "tier_1"
        d_order_value = 850.0
        d_discount = 15
        d_delivery_days = 4.0
        d_account_age = 90.0
        d_velocity = 0
        d_festive = False
        d_alt_addr = False

    # Input Form
    with st.form("order_input_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 💳 Payment & Commercials")
            payment_mode = st.selectbox(
                "Payment Mode",
                ["COD", "PREPAID"],
                index=0 if d_payment_mode == "COD" else 1,
                help="Cash on Delivery orders carry significantly higher baseline return rates.",
            )
            category = st.selectbox(
                "Product Vertical",
                ["fashion", "footwear", "accessories", "beauty", "home_kitchen", "electronics"],
                index=["fashion", "footwear", "accessories", "beauty", "home_kitchen", "electronics"].index(d_category),
                format_func=lambda x: x.replace("_", " ").title(),
                help="Fashion vertical carries highest return propensity due to sizing/color mismatch.",
            )
            order_value = st.number_input(
                "Order Total (₹)",
                min_value=50.0,
                max_value=50000.0,
                value=float(d_order_value),
                step=50.0,
            )

        with col2:
            st.markdown("#### 📍 Geography & Logistics")
            pincode_tier = st.selectbox(
                "Destination Tier",
                ["metro", "tier_1", "tier_2", "tier_3"],
                index=["metro", "tier_1", "tier_2", "tier_3"].index(d_tier),
                format_func=lambda x: x.replace("_", " ").title(),
                help="Urbanization proxy. Address clarity and courier presence decline in Tier-2/3.",
            )
            delivery_days_est = st.slider(
                "Estimated Delivery Transit (Days)",
                min_value=1.0,
                max_value=15.0,
                value=float(d_delivery_days),
                step=0.5,
                help="Longer transit delays cool buyer enthusiasm and increase doorstep rejections.",
            )
            discount_pct_ui = st.slider(
                "Promotional Discount (%)",
                min_value=0,
                max_value=75,
                value=int(d_discount),
                step=1,
                help="Deep discounts trigger impulsive, low-commitment checkouts.",
            )

        with col3:
            st.markdown("#### 👤 Customer Account & Context")
            account_age_days = st.number_input(
                "Account Age (Days)",
                min_value=0.0,
                max_value=1000.0,
                value=float(d_account_age),
                step=1.0,
                help="New accounts (0-7 days) present higher baseline cancellation risk.",
            )
            order_velocity_24h = st.selectbox(
                "Orders Placed in Last 24 Hours",
                [0, 1, 2],
                index=int(d_velocity),
                help="Multiple orders within 24h indicates potential burst ordering behavior.",
            )
            is_festive = st.checkbox(
                "Festive Season Order (e.g. Diwali Sale)",
                value=bool(d_festive),
                help="Holiday surges increase logistics bottleneck and impulsive buying.",
            )
            is_alternate_address = st.checkbox(
                "Alternate Shipping Address",
                value=bool(d_alt_addr),
                help="Shipments to non-primary addresses experience higher delivery attempt failures.",
            )

        submit_btn = st.form_submit_button("🔍 Predict RTO Risk", use_container_width=True, type="primary")

    if submit_btn or preset != "Custom Input":
        order_dict = {
            "payment_mode": payment_mode,
            "category": category,
            "pincode_tier": pincode_tier,
            "is_festive": is_festive,
            "is_alternate_address": is_alternate_address,
            "order_value": float(order_value),
            "discount_pct": float(discount_pct_ui / 100.0),
            "delivery_days_est": float(delivery_days_est),
            "account_age_days": float(account_age_days),
            "order_velocity_24h": int(order_velocity_24h),
        }

        try:
            res = predict_order(order_dict)

            st.markdown("---")
            st.markdown("### 📋 Prediction Results & Merchant Guidance")

            # Metrics row
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric(
                    label="Estimated RTO Probability",
                    value=f"{res['rto_probability_pct']}%",
                    delta=f"{res['risk_level']} Risk",
                    delta_color="inverse" if res["risk_level"] == "High" else "normal",
                )
            with m_col2:
                expected_loss = round(res["rto_probability"] * 200.0, 2)
                st.metric(
                    label="Expected Logistics Waste",
                    value=f"₹{expected_loss:.2f}",
                    help="Calculated as P(RTO) * ₹200 average round-trip logistics loss.",
                )
            with m_col3:
                st.markdown(f"**Classification Badge:**")
                st.markdown(f"<h3>{res['risk_badge']}</h3>", unsafe_allow_html=True)

            # Recommended Action Banner
            if res["risk_level"] == "High":
                st.error(f"**Action Required:** {res['recommended_action']}")
            elif res["risk_level"] == "Medium":
                st.warning(f"**Action Recommended:** {res['recommended_action']}")
            else:
                st.success(f"**Clear to Ship:** {res['recommended_action']}")

            # Risk Drivers
            with st.expander("🔎 Key Risk Factor Drivers (Explainability)", expanded=True):
                for f in res["risk_factors"]:
                    st.write(f"- {f}")

            # Developer Payload
            with st.expander("🛠️ View Input Feature Payload"):
                st.json(order_dict)

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")

# ---------------------------------------------------------------------------
# Tab 2: Model & Economics Benchmark
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Model Evaluation & Economics Benchmark (Held-out Validation Split)")
    st.markdown(
        "Evaluated on $N = 5,000$ chronological validation orders (Jan 22, 2025 to Mar 16, 2025). Ground truth base RTO prevalence: **15.50%**."
    )

    st.markdown("#### 1. Models Compared")
    bench_data = pd.DataFrame(
        [
            {
                "Model Architecture": "1. Baseline Logistic Regression (Unweighted)",
                "Accuracy": "84.46%",
                "Precision": "47.73%",
                "Recall (RTO)": "2.71%",
                "F1-Score": "0.0513",
                "ROC-AUC": "0.7994",
                "Status": "Rejected (misses 97% of returns)",
            },
            {
                "Model Architecture": "2. Balanced Logistic Regression (Balanced Weights)",
                "Accuracy": "68.98%",
                "Precision": "30.04%",
                "Recall (RTO)": "75.35%",
                "F1-Score": "0.4296",
                "ROC-AUC": "0.7993",
                "Status": "⭐ Selected Production Model",
            },
            {
                "Model Architecture": "3. Decision Tree Classifier (depth=5, Balanced)",
                "Accuracy": "68.98%",
                "Precision": "29.25%",
                "Recall (RTO)": "70.58%",
                "F1-Score": "0.4136",
                "ROC-AUC": "0.7813",
                "Status": "Rejected (lower AUC and step partitions)",
            },
        ]
    )
    st.dataframe(bench_data, use_container_width=True, hide_index=True)

    st.markdown("#### 2. Decision Threshold Analysis (Balanced Model)")
    thresh_data = pd.DataFrame(
        [
            {"Threshold": "0.30", "Accuracy": "51.64%", "Precision": "23.90%", "Recall": "97.03%", "F1": "0.3835", "RTO Caught (TP)": "752 / 775", "Missed (FN)": "23"},
            {"Threshold": "0.40", "Accuracy": "55.86%", "Precision": "24.98%", "Recall": "92.26%", "F1": "0.3932", "RTO Caught (TP)": "715 / 775", "Missed (FN)": "60"},
            {"Threshold": "0.50 (Winning)", "Accuracy": "68.98%", "Precision": "30.04%", "Recall": "75.35%", "F1": "0.4296", "RTO Caught (TP)": "584 / 775", "Missed (FN)": "191"},
        ]
    )
    st.dataframe(thresh_data, use_container_width=True, hide_index=True)

    st.markdown("#### 3. Financial Cost Breakdown (5,000 Validation Orders)")
    st.markdown(
        """
        - **Unweighted Model Loss:** $(754 \\text{ FN} \\times ₹200) + (23 \\text{ FP} \\times ₹20) = \\mathbf{₹1,51,260}$
        - **Balanced Model Loss:** $(191 \\text{ FN} \\times ₹200) + (1,360 \\text{ FP} \\times ₹20) = \\mathbf{₹65,400}$
        - **Net Merchant Savings:** **₹85,860 saved per 5,000 orders (56.8% cost reduction)**
        """
    )

# ---------------------------------------------------------------------------
# Tab 3: Batch Validation Sample
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Live Inference on Real Validation Orders")
    st.markdown("Load sample orders from `data/processed/val.parquet` to verify predictions on unseen orders.")

    if st.button("Load 5 Random Validation Orders"):
        val_path = Path("data/processed/val.parquet")
        if val_path.exists():
            val_df = pd.read_parquet(val_path).sample(5, random_state=np.random.randint(0, 10000))
            features = [
                "order_id", "payment_mode", "category", "pincode_tier",
                "order_value", "discount_pct", "delivery_days_est",
                "account_age_days", "order_velocity_24h", "is_festive",
                "is_alternate_address", "rto"
            ]
            sample_sub = val_df[features].copy()
            preds_list = []
            probs_list = []
            badges_list = []

            for _, row in sample_sub.iterrows():
                p_res = predict_order(row.to_dict())
                preds_list.append(int(p_res["is_rto_risk"]))
                probs_list.append(f"{p_res['rto_probability_pct']}%")
                badges_list.append(p_res["risk_badge"])

            sample_sub["P(RTO)"] = probs_list
            sample_sub["Predicted Risk"] = badges_list
            sample_sub["Actual RTO"] = sample_sub["rto"].apply(lambda x: "Returned (1)" if x == 1 else "Delivered (0)")

            display_cols = ["order_id", "payment_mode", "category", "pincode_tier", "order_value", "P(RTO)", "Predicted Risk", "Actual RTO"]
            st.dataframe(sample_sub[display_cols], use_container_width=True, hide_index=True)
        else:
            st.warning("data/processed/val.parquet not found.")

st.markdown("---")
st.caption("RTO Risk Predictor | Built with scikit-learn & Streamlit | Verified against Indian E-Commerce benchmarks.")
