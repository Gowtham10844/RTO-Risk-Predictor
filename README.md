# E-Commerce Post-Dispatch RTO Risk Predictor

An explainable, production-ready Machine Learning system that predicts **Return-to-Origin (RTO)** risk on Indian e-commerce orders prior to warehouse dispatch, enabling merchants to mitigate logistics losses while preserving customer trust.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Problem Statement

In Indian e-commerce, **Cash on Delivery (COD)** represents over 60% of consumer transactions. However, roughly **1 in 4 COD orders results in a Return-to-Origin (RTO)** — the parcel is dispatched, transported across the country, but rejected or uncollected at the customer's doorstep.

When an order returns undelivered:
- The seller absorbs two-way courier freight (~₹80 forward + ~₹80 reverse).
- Packaging materials (~₹20) are permanently damaged.
- Inventory is trapped in transit for 10–14 days, causing stockouts.
- **Average direct loss: ₹200 per returned order.**

Across the Indian retail ecosystem, RTO amounts to an estimated **₹8,000+ crore annual loss**. Crucially, once a parcel is handed to the delivery courier, this operational loss is locked in. Prevention must happen **post-order checkout, but pre-dispatch**.

---

## 2. Project Objective

1. **Predictive Risk Scoring:** Calculate a calibrated probability $P(\text{RTO}) \in [0, 1]$ for every incoming order strictly using attributes knowable at checkout.
2. **Actionable Merchant Intervention:** Categorize orders into actionable operational tiers:
   - **🟢 Low Risk ($P < 35\%$):** 1-click auto-dispatch with zero verification friction.
   - **🟡 Medium Risk ($35\% \le P < 50\%$):** Automated WhatsApp / SMS address confirmation link.
   - **🔴 High Risk ($P \ge 50\%$):** Proactive intervention (requesting a ₹50–₹100 advance deposit or prepaid conversion).
3. **Strict Explainability:** Restrict models to linear coefficients (**Logistic Regression**) and transparent splits (**Decision Trees**) so business operators understand *why* an order is flagged.
4. **Zero Data Leakage:** Ensure training pipelines use strict chronological splitting and encapsulation.

---

## 3. Dataset and Its Limitations

### Dataset Architecture
- **Volume:** 50,000 orders generated deterministically (seed `20260101`).
- **Real Geographic Skeleton:** 23,899 active Indian postal pincodes across 566 districts and 35 states/UTs from the official India Post directory (`data/external/india_pincodes.csv`).
- **Calibrated Marginals:** Intercepts solved to reproduce published Indian logistics benchmarks across payment modes, order value tiers, and categories.

### Disclosure of Limitations
> [!IMPORTANT]
> **Synthetic Calibration Disclosure:** Because real e-commerce transactional outcome logs contain sensitive consumer PII and proprietary courier agreements, this dataset is synthetically generated using seeded Bernoulli draws over calibrated log-odds. While marginal return rates and correlations accurately mirror Indian retail benchmarks, external production deployment requires fine-tuning on merchant-specific historical data.

---

## 4. Features Used

To guarantee zero data leakage, features are restricted strictly to **pre-dispatch variables**:

| Feature Name | Type | Domain / Values | Causal Relevance |
| :--- | :--- | :--- | :--- |
| `payment_mode` | Categorical | `COD`, `PREPAID` | Dominant driver: COD orders have an 18x higher RTO rate (25.6%) than Prepaid (1.4%). |
| `category` | Categorical | `fashion`, `footwear`, `accessories`, `beauty`, `home_kitchen`, `electronics` | Fashion exhibits the highest RTO rate (27.7%) due to sizing and fit mismatch. |
| `pincode_tier` | Categorical | `metro`, `tier_1`, `tier_2`, `tier_3` | Urbanization proxy; RTO increases monotonically from Metro (12.6%) to Tier 3 (26.5%). |
| `order_value` | Numerical | ₹99 to ₹41,896 | Price magnitude influences buyer financial commitment. |
| `discount_pct` | Numerical | 0.0% to 71.5% | Steep discounts trigger impulsive buying and buyer remorse upon delivery. |
| `delivery_days_est`| Numerical | 2.0 to 14.1 days | Transit delays cool buyer enthusiasm and increase cancellation propensity. |
| `account_age_days` | Numerical | 1 to 570 days | Brand-new accounts present higher baseline cancellation risk than mature accounts. |
| `order_velocity_24h`| Numerical | 0, 1, 2 | High order frequency in a short window signals erratic checkout sprees. |
| `is_festive` | Boolean | `True`, `False` | Seasonal holiday sales (e.g. Diwali) elevate courier bottlenecks and returns. |
| `is_alternate_address`| Boolean | `True`, `False` | Delivery to non-primary locations increases delivery attempt failures. |

### Excluded Columns (Leakage & Noise Prevention)
- **Surrogate IDs & PII:** `order_id`, `customer_id`, `customer_name`, `phone`, `address_line`.
- **High Cardinality Noise:** Raw 6-digit `pincode`, `city`, `district`, `state` (summarized via `pincode_tier`).
- **Temporal Sorting Column:** `order_ts` (used strictly to order chronological train/val/test splits).
- **Post-Dispatch Outcomes:** Courier delivery attempt status, delivery agent notes, transit tracking timestamps.

---

## 5. Machine Learning Workflow

```
Raw Checkout Order
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│              Scikit-Learn ColumnTransformer                     │
│  Categorical Pipeline: SimpleImputer + OneHotEncoder(ignore)    │
│  Numerical Pipeline:   SimpleImputer(median) + StandardScaler()  │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│     Balanced Logistic Regression Pipeline (L-BFGS, C=1.0)       │
│     w1 = N / (2 * N1) ≈ 2.95  |  w0 = N / (2 * N0) ≈ 0.60       │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
 Calibrated Probability P(RTO)
       │
       ├─► P < 0.35        ──► 🟢 Low Risk (Frictionless Auto-Dispatch)
       ├─► 0.35 <= P < 0.50 ──► 🟡 Medium Risk (Automated WhatsApp Verification)
       └─► P >= 0.50       ──► 🔴 High Risk (Mandatory Advance Deposit)
```

### Chronological Partitioning (No Future Leakage)
Instead of random splitting (which leaks future events into past training data), orders are split strictly along `order_ts`:
- **Training Set (70%):** 35,000 orders (Jan 1, 2024 to Jan 22, 2025)
- **Validation Set (10%):** 5,000 orders (Jan 22, 2025 to Mar 16, 2025) — for model & threshold selection
- **Held-Out Test Set (20%):** 10,000 orders (Mar 16, 2025 to Jun 30, 2025) — evaluated once at final verification

---

## 6. Models Compared

| Model Architecture | Objective / Configuration | Key Result |
| :--- | :--- | :--- |
| **1. Baseline Logistic Regression** | Unweighted loss ($C=1.0$, `solver='lbfgs'`) | Flawed. Optimizes accuracy (84.5%) by predicting "Delivered" for almost all orders, catching only **2.7% of returns**. |
| **2. Balanced Logistic Regression** | `class_weight='balanced'` ($w_1 \approx 2.95, w_0 \approx 0.60$) | **⭐ Selected Winner.** Balances precision and recall, catching **75.4% of RTOs** on validation and **72.4% on held-out test data**. |
| **3. Decision Tree Classifier** | `max_depth=5, class_weight='balanced'` | Provides readable if-then rules, but achieves lower ROC-AUC (0.7813) due to discrete step boundaries. |

---

## 7. Evaluation Metrics and Results

### Validation Set Comparison ($N = 5,000$ orders, Base RTO = $15.50\%$)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | RTO Caught (TP) | Missed (FN) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline LR (Unweighted)** | **84.46%** | 47.73% | 2.71% | 0.0513 | **0.7994** | 21 | 754 |
| **Balanced LR (Winning)** | 68.98% | **30.04%** | **75.35%** | **0.4296** | 0.7993 | **584** | **191** |
| **Balanced Decision Tree** | 68.98% | 29.25% | 70.58% | 0.4136 | 0.7813 | 547 | 228 |

### Final Held-Out Test Set Results ($N = 10,000$ orders, Base RTO = $14.73\%$)

```text
               precision    recall  f1-score   support

Delivered (0)     0.9366    0.7051    0.8045      8527
      RTO (1)     0.2977    0.7237    0.4218      1473

     accuracy                         0.7078     10000
    macro avg     0.6171    0.7144    0.6132     10000
 weighted avg     0.8425    0.7078    0.7481     10000
```

- **Test ROC-AUC:** **0.7924** (vs. 0.7993 on validation — minimal drop of $-0.0069$, verifying robust generalization)
- **Test Recall:** **72.37%** (catches 1,066 out of 1,473 RTOs)
- **Financial Impact on 10,000 Orders:**
  - Unmitigated Logistics Loss: ₹2,94,600
  - Net Model Operating Loss: ₹1,31,700
  - **Net Merchant Savings: ₹1,62,900 (55.3% reduction in RTO waste)**

---

## 8. How to Run the Project

### Prerequisites
- macOS / Linux / Windows
- Python 3.9 or higher

### Step 1: Clone Repository & Set Up Environment
```bash
git clone https://github.com/Gowtham10844/RTO-Risk-Predictor.git
cd RTO-Risk-Predictor

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Exploratory Analysis
```bash
python notebooks/exploration.py
```

### Step 3: Train & Benchmark Models
```bash
python src/train_model.py
```
*Outputs model comparison metrics, threshold sweeps, and serializes the winning pipeline to `models/rto_model.joblib`.*

### Step 4: Run One-Shot Test Evaluation
```bash
python src/evaluate_test.py
```

### Step 5: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
*Opens the interactive dashboard at `http://localhost:8501`.*

---

## 9. Web Application Overview

The Streamlit dashboard ([`app.py`](app.py)) provides three operational modes:

1. **Single Order Risk Scorer:**
   - Real-time parameter inputs (payment mode, order value, category, tier, discount, transit ETA, account age).
   - One-click testing presets (*High Risk COD*, *Safe Prepaid*, *Borderline Festive*).
   - Instant calculation of $P(\text{RTO})$, Expected Logistics Waste (₹), and Action Badges.
2. **Model & Economics Benchmark:**
   - Full side-by-side metric tables across baseline, balanced logistic regression, and decision trees.
   - P&L financial simulation showing rupee savings per 5,000 orders.
3. **Live Batch Validation Sample:**
   - Interactive inference engine sampling real unseen validation records from `data/processed/val.parquet`.

---

## 10. Limitations and Future Improvements

1. **Address Text Embeddings:** The current model excludes raw address strings to prevent memorization. Future versions can incorporate lightweight transformer embeddings (or regex-extracted street clarity metrics) to capture vague addresses without PII risk.
2. **Dynamic Risk Thresholds:** The decision threshold is currently static ($0.50$). In production, thresholds can dynamically adjust based on gross margin: high-margin products can tolerate higher RTO risk, while thin-margin goods require stricter cutoffs.
3. **Real-time Merchant Feedback Loop:** Incorporating courier delivery feedback to update customer reliability priors online.

---

## Project Structure
```text
RTO-Risk-Predictor/
├── .gitignore               # Git exclusion specifications
├── README.md                # Project documentation
├── PROGRESS.md              # 5-Day roadmap checklist (20/20 complete)
├── DAY1_NOTES.md            # Dataset schema & ML fundamentals
├── DAY2_NOTES.md            # Data preprocessing & baseline pipeline
├── DAY3_EVALUATION.md       # Model comparison & threshold analysis
├── INTERVIEW_PREP.md        # 2-minute pitch & 10 interview Q&As
├── requirements.txt         # Frozen Python dependencies
├── app.py                   # Streamlit web dashboard
├── config/
│   └── evidence.yaml        # Indian e-commerce benchmark calibration parameters
├── data/
│   ├── external/            # India Post official pincode directory
│   ├── raw/                 # Raw 50,000 orders CSV
│   └── processed/           # Chronological train, val, and test parquet splits
├── models/
│   └── rto_model.joblib     # Serialized winning pipeline
├── notebooks/
│   ├── exploration.ipynb    # Jupyter exploration notebook with outputs
│   └── exploration.py       # Standalone exploration script
└── src/
    ├── __init__.py
    ├── features.py          # Feature extraction & encoding logic
    ├── generate.py          # Synthetic dataset generation engine
    ├── train_model.py       # Model training, comparison & serialization
    ├── predict.py           # Inference & risk assessment helper
    └── evaluate_test.py     # Final test set evaluation script
```
