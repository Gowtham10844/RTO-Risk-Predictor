# Day 1: Dataset Architecture, Feature Inventory & ML Fundamentals

**Project:** E-Commerce Post-Dispatch RTO Risk Predictor  
**Date:** Day 1 Implementation  
**Dataset Source:** 50,000 synthetic Indian e-commerce checkout orders calibrated to real India Post geography and published industry COD benchmarks.

---

## 1. Dataset Overview & Data Quality Audit

| Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **Total Rows (Orders)** | 50,000 | Generated deterministically (seed `20260101`) |
| **Total Columns** | 24 (raw) / 42 (processed) | 24 observable at checkout |
| **Missing / Null Values** | 0 across all columns | Clean dataset with zero null values |
| **Duplicate Rows** | 0 | Every order record is unique |
| **Target Variable (`rto`)** | Binary (0 / 1) | **Class 0 (Delivered):** 41,815 (83.63%)<br>**Class 1 (RTO):** 8,185 (16.37%) |
| **Class Imbalance Ratio** | ~ 5.11 : 1 | Delivery is the majority class; RTO is the minority class |

---

## 2. Feature Column Inventory: Used vs. Excluded

In post-dispatch RTO risk prediction, **leakage prevention and causality** are critical. We only use information knowable **at checkout / before order dispatch**.

### A. Pre-Dispatch Features Included in Training (`X`)

| Column Name | Type | Domain / Values | Why We Include It (Causal Justification) |
| :--- | :--- | :--- | :--- |
| `payment_mode` (or `is_cod`) | Categorical / Boolean | `COD`, `PREPAID` | **Single strongest predictor.** In our dataset, COD has a **25.61% RTO rate** versus PREPAID at **1.41%**. Customers who have already paid upfront rarely refuse parcels at the door. |
| `order_value` | Numerical (float) | ₹189.0 to ₹41,896.0 | Captures buyer commitment. Orders in high price bands carry different financial risk thresholds. |
| `category` | Categorical | `fashion`, `home_kitchen`, `electronics`, `accessories`, `beauty`, `footwear` | **High variance by vertical.** Fashion has a **27.74% RTO rate** (sizing, color, and fit issues), while Electronics has only **6.25% RTO rate**. |
| `pincode_tier` | Categorical | `metro`, `tier_1`, `tier_2`, `tier_3` | **Urbanization proxy.** RTO increases monotonically as infrastructure tier drops: Metro (12.61%) → Tier 1 (14.57%) → Tier 2 (20.52%) → Tier 3 (26.50%). |
| `discount_pct` | Numerical (float) | 0.00 to 0.715 (0% to 71.5%) | Deep discounts trigger impulsive, low-commitment buying behavior, which leads to buyer remorse when the delivery arrives. |
| `is_festive` | Boolean / Binary | `True`, `False` | Seasonal shopping surges (Diwali, Great Indian Festival) elevate delivery delays and impulse cancellations (20.22% festive RTO vs 14.94% non-festive). |
| `delivery_days_est` | Numerical (float) | 1.0 to 14.7 days | Longer transit times correlate with higher return probability as customer enthusiasm cools down. |
| `account_age_days` | Numerical (float) | 0.0 to 728.7 days | Age of user account at checkout. Brand-new accounts (day 0) present significantly higher baseline fraud/RTO risk than loyal repeat customers. |
| `order_velocity_24h` | Numerical (int) | 0, 1, 2 | Rapid repeated checkouts in a 24-hour window indicate erratic or impulsive ordering patterns. |
| `is_alternate_address` | Boolean / Binary | `True`, `False` | Deliveries sent to unverified or temporary locations carry higher failed delivery attempt rates (17.51% vs 16.21%). |

---

### B. Columns Excluded from Training (`X`)

| Column Name | Reason for Exclusion |
| :--- | :--- |
| `order_id` | **High Cardinality Identifier.** Unique per row (50,000 unique values). Using IDs in models leads to arbitrary memorization and zero generalizability. |
| `order_ts` | **Chronological Splitting Only.** Must not be fed as a raw feature to prevent the model from learning timestamp shortcuts. Used to sort data into chronological train, validation, and test splits. |
| `customer_id` | **Customer PII / High Cardinality.** Cannot generalize to new, first-time customers (cold start). |
| `customer_name` | **Customer PII.** Cosmetic personal string. Including names introduces noise and demographic bias without predictive value. |
| `phone` | **Customer PII.** High cardinality phone numbers. |
| `address_line` | **Raw Unstructured Free Text.** Contains noise, spelling errors, and individual street names. In tabular models, raw address strings cause overfitting. Instead, engineered proxies (`addr_has_house_number`, `addr_quality_score`) are used if needed. |
| `pincode` | **High Cardinality Categorical / Arbitrary Number.** 6-digit postal codes treated as continuous integers create false mathematical relationships (e.g., treating pincode 560001 as "greater than" 110001). Categorical pincode tier (`pincode_tier`) is used instead. |
| `city`, `district`, `state` | **High Cardinality / Geographical Redundancy.** 528 cities and 369 districts would induce excessive sparsity in linear models. `pincode_tier` captures the urbanization effect cleanly. |
| `order_value_band` | **Redundant with `order_value`.** Discretized version of `order_value` (`under_500`, `500_1000`, `1000_plus`). Using both creates multicollinearity. |
| `signup_date` | **Redundant with `account_age_days`.** The elapsed duration is already captured as a stationary continuous variable. |
| `_pincode_rto_prior`, `_reliability_z`, `_address_quality`, `_p_rto_true` | **Latent Generator Variables (Data Leakage).** Unobservable real-world latents. Using them would constitute severe data leakage. |
| **Post-dispatch metrics** | Any courier attempt status, delivery boy notes, or tracking events are excluded because our prediction must happen **prior to dispatch**. |

---

## 3. Explaining 5 Core ML Concepts (Using Real Dataset Examples)

### Concept 1: Supervised Learning
* **Definition:** A machine learning paradigm where the algorithm learns a mathematical mapping function $f(X) \rightarrow y$ from a historical dataset of input features ($X$) paired with known, ground-truth outcomes ($y$).
* **Project Example:** We provide the model with historical orders containing attributes such as `payment_mode="COD"`, `category="fashion"`, `order_value=1299.0`, and its paired actual outcome `rto=1`. The model adjusts its internal weights so that when a new, unseen order arrives, it can estimate the likelihood of RTO.

### Concept 2: Binary Classification
* **Definition:** A supervised learning task where the target variable can take exactly one of two discrete, mutually exclusive classes or categories (typically encoded as 0 and 1).
* **Project Example:** For each order, there are exactly two eventual delivery states:
  - **Class 0 (Negative):** Order was successfully delivered and accepted by the customer (41,815 orders in our dataset, 83.63%).
  - **Class 1 (Positive):** Order was undelivered, rejected at the doorstep, and Returned to Origin (8,185 orders in our dataset, 16.37%).

### Concept 3: Features ($X$)
* **Definition:** The individual measurable properties, characteristics, or explanatory variables fed into the model to represent each observation.
* **Project Example:** For an order placed on our platform, our feature vector $X$ consists of pre-dispatch observable attributes:
  $$X = [\text{payment\_mode}, \text{pincode\_tier}, \text{category}, \text{order\_value}, \text{discount\_pct}, \text{is\_festive}, \text{delivery\_days\_est}, \dots]$$
  For instance, an order with $X = [\text{payment\_mode} = \text{"COD"}, \text{tier} = \text{"tier\_3"}, \text{category} = \text{"fashion"}, \text{order\_value} = 850.0]$ represents an order with multiple compounding risk factors.

### Concept 4: Target ($y$)
* **Definition:** The dependent variable or ground-truth label that the model is tasked with predicting.
* **Project Example:** Our target is `rto` $\in \{0, 1\}$.
  - In an e-commerce context, Class 1 ($y = 1$) is the "costly event" we want to identify early, as shipping a parcel back and forth without collecting payment loses approximately ₹150–₹250 in forward logistics, reverse logistics, packaging, and inventory lockup costs.

### Concept 5: Chronological Training vs. Testing Split
* **Definition:** The practice of dividing historical data into separate partitions to train the model on past data and evaluate its predictive accuracy on strictly future, held-out data.
* **Project Example & Why Random Shuffling Leaks:**
  In real-world e-commerce, time moves forward. If we used standard `train_test_split(shuffle=True)`, an order from November 2024 would be in the training set while an order from March 2024 would be in the test set. The model would be "learning from the future to predict the past," creating artificial performance inflation.
  Instead, our dataset uses a **strict chronological split** along `order_ts`:
  - **Train set:** 35,000 orders (Jan 1, 2024 to Jan 22, 2025) — 70% of data.
  - **Validation set:** 5,000 orders (Jan 22, 2025 to Mar 16, 2025) — 10% of data (for model selection & threshold tuning).
  - **Test set:** 10,000 orders (Mar 16, 2025 to Jun 30, 2025) — 20% of data (held out and untouched until Day 5 final evaluation).

---

## 4. Next Steps (Day 2 Preview)

On **Day 2**, we will:
1. Construct the scikit-learn `ColumnTransformer` (handling categorical encoding with `OneHotEncoder` and numerical scaling with `StandardScaler`).
2. Wrap the preprocessor and `LogisticRegression(max_iter=1000)` into a reproducible, leakage-free `Pipeline`.
3. Fit the pipeline strictly on `(X_train, y_train)` and generate predictions on the chronological validation set `X_val`.
