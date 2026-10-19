# Machine Learning Interview Preparation Guide: RTO Risk Predictor

This cheat sheet provides concise, high-impact, interview-ready responses filled in with the exact metrics, architectural decisions, and economic calculations from this project.

---

## 1. The 2-Minute Project Pitch

> **Interviewer:** *"Can you walk me through your E-Commerce Post-Dispatch RTO Risk Predictor project?"*

**Response:**
> "In Indian e-commerce, Cash on Delivery accounts for over 60% of all transactions, but nearly 1 in 4 COD orders results in a Return-to-Origin (RTO). When a parcel returns undelivered, the merchant absorbs two-way freight, damaged packaging, and trapped inventory — losing an average of ₹200 per order. Once a parcel is handed over to the courier, that financial loss is guaranteed.
>
> To solve this, I designed and built an end-to-end, explainable machine learning pipeline that predicts RTO risk **after checkout, but before warehouse dispatch**.
>
> I trained on a 50,000-order dataset calibrated against official India Post geography and published Indian e-commerce return benchmarks. To ensure zero data leakage, I enforced a strict **chronological train/validation/test split** along order timestamps rather than random shuffling.
>
> Because RTO is an imbalanced event (~15.5% prevalence), unweighted baseline models suffer from the 'accuracy trap' — achieving 84.5% accuracy while missing 97.3% of returns. I solved this by implementing a **Balanced Logistic Regression** pipeline using a scikit-learn `ColumnTransformer`. On our untouched, held-out test set of 10,000 future orders, the model achieved an **ROC-AUC of 0.7924 and caught 72.37% of all returns (1,066 returns caught)**, reducing net freight loss by **55.3% (₹1,62,900 saved)**.
>
> Finally, I serialized the pipeline using `joblib` and built an interactive **Streamlit dashboard** that provides real-time risk scores, cost-impact estimates, and automated intervention recommendations — such as requesting a ₹50 deposit or sending WhatsApp address confirmations."

---

## 2. Top 10 Technical & Business Interview Questions

### Q1: Why did you choose Logistic Regression instead of XGBoost, LightGBM, or Deep Learning?
**Answer:**
> "I prioritized **regulatory explainability, low-latency checkout inference, and direct financial interpretability**. 
> 
> In retail risk management, if a merchant asks a customer for an upfront deposit or blocks COD, the merchant operations team must know *why*. With Logistic Regression, every feature has an explicit log-odds coefficient and odds ratio. For instance, our model proves that choosing COD triples the odds of RTO ($\beta = +1.097$, Odds Ratio = $3.00$), and the Fashion vertical doubles the odds ($\beta = +0.735$, Odds Ratio = $2.08$), whereas prepayment slashes the odds by 87% ($\beta = -2.026$, Odds Ratio = $0.13$).
>
> Furthermore, with an ROC-AUC of ~0.80 on tabular data, Logistic Regression captured the primary linear risk boundaries without the overfitting risk, calibration drift, or memory footprint of deep neural nets."

---

### Q2: How did you prevent data leakage during preprocessing and feature selection?
**Answer:**
> "I addressed data leakage at two critical levels:
> 
> 1. **Temporal / Causality Leakage:** Standard random shuffling leaks future trends backwards. I implemented a strict **chronological split**: Training on past orders (Jan 2024–Jan 2025), validating on middle orders (Jan–Mar 2025), and evaluating strictly once on held-out future orders (Mar–Jun 2025). Furthermore, all post-dispatch attributes (delivery boy notes, attempt counts, tracking updates) were excluded from feature matrix $X$.
> 2. **Distribution / Preprocessing Leakage:** I encapsulated feature imputation and scaling inside a scikit-learn `ColumnTransformer` wrapped in a `Pipeline`. Crucially, `pipeline.fit()` was called **only on the training partition**. The mean, standard deviation, and one-hot categories learned from training were then applied downstream to validation and test data without re-fitting."

---

### Q3: Why is Accuracy a misleading metric in this project?
**Answer:**
> "Because our target is class-imbalanced: 84.5% of orders are delivered successfully, and only 15.5% are RTOs.
> 
> If a model predicts 'Delivered' (Class 0) for every single order, it achieves a high **84.50% accuracy**, yet it catches **0% of returns**. In our experiments, the unweighted baseline Logistic Regression achieved 84.46% accuracy, but had an abysmal **recall of only 2.71%** (catching only 21 of 775 RTOs). In a business where missed RTOs cost ₹200 each, that accuracy was completely useless. We evaluated primarily on **Recall, Precision, F1-Score, and ROC-AUC**."

---

### Q4: How did you handle class imbalance in your models?
**Answer:**
> "I utilized algorithmic cost-sensitive learning via `class_weight='balanced'`. 
> 
> Instead of synthetically duplicating data with SMOTE (which can distort probability calibration), `class_weight='balanced'` modifies the cross-entropy loss function by weighting each sample inversely to its class frequency:
> $$w_j = \frac{N}{2 \cdot N_j}$$
> In our training set, Class 1 (RTO) received a weight of $\approx 2.95$, while Class 0 (Delivered) received $\approx 0.60$. This forced the optimizer (L-BFGS) to treat a missed RTO as five times more severe than an error on a delivered order, raising RTO recall from **2.71% to 75.35%** on validation."

---

### Q5: Explain your Confusion Matrix and the business cost of False Positives vs. False Negatives.
**Answer:**
> "On our 10,000 held-out test orders, our confusion matrix was:
> - **True Negatives (TN):** 6,012 genuine orders cleared for auto-dispatch.
> - **True Positives (TP):** 1,066 high-risk RTOs caught before dispatch.
> - **False Negatives (FN):** 407 RTOs missed and shipped as normal.
> - **False Positives (FP):** 2,515 delivered orders flagged as high risk.
>
> In business terms, these errors have vastly asymmetric costs:
> - **Cost of FN (Missed RTO):** ₹200 (₹80 forward freight + ₹80 reverse freight + ₹20 destroyed packaging + ₹20 blocked inventory).
> - **Cost of FP (Flagging a genuine order):** Only ~₹20. When an order is flagged, we don't cancel it; we trigger an automated WhatsApp address confirmation link (~₹2) or request a ₹50 advance deposit.
> 
> Because an FN is **10x more costly** than an FP, catching 1,066 returns saved **₹1,62,900 net** on test orders alone (a 55.3% reduction in freight waste)."

---

### Q6: How did you select your probability decision threshold?
**Answer:**
> "Standard machine learning defaults to a 0.50 threshold. However, optimal thresholds depend on the economic loss matrix.
> 
> On the validation split, I evaluated thresholds across 0.30, 0.40, and 0.50:
> - At **0.30 threshold**: Recall was 97.03%, but FP exploded to 2,395 orders.
> - At **0.50 threshold (Balanced model)**: Achieved the optimal trade-off with **75.35% recall, 30.04% precision, and maximum F1 of 0.4296**.
> 
> In production, I translated this into a **three-tier operational policy**:
> - $P < 0.35$: Low Risk $\rightarrow$ Frictionless auto-dispatch.
> - $0.35 \le P < 0.50$: Medium Risk $\rightarrow$ Automated WhatsApp address/landmark confirmation.
> - $P \ge 0.50$: High Risk $\rightarrow$ Mandatory ₹50 advance deposit or Prepaid conversion."

---

### Q7: What was your data splitting strategy, and why didn't you use random `train_test_split`?
**Answer:**
> "I used a strict **chronological time-based split** along `order_ts`:
> - 70% Training: Jan 1, 2024 to Jan 22, 2025 (35,000 orders)
> - 10% Validation: Jan 22, 2025 to Mar 16, 2025 (5,000 orders)
> - 20% Test: Mar 16, 2025 to Jun 30, 2025 (10,000 orders)
> 
> If we used random splitting (`shuffle=True`), orders from November 2024 would train the model to predict orders from March 2024. That introduces lookahead bias: seasonal shopping trends, geographic shifts, and repeat customer history leak backwards into the training set. A time-based split is the only realistic way to simulate true production performance."

---

### Q8: How did you verify that your model did not overfit?
**Answer:**
> "I compared the validation metrics against the held-out test split, which had remained completely untouched throughout modeling and threshold tuning:
> - **ROC-AUC:** 0.7993 (Validation) vs. **0.7924 (Test)** $\rightarrow$ Delta of only **$-0.0069$** (<0.7% drop).
> - **Recall:** 75.35% (Validation) vs. **72.37% (Test)** $\rightarrow$ Delta of **$-0.0298$**.
> - **F1-Score:** 0.4296 (Validation) vs. **0.4218 (Test)** $\rightarrow$ Delta of **$-0.0078$**.
> 
> This consistency proves that the model learned true causal drivers (payment mode, category, tier, discount, delivery ETA) rather than memorizing training noise."

---

### Q9: How did you transition from an ML script to a production Streamlit web application?
**Answer:**
> "I serialized the entire pipeline using `joblib.dump(pipeline, 'models/rto_model.joblib')`. 
> 
> Next, I created a clean inference module (`src/predict.py`) that loads and caches the pipeline, accepts single-order JSON payloads, enforces strict data schemas, and extracts qualitative explanations.
> 
> Finally, in `app.py`, I built an intuitive Streamlit interface featuring:
> 1. Dropdown selectors matching our exact dataset categories (`fashion`, `electronics`, `metro`, `tier_3`, etc.).
> 2. Quick-scenario preset buttons for instant merchant testing.
> 3. Real-time metric cards displaying probability, financial exposure, and color-coded risk badges."

---

### Q10: What are the primary limitations of your model, and what would you build next?
**Answer:**
> "Two main limitations:
> 1. **Lack of NLP on Address Text:** To prevent memorization and privacy issues, I excluded raw address lines. In the future, I would use lightweight regex or a transformer to score address completeness (e.g. presence of house number, recognizable landmark, consonant gibberish score).
> 2. **Static Thresholding:** The current threshold is fixed at 0.50. In a commercial platform, thresholding should be dynamic based on **product profit margin**: A high-margin luxury handbag can tolerate higher RTO risk before intervention than a low-margin phone cable.
> 3. **Feedback Loop:** Adding automated courier tracking hooks to continuously update customer reliability scores post-delivery."
