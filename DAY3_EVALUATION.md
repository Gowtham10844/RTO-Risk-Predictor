# Model Comparison, Class Imbalance Handling & Threshold Tuning

**Project:** E-Commerce Post-Dispatch RTO Risk Predictor  
**Validation Evaluation Dataset:** $N = 5,000$ chronological held-out orders (Jan 22, 2025 to Mar 16, 2025)  
**Base RTO Rate in Validation Set:** 15.50% (775 RTO orders, 4,225 Delivered orders)  
**Implementation Script:** [`src/train_model.py`](file:///Users/saimpugowtham/Desktop/PROJECTS/RTO-Risk-Predictor/src/train_model.py)

---

## 1. Executive Summary

In e-commerce RTO risk management, optimizing for pure classification **accuracy** is fundamentally flawed due to class imbalance (~5.4 : 1). A naive model that predicts every single order will be delivered successfully achieves **84.50% accuracy**, yet allows **100% of RTO losses** to occur unchecked.

This evaluation compares three model configurations on the chronological validation split, analyzes the business economics of False Positives vs. False Negatives, sweeps decision thresholds ($0.30$, $0.40$, $0.50$), and selects the winning production model.

---

## 2. Model Performance Comparison Table (Threshold = 0.50)

| Model Architecture | Class Weighting | Accuracy | Precision (RTO) | Recall (RTO) | F1-Score | ROC-AUC | Key Behavior |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Logistic Regression (Baseline)** | None (Unweighted) | **0.8446** | **0.4773** | 0.0271 | 0.0513 | **0.7994** | High accuracy, but **catches only 21 of 775 RTOs (2.7% recall)**. Useless in production. |
| **2. Logistic Regression (Balanced)** | `class_weight="balanced"` | 0.6898 | 0.3004 | **0.7535** | **0.4296** | **0.7993** | **Best balance:** Catches **584 of 775 RTOs (75.4% recall)** with highest F1 and strong ROC-AUC. |
| **3. Decision Tree Classifier** | `max_depth=5, balanced` | 0.6898 | 0.2925 | 0.7058 | 0.4136 | 0.7813 | Lower ROC-AUC (0.7813) and lower recall (70.6%) due to axis-aligned step partitions. |

---

## 3. Confusion Matrix Breakdown & Classification Reports

### Experiment 1: Baseline Logistic Regression (Unweighted)
* **Confusion Matrix:**
  $$\begin{pmatrix} \text{TN} = 4,202 & \text{FP} = 23 \\ \text{FN} = 754 & \text{TP} = 21 \end{pmatrix}$$
* **Classification Report:**
  ```text
                 precision    recall  f1-score   support
  Delivered (0)     0.8479    0.9946    0.9154      4225
        RTO (1)     0.4773    0.0271    0.0513       775
       accuracy                         0.8446      5000
      macro avg     0.6626    0.5108    0.4833      5000
   weighted avg     0.7904    0.8446    0.7814      5000
  ```
* **Diagnosis:** The unweighted loss function is dominated by the 84.5% majority class. The model sets the decision boundary too conservatively, resulting in a **97.3% False Negative rate** on RTO orders.

---

### Experiment 2: Balanced Logistic Regression (`class_weight="balanced"`)
* **Confusion Matrix:**
  $$\begin{pmatrix} \text{TN} = 2,865 & \text{FP} = 1,360 \\ \text{FN} = 191 & \text{TP} = 584 \end{pmatrix}$$
* **Classification Report:**
  ```text
                 precision    recall  f1-score   support
  Delivered (0)     0.9375    0.6781    0.7870      4225
        RTO (1)     0.3004    0.7535    0.4296       775
       accuracy                         0.6898      5000
      macro avg     0.6190    0.7158    0.6083      5000
   weighted avg     0.8388    0.6898    0.7316      5000
  ```
* **Diagnosis:** Penalizing minority-class errors by the inverse class frequency ($w_1 \approx 2.95, w_0 \approx 0.60$) shifts the decision plane. RTO recall surges from **2.71% to 75.35%**, and F1-score increases more than eight-fold (**0.0513 → 0.4296**).

---

### Experiment 3: Decision Tree Classifier (`max_depth=5, balanced`)
* **Confusion Matrix:**
  $$\begin{pmatrix} \text{TN} = 2,902 & \text{FP} = 1,323 \\ \text{FN} = 228 & \text{TP} = 547 \end{pmatrix}$$
* **Classification Report:**
  ```text
                 precision    recall  f1-score   support
  Delivered (0)     0.9272    0.6869    0.7891      4225
        RTO (1)     0.2925    0.7058    0.4136       775
       accuracy                         0.6898      5000
      macro avg     0.6098    0.6963    0.6014      5000
   weighted avg     0.8288    0.6898    0.7309      5000
  ```
* **Diagnosis:** While interpretable, the shallow tree creates coarse, piecewise-constant probability bands across feature splits. It achieves lower ROC-AUC (0.7813) and catches 37 fewer returns than Balanced Logistic Regression.

---

## 4. Probability Threshold Comparison Table

To choose the operating threshold, we evaluate validation predictions across thresholds $0.30$, $0.40$, and $0.50$ (keeping the test set strictly untouched):

### Precision / Recall / F1 Trade-off Across Models

| Model | Threshold | Accuracy | Precision | Recall | F1-Score | True Positives (TP) | False Positives (FP) | False Negatives (FN) | True Negatives (TN) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Unweighted)** | `0.30` | 0.8084 | 0.3984 | 0.4632 | 0.4284 | 359 | 542 | 416 | 3,683 |
| | `0.40` | 0.8424 | 0.4780 | 0.1819 | 0.2636 | 141 | 154 | 634 | 4,071 |
| | `0.50` | 0.8446 | 0.4773 | 0.0271 | 0.0513 | 21 | 23 | 754 | 4,202 |
| **Logistic Regression (Balanced)** | `0.30` | 0.5164 | 0.2390 | **0.9703** | 0.3835 | 752 | 2,395 | **23** | 1,830 |
| | `0.40` | 0.5586 | 0.2498 | 0.9226 | 0.3932 | 715 | 2,147 | 60 | 2,078 |
| | `0.50` | **0.6898** | **0.3004** | **0.7535** | **0.4296** | **584** | **1,360** | **191** | **2,865** |
| **Decision Tree (Balanced)** | `0.30` | 0.5140 | 0.2381 | 0.9703 | 0.3823 | 752 | 2,407 | 23 | 1,818 |
| | `0.40` | 0.5142 | 0.2381 | 0.9703 | 0.3824 | 752 | 2,406 | 23 | 1,819 |
| | `0.50` | 0.6898 | 0.2925 | 0.7058 | 0.4136 | 547 | 1,323 | 228 | 2,902 |

---

## 5. Business Economics: Cost of Errors & Trade-off Analysis

In machine learning interviews and production design, evaluation metrics must connect to **business profit & loss (P&L)**:

### Definitions in Business Terms:
1. **True Positive (TP):** High-risk RTO order correctly identified.
   - **Action:** Merchant requests OTP confirmation, asks for ₹50 advance payment, or verifies address before dispatch.
   - **Benefit:** Prevents two-way logistics waste on an uncommitted buyer.
2. **False Positive (FP):** Genuine customer order incorrectly flagged as risky.
   - **Cost:** Verification friction (automated WhatsApp notification or SMS costs ~₹2). Even if a small fraction of customers drop off, the expected cost per FP is only **~₹20**.
3. **False Negative (FN):** High-risk RTO missed and shipped as normal COD.
   - **Cost:** **The most damaging error.** Forward courier freight (~₹80), return freight (~₹80), packaging destruction (~₹20), and blocked inventory (~₹20). Direct wasted cost is **~₹200 per order**.
4. **True Negative (TN):** Genuine order correctly predicted as low risk.
   - **Action:** 1-click frictionless dispatch directly to courier hub. Zero unnecessary overhead.

### Expected Net Loss Calculation:
$$\text{Expected Loss} = (\text{FN} \times ₹200) + (\text{FP} \times ₹20)$$

Evaluating on our 5,000 validation orders:
- **No Model / Baseline LR (Unweighted @ 0.50):**
  - $\text{FN} = 754$, $\text{FP} = 23$
  - $\text{Loss} = (754 \times ₹200) + (23 \times ₹20) = ₹1,50,800 + ₹460 = \mathbf{₹1,51,260}$
- **Balanced Logistic Regression (@ 0.50):**
  - $\text{FN} = 191$, $\text{FP} = 1,360$
  - $\text{Loss} = (191 \times ₹200) + (1,360 \times ₹20) = ₹38,200 + ₹27,200 = \mathbf{₹65,400}$
  - **Net Savings:** **₹85,860 saved on just 5,000 validation orders (56.8% cost reduction!)**
- **Balanced Logistic Regression (@ 0.40):**
  - $\text{FN} = 60$, $\text{FP} = 2,147$
  - $\text{Loss} = (60 \times ₹200) + (2,147 \times ₹20) = ₹12,000 + ₹42,940 = \mathbf{₹54,940}$

---

## 6. Final Model & Threshold Decision

### Winning Model: **Balanced Logistic Regression (`class_weight="balanced"`)**
* **Why it wins:**
  1. **Superior Ranking Power:** Highest ROC-AUC (**0.7993** vs 0.7813 for Decision Tree).
  2. **Superior Recall & F1:** Catches **75.35% of all RTOs** (vs 70.58% for Decision Tree and 2.71% for Unweighted LR), with the highest validation F1-score of **0.4296**.
  3. **Calibration & Interpretability:** Generates smooth, well-calibrated probabilities via sigmoid and provides transparent log-odds coefficients for merchant explainability.

### Production Threshold Strategy:
* **Low Risk ($P < 0.35$):** Auto-dispatch COD orders without friction.
* **Medium Risk ($0.35 \le P < 0.50$):** Automated WhatsApp address verification / IVR confirmation.
* **High Risk ($P \ge 0.50$):** Mandatory ₹50–₹100 advance payment conversion or prepaid-only requirement.
