# E-Commerce Post-Dispatch RTO Risk Predictor — Progress Tracker

## 5-Day Project Roadmap (20 Checklist Items)

### DAY 1: Understand the Dataset & ML Fundamentals
- [x] 1. Project setup: directory structure (`data/`, `notebooks/`), macOS virtual environment (`.venv`), core packages installed (`requirements.txt`).
- [x] 2. Dataset inspection: locate dataset in `data/`, load and inspect shape, schema, null values, duplicates, and categorical columns.
- [x] 3. Target variable verification: inspect distribution of target column (`0 = delivered`, `1 = RTO`).
- [x] 4. Day 1 documentation & ML fundamentals: create `DAY1_NOTES.md` documenting feature usage/exclusion and 5 core ML concepts.

### DAY 2: Data Preprocessing & Baseline Logistic Regression Model
- [x] 5. Feature selection: select pre-dispatch, leakage-safe features and exclude IDs/PII/post-dispatch fields.
- [x] 6. Chronological split: create train/val/test splits ordered chronologically by order timestamp.
- [x] 7. Scikit-Learn preprocessing pipeline: `ColumnTransformer` with imputation, encoding, and scaling wrapped in a `Pipeline`.
- [x] 8. Baseline Logistic Regression training & validation: fit pipeline strictly on training data and generate validation predictions.

### DAY 3: Model Comparison, Class Imbalance & Threshold Tuning
- [x] 9. Baseline evaluation: compute accuracy, precision, recall, F1, confusion matrix, and classification report on validation set.
- [x] 10. Model experiments: evaluate unweighted Logistic Regression, balanced Logistic Regression, and Decision Tree (`src/train_model.py`).
- [x] 11. Probability threshold comparison: evaluate validation metrics across thresholds (0.30, 0.40, 0.50) without touching test set.
- [x] 12. Day 3 evaluation report & concept walkthrough: create `DAY3_EVALUATION.md` justifying chosen model and threshold in business terms.

### DAY 4: Model Serialization & Streamlit Web Application
- [ ] 13. Model serialization: save winning pipeline to `models/rto_model.joblib` and create `src/predict.py`.
- [ ] 14. Streamlit setup: install streamlit and update `requirements.txt`.
- [ ] 15. Streamlit app development: create interactive `app.py` matching dataset schema with risk badges.
- [ ] 16. App verification & edge-case testing: test diverse input scenarios ensuring zero runtime errors.

### DAY 5: Final Test Set Evaluation, Documentation & Interview Prep
- [ ] 17. End-to-end verification & final test set evaluation: evaluate winning model once on held-out test split.
- [ ] 18. Documentation: create `.gitignore` and comprehensive 10-section `README.md`.
- [ ] 19. Interview prep guide: create `INTERVIEW_PREP.md` with 2-minute pitch and answers to 10 interview questions.
- [ ] 20. Final project wrap-up & git push instructions.
