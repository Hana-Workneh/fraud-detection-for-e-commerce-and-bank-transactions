# Fraud Detection: E-commerce + Bank Transactions

This repository contains an end-to-end fraud detection project using:
- **Fraud_Data.csv** (e-commerce transactions)
- **IpAddress_to_Country.csv** (IP-to-country mapping)
- **creditcard.csv** (bank card transactions)

## Project Structure
See the folder layout in the root directory. The `data/` folder is gitignored.

## Quickstart
1. Create and activate a virtual environment
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Put raw datasets into:
   - `data/raw/Fraud_Data.csv`
   - `data/raw/IpAddress_to_Country.csv`
   - `data/raw/creditcard.csv`

## Task 1 (Interim-1)
- EDA notebooks:
  - `notebooks/eda-fraud-data.ipynb`
  - `notebooks/eda-creditcard.ipynb`
- Feature engineering:
  - `notebooks/feature-engineering.ipynb`
- Optional scripts for reproducible preprocessing:
  - `scripts/task1_preprocess_fraud_data.py`
  - `scripts/task1_preprocess_creditcard.py`

## Outputs
Processed datasets should be saved to:
- `data/processed/fraud_ecommerce.parquet`
- `data/processed/creditcard.parquet`

## Task 2 — Model Building & Training (Interim-2)
✅ Data Preparation

- Train/test split with stratification to preserve fraud ratio

- Feature matrix X and target y prepared separately

- Categorical encoding + numeric scaling handled via ColumnTransformer

✅ Baseline Model: Logistic Regression

- A Logistic Regression baseline was trained for interpretability and benchmarking.

Evaluation includes:

- AUC-PR (Precision-Recall AUC)

- F1 Score

- Confusion Matrix

- Classification Report

Example results (E-commerce baseline):

- AUC-PR: 0.3954

- F1 (default threshold 0.5): 0.2767

✅ Threshold Tuning (Business-Driven)

- Because false positives and false negatives carry different costs, the decision threshold was tuned to maximize F1.

Best threshold (by F1):

- Threshold ≈ 0.70

- Best F1 ≈ 0.47

This improves balance between catching fraud and reducing unnecessary blocks.

All baseline modeling work is documented in:

notebooks/modeling.ipynb

## Next Steps
Task 2 (Remaining)

Train and compare an ensemble model (Random Forest / XGBoost / LightGBM)

Hyperparameter tuning (basic grid/random search)

Stratified K-Fold cross-validation

Task 3 — Explainability

SHAP summary plot (global)

SHAP force plots for:

True positive (fraud caught)

False positive (legitimate flagged)

False negative (fraud missed)

Convert insights into actionable business recommendations