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
