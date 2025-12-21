"""Task 1 preprocessing for creditcard.csv.

Usage:
  python scripts/task1_preprocess_creditcard.py --raw_path data/raw/creditcard.csv --out_path data/processed/creditcard.parquet

Notes:
- Performs light validation + dedup + ensures numeric types.
- Does NOT resample.
- Scaling can be done in modeling via Pipeline to avoid leakage.
"""

from __future__ import annotations
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_path", default="data/raw/creditcard.csv")
    ap.add_argument("--out_path", default="data/processed/creditcard.parquet")
    args = ap.parse_args()

    df = pd.read_csv(args.raw_path)
    df = df.drop_duplicates().copy()

    # Ensure numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.to_parquet(args.out_path, index=False)
    print(f"Saved: {args.out_path}")
    if "Class" in df.columns:
        print("Class balance (fraud=1):", df["Class"].mean())
    print("Rows:", len(df), "Cols:", df.shape[1])

if __name__ == "__main__":
    main()
