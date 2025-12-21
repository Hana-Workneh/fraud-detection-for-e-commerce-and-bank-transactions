"""Task 1 preprocessing for Fraud_Data.csv (e-commerce).

Usage:
  python scripts/task1_preprocess_fraud_data.py --raw_dir data/raw --out_path data/processed/fraud_ecommerce.parquet

Notes:
- Performs IP->country merge via range lookup using merge_asof.
- Engineers time-based and velocity features.
- Does NOT do resampling (that belongs in modeling, train-only).
"""

from __future__ import annotations
import argparse
import pandas as pd
import numpy as np

def load_inputs(raw_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fraud = pd.read_csv(f"{raw_dir}/Fraud_Data.csv")
    ip_map = pd.read_csv(f"{raw_dir}/IpAddress_to_Country.csv")
    return fraud, ip_map

def coerce_ip_to_int(series: pd.Series) -> pd.Series:
    # In many versions of this dataset, ip_address is already numeric-like.
    # If you have dotted IPv4 strings, you must convert them differently.
    return pd.to_numeric(series, errors="coerce")

def merge_ip_to_country(fraud: pd.DataFrame, ip_map: pd.DataFrame) -> pd.DataFrame:
    fraud = fraud.copy()
    ip_map = ip_map.copy()

    fraud["ip_int"] = coerce_ip_to_int(fraud["ip_address"])
    fraud = fraud.dropna(subset=["ip_int"]).copy()
    fraud["ip_int"] = fraud["ip_int"].astype(np.int64)

    ip_map["lower_bound_ip_address"] = pd.to_numeric(ip_map["lower_bound_ip_address"], errors="coerce").astype(np.int64)
    ip_map["upper_bound_ip_address"] = pd.to_numeric(ip_map["upper_bound_ip_address"], errors="coerce").astype(np.int64)

    fraud = fraud.sort_values("ip_int")
    ip_map = ip_map.sort_values("lower_bound_ip_address")

    merged = pd.merge_asof(
        fraud,
        ip_map,
        left_on="ip_int",
        right_on="lower_bound_ip_address",
        direction="backward",
    )

    in_range = merged["ip_int"].between(
        merged["lower_bound_ip_address"],
        merged["upper_bound_ip_address"],
        inclusive="both",
    )
    merged.loc[~in_range, "country"] = "Unknown"
    return merged

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()

    # Timestamps
    df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce")
    df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce")

    # Basic missing handling (adjust after inspecting in notebook)
    cat_cols = ["source", "browser", "sex", "country"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown")

    for c in ["age", "purchase_value"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())

    # Time features
    df["time_since_signup_sec"] = (df["purchase_time"] - df["signup_time"]).dt.total_seconds()
    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Flag impossible timestamps
    df["negative_time_since_signup"] = (df["time_since_signup_sec"] < 0).astype(int)
    df.loc[df["time_since_signup_sec"] < 0, "time_since_signup_sec"] = np.nan
    df["time_since_signup_sec"] = df["time_since_signup_sec"].fillna(df["time_since_signup_sec"].median())

    # Velocity features: transaction counts per user in rolling windows
    df = df.sort_values(["user_id", "purchase_time"]).reset_index(drop=True)

    def rolling_count(group: pd.DataFrame, window: str) -> pd.Series:
        g = group.set_index("purchase_time")
        return g["device_id"].rolling(window).count().reset_index(drop=True)

    for window in ["1h", "24h"]:
        df[f"user_txn_count_{window}"] = (
            df.groupby("user_id", group_keys=False)
              .apply(lambda g: rolling_count(g, window))
              .astype(float)
              .values
        )

    df["user_txn_index"] = df.groupby("user_id").cumcount()

    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--out_path", default="data/processed/fraud_ecommerce.parquet")
    args = ap.parse_args()

    fraud, ip_map = load_inputs(args.raw_dir)
    merged = merge_ip_to_country(fraud, ip_map)
    out = clean_and_engineer(merged)

    out.to_parquet(args.out_path, index=False)
    print(f"Saved: {args.out_path}")
    print("Class balance (fraud=1):", out["class"].mean() if "class" in out.columns else "<missing class>")
    print("Rows:", len(out), "Cols:", out.shape[1])

if __name__ == "__main__":
    main()
