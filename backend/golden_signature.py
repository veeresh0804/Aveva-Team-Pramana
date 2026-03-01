"""
Golden Signature Intelligence (Layer 3)
=========================================
Identifies the "fingerprint" of best-performing historical batches.
Top 10% batches by composite score form the golden centroid. Runtime
batches are compared via Euclidean distance to detect process drift.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

TARGET_COLS = ["Hardness", "Friability", "Dissolution_Rate",
               "Content_Uniformity", "Total_Energy_kWh"]

# Weights for composite score (higher = more important)
WEIGHTS = {
    "Hardness": 0.25,           # higher is better
    "Dissolution_Rate": 0.25,   # higher is better
    "Content_Uniformity": 0.15, # higher is better
    "Friability": -0.15,        # lower is better (negative weight)
    "Total_Energy_kWh": -0.20,  # lower is better (negative weight)
}


def create_golden_signature(df, feature_cols):
    """Compute golden signature centroid from top 10% performing batches."""

    scaler = MinMaxScaler()
    outputs = df[TARGET_COLS].copy()
    normalized = pd.DataFrame(
        scaler.fit_transform(outputs),
        columns=TARGET_COLS,
        index=df.index,
    )

    # Composite score (flip sign for "lower is better" metrics)
    scores = (
        WEIGHTS["Hardness"] * normalized["Hardness"]
        + WEIGHTS["Dissolution_Rate"] * normalized["Dissolution_Rate"]
        + WEIGHTS["Content_Uniformity"] * normalized["Content_Uniformity"]
        + WEIGHTS["Friability"] * (1 - normalized["Friability"])
        + WEIGHTS["Total_Energy_kWh"] * (1 - normalized["Total_Energy_kWh"])
    )

    threshold = np.percentile(scores, 90)
    golden_mask = scores >= threshold
    golden_batches = df[golden_mask]

    # Feature centroid
    golden_centroid = golden_batches[feature_cols].mean().values
    golden_std = golden_batches[feature_cols].std().values

    print(f"  Golden batches: {golden_mask.sum()} / {len(df)}")
    print(f"  Score threshold (90th pct): {threshold:.4f}")

    return golden_centroid, golden_std, golden_mask


def compute_deviation(batch_features, golden_centroid, golden_std):
    """Euclidean distance from golden signature (normalized)."""
    # Normalize by std to avoid scale issues
    safe_std = np.where(golden_std > 1e-10, golden_std, 1.0)
    normalized_diff = (batch_features - golden_centroid) / safe_std
    deviation = float(np.sqrt(np.sum(normalized_diff ** 2)))
    return deviation


def main():
    print("=" * 60)
    print("  COMPUTING GOLDEN SIGNATURE")
    print("=" * 60)

    df = pd.read_csv(os.path.join(DATA_DIR, "engineered_batch_dataset.csv"))

    non_feature_cols = ["Batch_ID"] + TARGET_COLS + [
        "Yield_Pct", "Performance_Pct", "Material_Type", "Batch_Size"
    ]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    golden_centroid, golden_std, golden_mask = create_golden_signature(df, feature_cols)

    # Save
    sig_data = {
        "centroid": golden_centroid,
        "std": golden_std,
        "feature_columns": feature_cols,
        "threshold_distance": 15.0,  # empirical threshold
    }
    with open(os.path.join(MODEL_DIR, "golden_signature.pkl"), "wb") as f:
        pickle.dump(sig_data, f)

    # Also save the golden batch mask for anomaly detector
    df["is_golden"] = golden_mask.astype(int)
    df.to_csv(os.path.join(DATA_DIR, "engineered_batch_dataset.csv"), index=False)

    print(f"\n✓ Golden signature saved to {MODEL_DIR}/golden_signature.pkl")

    # Demo: compute deviation for first batch
    sample = df[feature_cols].iloc[0].values
    dev = compute_deviation(sample, golden_centroid, golden_std)
    print(f"  Sample deviation (Batch 1): {dev:.2f}")


if __name__ == "__main__":
    main()
