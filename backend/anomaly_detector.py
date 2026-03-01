"""
Isolation Forest Anomaly Detection (Layer 4)
=============================================
Trained ONLY on golden batches → detects abnormal energy patterns
before quality degradation occurs.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import IsolationForest

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def train_anomaly_detector(df, feature_cols):
    """Train IsolationForest on golden batch features only."""
    golden_df = df[df["is_golden"] == 1]
    print(f"  Training on {len(golden_df)} golden batches")

    X_golden = golden_df[feature_cols].values

    iso_forest = IsolationForest(
        contamination=0.05,
        random_state=42,
        n_estimators=200,
    )
    iso_forest.fit(X_golden)

    return iso_forest


def detect_anomaly(iso_forest, batch_features):
    """Score a batch: 1 = normal, -1 = anomaly."""
    X = np.array(batch_features).reshape(1, -1) if np.ndim(batch_features) == 1 else batch_features
    prediction = iso_forest.predict(X)[0]
    score = iso_forest.decision_function(X)[0]

    status = "NORMAL" if prediction == 1 else "ANOMALY DETECTED"
    recommendation = ""
    if prediction == -1:
        recommendation = "Inspect equipment health and process parameters before proceeding."

    return {
        "status": status,
        "anomaly_score": round(float(score), 4),
        "recommendation": recommendation,
    }


def main():
    print("=" * 60)
    print("  TRAINING ANOMALY DETECTOR")
    print("=" * 60)

    df = pd.read_csv(os.path.join(DATA_DIR, "engineered_batch_dataset.csv"))

    target_cols = ["Hardness", "Friability", "Dissolution_Rate",
                   "Content_Uniformity", "Total_Energy_kWh"]
    non_feature_cols = ["Batch_ID"] + target_cols + [
        "Yield_Pct", "Performance_Pct", "Material_Type", "Batch_Size", "is_golden"
    ]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    iso_forest = train_anomaly_detector(df, feature_cols)

    # Save
    with open(os.path.join(MODEL_DIR, "anomaly_detector.pkl"), "wb") as f:
        pickle.dump(iso_forest, f)

    print(f"\n✓ Anomaly detector saved to {MODEL_DIR}/anomaly_detector.pkl")

    # Demo
    sample = df[feature_cols].iloc[0].values
    result = detect_anomaly(iso_forest, sample)
    print(f"  Sample result (Batch 1): {result}")

    # Stats across all batches
    all_preds = iso_forest.predict(df[feature_cols].values)
    n_anomalies = (all_preds == -1).sum()
    print(f"  Anomalies in full dataset: {n_anomalies}/{len(df)} ({100*n_anomalies/len(df):.1f}%)")


if __name__ == "__main__":
    main()
