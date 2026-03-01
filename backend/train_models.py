"""
11-Model Ensemble Prediction Engine (Layer 2)
===============================================
11 LightGBM models with slight hyperparameter variations wrapped in
MultiOutputRegressor for simultaneous prediction of quality + energy targets.
Provides uncertainty quantification via ensemble disagreement (std across models).
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET_COLS = ["Hardness", "Friability", "Dissolution_Rate",
               "Content_Uniformity", "Total_Energy_kWh"]


class EnsemblePredictor:
    """11-model LightGBM ensemble with uncertainty quantification."""

    def __init__(self, n_models=11):
        self.n_models = n_models
        self.models = []
        self.feature_columns = None

    def train(self, X_train, y_train):
        self.feature_columns = list(X_train.columns)
        self.models = []

        for i in range(self.n_models):
            model = LGBMRegressor(
                random_state=42 + i,
                subsample=0.75 + (i * 0.02),
                learning_rate=0.05 + (i * 0.005),
                n_estimators=150,
                max_depth=8,
                num_leaves=31,
                min_child_samples=10,
                verbose=-1,
            )
            multi_model = MultiOutputRegressor(model)
            multi_model.fit(X_train, y_train)
            self.models.append(multi_model)
            print(f"  Model {i+1}/{self.n_models} trained")

    def predict_with_uncertainty(self, X):
        """Returns (mean_predictions, std_predictions) across ensemble."""
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.array(X).reshape(1, -1) if np.ndim(X) == 1 else np.array(X)

        predictions = np.array([m.predict(X_arr) for m in self.models])
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        return mean_pred, std_pred


def main():
    print("=" * 60)
    print("  TRAINING 11-MODEL ENSEMBLE")
    print("=" * 60)

    # Load engineered dataset
    df = pd.read_csv(os.path.join(DATA_DIR, "engineered_batch_dataset.csv"))
    print(f"\nDataset shape: {df.shape}")

    # Separate features and targets
    non_feature_cols = ["Batch_ID"] + TARGET_COLS + [
        "Yield_Pct", "Performance_Pct", "Material_Type", "Batch_Size"
    ]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    X = df[feature_cols]
    y = df[TARGET_COLS]

    print(f"Feature columns: {len(feature_cols)}")
    print(f"Target columns:  {TARGET_COLS}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # Train ensemble
    ensemble = EnsemblePredictor(n_models=11)
    ensemble.train(X_train, y_train)

    # Evaluate
    y_pred_mean, y_pred_std = ensemble.predict_with_uncertainty(X_test)

    print("\n" + "─" * 50)
    print("  EVALUATION RESULTS")
    print("─" * 50)

    y_test_arr = y_test.values
    for i, target in enumerate(TARGET_COLS):
        mae = np.mean(np.abs(y_test_arr[:, i] - y_pred_mean[:, i]))
        rmse = np.sqrt(np.mean((y_test_arr[:, i] - y_pred_mean[:, i]) ** 2))
        mape = np.mean(np.abs((y_test_arr[:, i] - y_pred_mean[:, i]) / (y_test_arr[:, i] + 1e-10))) * 100
        mean_std = np.mean(y_pred_std[:, i])
        print(f"  {target:25s} | MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.2f}%  ±{mean_std:.3f}")

    overall_mape = np.mean(
        np.abs((y_test_arr - y_pred_mean) / (y_test_arr + 1e-10))
    ) * 100
    print(f"\n  Overall MAPE: {overall_mape:.2f}%  {'✓ PASS' if overall_mape < 10 else '✗ FAIL'}")

    # Save
    with open(os.path.join(MODEL_DIR, "ensemble_models.pkl"), "wb") as f:
        pickle.dump(ensemble, f)
    with open(os.path.join(MODEL_DIR, "feature_columns.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)

    print(f"\n✓ Saved ensemble_models.pkl and feature_columns.pkl to {MODEL_DIR}")


if __name__ == "__main__":
    main()
