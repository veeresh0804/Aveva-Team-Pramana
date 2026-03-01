"""
FastAPI Backend – AI-Driven Manufacturing Intelligence
=======================================================
Endpoints:
  POST /predict          – Batch prediction with uncertainty + anomaly + deviation
  POST /optimize         – NSGA-II Pareto optimization
  POST /explain          – SHAP feature importance + recommendations
  GET  /golden-signature – Current golden signature stats
  POST /anomaly-check    – Anomaly score for batch features
  POST /carbon           – Carbon emission calculation & compliance
  GET  /health           – Health check
  GET  /dataset-stats    – Dataset overview
"""

import os
import io
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict

# Local modules
from train_models import EnsemblePredictor  # needed for pickle deserialization
from feature_engineering import engineer_features_for_batch, engineer_all
from nsga2_optimizer import nsga2_optimize, select_balanced_solution
from explainability import explain_prediction, generate_recommendations, explain_all_targets
from carbon_engine import (
    calculate_carbon_emissions,
    batch_carbon_summary,
    adaptive_target,
)
from golden_signature import compute_deviation
from anomaly_detector import detect_anomaly

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

app = FastAPI(
    title="AI-Driven Manufacturing Intelligence",
    description="Multi-objective batch optimization with ensemble ML, NSGA-II, and SHAP explainability",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model storage ──────────────────────────────────────────────
ensemble = None
golden_sig = None
iso_forest = None
feature_cols = None
df_dataset = None

TARGET_COLS = ["Hardness", "Friability", "Dissolution_Rate",
               "Content_Uniformity", "Total_Energy_kWh"]


@app.on_event("startup")
async def load_models():
    global ensemble, golden_sig, iso_forest, feature_cols, df_dataset

    import importlib

    class _Unpickler(pickle.Unpickler):
        """Remap __main__ references to actual module names."""
        def find_class(self, module, name):
            if module == "__main__" and name == "EnsemblePredictor":
                mod = importlib.import_module("train_models")
                return getattr(mod, name)
            return super().find_class(module, name)

    with open(os.path.join(MODEL_DIR, "ensemble_models.pkl"), "rb") as f:
        ensemble = _Unpickler(f).load()
    with open(os.path.join(MODEL_DIR, "golden_signature.pkl"), "rb") as f:
        golden_sig = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "anomaly_detector.pkl"), "rb") as f:
        iso_forest = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "feature_columns.pkl"), "rb") as f:
        feature_cols = pickle.load(f)

    dataset_path = os.path.join(DATA_DIR, "engineered_batch_dataset.csv")
    if os.path.exists(dataset_path):
        df_dataset = pd.read_csv(dataset_path)

    print("✓ All models loaded successfully")


# ═══════════════════════════════════════════════════════════════════════
# Request / Response Models
# ═══════════════════════════════════════════════════════════════════════
class OptimizeRequest(BaseModel):
    Motor_Speed: float = 1500
    Temperature: float = 75
    Pressure: float = 4.0
    Flow_Rate: float = 22
    Hold_Time: float = 18
    pop_size: int = 40
    n_generations: int = 25


class CarbonRequest(BaseModel):
    predicted_energy_kwh: float
    hour: Optional[int] = None


class ExplainRequest(BaseModel):
    batch_features: Dict[str, float]


# ═══════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": ensemble is not None,
        "features": len(feature_cols) if feature_cols else 0,
    }


@app.get("/dataset-stats")
async def dataset_stats():
    if df_dataset is None:
        raise HTTPException(404, "Dataset not loaded")

    stats = {}
    for col in TARGET_COLS:
        if col in df_dataset.columns:
            stats[col] = {
                "mean": round(float(df_dataset[col].mean()), 2),
                "std": round(float(df_dataset[col].std()), 2),
                "min": round(float(df_dataset[col].min()), 2),
                "max": round(float(df_dataset[col].max()), 2),
            }

    return {
        "total_batches": len(df_dataset),
        "feature_count": len(feature_cols),
        "target_stats": stats,
        "feature_names": feature_cols[:10],
    }


@app.post("/predict")
async def predict_batch(file: UploadFile = File(...)):
    """Upload time-series CSV for a single batch → predictions + anomaly + deviation."""
    try:
        contents = await file.read()
        df_ts = pd.read_csv(io.BytesIO(contents))

        # Engineer features
        features = engineer_features_for_batch(df_ts)
        X = np.array([[features.get(c, 0.0) for c in feature_cols]])

        # Predict with uncertainty
        pred_mean, pred_std = ensemble.predict_with_uncertainty(X)

        # Anomaly check
        anomaly_result = detect_anomaly(iso_forest, X[0])

        # Golden deviation
        deviation = compute_deviation(
            X[0],
            golden_sig["centroid"],
            golden_sig["std"],
        )
        drift_status = "NORMAL" if deviation < golden_sig["threshold_distance"] else "PROCESS DRIFT DETECTED"

        # Carbon
        carbon_result = calculate_carbon_emissions(pred_mean[0, 4])

        predictions = {}
        for i, target in enumerate(TARGET_COLS):
            predictions[target] = {
                "value": round(float(pred_mean[0, i]), 2),
                "uncertainty": round(float(pred_std[0, i]), 3),
                "display": f"{pred_mean[0, i]:.2f} ± {pred_std[0, i]:.2f}",
            }

        return {
            "predictions": predictions,
            "reliability": {
                "anomaly": anomaly_result,
                "golden_deviation": round(deviation, 2),
                "drift_status": drift_status,
                "threshold": golden_sig["threshold_distance"],
            },
            "carbon": carbon_result,
        }

    except Exception as e:
        raise HTTPException(400, f"Prediction failed: {str(e)}")


@app.post("/predict-params")
async def predict_from_params(params: Dict[str, float]):
    """Predict from feature dictionary (no file upload needed)."""
    try:
        X = np.array([[params.get(c, 0.0) for c in feature_cols]])

        pred_mean, pred_std = ensemble.predict_with_uncertainty(X)

        anomaly_result = detect_anomaly(iso_forest, X[0])
        deviation = compute_deviation(X[0], golden_sig["centroid"], golden_sig["std"])
        drift_status = "NORMAL" if deviation < golden_sig["threshold_distance"] else "PROCESS DRIFT DETECTED"
        carbon_result = calculate_carbon_emissions(pred_mean[0, 4])

        predictions = {}
        for i, target in enumerate(TARGET_COLS):
            predictions[target] = {
                "value": round(float(pred_mean[0, i]), 2),
                "uncertainty": round(float(pred_std[0, i]), 3),
                "display": f"{pred_mean[0, i]:.2f} ± {pred_std[0, i]:.2f}",
            }

        return {
            "predictions": predictions,
            "reliability": {
                "anomaly": anomaly_result,
                "golden_deviation": round(deviation, 2),
                "drift_status": drift_status,
            },
            "carbon": carbon_result,
        }

    except Exception as e:
        raise HTTPException(400, f"Prediction failed: {str(e)}")


@app.post("/optimize")
async def optimize_batch(request: OptimizeRequest):
    """Run NSGA-II multi-objective optimization."""
    try:
        initial_params = {
            "Motor_Speed": request.Motor_Speed,
            "Temperature": request.Temperature,
            "Pressure": request.Pressure,
            "Flow_Rate": request.Flow_Rate,
            "Hold_Time": request.Hold_Time,
        }

        # Base features: median of dataset
        base_features = {}
        if df_dataset is not None:
            for c in feature_cols:
                if c in df_dataset.columns:
                    base_features[c] = float(df_dataset[c].median())

        pareto_solutions, pareto_objectives, hv_history = nsga2_optimize(
            initial_params,
            ensemble,
            feature_cols,
            base_features,
            pop_size=request.pop_size,
            n_generations=request.n_generations,
        )

        recommended = select_balanced_solution(pareto_solutions)

        return {
            "pareto_solutions": pareto_solutions,
            "hypervolume_convergence": [round(h, 2) for h in hv_history],
            "recommended_solution": recommended,
            "total_solutions": len(pareto_solutions),
            "generations": request.n_generations,
            "population_size": request.pop_size,
        }

    except Exception as e:
        raise HTTPException(400, f"Optimization failed: {str(e)}")


@app.post("/explain")
async def explain(request: ExplainRequest):
    """SHAP feature importance and recommendations."""
    try:
        X = pd.DataFrame([request.batch_features])

        # Ensure all feature columns exist
        for c in feature_cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[feature_cols]

        # SHAP for energy target
        energy_importance = explain_prediction(
            ensemble, X, feature_cols, target_index=4
        )
        recommendations = generate_recommendations(energy_importance)

        # SHAP for all targets
        all_targets = explain_all_targets(ensemble, X, feature_cols, TARGET_COLS)

        return {
            "energy_feature_importance": energy_importance.to_dict(orient="records"),
            "all_target_importance": all_targets,
            "recommendations": recommendations,
        }

    except Exception as e:
        raise HTTPException(400, f"Explanation failed: {str(e)}")


@app.get("/golden-signature")
async def get_golden_signature():
    """Current golden signature statistics."""
    if golden_sig is None:
        raise HTTPException(404, "Golden signature not loaded")

    # Top features of the golden centroid
    top_features = []
    for i, col in enumerate(golden_sig["feature_columns"][:15]):
        top_features.append({
            "feature": col,
            "golden_value": round(float(golden_sig["centroid"][i]), 3),
            "golden_std": round(float(golden_sig["std"][i]), 3),
        })

    return {
        "threshold_distance": golden_sig["threshold_distance"],
        "num_features": len(golden_sig["feature_columns"]),
        "top_features": top_features,
    }


@app.post("/anomaly-check")
async def anomaly_check(params: Dict[str, float]):
    """Check anomaly status for given batch features."""
    try:
        X = np.array([[params.get(c, 0.0) for c in feature_cols]])
        result = detect_anomaly(iso_forest, X[0])
        return result
    except Exception as e:
        raise HTTPException(400, f"Anomaly check failed: {str(e)}")


@app.post("/carbon")
async def carbon_check(request: CarbonRequest):
    """Carbon emission calculation and compliance status."""
    result = calculate_carbon_emissions(
        request.predicted_energy_kwh,
        request.hour,
    )
    return result


# ── Run ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
