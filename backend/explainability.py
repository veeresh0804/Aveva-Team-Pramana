"""
SHAP Explainability Engine (Layer 6)
======================================
Explains predictions using SHAP TreeExplainer on ensemble's first model.
Translates feature importance into actionable recommendations.
"""

import numpy as np
import pandas as pd
import shap


def explain_prediction(ensemble_predictor, X, feature_names, target_index=4):
    """
    SHAP analysis for a specific target (default: Total_Energy_kWh = index 4).
    Returns top-5 feature importances.
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    else:
        X_arr = np.array(X).reshape(1, -1) if np.ndim(X) == 1 else np.array(X)

    # Use the first model's estimator for the target
    mean_model = ensemble_predictor.models[0]
    target_estimator = mean_model.estimators_[target_index]

    explainer = shap.TreeExplainer(target_estimator)
    shap_values = explainer.shap_values(X_arr)

    # Mean absolute SHAP across samples
    if np.ndim(shap_values) == 1:
        abs_shap = np.abs(shap_values)
    else:
        abs_shap = np.abs(shap_values).mean(axis=0)

    df_importance = pd.DataFrame({
        "feature": feature_names,
        "shap_value": abs_shap,
    }).sort_values("shap_value", ascending=False).head(10)

    return df_importance


def generate_recommendations(shap_analysis):
    """Translate SHAP feature importance into human-readable recommendations."""
    recommendations = []

    feature_advice = {
        "Mean_Motor_Speed": {
            "parameter": "Motor Speed",
            "action": "Reduce by 2-3% from current setpoint",
            "expected_saving": "~8 kWh per batch",
            "category": "energy",
        },
        "Std_Motor_Speed": {
            "parameter": "Motor Speed Stability",
            "action": "Check motor controller for speed fluctuations",
            "expected_saving": "Improved process consistency",
            "category": "reliability",
        },
        "Mean_Power": {
            "parameter": "Baseline Power Draw",
            "action": "Schedule preventive maintenance if elevated",
            "expected_saving": "~5-10 kWh per batch",
            "category": "energy",
        },
        "Max_Vibration": {
            "parameter": "Peak Vibration",
            "action": "Inspect bearings and alignment",
            "expected_saving": "Prevent unplanned downtime",
            "category": "reliability",
        },
        "RMS_Vibration": {
            "parameter": "Overall Vibration Level",
            "action": "Schedule vibration analysis if trending upward",
            "expected_saving": "Asset life extension",
            "category": "reliability",
        },
        "Mean_Temperature": {
            "parameter": "Process Temperature",
            "action": "Optimize to 75±3°C for best trade-off",
            "expected_saving": "~3-5 kWh per batch",
            "category": "energy",
        },
        "Temperature_Drift": {
            "parameter": "Temperature Stability",
            "action": "Check heating element calibration",
            "expected_saving": "Improved quality consistency",
            "category": "quality",
        },
        "Spectral_Energy": {
            "parameter": "Power Signal Spectral Energy",
            "action": "High spectral energy indicates mechanical issues — inspect actuators",
            "expected_saving": "Early fault detection",
            "category": "reliability",
        },
        "Spectral_Entropy": {
            "parameter": "Power Signal Complexity",
            "action": "Elevated entropy suggests irregular operation — check controls",
            "expected_saving": "Process stability improvement",
            "category": "reliability",
        },
        "Phase1_Processing_Duration": {
            "parameter": "Processing Phase Duration",
            "action": "Optimize to 18-20 minutes for energy-quality balance",
            "expected_saving": "~5 kWh per batch",
            "category": "energy",
        },
        "Mean_Pressure": {
            "parameter": "Operating Pressure",
            "action": "Maintain within 4.0-4.5 Bar optimal range",
            "expected_saving": "~2 kWh per batch + quality improvement",
            "category": "energy",
        },
        "Mean_Flow_Rate": {
            "parameter": "Flow Rate",
            "action": "Optimize coolant/fluid flow to 20-24 LPM",
            "expected_saving": "~3 kWh per batch",
            "category": "energy",
        },
    }

    for _, row in shap_analysis.head(5).iterrows():
        feat = row["feature"]
        if feat in feature_advice:
            rec = feature_advice[feat].copy()
            rec["impact_score"] = round(float(row["shap_value"]), 4)
            recommendations.append(rec)
        else:
            recommendations.append({
                "parameter": feat.replace("_", " ").title(),
                "action": f"Investigate '{feat}' — it significantly impacts predictions",
                "expected_saving": "Further analysis needed",
                "category": "investigation",
                "impact_score": round(float(row["shap_value"]), 4),
            })

    return recommendations


def explain_all_targets(ensemble_predictor, X, feature_names, target_names):
    """Run SHAP for each target and return combined results."""
    results = {}
    for i, target in enumerate(target_names):
        importance = explain_prediction(ensemble_predictor, X, feature_names, target_index=i)
        results[target] = importance.to_dict(orient="records")
    return results
