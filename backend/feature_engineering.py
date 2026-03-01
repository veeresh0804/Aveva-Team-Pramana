"""
Feature Engineering Engine (Layer 1)
=====================================
Transforms raw time-series sensor data into batch-level intelligence features:
  A) Statistical time-domain features
  B) Phase-wise segmented features
  C) FFT frequency-domain features (innovation layer)
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ═══════════════════════════════════════════════════════════════════════
# A) Statistical (Time-Domain) Features
# ═══════════════════════════════════════════════════════════════════════
def extract_statistical_features(batch_df):
    """Extract time-domain statistics for each sensor signal."""
    features = {}

    # Power features
    power = batch_df["Power_Consumption_kW"].values
    features["Total_Energy_kWh_Calc"] = np.sum(power) / 60.0
    features["Mean_Power"] = np.mean(power)
    features["Std_Power"] = np.std(power)
    features["Max_Power"] = np.max(power)
    features["Min_Power"] = np.min(power)
    features["Range_Power"] = np.ptp(power)
    features["Skew_Power"] = float(sp_stats.skew(power))
    features["Kurt_Power"] = float(sp_stats.kurtosis(power))
    features["RMS_Power"] = np.sqrt(np.mean(power ** 2))

    # Vibration features
    vib = batch_df["Vibration_mm_s"].values
    features["Mean_Vibration"] = np.mean(vib)
    features["Std_Vibration"] = np.std(vib)
    features["Max_Vibration"] = np.max(vib)
    features["RMS_Vibration"] = np.sqrt(np.mean(vib ** 2))

    # Temperature features
    temp = batch_df["Temperature_C"].values
    features["Mean_Temperature"] = np.mean(temp)
    features["Std_Temperature"] = np.std(temp)
    features["Temperature_Drift"] = float(temp[-1] - temp[0])
    features["Max_Temperature"] = np.max(temp)

    # Pressure features
    pres = batch_df["Pressure_Bar"].values
    features["Mean_Pressure"] = np.mean(pres)
    features["Pressure_Variance"] = np.var(pres)

    # Motor Speed features
    speed = batch_df["Motor_Speed_RPM"].values
    features["Mean_Motor_Speed"] = np.mean(speed)
    features["Std_Motor_Speed"] = np.std(speed)

    # Flow Rate features
    flow = batch_df["Flow_Rate_LPM"].values
    features["Mean_Flow_Rate"] = np.mean(flow)
    features["Std_Flow_Rate"] = np.std(flow)

    return features


# ═══════════════════════════════════════════════════════════════════════
# B) Phase-wise Segmented Features
# ═══════════════════════════════════════════════════════════════════════
def extract_phase_features(batch_df):
    """Extract per-phase energy, duration, and sensor statistics."""
    features = {}
    phases = ["Heating", "Processing", "Cooling", "Idle"]

    for i, phase in enumerate(phases):
        phase_data = batch_df[batch_df["Process_Phase"] == phase]
        prefix = f"Phase{i}_{phase}"

        if len(phase_data) == 0:
            features[f"{prefix}_Duration"] = 0
            features[f"{prefix}_Energy_kWh"] = 0
            features[f"{prefix}_Mean_Power"] = 0
            features[f"{prefix}_Vibration_Mean"] = 0
            features[f"{prefix}_Power_Variance"] = 0
            continue

        features[f"{prefix}_Duration"] = len(phase_data)
        features[f"{prefix}_Energy_kWh"] = np.sum(phase_data["Power_Consumption_kW"].values) / 60.0
        features[f"{prefix}_Mean_Power"] = np.mean(phase_data["Power_Consumption_kW"].values)
        features[f"{prefix}_Vibration_Mean"] = np.mean(phase_data["Vibration_mm_s"].values)
        features[f"{prefix}_Power_Variance"] = np.var(phase_data["Power_Consumption_kW"].values)

    return features


# ═══════════════════════════════════════════════════════════════════════
# C) FFT Frequency-Domain Features (Innovation Layer)
# ═══════════════════════════════════════════════════════════════════════
def extract_fft_features(power_signal):
    """
    Extract frequency-domain features from the power consumption signal.
    - Normal motor → smooth frequency signature
    - Worn actuator → high-frequency noise spikes
    - Load imbalance → new frequency components
    """
    features = {}

    n = len(power_signal)
    if n < 4:
        for k in ["Spectral_Energy", "Spectral_Entropy",
                   "Dominant_Freq_1", "Dominant_Freq_2", "Dominant_Freq_3",
                   "Dominant_Amp_1", "Dominant_Amp_2", "Dominant_Amp_3"]:
            features[k] = 0.0
        return features

    # Apply FFT
    fft_vals = np.fft.rfft(power_signal - np.mean(power_signal))
    amplitudes = np.abs(fft_vals)
    frequencies = np.fft.rfftfreq(n, d=1.0)  # 1-min sampling → cycles/min

    # Spectral Energy
    features["Spectral_Energy"] = float(np.sum(amplitudes ** 2))

    # Spectral Entropy
    amp_norm = amplitudes / (np.sum(amplitudes) + 1e-10)
    amp_norm = amp_norm[amp_norm > 0]
    features["Spectral_Entropy"] = float(-np.sum(amp_norm * np.log(amp_norm + 1e-10)))

    # Top-3 dominant frequencies
    top_indices = np.argsort(amplitudes[1:])[-3:][::-1] + 1  # skip DC
    for rank, idx in enumerate(top_indices):
        features[f"Dominant_Freq_{rank + 1}"] = float(frequencies[idx])
        features[f"Dominant_Amp_{rank + 1}"] = float(amplitudes[idx])

    # Dominant frequency amplitude (max)
    features["Dominant_Frequency_Amplitude"] = float(np.max(amplitudes[1:]))

    return features


# ═══════════════════════════════════════════════════════════════════════
# Pipeline: batch-level feature extraction
# ═══════════════════════════════════════════════════════════════════════
def engineer_features_for_batch(batch_df):
    """Full feature vector for a single batch."""
    feats = {}
    feats.update(extract_statistical_features(batch_df))
    feats.update(extract_phase_features(batch_df))
    feats.update(extract_fft_features(batch_df["Power_Consumption_kW"].values))
    return feats


def engineer_all(df_timeseries):
    """Process all batches from time-series dataframe."""
    batch_ids = df_timeseries["Batch_ID"].unique()
    records = []
    for bid in batch_ids:
        batch_df = df_timeseries[df_timeseries["Batch_ID"] == bid]
        feats = engineer_features_for_batch(batch_df)
        feats["Batch_ID"] = bid
        records.append(feats)
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("Loading time-series data ...")
    ts_path = os.path.join(DATA_DIR, "Batch_Process_Data.csv")
    prod_path = os.path.join(DATA_DIR, "Batch_Production_Data.csv")

    df_ts = pd.read_csv(ts_path)
    df_prod = pd.read_csv(prod_path)

    print(f"  Time-series rows: {len(df_ts):,}")
    print(f"  Batches: {df_ts['Batch_ID'].nunique()}")

    print("Engineering features ...")
    df_features = engineer_all(df_ts)

    # Merge with production (labels)
    df_final = df_features.merge(df_prod, on="Batch_ID")
    out_path = os.path.join(DATA_DIR, "engineered_batch_dataset.csv")
    df_final.to_csv(out_path, index=False)

    print(f"\n✓ Engineered dataset: {out_path}")
    print(f"  Shape: {df_final.shape}")
    print(f"  Feature columns: {len(df_features.columns) - 1}")
    print(f"  Sample features: {list(df_features.columns[:10])}")


if __name__ == "__main__":
    main()
