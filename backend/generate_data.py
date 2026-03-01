"""
Synthetic Automotive Batch Manufacturing Data Generator
=======================================================
Generates realistic time-series process data and batch-level production data
for 500 automotive manufacturing batches with 4 process phases.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

NUM_BATCHES = 500
PHASES = ["Heating", "Processing", "Cooling", "Idle"]
PHASE_DURATIONS_MIN = {"Heating": (15, 25), "Processing": (30, 50),
                        "Cooling": (20, 30), "Idle": (5, 10)}

# ── Batch-level latent factors ─────────────────────────────────────────
def _batch_factors(batch_id):
    """Generate correlated latent factors that drive both process and quality."""
    rng = np.random.RandomState(batch_id)
    machine_age = rng.uniform(0.0, 1.0)        # 0 = new, 1 = worn
    ambient_temp = rng.uniform(15, 35)          # °C
    material_quality = rng.uniform(0.7, 1.0)    # 1 = perfect raw material
    operator_skill = rng.uniform(0.6, 1.0)
    return machine_age, ambient_temp, material_quality, operator_skill


# ── Time-series generation for one batch ───────────────────────────────
def generate_batch_timeseries(batch_id):
    machine_age, ambient_temp, material_quality, operator_skill = _batch_factors(batch_id)
    rng = np.random.RandomState(batch_id * 7 + 3)

    rows = []
    t = 0
    for phase in PHASES:
        lo, hi = PHASE_DURATIONS_MIN[phase]
        duration = rng.randint(lo, hi + 1)

        for minute in range(duration):
            # --- Power ---
            base_power = {"Heating": 85, "Processing": 110, "Cooling": 40, "Idle": 12}
            power = base_power[phase]
            power += machine_age * 18               # worn machines draw more
            power += (ambient_temp - 25) * 0.4       # hotter ambient → more cooling
            power *= (1 + rng.normal(0, 0.04))        # random noise
            if phase == "Heating":
                power += 15 * np.sin(np.pi * minute / duration)  # ramp-up pattern

            # --- Vibration ---
            base_vib = {"Heating": 2.0, "Processing": 3.5, "Cooling": 1.5, "Idle": 0.5}
            vibration = base_vib[phase]
            vibration += machine_age * 2.5            # worn → more vibration
            vibration *= (1 + rng.normal(0, 0.08))

            # --- Temperature ---
            base_temp = {"Heating": 70 + 20 * (minute / max(duration - 1, 1)),
                         "Processing": 90 + rng.normal(0, 2),
                         "Cooling": 90 - 40 * (minute / max(duration - 1, 1)),
                         "Idle": 30 + rng.normal(0, 1)}
            temperature = base_temp[phase] + ambient_temp * 0.1

            # --- Pressure ---
            base_pres = {"Heating": 3.5, "Processing": 4.5, "Cooling": 3.0, "Idle": 1.0}
            pressure = base_pres[phase] + rng.normal(0, 0.15)

            # --- Motor Speed ---
            base_speed = {"Heating": 1400, "Processing": 1600, "Cooling": 1200, "Idle": 0}
            motor_speed = base_speed[phase]
            motor_speed += operator_skill * 50
            motor_speed *= (1 + rng.normal(0, 0.02))

            # --- Flow Rate ---
            base_flow = {"Heating": 20, "Processing": 25, "Cooling": 18, "Idle": 0}
            flow_rate = base_flow[phase] + rng.normal(0, 1.5)

            rows.append({
                "Batch_ID": f"B{batch_id:04d}",
                "Timestamp_Min": t,
                "Power_Consumption_kW": round(max(power, 0), 2),
                "Vibration_mm_s": round(max(vibration, 0.1), 3),
                "Temperature_C": round(temperature, 2),
                "Pressure_Bar": round(max(pressure, 0.5), 2),
                "Motor_Speed_RPM": round(max(motor_speed, 0), 1),
                "Flow_Rate_LPM": round(max(flow_rate, 0), 2),
                "Process_Phase": phase,
            })
            t += 1

    return rows


# ── Batch production (quality/yield/performance) ──────────────────────
def generate_batch_production(batch_id, ts_rows):
    machine_age, ambient_temp, material_quality, operator_skill = _batch_factors(batch_id)
    rng = np.random.RandomState(batch_id * 13 + 7)

    total_energy = sum(r["Power_Consumption_kW"] / 60.0 for r in ts_rows)  # kWh

    # Quality metrics (correlated with process factors)
    hardness = 7.5 + material_quality * 2.0 - machine_age * 1.0 + operator_skill * 0.5 + rng.normal(0, 0.3)
    friability = 1.8 - material_quality * 0.8 + machine_age * 0.4 + rng.normal(0, 0.15)
    dissolution = 80 + material_quality * 15 - machine_age * 5 + operator_skill * 3 + rng.normal(0, 2)
    content_uniformity = 95 + material_quality * 4 - machine_age * 2 + operator_skill * 1.5 + rng.normal(0, 1)

    # Yield & Performance
    yield_pct = 90 + material_quality * 8 - machine_age * 5 + operator_skill * 2 + rng.normal(0, 1.5)
    performance = 85 + operator_skill * 10 - machine_age * 6 + rng.normal(0, 2)

    return {
        "Batch_ID": f"B{batch_id:04d}",
        "Hardness": round(np.clip(hardness, 5, 12), 2),
        "Friability": round(np.clip(friability, 0.3, 3.0), 2),
        "Dissolution_Rate": round(np.clip(dissolution, 60, 100), 2),
        "Content_Uniformity": round(np.clip(content_uniformity, 85, 100), 2),
        "Yield_Pct": round(np.clip(yield_pct, 75, 100), 2),
        "Performance_Pct": round(np.clip(performance, 70, 100), 2),
        "Total_Energy_kWh": round(total_energy, 2),
        "Material_Type": rng.choice(["TypeA", "TypeB", "TypeC"]),
        "Batch_Size": rng.choice([100, 150, 200, 250]),
    }


# ── Main ───────────────────────────────────────────────────────────────
def main():
    print("Generating synthetic manufacturing data ...")
    all_ts_rows = []
    all_prod_rows = []

    for bid in range(1, NUM_BATCHES + 1):
        ts_rows = generate_batch_timeseries(bid)
        all_ts_rows.extend(ts_rows)
        all_prod_rows.append(generate_batch_production(bid, ts_rows))
        if bid % 100 == 0:
            print(f"  {bid}/{NUM_BATCHES} batches generated")

    df_ts = pd.DataFrame(all_ts_rows)
    df_prod = pd.DataFrame(all_prod_rows)

    ts_path = os.path.join(DATA_DIR, "Batch_Process_Data.csv")
    prod_path = os.path.join(DATA_DIR, "Batch_Production_Data.csv")

    df_ts.to_csv(ts_path, index=False)
    df_prod.to_csv(prod_path, index=False)

    print(f"\n✓ Time-series data : {ts_path}  ({len(df_ts):,} rows)")
    print(f"✓ Production data  : {prod_path}  ({len(df_prod):,} rows)")
    print(f"✓ Batches: {NUM_BATCHES}, Phases: {PHASES}")


if __name__ == "__main__":
    main()
