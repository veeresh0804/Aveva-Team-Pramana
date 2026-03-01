"""
Carbon Intelligence & Adaptive Targeting (Layer 7)
=====================================================
Dynamic carbon emission calculation, regulatory compliance checking,
and adaptive target setting based on time-of-day grid carbon intensity.
"""

import numpy as np
from datetime import datetime


# Carbon intensity varies by time of day (grid mix changes)
# Peak hours → higher carbon intensity (more gas/coal)
CARBON_INTENSITY_SCHEDULE = {
    "off_peak":  0.75,   # kg CO2 per kWh  (night: 22:00 - 06:00)
    "shoulder":  0.82,   # kg CO2 per kWh  (morning/evening)
    "peak":      0.90,   # kg CO2 per kWh  (daytime: 10:00 - 17:00)
}

# Regulatory and organizational targets
REGULATORY_LIMIT_KG = 450.0   # kg CO2 per batch (regulatory ceiling)
ORGANIZATIONAL_TARGET_KG = 400.0  # internal sustainability target (stretch goal)


def get_grid_carbon_intensity(hour=None):
    """Get carbon intensity based on time of day."""
    if hour is None:
        hour = datetime.now().hour

    if 22 <= hour or hour < 6:
        period = "off_peak"
    elif 10 <= hour < 17:
        period = "peak"
    else:
        period = "shoulder"

    return CARBON_INTENSITY_SCHEDULE[period], period


def calculate_carbon_emissions(predicted_energy_kwh, hour=None):
    """Calculate carbon emissions and compliance status."""
    carbon_intensity, period = get_grid_carbon_intensity(hour)
    carbon_kg = predicted_energy_kwh * carbon_intensity

    result = {
        "predicted_energy_kwh": round(predicted_energy_kwh, 2),
        "carbon_intensity_kg_per_kwh": carbon_intensity,
        "grid_period": period,
        "carbon_emissions_kg": round(carbon_kg, 2),
        "regulatory_limit_kg": REGULATORY_LIMIT_KG,
        "organizational_target_kg": ORGANIZATIONAL_TARGET_KG,
    }

    # Regulatory compliance
    if carbon_kg > REGULATORY_LIMIT_KG:
        result["regulatory_status"] = "EXCEEDS LIMIT"
        result["regulatory_excess_kg"] = round(carbon_kg - REGULATORY_LIMIT_KG, 2)
        result["recommendation"] = (
            "Critical: Exceeds regulatory limit. Consider lower-energy Pareto solution "
            "or reschedule batch to off-peak hours."
        )
    elif carbon_kg > ORGANIZATIONAL_TARGET_KG:
        result["regulatory_status"] = "COMPLIANT (above internal target)"
        result["regulatory_headroom_kg"] = round(REGULATORY_LIMIT_KG - carbon_kg, 2)
        result["target_excess_kg"] = round(carbon_kg - ORGANIZATIONAL_TARGET_KG, 2)
        result["recommendation"] = (
            "Within regulatory limits but above organizational target. "
            "Consider parameter optimization."
        )
    else:
        result["regulatory_status"] = "FULLY COMPLIANT"
        result["regulatory_headroom_kg"] = round(REGULATORY_LIMIT_KG - carbon_kg, 2)
        result["target_headroom_kg"] = round(ORGANIZATIONAL_TARGET_KG - carbon_kg, 2)
        result["recommendation"] = "On track for sustainability goals."

    return result


def adaptive_target(historical_emissions, reduction_rate=0.02):
    """
    Compute adaptive emission target based on historical performance.
    Reduces target by `reduction_rate` (default 2%) from rolling average.
    """
    if not historical_emissions:
        return REGULATORY_LIMIT_KG

    rolling_avg = np.mean(historical_emissions[-30:])  # last 30 batches
    adaptive = rolling_avg * (1 - reduction_rate)
    # Never exceed regulatory limit
    return min(round(adaptive, 2), REGULATORY_LIMIT_KG)


def batch_carbon_summary(batch_predictions, hour=None):
    """Full carbon intelligence summary for a batch."""
    energy = batch_predictions.get("Total_Energy_kWh", 500)
    carbon_result = calculate_carbon_emissions(energy, hour)

    # Add context
    carbon_result["savings_if_off_peak"] = round(
        energy * (CARBON_INTENSITY_SCHEDULE["peak"] - CARBON_INTENSITY_SCHEDULE["off_peak"]),
        2,
    )
    carbon_result["annual_savings_potential_kg"] = round(
        carbon_result.get("regulatory_headroom_kg", 0) * 365 * 3,  # 3 batches/day
        0,
    )

    return carbon_result
