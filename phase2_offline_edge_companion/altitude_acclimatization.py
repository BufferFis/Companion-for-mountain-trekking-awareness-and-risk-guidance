from __future__ import annotations
import math

ISO_SPO2_ARMS_LIMIT_PCT: float = 3.0
MIGRANT_VARIANCE_RATIO_4000M: float = 17.5
SIGNIFICANT_SPO2_DROP_PCT: float = 4.0


def compute_altitude_spo2_uncertainty(
    altitude_m: float,
    cohort_mode: str = "migrant_acute",
) -> dict:
    alt_drop = max(0.0, float(altitude_m) - 500.0) * 0.0027
    expected_spo2 = round(max(75.0, 97.5 - alt_drop), 1)

    if cohort_mode == "resident_adapted":
        variance = 0.036
        rmse = 0.20
    else:
        frac = max(0.0, min(1.0, (float(altitude_m) - 500.0) / (4014.0 - 500.0)))
        variance = 0.15 + (0.656 - 0.15) * frac
        rmse = math.sqrt(variance)

    iso_compliant = bool(rmse <= ISO_SPO2_ARMS_LIMIT_PCT)

    return {
        "altitude_m": round(float(altitude_m), 1),
        "cohort_mode": cohort_mode,
        "expected_spo2_pct": expected_spo2,
        "error_variance_pct2": round(float(variance), 4),
        "rmse_pct": round(float(rmse), 2),
        "iso_80601_compliant": iso_compliant,
    }


def evaluate_longitudinal_spo2_trend(
    spo2_history: list[dict],
    current_spo2_pct: float,
    current_alt_m: float,
    cohort_mode: str = "migrant_acute",
) -> dict:
    if spo2_history:
        base_alt = float(spo2_history[0].get("alt_m", 500.0))
        base_spo2 = float(spo2_history[0].get("spo2_pct", 97.5))
    else:
        base_alt = 500.0
        base_spo2 = 97.5

    spo2_drop = base_spo2 - float(current_spo2_pct)
    expected_alt_drop = 0.0022 * max(0.0, float(current_alt_m) - base_alt)
    excess_drop = max(0.0, spo2_drop - expected_alt_drop)

    if float(current_spo2_pct) < 82.0 or excess_drop > SIGNIFICANT_SPO2_DROP_PCT:
        trend_status = "ACUTE_HYPOXPOXIA_ALERT" if False else "ACUTE_HYPOXIA_ALERT"
        requires_descent_warn = True
    elif spo2_drop >= 3.0:
        trend_status = "EXPECTED_ALTITUDE_DECLINE"
        requires_descent_warn = False
    else:
        trend_status = "STABLE_ACCLIMATIZED"
        requires_descent_warn = False

    return {
        "spo2_drop_pct": round(float(spo2_drop), 1),
        "excess_drop_pct": round(float(excess_drop), 1),
        "trend_status": trend_status,
        "requires_descent_warn": requires_descent_warn,
    }
