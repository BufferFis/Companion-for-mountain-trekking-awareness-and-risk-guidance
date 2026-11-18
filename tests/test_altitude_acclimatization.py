from __future__ import annotations
from phase2_offline_edge_companion.altitude_acclimatization import (
    compute_altitude_spo2_uncertainty,
    evaluate_longitudinal_spo2_trend,
)


def test_xu_2026_migrant_vs_resident_variance_ratio_at_4014m() -> None:
    migrant = compute_altitude_spo2_uncertainty(4014.0, cohort_mode="migrant_acute")
    resident = compute_altitude_spo2_uncertainty(4014.0, cohort_mode="resident_adapted")

    assert migrant["iso_80601_compliant"] is True
    assert resident["iso_80601_compliant"] is True
    ratio = migrant["error_variance_pct2"] / max(resident["error_variance_pct2"], 1e-6)
    assert ratio >= 10.0


def test_xu_2026_longitudinal_spo2_decline_detection() -> None:
    history = [
        {"alt_m": 500.0, "spo2_pct": 97.5},
        {"alt_m": 2560.0, "spo2_pct": 93.4},
    ]
    normal_ascent = evaluate_longitudinal_spo2_trend(history, current_spo2_pct=92.0, current_alt_m=3460.0)
    assert normal_ascent["trend_status"] in ("EXPECTED_ALTITUDE_DECLINE", "STABLE_ACCLIMATIZED")
    assert normal_ascent["requires_descent_warn"] is False

    acute_hypoxia = evaluate_longitudinal_spo2_trend(history, current_spo2_pct=78.5, current_alt_m=3460.0)
    assert acute_hypoxia["trend_status"] == "ACUTE_HYPOXIA_ALERT"
    assert acute_hypoxia["requires_descent_warn"] is True
