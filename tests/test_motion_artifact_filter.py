from __future__ import annotations
import numpy as np
from phase2_offline_edge_companion.motion_artifact_filter import (
    compute_imu_hscr,
    classify_motion_periodicity,
    cancel_motion_artifact_cross_film,
)


def test_hu_2026_hscr_distinguishes_periodic_from_scrambling() -> None:
    rng = np.random.default_rng(42)
    t = np.linspace(0.0, 8.0, 200)
    # Periodic walking/running cadence (1.8 Hz + 3.6 Hz harmonic)
    periodic_imu = 1.0 + 0.8 * np.sin(2 * np.pi * 1.8 * t) + 0.3 * np.sin(2 * np.pi * 3.6 * t)
    # Broadband non-periodic rock scrambling
    scrambling_imu = 1.0 + rng.normal(0.0, 0.9, size=200)

    res_p = compute_imu_hscr(periodic_imu, fs_hz=25.0)
    res_s = compute_imu_hscr(scrambling_imu, fs_hz=25.0)

    assert res_p["hscr_score"] >= 0.55
    assert res_p["motion_regime"] == "periodic_locomotion"
    assert res_s["hscr_score"] < res_p["hscr_score"]
    assert res_s["motion_regime"] == "non_periodic_scrambling"


def test_classify_motion_periodicity_regimes() -> None:
    assert classify_motion_periodicity(hscr_score=0.10, motion_variance=0.005) == "stationary"
    assert classify_motion_periodicity(hscr_score=0.82, motion_variance=0.25) == "periodic_locomotion"
    assert classify_motion_periodicity(hscr_score=0.28, motion_variance=0.45) == "non_periodic_scrambling"


def test_cancel_motion_artifact_cross_film_reduces_noise() -> None:
    t = np.linspace(0.0, 4.0, 100)
    clean_cardiac = np.sin(2 * np.pi * 1.2 * t)
    motion_interference = 1.5 * np.sin(2 * np.pi * 2.3 * t)
    corrupted_ppg = clean_cardiac + motion_interference
    imu_xyz = np.column_stack([motion_interference, np.zeros(100), np.ones(100)])

    out = cancel_motion_artifact_cross_film(corrupted_ppg, imu_xyz, fs_hz=25.0)
    assert out["cleaned_ppg"].shape == corrupted_ppg.shape
    assert out["suppression_db"] >= 0.0
