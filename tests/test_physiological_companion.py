from __future__ import annotations
import numpy as np
from phase2_offline_edge_companion.physiological_companion import (
    evaluate_imu_spo2_gate,
    compute_w_sqi,
    evaluate_activity_aware_exertion,
    log_to_blackbox_queue,
    read_blackbox_queue,
)


def test_xu_2026_imu_spo2_motion_gate_blocks_movement() -> None:
    moving_imu = np.random.normal(0, 0.4, size=(50, 3))
    gate = evaluate_imu_spo2_gate(moving_imu, prior_stillness_s=8.0)
    assert gate["is_stationary"] is False
    assert gate["stillness_duration_s"] == 0.0
    assert gate["spo2_unlocked"] is False
    assert gate["gate_status"] == "SPO2_GATED_MOTION"


def test_xu_2026_imu_spo2_unlocks_after_12s_stillness() -> None:
    still_imu = np.ones((50, 3)) * 0.577
    gate_early = evaluate_imu_spo2_gate(still_imu, prior_stillness_s=5.0)
    assert gate_early["is_stationary"] is True
    assert gate_early["spo2_unlocked"] is False

    gate_unlocked = evaluate_imu_spo2_gate(still_imu, prior_stillness_s=11.5)
    assert gate_unlocked["stillness_duration_s"] >= 12.0
    assert gate_unlocked["spo2_unlocked"] is True
    assert gate_unlocked["gate_status"] == "SPO2_ACTIVE_VALID"


def test_hu_2026_w_sqi_suppresses_arm_swing_noise() -> None:
    t = np.linspace(0, 2.0, 50)
    clean_ppg = np.sin(2 * np.pi * 1.2 * t)
    still_imu = np.ones(50)
    sqi_clean = compute_w_sqi(clean_ppg, still_imu)
    assert sqi_clean >= 60.0

    noisy_ppg = np.sin(2 * np.pi * 2.0 * t) + np.random.normal(0, 0.5, 50)
    noisy_imu = np.sin(2 * np.pi * 2.0 * t)
    sqi_noisy = compute_w_sqi(noisy_ppg, noisy_imu)
    assert sqi_noisy < sqi_clean


def test_samavati_2026_activity_aware_exertion_states() -> None:
    base = {"resting_hr_bpm": 68.0}

    suppressed = evaluate_activity_aware_exertion(
        observed_hr_bpm=155.0,
        w_sqi_pct=45.0,
        imu_effort=0.3,
        ascent_rate_m_min=10.0,
        altitude_m=3000.0,
        in_hazard_zone=False,
        baseline_params=base,
    )
    assert suppressed["alert_suppressed"] is True
    assert suppressed["classification"] == "SUPPRESSED_LOW_SQI"

    climbing = evaluate_activity_aware_exertion(
        observed_hr_bpm=145.0,
        w_sqi_pct=85.0,
        imu_effort=0.4,
        ascent_rate_m_min=12.0,
        altitude_m=3000.0,
        in_hazard_zone=False,
        baseline_params=base,
    )
    assert climbing["classification"] == "NORMAL EXERTION"
    assert climbing["requires_medical_alert"] is False

    resting_tachy = evaluate_activity_aware_exertion(
        observed_hr_bpm=150.0,
        w_sqi_pct=85.0,
        imu_effort=0.01,
        ascent_rate_m_min=0.0,
        altitude_m=3000.0,
        in_hazard_zone=False,
        baseline_params=base,
    )
    assert resting_tachy["classification"] == "ELEVATED RISK REASSESSMENT"
    assert resting_tachy["requires_medical_alert"] is True


def test_sana_2026_blackbox_jsonl_persistence(tmp_path: object) -> None:
    q = str(tmp_path / "sync.jsonl")
    pkt = {"event": "HAZARD_ALERT", "hr": 145}
    n = log_to_blackbox_queue(q, pkt)
    assert n == 1
    records = read_blackbox_queue(q)
    assert len(records) == 1
    assert records[0]["event"] == "HAZARD_ALERT"
