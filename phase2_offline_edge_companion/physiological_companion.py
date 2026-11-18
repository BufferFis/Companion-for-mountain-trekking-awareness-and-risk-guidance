from __future__ import annotations
import json
import os
import numpy as np

STILLNESS_VARIANCE_THRESHOLD_G2: float = 0.04
STILLNESS_REQUIRED_HOLD_SEC: float = 12.0
W_SQI_MIN_THRESHOLD_PCT: float = 60.0
CADENCE_BAND_HZ: tuple[float, float] = (1.0, 3.0)


def simulate_sensor_packet(
    timestamp_s: float,
    activity_mode: str = "hiking",
    altitude_m: float = 3200.0,
    inject_motion_noise: bool = False,
    inject_hypoxia: bool = False,
    inject_resting_tachycardia: bool = False,
    seed: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    fs = 25.0
    n_samples = 50
    t = np.linspace(timestamp_s, timestamp_s + 2.0, n_samples)

    if activity_mode == "stationary":
        base_hr = 68.0
        accel_var = 0.005
        ascent_rate = 0.0
    elif activity_mode == "steep_climbing":
        base_hr = 152.0
        accel_var = 0.35
        ascent_rate = 14.0
    else:
        base_hr = 118.0
        accel_var = 0.18
        ascent_rate = 4.0

    if inject_resting_tachycardia:
        base_hr = 148.0
        accel_var = 0.008
        activity_mode = "stationary"

    motion_noise_factor = 2.5 if inject_motion_noise else 1.0
    accel_x = rng.normal(0, np.sqrt(accel_var) * motion_noise_factor, n_samples)
    accel_y = rng.normal(0, np.sqrt(accel_var) * motion_noise_factor, n_samples)
    accel_z = rng.normal(1.0, np.sqrt(accel_var) * motion_noise_factor, n_samples)
    imu_accel_xyz = np.column_stack([accel_x, accel_y, accel_z])
    imu_accel_mag = np.linalg.norm(imu_accel_xyz, axis=1)

    hr_hz = base_hr / 60.0
    clean_ppg = np.sin(2 * np.pi * hr_hz * t) + 0.3 * np.sin(4 * np.pi * hr_hz * t)
    if inject_motion_noise:
        cadence_hz = rng.uniform(1.8, 2.5)
        clean_ppg += 2.0 * np.sin(2 * np.pi * cadence_hz * t) + rng.normal(0, 0.4, n_samples)
    else:
        clean_ppg += rng.normal(0, 0.05, n_samples)

    if inject_hypoxia:
        spo2 = round(rng.uniform(76.0, 82.0), 1)
    else:
        alt_drop = max(0.0, (altitude_m - 2000.0) / 1000.0 * 2.0)
        spo2 = round(rng.uniform(94.0 - alt_drop, 97.0 - alt_drop), 1)

    pressure_hpa = round(1013.25 * (1.0 - 2.25577e-5 * altitude_m) ** 5.25588, 1)

    return {
        "timestamp_s": round(timestamp_s, 2),
        "activity_mode": activity_mode,
        "max30102": {
            "ppg_waveform": clean_ppg.tolist(),
            "heart_rate_bpm": round(base_hr + rng.uniform(-2.0, 2.0), 1),
            "spo2_pct": spo2,
        },
        "mpu6050": {
            "accel_xyz": imu_accel_xyz.tolist(),
            "accel_mag": imu_accel_mag.tolist(),
            "motion_variance": round(float(np.var(imu_accel_mag)), 4),
        },
        "bmp280": {
            "pressure_hpa": pressure_hpa,
            "altitude_m": round(altitude_m, 1),
            "ascent_rate_m_min": round(ascent_rate, 1),
        },
    }


def evaluate_imu_spo2_gate(
    imu_accel_xyz: np.ndarray,
    prior_stillness_s: float,
    dt_s: float = 1.0,
    variance_threshold: float = STILLNESS_VARIANCE_THRESHOLD_G2,
    required_hold_s: float = STILLNESS_REQUIRED_HOLD_SEC,
) -> dict:
    if len(imu_accel_xyz.shape) == 2:
        mag = np.linalg.norm(imu_accel_xyz, axis=1)
    else:
        mag = imu_accel_xyz

    variance = float(np.var(mag))
    is_stationary = variance <= variance_threshold

    if is_stationary:
        accum_stillness = prior_stillness_s + dt_s
    else:
        accum_stillness = 0.0

    spo2_unlocked = accum_stillness >= required_hold_s
    status = "SPO2_ACTIVE_VALID" if spo2_unlocked else "SPO2_GATED_MOTION"

    return {
        "motion_variance": round(variance, 4),
        "is_stationary": is_stationary,
        "stillness_duration_s": round(accum_stillness, 1),
        "spo2_unlocked": spo2_unlocked,
        "gate_status": status,
    }


def compute_w_sqi(
    ppg_waveform: np.ndarray,
    imu_accel_mag: np.ndarray,
    fs_hz: float = 25.0,
) -> float:
    n = len(ppg_waveform)
    if n < 8:
        return 50.0

    ppg_detrend = ppg_waveform - np.mean(ppg_waveform)
    imu_detrend = imu_accel_mag - np.mean(imu_accel_mag)

    fft_ppg = np.abs(np.fft.rfft(ppg_detrend))
    fft_imu = np.abs(np.fft.rfft(imu_detrend))
    freqs = np.fft.rfftfreq(n, 1.0 / fs_hz)

    band_mask = (freqs >= 0.8) & (freqs <= 3.5)
    if not np.any(band_mask) or np.sum(fft_ppg[band_mask]) < 1e-4:
        return 0.0

    ppg_energy_in_band = np.sum(fft_ppg[band_mask])
    peak_ppg = np.max(fft_ppg[band_mask])
    spectral_concentration = peak_ppg / max(ppg_energy_in_band, 1e-4)

    cadence_mask = (freqs >= CADENCE_BAND_HZ[0]) & (freqs <= CADENCE_BAND_HZ[1])
    imu_motion_energy = np.sum(fft_imu[cadence_mask]) if np.any(cadence_mask) else 0.0
    ppg_motion_energy = np.sum(fft_ppg[cadence_mask]) if np.any(cadence_mask) else 0.0

    overlap = (imu_motion_energy * ppg_motion_energy) / (ppg_energy_in_band * (imu_motion_energy + 1e-3))
    raw_score = (spectral_concentration * 150.0) - (overlap * 45.0)
    return round(float(np.clip(raw_score, 0.0, 100.0)), 1)


def build_or_load_exertion_tflite(model_path: str) -> object:
    return None


def evaluate_activity_aware_exertion(
    observed_hr_bpm: float,
    w_sqi_pct: float,
    imu_effort: float,
    ascent_rate_m_min: float,
    altitude_m: float,
    in_hazard_zone: bool,
    baseline_params: dict,
    tflite_model_path: str | None = None,
) -> dict:
    resting_hr = baseline_params.get("resting_hr_bpm", 70.0)
    alt_factor = max(0.0, (altitude_m - 2000.0) / 1000.0 * 6.0)
    climb_factor = max(0.0, ascent_rate_m_min * 4.5)
    motion_factor = imu_effort * 60.0

    expected_hr = resting_hr + alt_factor + climb_factor + motion_factor
    hr_delta = observed_hr_bpm - expected_hr

    if w_sqi_pct < W_SQI_MIN_THRESHOLD_PCT:
        return {
            "expected_hr_bpm": round(expected_hr, 1),
            "hr_delta_bpm": round(hr_delta, 1),
            "alert_suppressed": True,
            "classification": "SUPPRESSED_LOW_SQI",
            "requires_medical_alert": False,
        }

    is_climbing = ascent_rate_m_min > 5.0 or imu_effort > 0.25
    is_tachycardic = observed_hr_bpm > 140.0 or hr_delta > 25.0

    if is_tachycardic and (not is_climbing or in_hazard_zone):
        cls_state = "ELEVATED RISK REASSESSMENT"
        med_alert = True
    elif is_climbing:
        cls_state = "NORMAL EXERTION"
        med_alert = False
    else:
        cls_state = "NOMINAL"
        med_alert = False

    return {
        "expected_hr_bpm": round(expected_hr, 1),
        "hr_delta_bpm": round(hr_delta, 1),
        "alert_suppressed": False,
        "classification": cls_state,
        "requires_medical_alert": med_alert,
    }


def log_to_blackbox_queue(
    queue_path: str,
    packet: dict,
) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(queue_path)), exist_ok=True)
    with open(queue_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(packet) + "\n")
    with open(queue_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def read_blackbox_queue(queue_path: str) -> list[dict]:
    if not os.path.exists(queue_path):
        return []
    records = []
    with open(queue_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def process_physiological_step(
    sensor_packet: dict,
    prior_stillness_s: float,
    in_hazard_zone: bool,
    baseline_params: dict,
    queue_path: str,
    tflite_model_path: str | None = None,
) -> dict:
    imu_accel = np.array(sensor_packet["mpu6050"]["accel_xyz"], dtype=np.float32)
    ppg_wave = np.array(sensor_packet["max30102"]["ppg_waveform"], dtype=np.float32)
    imu_mag = np.array(sensor_packet["mpu6050"]["accel_mag"], dtype=np.float32)

    gate_info = evaluate_imu_spo2_gate(imu_accel, prior_stillness_s)
    w_sqi = compute_w_sqi(ppg_wave, imu_mag)

    imu_effort = float(sensor_packet["mpu6050"]["motion_variance"])
    ascent_rate = float(sensor_packet["bmp280"]["ascent_rate_m_min"])
    alt_m = float(sensor_packet["bmp280"]["altitude_m"])
    hr_bpm = float(sensor_packet["max30102"]["heart_rate_bpm"])

    exertion_info = evaluate_activity_aware_exertion(
        observed_hr_bpm=hr_bpm,
        w_sqi_pct=w_sqi,
        imu_effort=imu_effort,
        ascent_rate_m_min=ascent_rate,
        altitude_m=alt_m,
        in_hazard_zone=in_hazard_zone,
        baseline_params=baseline_params,
        tflite_model_path=tflite_model_path,
    )

    combined = {
        "timestamp_s": sensor_packet["timestamp_s"],
        "spo2_gate": gate_info,
        "w_sqi_pct": w_sqi,
        "exertion": exertion_info,
        "spo2_display": sensor_packet["max30102"]["spo2_pct"] if gate_info["spo2_unlocked"] else None,
        "hr_display": hr_bpm,
    }

    if exertion_info["requires_medical_alert"] or in_hazard_zone:
        log_to_blackbox_queue(queue_path, combined)

    return combined
