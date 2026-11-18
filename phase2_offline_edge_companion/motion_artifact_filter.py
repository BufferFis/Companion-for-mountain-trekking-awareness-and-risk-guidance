from __future__ import annotations
import numpy as np

HSCR_PERIODIC_THRESHOLD: float = 0.55
HSCR_MAX_FREQ_HZ: float = 5.0
HSCR_BANDWIDTH_HZ: float = 0.15
STATIONARY_VAR_THRESHOLD_G2: float = 0.04


def classify_motion_periodicity(
    hscr_score: float,
    motion_variance: float,
    stationary_var_threshold: float = STATIONARY_VAR_THRESHOLD_G2,
) -> str:
    if motion_variance <= stationary_var_threshold:
        return "stationary"
    if hscr_score >= HSCR_PERIODIC_THRESHOLD:
        return "periodic_locomotion"
    return "non_periodic_scrambling"


def compute_imu_hscr(
    imu_accel_mag: np.ndarray,
    fs_hz: float = 25.0,
    delta_f_hz: float = HSCR_BANDWIDTH_HZ,
) -> dict:
    x_raw = np.asarray(imu_accel_mag, dtype=np.float32).flatten()
    motion_variance = float(np.var(x_raw))
    x = x_raw - float(np.mean(x_raw))

    n = len(x)
    if n == 0:
        return {
            "hscr_score": 0.0,
            "dominant_freq_hz": 0.0,
            "motion_regime": "stationary",
        }

    freqs = np.fft.rfftfreq(n, 1.0 / fs_hz)
    power = np.abs(np.fft.rfft(x)) ** 2

    total_mask = (freqs >= 0.1) & (freqs <= HSCR_MAX_FREQ_HZ)
    e_total = float(np.sum(power[total_mask])) + 1e-9

    f0_mask = (freqs >= 0.5) & (freqs <= 3.5)
    if np.any(f0_mask) and np.max(power[f0_mask]) > 0:
        idx0 = np.argmax(np.where(f0_mask, power, 0.0))
        f0 = float(freqs[idx0])
    else:
        f0 = 1.0

    harmonic_mask = np.zeros_like(freqs, dtype=bool)
    for k in (1, 2, 3):
        fk = k * f0
        if fk <= HSCR_MAX_FREQ_HZ + delta_f_hz:
            harmonic_mask |= (freqs >= fk - delta_f_hz) & (freqs <= fk + delta_f_hz)

    e_harmonics = float(np.sum(power[harmonic_mask & total_mask]))
    hscr = float(np.clip(e_harmonics / e_total, 0.0, 1.0))
    regime = classify_motion_periodicity(hscr, motion_variance)

    return {
        "hscr_score": round(hscr, 3),
        "dominant_freq_hz": round(f0, 2),
        "motion_regime": regime,
    }


def cancel_motion_artifact_cross_film(
    ppg_waveform: np.ndarray,
    imu_accel_xyz: np.ndarray,
    fs_hz: float = 25.0,
) -> dict:
    p = np.asarray(ppg_waveform, dtype=np.float32).flatten()
    imu_arr = np.asarray(imu_accel_xyz, dtype=np.float32)
    if imu_arr.ndim == 2:
        imu_mag = np.linalg.norm(imu_arr, axis=1)
    else:
        imu_mag = imu_arr.flatten()

    p_mean = float(np.mean(p))
    p_detrend = p - p_mean
    i_mean = float(np.mean(imu_mag))
    i_detrend = imu_mag - i_mean

    denom = float(np.dot(i_detrend, i_detrend))
    if denom > 1e-6:
        alpha = float(np.dot(p_detrend, i_detrend)) / denom
    else:
        alpha = 0.0

    artifact = alpha * i_detrend
    cleaned = p_mean + (p_detrend - artifact)

    std_p = float(np.std(p_detrend))
    std_i = float(np.std(i_detrend))
    if std_p > 1e-6 and std_i > 1e-6:
        corr_before = abs(float(np.corrcoef(p_detrend, i_detrend)[0, 1]))
    else:
        corr_before = 0.0

    cl_detrend = cleaned - float(np.mean(cleaned))
    std_cl = float(np.std(cl_detrend))
    if std_cl > 1e-6 and std_i > 1e-6:
        corr_after = abs(float(np.corrcoef(cl_detrend, i_detrend)[0, 1]))
    else:
        corr_after = 0.0

    suppression_db = float(max(0.0, 10.0 * np.log10(max(corr_before, 1e-4) / max(corr_after, 1e-4))))

    return {
        "cleaned_ppg": cleaned.astype(np.float32),
        "artifact_estimate": artifact.astype(np.float32),
        "suppression_db": round(suppression_db, 2),
    }
