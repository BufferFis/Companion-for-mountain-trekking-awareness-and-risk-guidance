from __future__ import annotations
import numpy as np
from phase2_offline_edge_companion.ppg_vital_tflite import (
    MAX_MEDVSE_MODEL_BYTES,
    normalize_ppg_window_zscore,
    filter_ppg_dct_bandpass,
    build_and_export_medvse_tflite,
    infer_vitals_from_ppg_window,
)


def test_normalize_ppg_window_zscore_zero_mean_unit_std() -> None:
    raw = np.linspace(10.0, 50.0, 250, dtype=np.float32) + 5.0 * np.sin(np.linspace(0, 20, 250))
    norm = normalize_ppg_window_zscore(raw)
    assert norm.shape == raw.shape
    assert abs(float(np.mean(norm))) < 1e-4
    assert abs(float(np.std(norm)) - 1.0) < 1e-3


def test_filter_ppg_dct_bandpass_removes_dc_and_high_freq() -> None:
    t = np.linspace(0.0, 10.0, 250, dtype=np.float32)
    # 1.2 Hz cardiac (72 BPM) + 9 Hz high-freq noise + DC offset
    sig = 100.0 + np.sin(2 * np.pi * 1.2 * t) + 0.8 * np.sin(2 * np.pi * 9.0 * t)
    norm = normalize_ppg_window_zscore(sig)
    filtered = filter_ppg_dct_bandpass(norm, fs_hz=25.0, band_hz=(0.5, 4.0))
    assert filtered.shape == sig.shape
    assert np.isfinite(filtered).all()


def test_medvse_tflite_export_under_1mb(tmp_path: object) -> None:
    out_file = str(tmp_path / "medvse_vitals.tflite")
    meta = build_and_export_medvse_tflite(out_file)
    assert meta["within_1mb"] is True
    assert 0 < meta["size_bytes"] < MAX_MEDVSE_MODEL_BYTES
    assert meta["param_count"] <= 53_713


def test_infer_vitals_from_ppg_window_bounds() -> None:
    t = np.linspace(0.0, 10.0, 250, dtype=np.float32)
    ppg = np.sin(2 * np.pi * 1.3 * t) + 0.2 * np.sin(2 * np.pi * 0.25 * t)
    res = infer_vitals_from_ppg_window(ppg, fs_hz=25.0)
    assert 40.0 <= res["hr_bpm"] <= 210.0
    assert 70.0 <= res["spo2_pct"] <= 100.0
    assert 8.0 <= res["rr_breaths_min"] <= 40.0
