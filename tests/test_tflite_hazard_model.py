from __future__ import annotations
import numpy as np
from phase1_basecamp_compiler.tflite_hazard_model import (
    generate_calibrated_training_data,
    build_and_train_hazard_model,
    export_quantized_tflite,
    verify_model_memory_budget,
    predict_initiation_grid,
    _calc_swain_prob,
    MAX_MODEL_BYTES,
)


def test_swain_2026_calibration_high_risk() -> None:
    elev = np.array([3200.0])
    slope = np.array([42.0])
    aspect = np.array([180.0])
    curv = np.array([0.02])
    rain = np.array([260.0])
    p = _calc_swain_prob(elev, slope, aspect, curv, rain)
    assert p[0] > 0.75


def test_swain_2026_calibration_low_risk() -> None:
    elev = np.array([2200.0])
    slope = np.array([8.0])
    aspect = np.array([90.0])
    curv = np.array([-0.01])
    rain = np.array([40.0])
    p = _calc_swain_prob(elev, slope, aspect, curv, rain)
    assert p[0] < 0.25


def test_tflite_model_size_under_1mb(tmp_path: object) -> None:
    p = str(tmp_path / "test.tflite")
    X, y = generate_calibrated_training_data(n_samples=500)
    m = build_and_train_hazard_model(X, y, epochs=1)
    sz = export_quantized_tflite(m, p)
    assert sz < MAX_MODEL_BYTES
    v = verify_model_memory_budget(p)
    assert v["within_1mb_budget"] is True


def test_predict_initiation_grid_shape_and_bounds(synthetic_terrain_grid: object, tmp_path: object) -> None:
    p = str(tmp_path / "model.tflite")
    X, y = generate_calibrated_training_data(n_samples=300)
    m = build_and_train_hazard_model(X, y, epochs=1)
    export_quantized_tflite(m, p)
    grid = predict_initiation_grid(p, synthetic_terrain_grid)
    assert grid.shape == synthetic_terrain_grid.elevation_m.shape
    assert np.all(grid >= 0.0) and np.all(grid <= 1.0)
