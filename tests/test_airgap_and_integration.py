from __future__ import annotations
import socket
from phase2_offline_edge_companion.edge_runtime import (
    verify_airgap_compliance,
    EdgeCompanionRuntime,
)
from phase2_offline_edge_companion.physiological_companion import simulate_sensor_packet


def test_phase2_zero_network_imports_ast_audit() -> None:
    assert verify_airgap_compliance("phase2_offline_edge_companion") is True


def test_phase2_runtime_with_socket_blocked(sample_trek_cartridge: dict, tmp_path: object, monkeypatch: object) -> None:
    p = str(tmp_path / "cartridge.json")
    import json
    with open(p, "w", encoding="utf-8") as f:
        json.dump(sample_trek_cartridge, f)

    def blocked_socket(*args, **kwargs):
        raise RuntimeError("AIR-GAP VIOLATION: Socket call attempted!")

    monkeypatch.setattr(socket, "socket", blocked_socket)

    runtime = EdgeCompanionRuntime(cartridge_path=p, queue_path=str(tmp_path / "queue.jsonl"))
    pkt = simulate_sensor_packet(0.0)
    res = runtime.step(30.9960, 78.9390, 2600.0, pkt)
    assert "navigation" in res
    assert "physiology" in res


def test_end_to_end_basecamp_to_trail_pipeline(tmp_path: object) -> None:
    from phase1_basecamp_compiler.compile_cartridge import run_basecamp_compilation
    out_dir = str(tmp_path / "pipeline_test")
    res = run_basecamp_compilation(output_dir=out_dir, force_offline=True)
    cart_path = res["cartridge_meta"]["cartridge_path"]

    runtime = EdgeCompanionRuntime(cartridge_path=cart_path, queue_path=str(tmp_path / "sync.jsonl"))
    pkt = simulate_sensor_packet(0.0, activity_mode="hiking")
    step_res = runtime.step(30.9960, 78.9390, 2600.0, pkt)

    assert step_res["navigation"]["status"] in ("SAFE", "WARNING_BUFFER", "CRITICAL_BREACH")
    assert "w_sqi_pct" in step_res["physiology"]

    # Extended Phase 2 offline methods test
    import numpy as np
    t = np.linspace(0, 4, 100)
    sig = np.sin(2 * np.pi * 1.5 * t)
    motion = runtime.evaluate_motion_regime(sig)
    assert "hscr_score" in motion

    cleaned = runtime.filter_motion_artifact(sig, np.column_stack([sig, sig, sig]))
    assert "cleaned_ppg" in cleaned

    vitals = runtime.infer_vital_signs(sig)
    assert "hr_bpm" in vitals

    acclim = runtime.check_altitude_acclimatization(3400.0)
    assert acclim["iso_80601_compliant"] is True

    trend = runtime.evaluate_spo2_trend([{"alt_m": 500, "spo2_pct": 98.0}], 93.0, 3400.0)
    assert "trend_status" in trend

    cb = runtime.check_cloudburst_anomaly([(0, 750), (3600, 749)], [(0, 2000), (3600, 2000)])
    assert "cloudburst_detected" in cb

    batch = runtime.sync_burst(batch_limit=5)
    assert isinstance(batch, list)
