from __future__ import annotations
from phase2_offline_edge_companion.blackbox_sync_manager import (
    detect_barometric_cloudburst_anomaly,
    enqueue_prioritized_telemetry,
    prepare_burst_sync_batch,
    acknowledge_synced_packets,
)


def test_sana_2026_barometric_cloudburst_compensation() -> None:
    # Stationary hiker at 2800m experiencing rapid -4.0 hPa/hr storm pressure drop
    p_hist = [(0.0, 720.0), (1800.0, 718.0), (3600.0, 716.0)]
    alt_hist = [(0.0, 2800.0), (1800.0, 2800.0), (3600.0, 2800.0)]
    storm = detect_barometric_cloudburst_anomaly(p_hist, alt_hist)
    assert storm["cloudburst_detected"] is True
    assert storm["baro_status"] == "CLOUDBURST_BARO_DROP"
    assert storm["local_hazard_multiplier"] > 1.0

    # Hiker climbing 50m (pressure drop due to elevation gain only, weather stable)
    p_climb = [(0.0, 720.0), (3600.0, 714.5)]
    alt_climb = [(0.0, 2800.0), (3600.0, 2850.0)]
    climb_res = detect_barometric_cloudburst_anomaly(p_climb, alt_climb)
    assert climb_res["cloudburst_detected"] is False
    assert climb_res["baro_status"] == "BARO_STABLE"


def test_sana_2026_prioritized_burst_sync_and_ack(tmp_path: object) -> None:
    q_path = str(tmp_path / "priority_sync.jsonl")
    enqueue_prioritized_telemetry(q_path, {"msg": "routine"}, priority_tier="P3_ROUTINE_VITALS")
    enqueue_prioritized_telemetry(q_path, {"msg": "breach"}, priority_tier="P1_HAZARD_BREACH")
    enqueue_prioritized_telemetry(q_path, {"msg": "sos"}, priority_tier="P0_CRITICAL_SOS")

    batch = prepare_burst_sync_batch(q_path, batch_limit=2)
    assert len(batch) == 2
    assert batch[0]["priority_tier"] == "P0_CRITICAL_SOS"
    assert batch[1]["priority_tier"] == "P1_HAZARD_BREACH"

    acked = acknowledge_synced_packets(q_path, [batch[0]["seq_id"]])
    assert acked == 1
    remaining = prepare_burst_sync_batch(q_path, batch_limit=10)
    assert len(remaining) == 2
