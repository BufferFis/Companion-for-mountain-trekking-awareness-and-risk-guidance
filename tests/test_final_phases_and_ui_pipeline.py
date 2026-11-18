from __future__ import annotations
import os
import pytest
from phase1_basecamp_compiler.gpx_and_firmware_exporter import (
    parse_gpx_waypoints,
    export_safe_route_gpx,
    export_embedded_c_header,
)
from phase2_offline_edge_companion.edge_runtime import EdgeCompanionRuntime
from phase2_offline_edge_companion.physiological_companion import simulate_sensor_packet


SAMPLE_GPX_XML = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="MountainCompanion">
  <trk>
    <name>Kodaikanal Ridge Trail</name>
    <trkseg>
      <trkpt lat="10.2145" lon="77.4935"><ele>2050.0</ele></trkpt>
      <trkpt lat="10.2000" lon="77.4800"><ele>1920.0</ele></trkpt>
      <trkpt lat="10.1850" lon="77.4620"><ele>1780.0</ele></trkpt>
    </trkseg>
  </trk>
</gpx>
"""


def test_gpx_parse_and_export_roundtrip(sample_trek_cartridge: dict, tmp_path: object) -> None:
    pts = parse_gpx_waypoints(SAMPLE_GPX_XML, max_waypoints=10)
    assert len(pts) == 3
    assert pts[0]["step"] == 1
    assert abs(pts[0]["lat"] - 10.2145) < 1e-4
    assert abs(pts[0]["alt"] - 2050.0) < 1e-1

    out_gpx = str(tmp_path / "route.gpx")
    written_path = export_safe_route_gpx(sample_trek_cartridge, output_path=out_gpx)
    assert os.path.exists(written_path)
    with open(written_path, "r", encoding="utf-8") as f:
        xml_out = f.read()
    assert "<gpx" in xml_out
    assert "<trkpt" in xml_out
    reparsed = parse_gpx_waypoints(xml_out)
    assert len(reparsed) >= 2


def test_export_embedded_c_header_valid_progmem(sample_trek_cartridge: dict, tmp_path: object) -> None:
    out_h = str(tmp_path / "trek_cartridge_firmware.h")
    meta = export_embedded_c_header(sample_trek_cartridge, output_path=out_h)
    assert os.path.exists(out_h)
    assert meta["size_bytes"] > 100
    assert meta["node_count"] == len(sample_trek_cartridge["danger_nodes"])
    with open(out_h, "r", encoding="utf-8") as f:
        h_text = f.read()
    assert "#ifndef TREK_CARTRIDGE_FIRMWARE_H" in h_text
    assert "TrekDangerNode_t" in h_text
    assert "TREK_DANGER_NODES" in h_text


def test_unified_companion_step_all_six_papers_integrated(sample_trek_cartridge: dict, tmp_path: object) -> None:
    import json
    cart_path = str(tmp_path / "cartridge.json")
    with open(cart_path, "w", encoding="utf-8") as f:
        json.dump(sample_trek_cartridge, f)

    q_path = str(tmp_path / "queue.jsonl")
    rt = EdgeCompanionRuntime(cartridge_path=cart_path, queue_path=q_path)

    node = sample_trek_cartridge["danger_nodes"][0]
    pkt = simulate_sensor_packet(timestamp_s=0.0, activity_mode="stationary", altitude_m=3000.0)
    res = rt.run_unified_companion_step(
        hiker_lat=node["lat"],
        hiker_lon=node["lon"],
        hiker_alt=3000.0,
        sensor_packet=pkt,
        clock_hour_idx=0,
        cohort_mode="migrant_acute",
    )
    assert "navigation" in res
    assert "physiology" in res
    assert "motion_filter" in res
    assert "medvse_vitals" in res
    assert "acclimatization" in res
    assert "baro_cloudburst" in res
    assert res["navigation"]["status"] == "CRITICAL_BREACH"
    assert os.path.exists(q_path)


def test_post_trek_burst_sync_and_audit_report(sample_trek_cartridge: dict, tmp_path: object) -> None:
    import json
    cart_path = str(tmp_path / "cartridge.json")
    with open(cart_path, "w", encoding="utf-8") as f:
        json.dump(sample_trek_cartridge, f)

    q_path = str(tmp_path / "queue.jsonl")
    rt = EdgeCompanionRuntime(cartridge_path=cart_path, queue_path=q_path)
    node = sample_trek_cartridge["danger_nodes"][0]
    pkt = simulate_sensor_packet(timestamp_s=3600.0, activity_mode="hiking", altitude_m=3000.0, inject_hypoxia=True)
    rt.run_unified_companion_step(node["lat"], node["lon"], 3000.0, pkt, clock_hour_idx=0)

    batch = rt.sync_burst(batch_limit=10)
    assert len(batch) >= 1
    acked = rt.ack_sync([r["seq_id"] for r in batch])
    assert acked >= 1

    audit = rt.generate_post_trek_audit_report()
    assert audit["total_events"] >= 1
    assert audit["synced_count"] >= 1
    assert "report_markdown" in audit
