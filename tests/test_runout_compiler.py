from __future__ import annotations
from phase1_basecamp_compiler.runout_compiler import (
    classify_runout_regime,
    trace_single_debris_flow,
    compress_sparse_danger_nodes,
    export_trek_cartridge,
    MAX_CARTRIDGE_BYTES,
)


def test_classify_runout_regime_boundaries() -> None:
    assert classify_runout_regime(40.0) == "initiation"
    assert classify_runout_regime(30.0) == "transport"
    assert classify_runout_regime(20.0) == "runout_fan"
    assert classify_runout_regime(12.0) == "stopped"


def test_detachment_zone_radius_expansion(synthetic_terrain_grid: object) -> None:
    synthetic_terrain_grid.slope_deg[5, 5] = 42.0
    nodes = trace_single_debris_flow(5, 5, 0.85, synthetic_terrain_grid, base_radius_m=60.0)
    assert len(nodes) > 0
    assert nodes[0]["danger_radius_m"] == 90.0


def test_stopping_at_15_deg_slope(synthetic_terrain_grid: object) -> None:
    synthetic_terrain_grid.slope_deg[:, :] = 12.0
    nodes = trace_single_debris_flow(5, 5, 0.85, synthetic_terrain_grid)
    assert len(nodes) == 0


def test_sparse_cartridge_size_under_25kb(tmp_path: object) -> None:
    p = str(tmp_path / "cartridge.json")
    nodes = [
        {
            "id": i,
            "lat": 31.0 + i * 0.001,
            "lon": 78.9 + i * 0.001,
            "alt": 3000.0,
            "danger_radius_m": 75.0,
            "buffer_radius_m": 500.0,
            "zone_type": "transport",
            "slide_azimuth_deg": 190.0,
            "risk_score": 0.82,
        }
        for i in range(20)
    ]
    meta = export_trek_cartridge(p, (30.9, 78.9), (31.1, 79.1), nodes, [], {})
    assert meta["size_bytes"] < MAX_CARTRIDGE_BYTES


def test_cartridge_json_schema_compliance(sample_trek_cartridge: dict) -> None:
    assert "danger_nodes" in sample_trek_cartridge
    for n in sample_trek_cartridge["danger_nodes"]:
        assert "id" in n
        assert "lat" in n
        assert "lon" in n
        assert "danger_radius_m" in n
        assert n["buffer_radius_m"] == 500.0
        assert n["zone_type"] in ("initiation", "transport", "runout_fan")
