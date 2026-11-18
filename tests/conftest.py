from __future__ import annotations
import pytest
from phase1_basecamp_compiler.terrain_ingestor import (
    build_corridor_grid,
    compute_topographic_derivatives,
    generate_synthetic_himalayan_dem,
    ingest_trek_corridor,
)


@pytest.fixture
def sample_trek_coords() -> tuple[tuple[float, float], tuple[float, float]]:
    return (30.9960, 78.9390), (31.0280, 78.9850)


@pytest.fixture
def synthetic_terrain_grid(sample_trek_coords: tuple[tuple[float, float], tuple[float, float]]) -> object:
    start, dest = sample_trek_coords
    return ingest_trek_corridor(start, dest, buffer_km=1.0, resolution_m=120.0, force_offline_fallback=True)


@pytest.fixture
def sample_trek_cartridge(tmp_path: object) -> dict:
    from phase1_basecamp_compiler.runout_compiler import export_trek_cartridge
    p = str(tmp_path / "trek_cartridge.json")
    nodes = [
        {
            "id": 1,
            "lat": 31.0100,
            "lon": 78.9500,
            "alt": 3100.0,
            "danger_radius_m": 90.0,
            "buffer_radius_m": 500.0,
            "zone_type": "initiation",
            "slide_azimuth_deg": 180.0,
            "risk_score": 0.88,
        }
    ]
    waypoints = [
        {"step": 1, "lat": 30.9960, "lon": 78.9390, "alt": 2600.0},
        {"step": 2, "lat": 31.0280, "lon": 78.9850, "alt": 3400.0},
    ]
    base = {"resting_hr_bpm": 68.0, "max_hr_bpm": 180.0}
    export_trek_cartridge(p, (30.9960, 78.9390), (31.0280, 78.9850), nodes, waypoints, base)
    with open(p, "r", encoding="utf-8") as f:
        import json
        return json.load(f)
