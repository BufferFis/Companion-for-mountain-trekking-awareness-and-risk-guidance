from __future__ import annotations
import numpy as np
from phase1_basecamp_compiler.terrain_ingestor import (
    build_corridor_grid,
    compute_topographic_derivatives,
    generate_synthetic_himalayan_dem,
    ingest_trek_corridor,
    search_place_coordinates,
)


def test_build_corridor_grid_dimensions() -> None:
    lat_g, lon_g, x_g, y_g = build_corridor_grid((30.0, 78.0), (30.05, 78.05), buffer_km=1.0, resolution_m=100.0)
    assert lat_g.shape == lon_g.shape == x_g.shape == y_g.shape
    assert lat_g.shape[0] > 10
    assert lat_g.shape[1] > 10


def test_compute_topographic_derivatives_planar_slope() -> None:
    H, W = 30, 30
    dy, dx = 100.0, 100.0
    y_coords = np.arange(H) * dy
    elev = np.tile(y_coords[:, None], (1, W))
    slope, aspect, curv = compute_topographic_derivatives(elev, dx, dy)
    assert np.allclose(slope, 45.0, atol=0.1)
    assert np.allclose(aspect, 180.0, atol=0.1)
    assert np.allclose(curv, 0.0, atol=1e-4)


def test_synthetic_himalayan_dem_fallback() -> None:
    lat_g, lon_g, _, _ = build_corridor_grid((30.9, 78.9), (31.0, 79.0), buffer_km=1.0, resolution_m=100.0)
    elev, rain = generate_synthetic_himalayan_dem(lat_g, lon_g, (30.9, 78.9), (31.0, 79.0))
    assert np.mean(elev) > 2000.0
    assert np.max(rain) > 231.0


def test_ingest_trek_corridor_offline_mode() -> None:
    terrain = ingest_trek_corridor((30.99, 78.93), (31.02, 78.98), force_offline_fallback=True)
    assert terrain.is_fallback is True
    assert terrain.elevation_m.shape == terrain.slope_deg.shape


def test_search_place_coordinates_presets_and_fallback() -> None:
    res_koda = search_place_coordinates("Kodaikanal")
    assert len(res_koda) > 0
    assert 9.0 <= res_koda[0]["lat"] <= 11.5
    assert 76.5 <= res_koda[0]["lon"] <= 78.5

    res_ooty = search_place_coordinates("Ooty")
    assert len(res_ooty) > 0
    assert 11.0 <= res_ooty[0]["lat"] <= 12.0


def test_adaptive_grid_prevents_rate_limit_on_long_corridor() -> None:
    terrain = ingest_trek_corridor(
        start_coords=(11.4102, 76.6950),
        dest_coords=(10.2381, 77.4892),
        buffer_km=2.0,
        force_offline_fallback=False,
        target_grid_size=40,
        sample_grid_dim=10,
    )
    assert terrain.is_fallback is False
    assert terrain.elevation_m.shape == (40, 40)
    assert terrain.slope_deg.shape == (40, 40)
    assert terrain.corridor_length_km > 100.0
    assert len(terrain.hourly_timeline) == 168
