from __future__ import annotations
import math
from phase2_offline_edge_companion.offline_evasion_nav import (
    haversine_distance_m,
    initial_bearing_deg,
    destination_point,
    compute_immediate_escape_vector,
    compute_safe_destination_detour,
    evaluate_hiker_hazard_state,
    evaluate_dynamic_surroundings_at_clock,
)


def test_haversine_and_bearing_accuracy() -> None:
    p1 = (30.0, 78.0)
    p2 = (30.0, 78.01)
    d = haversine_distance_m(p1[0], p1[1], p2[0], p2[1])
    assert 900.0 < d < 1050.0

    b = initial_bearing_deg(p1[0], p1[1], 30.01, 78.0)
    assert abs(b - 0.0) < 0.5 or abs(b - 360.0) < 0.5


def test_immediate_escape_vector_perpendicularity() -> None:
    node = {
        "lat": 31.000,
        "lon": 78.950,
        "alt": 3000.0,
        "danger_radius_m": 80.0,
        "slide_azimuth_deg": 180.0,
    }
    hiker_lat, hiker_lon = 31.0001, 78.9501
    esc = compute_immediate_escape_vector(hiker_lat, hiker_lon, 3000.0, node)
    bearing = esc["escape_bearing_deg"]
    diff1 = abs((bearing - (180.0 + 90.0)) % 360.0)
    diff2 = abs((bearing - (180.0 - 90.0)) % 360.0)
    assert diff1 < 0.1 or diff2 < 0.1
    assert esc["escape_distance_m"] > 0.0


def test_safe_destination_detour_clears_cluster() -> None:
    cluster = [
        {"lat": 31.010, "lon": 78.950, "danger_radius_m": 60.0},
        {"lat": 31.012, "lon": 78.952, "danger_radius_m": 80.0},
    ]
    detour = compute_safe_destination_detour(31.005, 78.945, (31.030, 78.970), cluster)
    assert "waypoint_lat" in detour
    assert len(detour["path_coords"]) == 3


def test_hazard_state_transitions(sample_trek_cartridge: dict) -> None:
    node = sample_trek_cartridge["danger_nodes"][0]

    st_breach = evaluate_hiker_hazard_state(node["lat"], node["lon"], 3000.0, sample_trek_cartridge)
    assert st_breach["status"] == "CRITICAL_BREACH"

    pt_buffer = destination_point(node["lat"], node["lon"], 90.0, 300.0)
    st_buf = evaluate_hiker_hazard_state(pt_buffer[0], pt_buffer[1], 3000.0, sample_trek_cartridge)
    assert st_buf["status"] == "WARNING_BUFFER"

    pt_safe = destination_point(node["lat"], node["lon"], 90.0, 1500.0)
    st_safe = evaluate_hiker_hazard_state(pt_safe[0], pt_safe[1], 3000.0, sample_trek_cartridge)
    assert st_safe["status"] == "SAFE"


def test_arm_rtc_clock_dynamic_surroundings_modulation(sample_trek_cartridge: dict) -> None:
    sample_trek_cartridge["hourly_timeline"] = [
        {"h": 0, "t": "2026-11-07 00:00", "p": 0.0, "r": 25.0, "sm": 0.22, "tc": 16.0, "wc": 1, "s": 0.80},
        {"h": 1, "t": "2026-11-07 01:00", "p": 8.5, "r": 95.0, "sm": 0.42, "tc": 14.0, "wc": 65, "s": 1.65},
    ]
    env_dry = evaluate_dynamic_surroundings_at_clock(sample_trek_cartridge, clock_hour_idx=0)
    env_storm = evaluate_dynamic_surroundings_at_clock(sample_trek_cartridge, clock_hour_idx=1)

    r_dry = env_dry["dynamic_nodes"][0]["danger_radius_m"]
    r_storm = env_storm["dynamic_nodes"][0]["danger_radius_m"]
    assert r_storm > r_dry
    assert env_storm["dynamic_nodes"][0]["risk_score"] > env_dry["dynamic_nodes"][0]["risk_score"]


def test_uphill_hiker_not_flagged_in_danger(sample_trek_cartridge: dict) -> None:
    node = sample_trek_cartridge["danger_nodes"][0]
    # Node crown is at 3100m. Hiker is uphill at 3150m, 30m away horizontally
    pt = destination_point(node["lat"], node["lon"], 180.0, 30.0)
    st = evaluate_hiker_hazard_state(pt[0], pt[1], 3150.0, sample_trek_cartridge)
    assert st["status"] == "SAFE"
    assert st["is_uphill_safe"] is True
    assert len(st["uphill_ignored_nodes"]) >= 1
    assert st["immediate_escape"] is None


def test_upslope_behind_crown_not_critical_breach(sample_trek_cartridge: dict) -> None:
    node = sample_trek_cartridge["danger_nodes"][0]
    # Slide azimuth is 180 deg (South). Hiker is 50m North (upslope behind crown), same elevation
    pt_north = destination_point(node["lat"], node["lon"], 0.0, 50.0)
    st = evaluate_hiker_hazard_state(pt_north[0], pt_north[1], 3100.0, sample_trek_cartridge)
    assert st["status"] != "CRITICAL_BREACH"
