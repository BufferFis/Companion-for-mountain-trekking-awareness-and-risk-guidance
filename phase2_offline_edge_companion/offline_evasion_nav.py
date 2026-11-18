from __future__ import annotations
import json
import math

EARTH_RADIUS_M: float = 6_371_000.0
CARDINAL_DIRECTIONS: tuple[str, ...] = (
    "N", "NE", "E", "SE", "S", "SW", "W", "NW"
)
UPHILL_CROWN_MARGIN_M: float = 10.0
DOWNSTREAM_CONE_HALF_ANGLE_DEG: float = 75.0
UPSLOPE_CUTOFF_ANGLE_DEG: float = 105.0


def angular_difference_deg(angle_a: float, angle_b: float) -> float:
    return abs((angle_a - angle_b + 180.0) % 360.0 - 180.0)


def build_directional_cone_polygon(node: dict, n_arc_pts: int = 9) -> list[tuple[float, float]]:
    lat = float(node["lat"])
    lon = float(node["lon"])
    az = float(node.get("slide_azimuth_deg", 180.0))
    rad = float(node.get("danger_radius_m", 60.0))
    crown_back = destination_point(lat, lon, (az + 180.0) % 360.0, min(10.0, rad * 0.15))
    pts = [crown_back]
    half_angle = DOWNSTREAM_CONE_HALF_ANGLE_DEG
    for i in range(n_arc_pts):
        frac = -1.0 + 2.0 * (i / max(1, n_arc_pts - 1))
        b = (az + frac * half_angle) % 360.0
        pts.append(destination_point(lat, lon, b, rad))
    pts.append(crown_back)
    return pts


def is_hiker_in_directional_hazard(
    hiker_lat: float,
    hiker_lon: float,
    hiker_alt: float,
    node: dict,
) -> tuple[str, float]:
    node_lat = float(node["lat"])
    node_lon = float(node["lon"])
    node_alt = float(node.get("alt", 0.0))
    danger_r = float(node.get("danger_radius_m", 60.0))
    buffer_r = float(node.get("buffer_radius_m", 500.0))
    slide_az = float(node.get("slide_azimuth_deg", 0.0))

    d_crown = haversine_distance_m(hiker_lat, hiker_lon, node_lat, node_lon)

    # 1. Check downstream traced path vertices if present
    min_path_d = d_crown
    for pt in node.get("path", []):
        pt_lat, pt_lon = float(pt[0]), float(pt[1])
        pt_alt = float(pt[2]) if len(pt) > 2 else node_alt
        if hiker_alt <= pt_alt + UPHILL_CROWN_MARGIN_M:
            d_pt = haversine_distance_m(hiker_lat, hiker_lon, pt_lat, pt_lon)
            if d_pt < min_path_d:
                min_path_d = d_pt

    # 2. Uphill Altitude Gate: landslide initiates below hiker -> gravity flows away downhill
    if hiker_alt > node_alt + UPHILL_CROWN_MARGIN_M:
        return ("SAFE_UPHILL", d_crown)

    # 3. Directional Bearing Check from crown to hiker
    if d_crown > 1.0:
        b_to_hiker = initial_bearing_deg(node_lat, node_lon, hiker_lat, hiker_lon)
        angle_off = angular_difference_deg(b_to_hiker, slide_az)
    else:
        angle_off = 0.0

    # Behind crown upslope (>105 deg off slide direction) and outside immediate 25% scarp rim
    if angle_off > UPSLOPE_CUTOFF_ANGLE_DEG and d_crown > (danger_r * 0.25):
        if d_crown <= buffer_r:
            return ("WARNING_BUFFER", d_crown)
        return ("SAFE_UPHILL", d_crown)

    # Inside downstream cone (<=75 deg) or at immediate crown or along traced path
    in_cone = (angle_off <= DOWNSTREAM_CONE_HALF_ANGLE_DEG) or (d_crown <= danger_r * 0.35)
    if (in_cone and d_crown <= danger_r) or (min_path_d <= danger_r * 0.85):
        return ("CRITICAL_BREACH", min(d_crown, min_path_d))

    if d_crown <= buffer_r or min_path_d <= buffer_r:
        return ("WARNING_BUFFER", min(d_crown, min_path_d))

    return ("SAFE", min(d_crown, min_path_d))


def load_trek_cartridge(cartridge_path: str) -> dict:
    with open(cartridge_path, "r", encoding="utf-8") as f:
        return json.load(f)


def haversine_distance_m(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return EARTH_RADIUS_M * c


def initial_bearing_deg(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


def destination_point(
    lat: float,
    lon: float,
    bearing_deg: float,
    distance_m: float,
) -> tuple[float, float]:
    delta = distance_m / EARTH_RADIUS_M
    theta = math.radians(bearing_deg)
    phi1 = math.radians(lat)
    lambda1 = math.radians(lon)

    phi2 = math.asin(
        math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(theta)
    )
    lambda2 = lambda1 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2),
    )
    return math.degrees(phi2), (math.degrees(lambda2) + 540.0) % 360.0 - 180.0


def bearing_to_cardinal(bearing_deg: float) -> str:
    norm_deg = (bearing_deg % 360.0 + 22.5) % 360.0
    idx = int(norm_deg // 45.0)
    return CARDINAL_DIRECTIONS[idx]


def evaluate_dynamic_surroundings_at_clock(
    cartridge: dict,
    clock_hour_idx: int = 0,
) -> dict:
    """Evaluate dynamic hazard surroundings at On-Board Arm RTC Clock hour (0..167)."""
    timeline = cartridge.get("hourly_timeline", [])
    base_nodes = cartridge.get("danger_nodes", [])

    if timeline:
        idx = max(0, min(clock_hour_idx, len(timeline) - 1))
        wx_step = timeline[idx]
        scale = float(wx_step.get("s", 1.0))
        iso_time = str(wx_step.get("t", f"Hour +{idx}"))
        precip_hr = float(wx_step.get("p", 0.0))
        rolling_rain = float(wx_step.get("r", 0.0))
        soil_moisture = float(wx_step.get("sm", 0.25))
        temp_c = float(wx_step.get("tc", 15.0))
    else:
        idx = clock_hour_idx
        scale = 1.0
        iso_time = f"Trek Hour +{idx:02d}:00"
        precip_hr = 0.0
        rolling_rain = 50.0
        soil_moisture = 0.30
        temp_c = 15.0

    dynamic_nodes: list[dict] = []
    active_count = 0
    watch_count = 0

    for n in base_nodes:
        base_risk = float(n.get("risk_score", 0.78))
        base_rad = float(n.get("danger_radius_m", 60.0))
        base_buf = float(n.get("buffer_radius_m", 500.0))

        # Dynamic risk and radius modulated by forecasted rainfall + soil moisture at clock_hour_idx
        dyn_risk = round(min(0.99, max(0.15, base_risk * (0.75 + 0.25 * scale))), 2)
        dyn_rad = round(base_rad * scale, 1)
        dyn_buf = round(base_buf * min(1.3, max(0.85, scale)), 1)

        if dyn_risk >= 0.75 or precip_hr >= 2.0:
            state = "ACTIVE_CRITICAL"
            active_count += 1
        elif dyn_risk >= 0.60:
            state = "ELEVATED_WATCH"
            watch_count += 1
        else:
            state = "BASELINE_MONITOR"

        node_copy = dict(n)
        node_copy["base_danger_radius_m"] = base_rad
        node_copy["danger_radius_m"] = dyn_rad
        node_copy["buffer_radius_m"] = dyn_buf
        node_copy["risk_score"] = dyn_risk
        node_copy["activity_state"] = state
        dynamic_nodes.append(node_copy)

    return {
        "clock_hour_idx": idx,
        "rtc_iso_time": iso_time,
        "precip_mm_hr": precip_hr,
        "rolling_rain_mm": rolling_rain,
        "soil_moisture": soil_moisture,
        "temp_c": temp_c,
        "hazard_scale": scale,
        "active_critical_count": active_count,
        "elevated_watch_count": watch_count,
        "dynamic_nodes": dynamic_nodes,
    }


def compute_immediate_escape_vector(
    hiker_lat: float,
    hiker_lon: float,
    hiker_alt: float,
    danger_node: dict,
) -> dict:
    slide_az = float(danger_node.get("slide_azimuth_deg", 0.0))
    danger_rad = float(danger_node.get("danger_radius_m", 60.0))
    node_lat = float(danger_node["lat"])
    node_lon = float(danger_node["lon"])

    b_right = (slide_az + 90.0) % 360.0
    b_left = (slide_az - 90.0) % 360.0

    dist_to_center = haversine_distance_m(hiker_lat, hiker_lon, node_lat, node_lon)
    escape_dist = max(danger_rad - dist_to_center + 15.0, 20.0)

    t_right = destination_point(hiker_lat, hiker_lon, b_right, escape_dist)
    t_left = destination_point(hiker_lat, hiker_lon, b_left, escape_dist)

    d_right_node = haversine_distance_m(t_right[0], t_right[1], node_lat, node_lon)
    d_left_node = haversine_distance_m(t_left[0], t_left[1], node_lat, node_lon)

    if d_right_node >= d_left_node:
        chosen_b = b_right
        target_lat, target_lon = t_right
    else:
        chosen_b = b_left
        target_lat, target_lon = t_left

    cardinal = bearing_to_cardinal(chosen_b)
    directive = f"ESCAPE {cardinal} ({chosen_b:.0f} deg) run {escape_dist:.0f}m perpendicular to slide"

    return {
        "escape_bearing_deg": round(chosen_b, 1),
        "cardinal_dir": cardinal,
        "escape_distance_m": round(escape_dist, 1),
        "target_lat": round(target_lat, 5),
        "target_lon": round(target_lon, 5),
        "directive_text": directive,
    }


def compute_safe_destination_detour(
    hiker_lat: float,
    hiker_lon: float,
    dest_coords: tuple[float, float],
    contiguous_cluster: list[dict],
) -> dict:
    if not contiguous_cluster:
        b_direct = initial_bearing_deg(hiker_lat, hiker_lon, dest_coords[0], dest_coords[1])
        d_direct = haversine_distance_m(hiker_lat, hiker_lon, dest_coords[0], dest_coords[1])
        return {
            "waypoint_lat": round(dest_coords[0], 5),
            "waypoint_lon": round(dest_coords[1], 5),
            "detour_bearing_deg": round(b_direct, 1),
            "angular_deviation_deg": 0.0,
            "detour_distance_m": round(d_direct, 1),
            "path_coords": [(round(hiker_lat, 5), round(hiker_lon, 5)), (round(dest_coords[0], 5), round(dest_coords[1], 5))],
        }

    c_lat = sum(n["lat"] for n in contiguous_cluster) / len(contiguous_cluster)
    c_lon = sum(n["lon"] for n in contiguous_cluster) / len(contiguous_cluster)

    max_extent = max(
        haversine_distance_m(c_lat, c_lon, n["lat"], n["lon"]) + float(n["danger_radius_m"])
        for n in contiguous_cluster
    )
    detour_radius = max_extent + 100.0

    b_direct = initial_bearing_deg(hiker_lat, hiker_lon, dest_coords[0], dest_coords[1])
    b_hiker_to_cluster = initial_bearing_deg(hiker_lat, hiker_lon, c_lat, c_lon)

    b_tangent_1 = (b_hiker_to_cluster + 80.0) % 360.0
    b_tangent_2 = (b_hiker_to_cluster - 80.0) % 360.0

    wp1 = destination_point(c_lat, c_lon, b_tangent_1, detour_radius)
    wp2 = destination_point(c_lat, c_lon, b_tangent_2, detour_radius)

    dev1 = abs((initial_bearing_deg(hiker_lat, hiker_lon, wp1[0], wp1[1]) - b_direct + 180.0) % 360.0 - 180.0)
    dev2 = abs((initial_bearing_deg(hiker_lat, hiker_lon, wp2[0], wp2[1]) - b_direct + 180.0) % 360.0 - 180.0)

    chosen_wp = wp1 if dev1 <= dev2 else wp2
    chosen_dev = dev1 if dev1 <= dev2 else dev2
    b_to_wp = initial_bearing_deg(hiker_lat, hiker_lon, chosen_wp[0], chosen_wp[1])
    d_total = haversine_distance_m(hiker_lat, hiker_lon, chosen_wp[0], chosen_wp[1]) + haversine_distance_m(chosen_wp[0], chosen_wp[1], dest_coords[0], dest_coords[1])

    return {
        "waypoint_lat": round(chosen_wp[0], 5),
        "waypoint_lon": round(chosen_wp[1], 5),
        "detour_bearing_deg": round(b_to_wp, 1),
        "angular_deviation_deg": round(chosen_dev, 1),
        "detour_distance_m": round(d_total, 1),
        "path_coords": [
            (round(hiker_lat, 5), round(hiker_lon, 5)),
            (round(chosen_wp[0], 5), round(chosen_wp[1], 5)),
            (round(dest_coords[0], 5), round(dest_coords[1], 5)),
        ],
    }


def evaluate_hiker_hazard_state(
    hiker_lat: float,
    hiker_lon: float,
    hiker_alt: float,
    cartridge: dict,
    clock_hour_idx: int | None = None,
) -> dict:
    if clock_hour_idx is not None and cartridge.get("hourly_timeline"):
        env_state = evaluate_dynamic_surroundings_at_clock(cartridge, clock_hour_idx)
        nodes = env_state["dynamic_nodes"]
    else:
        env_state = None
        nodes = cartridge.get("danger_nodes", [])

    dest_coords = tuple(cartridge.get("dest_coords", (0.0, 0.0)))

    breached = []
    buffered = []
    uphill_ignored_nodes = []
    nearest_node = None
    min_dist = float("inf")

    for node in nodes:
        node_status, effective_d = is_hiker_in_directional_hazard(hiker_lat, hiker_lon, hiker_alt, node)
        if effective_d < min_dist:
            min_dist = effective_d
            nearest_node = node

        if node_status == "CRITICAL_BREACH":
            breached.append(node)
        elif node_status == "WARNING_BUFFER":
            buffered.append(node)
        elif node_status == "SAFE_UPHILL":
            uphill_ignored_nodes.append(node)

    if breached:
        status = "CRITICAL_BREACH"
        active_node = breached[0]
        escape = compute_immediate_escape_vector(hiker_lat, hiker_lon, hiker_alt, active_node)
        detour = compute_safe_destination_detour(hiker_lat, hiker_lon, dest_coords, breached)
    elif buffered:
        status = "WARNING_BUFFER"
        active_node = buffered[0]
        escape = compute_immediate_escape_vector(hiker_lat, hiker_lon, hiker_alt, active_node)
        detour = compute_safe_destination_detour(hiker_lat, hiker_lon, dest_coords, buffered)
    else:
        status = "SAFE"
        escape = None
        detour = None

    return {
        "status": status,
        "is_uphill_safe": len(uphill_ignored_nodes) > 0 and not breached and not buffered,
        "nearest_node": nearest_node,
        "distance_to_nearest_m": round(min_dist, 1) if nearest_node else -1.0,
        "breached_nodes": breached,
        "buffer_nodes": buffered,
        "uphill_ignored_nodes": uphill_ignored_nodes,
        "immediate_escape": escape,
        "safe_detour": detour,
        "clock_environment": env_state,
    }
