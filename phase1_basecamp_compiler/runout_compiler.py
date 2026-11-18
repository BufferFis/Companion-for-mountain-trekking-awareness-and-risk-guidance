from __future__ import annotations
import json
import math
import os
from dataclasses import dataclass
import numpy as np

DETACHMENT_SLOPE_DEG: float = 35.0
TRANSPORT_SLOPE_DEG: float = 25.0
STOPPING_SLOPE_DEG: float = 15.0
DETACHMENT_RADIUS_EXPANSION: float = 1.5
DEFAULT_BASE_DANGER_RADIUS_M: float = 60.0
SPATIAL_BUFFER_RADIUS_M: float = 500.0
MAX_CARTRIDGE_BYTES: int = 25_600


@dataclass
class DangerNode:
    id: int
    lat: float
    lon: float
    alt: float
    danger_radius_m: float
    buffer_radius_m: float
    zone_type: str
    slide_azimuth_deg: float
    risk_score: float


def classify_runout_regime(slope_deg: float) -> str:
    if slope_deg > DETACHMENT_SLOPE_DEG:
        return "initiation"
    if slope_deg > TRANSPORT_SLOPE_DEG:
        return "transport"
    if slope_deg > STOPPING_SLOPE_DEG:
        return "runout_fan"
    return "stopped"


def trace_single_debris_flow(
    row_idx: int,
    col_idx: int,
    p_init_val: float,
    terrain: object,
    base_radius_m: float = DEFAULT_BASE_DANGER_RADIUS_M,
    max_steps: int = 100,
) -> list[dict]:
    nodes = []
    raw_path: list[list[float]] = []
    H, W = terrain.slope_deg.shape
    curr_r, curr_c = float(row_idx), float(col_idx)
    momentum = 1.0
    risk = float(p_init_val)
    last_alt = None

    for _ in range(max_steps):
        r_i = int(round(curr_r))
        c_i = int(round(curr_c))
        if r_i < 0 or r_i >= H or c_i < 0 or c_i >= W:
            break

        alt = float(terrain.elevation_m[r_i, c_i])
        # Monotonic downslope check: debris cannot flow uphill
        if last_alt is not None and alt > last_alt + 2.0:
            break
        last_alt = alt

        slope = float(terrain.slope_deg[r_i, c_i])
        regime = classify_runout_regime(slope)
        if regime == "stopped" or momentum < 0.1:
            break

        azimuth = float(terrain.aspect_deg[r_i, c_i])
        lat = float(terrain.lat_grid[r_i, c_i])
        lon = float(terrain.lon_grid[r_i, c_i])

        raw_path.append([round(lat, 5), round(lon, 5), round(alt, 1)])

        if regime == "initiation":
            radius = base_radius_m * DETACHMENT_RADIUS_EXPANSION
            momentum = 1.0
        elif regime == "transport":
            radius = base_radius_m
            momentum = min(momentum, 0.9)
        else:
            radius = base_radius_m * 1.2
            momentum *= 0.75
            risk *= 0.85

        node = {
            "id": 0,
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "alt": round(alt, 1),
            "danger_radius_m": round(radius, 1),
            "buffer_radius_m": SPATIAL_BUFFER_RADIUS_M,
            "zone_type": regime,
            "slide_azimuth_deg": round(azimuth, 1),
            "risk_score": round(risk, 2),
        }
        nodes.append(node)

        az_rad = math.radians(azimuth)
        curr_r += -math.cos(az_rad)
        curr_c += math.sin(az_rad)

    if raw_path and nodes:
        # Downsample to max 5 key points to fit <25KB cartridge budget
        stride = max(1, len(raw_path) // 4)
        compact = raw_path[::stride][:4]
        if raw_path[-1] not in compact:
            compact.append(raw_path[-1])
        toe_elevation = raw_path[-1][2]

        for n in nodes:
            n["path"] = compact
            n["toe_alt"] = toe_elevation

    return nodes


def propagate_all_runouts(
    terrain: object,
    p_init_grid: np.ndarray,
    threshold: float = 0.75,
) -> list[dict]:
    all_nodes = []
    init_indices = np.argwhere(p_init_grid > threshold)
    for r_idx, c_idx in init_indices:
        p_val = float(p_init_grid[r_idx, c_idx])
        flow = trace_single_debris_flow(r_idx, c_idx, p_val, terrain)
        all_nodes.extend(flow)
    return all_nodes


def compress_sparse_danger_nodes(
    raw_nodes: list[dict],
    merge_distance_m: float = 120.0,
    max_bytes: int = 11_500,
) -> list[dict]:
    if not raw_nodes:
        return []

    # Fast O(N) exact coordinate deduplication before spatial clustering
    by_coord: dict[tuple[float, float], dict] = {}
    for node in raw_nodes:
        key = (float(node["lat"]), float(node["lon"]))
        existing = by_coord.get(key)
        if existing is None:
            by_coord[key] = dict(node)
        else:
            if node["danger_radius_m"] > existing["danger_radius_m"]:
                existing["danger_radius_m"] = node["danger_radius_m"]
            if node["risk_score"] > existing["risk_score"]:
                existing["risk_score"] = node["risk_score"]

    merged: list[dict] = []
    for node in by_coord.values():
        n_lat = float(node["lat"])
        n_lon = float(node["lon"])
        found_cluster = False
        for m in merged:
            m_lat = float(m["lat"])
            d_lat = (n_lat - m_lat) * 111320.0
            d_lon = (n_lon - float(m["lon"])) * 111320.0 * math.cos(math.radians(m_lat))
            if math.hypot(d_lat, d_lon) < merge_distance_m:
                if node["danger_radius_m"] > m["danger_radius_m"]:
                    m["danger_radius_m"] = node["danger_radius_m"]
                if node["risk_score"] > m["risk_score"]:
                    m["risk_score"] = node["risk_score"]
                # Keep highest elevation as crown altitude
                if float(node.get("alt", 0.0)) > float(m.get("alt", 0.0)):
                    m["alt"] = node["alt"]
                # Keep longest downstream path
                if len(node.get("path", [])) > len(m.get("path", [])):
                    m["path"] = node["path"]
                    if "toe_alt" in node:
                        m["toe_alt"] = node["toe_alt"]
                found_cluster = True
                break
        if not found_cluster:
            merged.append(node)

    merged.sort(key=lambda x: x["risk_score"], reverse=True)

    result: list[dict] = []
    running_bytes = 2  # for "[]"
    for idx, item in enumerate(merged, start=1):
        item["id"] = idx
        item_bytes = len(json.dumps(item, separators=(",", ":")).encode("utf-8")) + (1 if result else 0)
        if running_bytes + item_bytes > max_bytes:
            break
        result.append(item)
        running_bytes += item_bytes

    return result


def generate_safe_trail_waypoints(
    start_coords: tuple[float, float],
    dest_coords: tuple[float, float],
    terrain: object,
    n_waypoints: int = 30,
) -> list[dict]:
    waypoints = []
    lats = np.linspace(start_coords[0], dest_coords[0], n_waypoints)
    lons = np.linspace(start_coords[1], dest_coords[1], n_waypoints)

    min_lat = np.min(terrain.lat_grid)
    max_lat = np.max(terrain.lat_grid)
    min_lon = np.min(terrain.lon_grid)
    max_lon = np.max(terrain.lon_grid)

    H, W = terrain.elevation_m.shape
    for idx, (lt, ln) in enumerate(zip(lats, lons), start=1):
        r = int(np.clip((lt - min_lat) / max(max_lat - min_lat, 1e-6) * (H - 1), 0, H - 1))
        c = int(np.clip((ln - min_lon) / max(max_lon - min_lon, 1e-6) * (W - 1), 0, W - 1))
        alt = float(terrain.elevation_m[r, c])
        waypoints.append({
            "step": idx,
            "lat": round(float(lt), 5),
            "lon": round(float(ln), 5),
            "alt": round(alt, 1),
        })
    return waypoints


def export_trek_cartridge(
    output_path: str,
    start_coords: tuple[float, float],
    dest_coords: tuple[float, float],
    danger_nodes: list[dict],
    trail_waypoints: list[dict],
    baseline_params: dict,
    hourly_timeline: list[dict] | None = None,
) -> dict:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cartridge_data = {
        "format": "trek_cartridge_v2_rtc",
        "start_coords": [round(start_coords[0], 5), round(start_coords[1], 5)],
        "dest_coords": [round(dest_coords[0], 5), round(dest_coords[1], 5)],
        "danger_nodes": danger_nodes,
        "trail_waypoints": trail_waypoints,
        "baseline_params": baseline_params,
        "hourly_timeline": hourly_timeline or [],
    }

    serialized = json.dumps(cartridge_data, separators=(",", ":"))
    raw_bytes = serialized.encode("utf-8")
    while len(raw_bytes) >= MAX_CARTRIDGE_BYTES and len(cartridge_data["danger_nodes"]) > 5:
        cartridge_data["danger_nodes"] = cartridge_data["danger_nodes"][:-5]
        serialized = json.dumps(cartridge_data, separators=(",", ":"))
        raw_bytes = serialized.encode("utf-8")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(serialized)

    size = os.path.getsize(output_path)
    return {
        "cartridge_path": output_path,
        "size_bytes": size,
        "node_count": len(cartridge_data["danger_nodes"]),
        "timeline_hours": len(cartridge_data["hourly_timeline"]),
        "within_25kb": size < MAX_CARTRIDGE_BYTES,
    }
