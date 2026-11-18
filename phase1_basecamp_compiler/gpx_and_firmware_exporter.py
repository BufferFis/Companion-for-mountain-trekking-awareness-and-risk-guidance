from __future__ import annotations
import os
import xml.etree.ElementTree as ET


def parse_gpx_waypoints(gpx_xml_content: str, max_waypoints: int = 50) -> list[dict]:
    """
    Parse standard GPX XML (<trkpt>, <rtept>, or <wpt>) into corridor waypoints:
    [{"step": 1, "lat": float, "lon": float, "alt": float}, ...]
    Downsamples to at most `max_waypoints` evenly spaced points.
    """
    root = ET.fromstring(gpx_xml_content.strip())
    raw_pts: list[tuple[float, float, float]] = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag in ("trkpt", "rtept", "wpt"):
            lat = float(elem.attrib["lat"])
            lon = float(elem.attrib["lon"])
            alt = 2000.0
            for child in elem:
                ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if ctag == "ele" and child.text:
                    alt = float(child.text.strip())
            raw_pts.append((lat, lon, alt))

    if not raw_pts:
        raise ValueError("No valid GPX waypoints (<trkpt>, <rtept>, or <wpt>) found")

    if len(raw_pts) > max_waypoints:
        stride = max(1, len(raw_pts) // max_waypoints)
        sampled = raw_pts[::stride][: max_waypoints - 1]
        if raw_pts[-1] not in sampled:
            sampled.append(raw_pts[-1])
        raw_pts = sampled

    return [
        {"step": idx, "lat": round(lat, 5), "lon": round(lon, 5), "alt": round(alt, 1)}
        for idx, (lat, lon, alt) in enumerate(raw_pts, start=1)
    ]


def export_safe_route_gpx(cartridge: dict, output_path: str = "data/compiled_trek_route.gpx") -> str:
    """
    Export compiled trail_waypoints and danger_nodes from cartridge into a valid GPX 1.1 XML file
    compatible with Garmin / OsmAnd / organic offline maps.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="MountainSafetyCompanion" xmlns="http://www.topografix.com/GPX/1/1">',
    ]
    for node in cartridge.get("danger_nodes", []):
        lines.append(
            f'  <wpt lat="{node["lat"]}" lon="{node["lon"]}">'
            f'<ele>{node.get("alt", 2000.0)}</ele>'
            f'<name>HAZARD_{node["id"]}_{node["zone_type"].upper()}</name>'
            f'<desc>Radius {node["danger_radius_m"]}m Azimuth {node["slide_azimuth_deg"]}deg Risk {node["risk_score"]}</desc>'
            f'</wpt>'
        )
    lines.append("  <trk><name>Compiled Mountain Corridor</name><trkseg>")
    for wp in cartridge.get("trail_waypoints", []):
        lines.append(
            f'    <trkpt lat="{wp["lat"]}" lon="{wp["lon"]}"><ele>{wp.get("alt", 2000.0)}</ele></trkpt>'
        )
    lines.append("  </trkseg></trk>")
    lines.append("</gpx>")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return output_path


def export_embedded_c_header(
    cartridge: dict,
    output_path: str = "data/trek_cartridge_firmware.h",
) -> dict:
    """
    Generate a zero-allocation C/C++ PROGMEM header file (.h) from trek_cartridge.json
    for direct compilation onto ESP32 / STM32 / Arm Cortex-M microcontrollers.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    nodes = cartridge.get("danger_nodes", [])
    timeline = cartridge.get("hourly_timeline", [])

    lines = [
        "#ifndef TREK_CARTRIDGE_FIRMWARE_H",
        "#define TREK_CARTRIDGE_FIRMWARE_H",
        "#include <stdint.h>",
        "",
        "typedef struct {",
        "    uint16_t id;",
        "    float lat;",
        "    float lon;",
        "    float crown_alt_m;",
        "    float danger_radius_m;",
        "    float buffer_radius_m;",
        "    float slide_azimuth_deg;",
        "    float risk_score;",
        "} TrekDangerNode_t;",
        "",
        "typedef struct {",
        "    uint16_t hour_idx;",
        "    float precip_mm;",
        "    float rolling_rain_mm;",
        "    float soil_moisture;",
        "    float hazard_scale;",
        "} TrekHourlyForecast_t;",
        "",
        f"#define TREK_DANGER_NODE_COUNT {len(nodes)}",
        f"#define TREK_FORECAST_HOUR_COUNT {len(timeline)}",
        "",
        "static const TrekDangerNode_t TREK_DANGER_NODES[] = {",
    ]
    for n in nodes:
        lines.append(
            f"    {{{int(n['id'])}, {float(n['lat']):.5f}f, {float(n['lon']):.5f}f, "
            f"{float(n.get('alt', 0.0)):.1f}f, {float(n['danger_radius_m']):.1f}f, "
            f"{float(n.get('buffer_radius_m', 500.0)):.1f}f, {float(n.get('slide_azimuth_deg', 0.0)):.1f}f, "
            f"{float(n.get('risk_score', 0.5)):.2f}f}},"
        )
    lines.append("};")
    lines.append("")
    lines.append("static const TrekHourlyForecast_t TREK_HOURLY_FORECAST[] = {")
    for h in timeline:
        lines.append(
            f"    {{{int(h.get('h', 0))}, {float(h.get('p', 0.0)):.2f}f, "
            f"{float(h.get('r', 0.0)):.2f}f, {float(h.get('sm', 0.25)):.3f}f, "
            f"{float(h.get('s', 1.0)):.2f}f}},"
        )
    lines.append("};")
    lines.append("#endif /* TREK_CARTRIDGE_FIRMWARE_H */")

    content = "\n".join(lines) + "\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    size_bytes = os.path.getsize(output_path)
    return {
        "header_path": output_path,
        "size_bytes": size_bytes,
        "node_count": len(nodes),
        "forecast_hours": len(timeline),
    }
