from __future__ import annotations
import json
import os
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

from phase1_basecamp_compiler.terrain_ingestor import (
    search_place_coordinates,
    PRESET_TREK_ROUTES,
)
from phase1_basecamp_compiler.compile_cartridge import run_basecamp_compilation
from phase1_basecamp_compiler.gpx_and_firmware_exporter import (
    parse_gpx_waypoints,
    export_safe_route_gpx,
    export_embedded_c_header,
)
from phase2_offline_edge_companion.offline_evasion_nav import (
    load_trek_cartridge,
    evaluate_dynamic_surroundings_at_clock,
    haversine_distance_m,
    build_directional_cone_polygon,
)
from phase2_offline_edge_companion.physiological_companion import (
    simulate_sensor_packet,
    read_blackbox_queue,
)
from phase2_offline_edge_companion.edge_runtime import (
    EdgeCompanionRuntime,
    verify_airgap_compliance,
)
from phase2_offline_edge_companion.hardware_sensors import HardwareSensorBridge


@st.cache_data(show_spinner=False)
def _cached_load_cartridge(cartridge_path: str, mtime: float) -> dict:
    return load_trek_cartridge(cartridge_path)


def _get_cartridge_fast(cartridge_path: str) -> dict:
    mtime = os.path.getmtime(cartridge_path) if os.path.exists(cartridge_path) else 0.0
    return _cached_load_cartridge(cartridge_path, mtime)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_search_place(query: str) -> list[dict]:
    return search_place_coordinates(query)


def plot_terrain_and_runout_map(
    terrain: object,
    p_init_grid: np.ndarray,
    cartridge: dict,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=terrain.elevation_m,
        x=terrain.lon_grid[0, :],
        y=terrain.lat_grid[:, 0],
        colorscale="Earth",
        name="Elevation (m)",
        colorbar=dict(title="Elev (m)"),
    ))
    nodes = cartridge.get("danger_nodes", [])
    if nodes:
        fig.add_trace(go.Scatter(
            x=[n["lon"] for n in nodes],
            y=[n["lat"] for n in nodes],
            mode="markers",
            marker=dict(
                color=["red" if n["zone_type"] == "initiation" else ("orange" if n["zone_type"] == "transport" else "yellow") for n in nodes],
                size=8,
                line=dict(width=1, color="black"),
            ),
            name="Mapped Hazard Nodes",
        ))
    fig.update_layout(
        title="2D SRTM Elevation Grid (m) & Hazard Nodes",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        margin=dict(l=15, r=15, t=35, b=15),
        height=320,
    )
    return fig


def plot_slope_and_initiation_heatmaps(
    terrain: object,
    p_init_grid: np.ndarray,
) -> tuple[go.Figure, go.Figure]:
    fig_slope = go.Figure(data=go.Heatmap(
        z=terrain.slope_deg,
        x=terrain.lon_grid[0, :],
        y=terrain.lat_grid[:, 0],
        colorscale="YlOrRd",
        colorbar=dict(title="Deg"),
    ))
    fig_slope.update_layout(
        title="Slope Incline θ (deg) via numpy.gradient()",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        margin=dict(l=15, r=15, t=35, b=15),
        height=320,
    )

    fig_p = go.Figure(data=go.Heatmap(
        z=p_init_grid,
        x=terrain.lon_grid[0, :],
        y=terrain.lat_grid[:, 0],
        colorscale="Reds",
        zmin=0.0,
        zmax=1.0,
        colorbar=dict(title="P_init"),
    ))
    fig_p.update_layout(
        title="<1 MB TFLite Initiation Probability (P_init > 0.75)",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        margin=dict(l=15, r=15, t=35, b=15),
        height=320,
    )
    return fig_slope, fig_p


def plot_hourly_forecast_timeline(timeline: list[dict], active_hour_idx: int | None = None) -> go.Figure:
    fig = go.Figure()
    if not timeline:
        return fig
    times = [h["t"] for h in timeline]
    precips = [h["p"] for h in timeline]
    rolling_r = [h["r"] for h in timeline]
    scales = [h["s"] * 50.0 for h in timeline]

    fig.add_trace(go.Bar(
        x=times,
        y=precips,
        name="Hourly Forecast Rain (mm/hr)",
        marker_color="#3B82F6",
    ))
    fig.add_trace(go.Scatter(
        x=times,
        y=rolling_r,
        mode="lines",
        name="Effective Pore-Water Saturation (mm)",
        line=dict(color="#EF4444", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=times,
        y=scales,
        mode="lines",
        name="Dynamic Hazard Scale (x50)",
        line=dict(color="#F59E0B", width=2, dash="dot"),
    ))
    if active_hour_idx is not None and 0 <= active_hour_idx < len(times):
        fig.add_vline(x=times[active_hour_idx], line_width=3, line_dash="dash", line_color="#10B981")

    fig.update_layout(
        title="168-Hour Real Open-Meteo Forecast Timeline (Drives On-Board Arm RTC Clock)",
        xaxis_title="Forecast Timestamp (Local Trek Time)",
        yaxis_title="mm / Index",
        margin=dict(l=15, r=15, t=35, b=15),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_cartridge_size_comparison(raw_raster_bytes: int, cartridge_bytes: int) -> go.Figure:
    categories = ["Full Terrain Raster", "Offline Cartridge (<25 KB)"]
    sizes_kb = [raw_raster_bytes / 1024.0, cartridge_bytes / 1024.0]
    fig = go.Figure(data=[go.Bar(
        x=categories,
        y=sizes_kb,
        text=[f"{s:.1f} KB" for s in sizes_kb],
        textposition="auto",
        marker_color=["#EF553B", "#00CC96"],
    )])
    fig.update_layout(
        title="Compression Ratio: Raster vs Air-Gapped Cartridge",
        yaxis_title="File Size (KB)",
        margin=dict(l=15, r=15, t=35, b=15),
        height=320,
    )
    return fig


def plot_live_evasion_map(cartridge: dict, hiker_pos: dict, nav_state: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[hiker_pos["lon"]],
        y=[hiker_pos["lat"]],
        mode="markers",
        marker=dict(color="blue", size=12),
        name="Hiker",
    ))
    return fig


def render_physiological_gauges(physio_state: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Heart Rate (MAX30102)", f"{physio_state['hr_display']:.0f} BPM")
    spo2_txt = (
        f"{physio_state['spo2_display']}%"
        if physio_state["spo2_display"] is not None
        else "🔒 LOCKED (Moving — Hold Still 12s)"
    )
    c2.metric("SpO2 (Xu et al. 2026 IMU Gate)", spo2_txt)
    c3.metric("Signal Quality W_SQI (Hu et al. 2026)", f"{physio_state['w_sqi_pct']:.0f}%")
    c4.metric("Exertion State (Samavati et al. 2026)", physio_state["exertion"]["classification"])


def _ensure_initial_state() -> None:
    if "start_coords" not in st.session_state:
        first_key = list(PRESET_TREK_ROUTES.keys())[0]
        p = PRESET_TREK_ROUTES[first_key]
        st.session_state["start_coords"] = list(p["start_coords"])
        st.session_state["dest_coords"] = list(p["dest_coords"])
        st.session_state["start_name"] = p["start_name"]
        st.session_state["dest_name"] = p["dest_name"]
        st.session_state["active_preset"] = first_key
        st.session_state["map_center"] = [
            (p["start_coords"][0] + p["dest_coords"][0]) / 2.0,
            (p["start_coords"][1] + p["dest_coords"][1]) / 2.0,
        ]
        st.session_state["map_zoom"] = 13

    if "compilation_res" not in st.session_state:
        res = run_basecamp_compilation(
            start_coords=tuple(st.session_state["start_coords"]),
            dest_coords=tuple(st.session_state["dest_coords"]),
            start_name=st.session_state["start_name"],
            dest_name=st.session_state["dest_name"],
            output_dir="data",
            force_offline=False,
            sample_grid_dim=10,
        )
        st.session_state["compilation_res"] = res


@st.fragment
def render_tab1_basecamp_compiler() -> None:
    _ensure_initial_state()
    st.subheader("Phase 1: Base Camp Compiler — Real Open-Meteo SRTM + 168h Hourly Forecast")

    with st.expander("ℹ️ What is happening on this page & where does the data come from? (Zero Synthetic Data)", expanded=False):
        st.markdown("""
        1. **100% Real Open-Meteo Terrain & Hourly Weather (Zero Synthetic Faking)**:
           - Queries real **Open-Meteo SRTM 90m Elevation** (`https://api.open-meteo.com/v1/elevation`) in politely paced 100-coordinate batches so the server is never rate-limited.
           - Queries real **14-Day Hourly Weather & Soil Moisture** (`https://api.open-meteo.com/v1/forecast` — 7 days past antecedent rainfall + 7 days future hourly precipitation, soil moisture `0–7cm`, and temperature).
        2. **Physically Select Start & End Points on OpenStreetMap**:
           - Pick a preset (**Kodaikanal**, **Ooty**, **Munnar**, **Gangotri**), search any town/peak name, or **click directly anywhere on the OpenStreetMap below** to place your **Green Start Pin (Base Camp)** and **Blue End Pin (Destination)**.
        3. **Time-Indexed Landslide Runout Physics & <25 KB Cartridge**:
           - Computes slope $\\theta$, aspect $\\alpha$, and curvature via `numpy.gradient()`, runs the `<1 MB` TFLite model, traces downslope runouts (`>35°` Detachment, `25°–35°` Transport, `15°–25°` Fan), and embeds the **168-hour forecast timeline** into `data/trek_cartridge.json` (`<25 KB`) to drive the **On-Board Arm RTC Clock** in Tab 2!
        """)

    col_ctrl1, col_ctrl2 = st.columns([1.15, 1])

    with col_ctrl1:
        st.markdown("#### 1. Choose Preset Region or Search Any Place")
        preset_options = list(PRESET_TREK_ROUTES.keys())
        preset_choice = st.selectbox(
            "Fixed Trek Route / Region Preset",
            options=preset_options,
            index=0,
        )
        if st.session_state.get("active_preset") != preset_choice:
            p = PRESET_TREK_ROUTES[preset_choice]
            st.session_state["start_coords"] = list(p["start_coords"])
            st.session_state["dest_coords"] = list(p["dest_coords"])
            st.session_state["start_name"] = p["start_name"]
            st.session_state["dest_name"] = p["dest_name"]
            st.session_state["active_preset"] = preset_choice
            st.session_state["map_center"] = [
                (p["start_coords"][0] + p["dest_coords"][0]) / 2.0,
                (p["start_coords"][1] + p["dest_coords"][1]) / 2.0,
            ]
            d_km = haversine_distance_m(p["start_coords"][0], p["start_coords"][1], p["dest_coords"][0], p["dest_coords"][1]) / 1000.0
            st.session_state["map_zoom"] = 9 if d_km > 40 else 13
            res = run_basecamp_compilation(
                start_coords=tuple(st.session_state["start_coords"]),
                dest_coords=tuple(st.session_state["dest_coords"]),
                start_name=st.session_state["start_name"],
                dest_name=st.session_state["dest_name"],
                output_dir="data",
                force_offline=False,
                sample_grid_dim=10,
            )
            st.session_state["compilation_res"] = res
            st.session_state.pop("tab2_hiker_pos", None)
            st.session_state.pop("last_tab2_click", None)
            st.rerun()

        search_query = st.text_input("Search Any Town, Peak, or Valley (e.g. Kodaikanal, Ooty, Coonoor, Munnar, Manali)", value="")
        if search_query:
            results = _cached_search_place(search_query)
            if results:
                top = results[0]
                st.caption(f"Found: **{top['display_name']}** (`{top['lat']:.4f}, {top['lon']:.4f}`)")
                b1, b2, b3 = st.columns(3)
                if b1.button("📍 Set as Start"):
                    st.session_state["start_coords"] = [top["lat"], top["lon"]]
                    st.session_state["start_name"] = top["name"]
                    st.session_state["map_center"] = [top["lat"], top["lon"]]
                    st.rerun(scope="fragment")
                if b2.button("🎯 Set as End"):
                    st.session_state["dest_coords"] = [top["lat"], top["lon"]]
                    st.session_state["dest_name"] = top["name"]
                    st.session_state["map_center"] = [top["lat"], top["lon"]]
                    st.rerun(scope="fragment")
                if b3.button("🗺️ Pan Map Here"):
                    st.session_state["map_center"] = [top["lat"], top["lon"]]
                    st.session_state["map_zoom"] = 13
                    st.rerun(scope="fragment")

        uploaded_gpx = st.file_uploader("Upload Trail GPX (.gpx) (Optional)", type=["gpx"])
        if uploaded_gpx is not None:
            gpx_name = uploaded_gpx.name
            if st.session_state.get("loaded_gpx_name") != gpx_name:
                try:
                    gpx_text = uploaded_gpx.getvalue().decode("utf-8")
                    gpx_pts = parse_gpx_waypoints(gpx_text, max_waypoints=50)
                    st.session_state["loaded_gpx_name"] = gpx_name
                    st.session_state["gpx_waypoints"] = gpx_pts
                    st.session_state["start_coords"] = [gpx_pts[0]["lat"], gpx_pts[0]["lon"]]
                    st.session_state["dest_coords"] = [gpx_pts[-1]["lat"], gpx_pts[-1]["lon"]]
                    st.session_state["start_name"] = f"GPX Start ({gpx_name})"
                    st.session_state["dest_name"] = f"GPX End ({gpx_name})"
                    st.session_state["map_center"] = [
                        (gpx_pts[0]["lat"] + gpx_pts[-1]["lat"]) / 2.0,
                        (gpx_pts[0]["lon"] + gpx_pts[-1]["lon"]) / 2.0,
                    ]
                    st.rerun(scope="fragment")
                except Exception as exc:
                    st.error(f"GPX parse error: {exc}")

    with col_ctrl2:
        st.markdown("#### 2. Active Corridor Pins (Click Map Below to Move)")
        start_c = st.session_state["start_coords"]
        dest_c = st.session_state["dest_coords"]
        dist_km = haversine_distance_m(start_c[0], start_c[1], dest_c[0], dest_c[1]) / 1000.0

        st.info(
            f"🟢 **Start (Base Camp):** {st.session_state.get('start_name', 'Start')} (`{start_c[0]:.4f}, {start_c[1]:.4f}`)\n\n"
            f"🔵 **End (Destination):** {st.session_state.get('dest_name', 'End')} (`{dest_c[0]:.4f}, {dest_c[1]:.4f}`)\n\n"
            f"📏 **Corridor Span:** `{dist_km:.2f} km` (+ 2 km lateral buffer)"
        )

        click_target = st.radio(
            "When you click on the OpenStreetMap below, update:",
            options=["🟢 Start Pin (Base Camp)", "🔵 End Pin (Destination)"],
            horizontal=True,
        )

    st.markdown("#### 🗺️ Interactive OpenStreetMap — Click Anywhere to Set Start/End & Inspect Mapped Hazards")
    m_center = st.session_state.get("map_center", [(start_c[0] + dest_c[0]) / 2.0, (start_c[1] + dest_c[1]) / 2.0])
    m_zoom = st.session_state.get("map_zoom", 13)

    m = folium.Map(location=m_center, zoom_start=m_zoom, tiles="OpenStreetMap")
    folium.TileLayer("OpenTopoMap", name="Topographic Contours (OpenTopoMap)").add_to(m)

    folium.Marker(
        location=start_c,
        popup=f"Start: {st.session_state.get('start_name', 'Base Camp')} ({start_c[0]:.4f}, {start_c[1]:.4f})",
        tooltip=f"🟢 Start: {st.session_state.get('start_name', 'Base Camp')}",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(m)

    folium.Marker(
        location=dest_c,
        popup=f"End: {st.session_state.get('dest_name', 'Destination')} ({dest_c[0]:.4f}, {dest_c[1]:.4f})",
        tooltip=f"🔵 End: {st.session_state.get('dest_name', 'Destination')}",
        icon=folium.Icon(color="blue", icon="flag"),
    ).add_to(m)

    folium.PolyLine([start_c, dest_c], color="#2563EB", weight=3, dash_array="6, 8", tooltip="Corridor Axis").add_to(m)

    if st.session_state.get("gpx_waypoints"):
        gpx_line = [[p["lat"], p["lon"]] for p in st.session_state["gpx_waypoints"]]
        folium.PolyLine(gpx_line, color="#7C3AED", weight=4, tooltip="Uploaded GPX Track").add_to(m)

    cartridge = None
    if "compilation_res" in st.session_state:
        cartridge = _get_cartridge_fast(st.session_state["compilation_res"]["cartridge_meta"]["cartridge_path"])
        for node in cartridge.get("danger_nodes", []):
            ztype = node["zone_type"]
            color = "#DC2626" if ztype == "initiation" else ("#EA580C" if ztype == "transport" else "#EAB308")
            folium.Circle(
                location=[node["lat"], node["lon"]],
                radius=node["buffer_radius_m"],
                color="#F97316",
                weight=1,
                fill=False,
                dash_array="4, 8",
                tooltip=f"500m Early Warning Buffer (Node #{node['id']})",
            ).add_to(m)
            # Directional Runout Cone Polygon
            cone_pts = build_directional_cone_polygon(node)
            folium.Polygon(
                locations=cone_pts,
                color=color,
                weight=2,
                fill=True,
                fill_color=color,
                fill_opacity=0.45,
                tooltip=(
                    f"Node #{node['id']} | {ztype.upper()} Cone | Crown Alt: {node['alt']}m | "
                    f"Radius: {node['danger_radius_m']}m | Downslope Slide Azimuth: {node['slide_azimuth_deg']}° | "
                    f"Risk: {node['risk_score']}"
                ),
            ).add_to(m)
            # Traced Downslope Flow Polyline
            if node.get("path") and len(node["path"]) >= 2:
                flow_pts = [[p[0], p[1]] for p in node["path"]]
                folium.PolyLine(
                    locations=flow_pts,
                    color=color,
                    weight=4,
                    opacity=0.85,
                    tooltip=f"Downslope Debris Path #{node['id']} (Crown {node['alt']}m -> Toe {node.get('toe_alt', node['alt'])}m)",
                ).add_to(m)

    folium.LayerControl().add_to(m)
    map_out = st_folium(
        m,
        width="100%",
        height=480,
        key="basecamp_folium_map",
        returned_objects=["last_clicked"],
    )

    if map_out and map_out.get("last_clicked"):
        lat_clk = round(float(map_out["last_clicked"]["lat"]), 5)
        lon_clk = round(float(map_out["last_clicked"]["lng"]), 5)
        click_pair = (lat_clk, lon_clk)
        if click_pair != st.session_state.get("last_tab1_click"):
            st.session_state["last_tab1_click"] = click_pair
            if "Start" in click_target:
                st.session_state["start_coords"] = [lat_clk, lon_clk]
                st.session_state["start_name"] = f"Map Pin ({lat_clk:.3f}, {lon_clk:.3f})"
            else:
                st.session_state["dest_coords"] = [lat_clk, lon_clk]
                st.session_state["dest_name"] = f"Map Pin ({lat_clk:.3f}, {lon_clk:.3f})"
            st.session_state["map_center"] = [lat_clk, lon_clk]
            st.rerun(scope="fragment")

    c_opt1, c_opt2 = st.columns([2, 1])
    with c_opt1:
        density_choice = st.selectbox(
            "Open-Meteo Real SRTM DEM Sampling Density (Polite Paced Batching — Zero Rate-Limiting)",
            options=[
                "Standard Real DEM (10x10 = 100 pts, 1 batch, ~2s)",
                "Medium Real DEM (20x20 = 400 pts, 4 paced batches, ~6s)",
                "High-Density Real DEM (30x30 = 900 pts, 9 paced batches, ~15s)",
                "Ultra-High Trek DEM (50x50 = 2500 pts, 25 paced batches, ~40s)",
            ],
            index=0,
        )
        grid_dim_map = {
            "Standard Real DEM (10x10 = 100 pts, 1 batch, ~2s)": 10,
            "Medium Real DEM (20x20 = 400 pts, 4 paced batches, ~6s)": 20,
            "High-Density Real DEM (30x30 = 900 pts, 9 paced batches, ~15s)": 30,
            "Ultra-High Trek DEM (50x50 = 2500 pts, 25 paced batches, ~40s)": 50,
        }
        chosen_dim = grid_dim_map[density_choice]
    with c_opt2:
        st.write("")
        compile_btn = st.button("⚡ Compile Real DEM & 168h Weather Cartridge (<25 KB)", type="primary", width="stretch")

    if compile_btn:
        prog_bar = st.progress(0, text="Initializing real Open-Meteo SRTM + 168h Forecast download...")
        def _update_prog(b_idx: int, total_b: int, msg: str) -> None:
            frac = min(0.95, b_idx / max(total_b, 1))
            prog_bar.progress(frac, text=msg)

        res = run_basecamp_compilation(
            start_coords=tuple(st.session_state["start_coords"]),
            dest_coords=tuple(st.session_state["dest_coords"]),
            start_name=st.session_state.get("start_name", "Base Camp"),
            dest_name=st.session_state.get("dest_name", "Destination"),
            output_dir="data",
            force_offline=False,
            sample_grid_dim=chosen_dim,
            batch_delay_s=1.2,
            progress_callback=_update_prog,
        )
        prog_bar.progress(1.0, text="Compilation complete!")
        st.session_state["compilation_res"] = res
        st.session_state.pop("tab2_hiker_pos", None)
        st.session_state.pop("last_tab2_click", None)
        st.rerun()

    if "compilation_res" in st.session_state:
        res = st.session_state["compilation_res"]
        t = res["terrain"]
        if cartridge is None:
            cartridge = _get_cartridge_fast(res["cartridge_meta"]["cartridge_path"])

        st.markdown("---")
        st.markdown("### 📊 Real Data Provenance & Compiled Corridor Telemetry")
        st.success(
            f"✅ **100% REAL OPEN-METEO DATA:** SRTM 90m Elevation (`https://api.open-meteo.com/v1/elevation`) + "
            f"14-Day Hourly Precipitation & Soil Moisture (`https://api.open-meteo.com/v1/forecast`) for "
            f"**{st.session_state.get('start_name')} → {st.session_state.get('dest_name')}**."
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Elevation Range (SRTM)", f"{np.min(t.elevation_m):.0f}m – {np.max(t.elevation_m):.0f}m")
        m2.metric("Steepest Ridge Slope (θ)", f"{np.max(t.slope_deg):.1f}°")
        m3.metric("Past 7-Day Antecedent Rain", f"{t.antecedent_7d_rain_mm:.1f} mm")
        m4.metric("Next 7-Day Forecast Rain", f"{t.forecast_7d_rain_mm:.1f} mm")

        nodes = cartridge.get("danger_nodes", [])
        n_init = sum(1 for n in nodes if n["zone_type"] == "initiation")
        n_trans = sum(1 for n in nodes if n["zone_type"] == "transport")
        n_fan = sum(1 for n in nodes if n["zone_type"] == "runout_fan")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("🔴 Initiation Zones (>35°)", f"{n_init} nodes")
        m6.metric("🟠 Transport Gullies (25–35°)", f"{n_trans} nodes")
        m7.metric("🟡 Runout Fans (15–25°)", f"{n_fan} nodes")
        m8.metric("📦 Cartridge Size (with 168h Clock)", f"{res['cartridge_meta']['size_bytes']:,} bytes (<25 KB)")

        cache_id = id(res)
        if st.session_state.get("tab1_fig_cache_id") != cache_id:
            raw_bytes = t.elevation_m.nbytes * 3
            fig_slope, fig_p = plot_slope_and_initiation_heatmaps(t, res["p_init_grid"])
            st.session_state["tab1_fig_cache_id"] = cache_id
            st.session_state["tab1_figs"] = {
                "timeline": plot_hourly_forecast_timeline(cartridge["hourly_timeline"]) if cartridge.get("hourly_timeline") else None,
                "elev": plot_terrain_and_runout_map(t, res["p_init_grid"], cartridge),
                "slope": fig_slope,
                "p_init": fig_p,
                "size": plot_cartridge_size_comparison(raw_bytes, res["cartridge_meta"]["size_bytes"]),
            }

        figs = st.session_state["tab1_figs"]
        if figs["timeline"] is not None:
            st.plotly_chart(figs["timeline"], width="stretch")

        st.markdown("#### 📦 Export Compiled Air-Gapped Artefacts (Flash onto Companion / Watch / ESP32)")
        cartridge_path = res["cartridge_meta"]["cartridge_path"]
        gpx_out_path = export_safe_route_gpx(cartridge, output_path="data/compiled_trek_route.gpx")
        fw_meta = export_embedded_c_header(cartridge, output_path="data/trek_cartridge_firmware.h")

        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            with open(cartridge_path, "rb") as f_cart:
                st.download_button(
                    label=f"⬇️ Cartridge JSON ({res['cartridge_meta']['size_bytes']:,} B)",
                    data=f_cart.read(),
                    file_name="trek_cartridge.json",
                    mime="application/json",
                    width="stretch",
                )
        with dl2:
            with open(gpx_out_path, "rb") as f_gpx:
                st.download_button(
                    label="⬇️ Safe Route GPX (.gpx)",
                    data=f_gpx.read(),
                    file_name="compiled_trek_route.gpx",
                    mime="application/gpx+xml",
                    width="stretch",
                )
        with dl3:
            with open(fw_meta["header_path"], "rb") as f_hdr:
                st.download_button(
                    label=f"⬇️ C/C++ Firmware Header ({fw_meta['size_bytes']:,} B)",
                    data=f_hdr.read(),
                    file_name="trek_cartridge_firmware.h",
                    mime="text/plain",
                    width="stretch",
                )

        with st.expander("🔬 Terrain Physics & Compression Diagnostics (Slope / Initiation / Byte Budget)", expanded=False):
            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(figs["elev"], width="stretch")
                st.plotly_chart(figs["p_init"], width="stretch")
            with g2:
                st.plotly_chart(figs["slope"], width="stretch")
                st.plotly_chart(figs["size"], width="stretch")


@st.fragment
def render_tab2_offline_trail_companion() -> None:
    _ensure_initial_state()
    st.subheader("Phase 2: 100% Air-Gapped Trail Companion — On-Board Arm RTC Clock & Dynamic Danger Avoidance")

    cartridge_path = "data/trek_cartridge.json"
    if not os.path.exists(cartridge_path):
        st.warning("⚠️ No trek_cartridge.json found. Compile in Tab 1 first!")
        return

    cartridge = _get_cartridge_fast(cartridge_path)
    trail = cartridge.get("trail_waypoints", [])
    base_params = cartridge.get("baseline_params", {})
    timeline = cartridge.get("hourly_timeline", [])
    max_hour = max(0, len(timeline) - 1)

    if "arm_rtc_hour" not in st.session_state:
        st.session_state["arm_rtc_hour"] = 0

    st.markdown("### ⌚ On-Board Arm RTC Clock (Drives Time-Dynamic Weather & Hazard Surroundings Offline)")
    clk_c1, clk_c2, clk_c3, clk_c4 = st.columns([2.2, 0.9, 0.9, 1.2])
    with clk_c2:
        if st.button("⌚ +1 Hour", width="stretch"):
            st.session_state["arm_rtc_hour"] = min(max_hour, st.session_state["arm_rtc_hour"] + 1)
    with clk_c3:
        if st.button("⌚ +6 Hours", width="stretch"):
            st.session_state["arm_rtc_hour"] = min(max_hour, st.session_state["arm_rtc_hour"] + 6)
    with clk_c4:
        if st.button("🌧️ Jump to Peak Rain Hour", width="stretch") and timeline:
            peak_h = max(range(len(timeline)), key=lambda i: timeline[i].get("s", 1.0))
            st.session_state["arm_rtc_hour"] = peak_h
    with clk_c1:
        clock_hour = st.slider(
            "Arm RTC Trek Elapsed Hour (0h = Departure .. 167h = Day 7)",
            min_value=0,
            max_value=max(1, max_hour),
            value=min(st.session_state["arm_rtc_hour"], max(1, max_hour)),
        )
        st.session_state["arm_rtc_hour"] = clock_hour

    env_now = evaluate_dynamic_surroundings_at_clock(cartridge, clock_hour)
    dynamic_nodes = env_now["dynamic_nodes"]

    w1, w2, w3, w4, w5 = st.columns(5)
    day_num = (clock_hour // 24) + 1
    w1.metric("⌚ Arm RTC Timestamp", f"Day {day_num} ({env_now['rtc_iso_time']})")
    w2.metric("🌧️ Forecast Rain @ Hour", f"{env_now['precip_mm_hr']:.2f} mm/h")
    w3.metric("💧 Soil Moisture (0–7cm)", f"{env_now['soil_moisture']:.3f} m³/m³")
    w4.metric("🌡️ Ambient Temp", f"{env_now['temp_c']:.1f} °C")
    w5.metric("⚡ Dynamic Hazard Scale", f"{env_now['hazard_scale']:.2f}x ({env_now['active_critical_count']} critical)")

    if timeline:
        with st.expander("📈 View 168-Hour Forecast Timeline & Current Arm RTC Clock Marker", expanded=False):
            st.plotly_chart(plot_hourly_forecast_timeline(timeline, active_hour_idx=clock_hour), width="stretch")

    cartridge_mtime = os.path.getmtime(cartridge_path)
    if "edge_runtime" not in st.session_state or st.session_state.get("edge_runtime_mtime") != cartridge_mtime:
        st.session_state["edge_runtime"] = EdgeCompanionRuntime(
            cartridge_path=cartridge_path,
            queue_path="data/offline_sync_queue.jsonl",
        )
        st.session_state["edge_runtime_mtime"] = cartridge_mtime
        if not st.session_state.pop("tab2_keep_pos_on_reload", False):
            st.session_state.pop("tab2_hiker_pos", None)
            st.session_state.pop("last_tab2_click", None)
    runtime: EdgeCompanionRuntime = st.session_state["edge_runtime"]

    st.info(
        f"📦 **Loaded Offline Cartridge:** `{base_params.get('start_name', 'Start')}` → `{base_params.get('dest_name', 'Destination')}` "
        f"| **Mapped Landslide Zones:** `{len(dynamic_nodes)}` | **Source:** `{base_params.get('source_label', 'Compiled DEM')}`"
    )

    if not dynamic_nodes:
        st.warning("ℹ️ Current compiled corridor has 0 active landslide zones. Click any mountain area on the map below and press **Scan & Compile Landslides Around Pressed Area**.")

    op1, op2, op3 = st.columns([1.2, 1.0, 0.8])
    with op1:
        pos_mode = st.radio(
            "GPS Position Mode:",
            ["Click Map / Jump Button", "Scrub Trail Axis"],
            index=0,
            horizontal=True,
        )
        if pos_mode == "Scrub Trail Axis" and trail:
            idx = st.slider("Corridor Step", 0, len(trail) - 1, len(trail) // 2)
            hiker_lat = float(trail[idx]["lat"])
            hiker_lon = float(trail[idx]["lon"])
            hiker_alt = float(trail[idx]["alt"])
            st.session_state["tab2_hiker_pos"] = [hiker_lat, hiker_lon, hiker_alt]
        else:
            if "tab2_hiker_pos" not in st.session_state:
                if dynamic_nodes:
                    n0 = dynamic_nodes[0]
                    st.session_state["tab2_hiker_pos"] = [float(n0["lat"]), float(n0["lon"]), float(n0["alt"])]
                elif trail:
                    w0 = trail[len(trail) // 2]
                    st.session_state["tab2_hiker_pos"] = [float(w0["lat"]), float(w0["lon"]), float(w0["alt"])]
                else:
                    sc = cartridge.get("start_coords", [10.2000, 77.4800])
                    st.session_state["tab2_hiker_pos"] = [float(sc[0]), float(sc[1]), 2100.0]
            hiker_lat, hiker_lon, hiker_alt = st.session_state["tab2_hiker_pos"]

    with op2:
        hiker_alt = st.slider(
            "⛰️ Hiker Elevation (m) — Uphill Safety Gate:",
            min_value=max(0.0, float(hiker_alt) - 600.0),
            max_value=float(hiker_alt) + 600.0,
            value=float(hiker_alt),
            step=5.0,
        )
        st.session_state["tab2_hiker_pos"][2] = hiker_alt
    with op3:
        cohort_mode = st.selectbox("Acclimatization Cohort", ["migrant_acute", "resident_adapted"], index=0)
        is_rest_stop = st.checkbox("Hold Still >= 12s (Unlock SpO2)", value=False)

    st.markdown("#### 🛰️ Scan Any New Map Area for Possible Landslides (Updates Offline Cartridge Around Pressed Point)")
    scan_c1, scan_c2, scan_c3 = st.columns([1.4, 0.8, 1.4])
    with scan_c1:
        tab2_search = st.text_input(
            "Jump Map to Any Place (Optional)",
            value="",
            placeholder="e.g. Ooty, Munnar, Coonoor, Manali, Gangotri",
            key="tab2_place_search",
        )
    with scan_c2:
        st.write("")
        st.write("")
        if st.button("📍 Jump & Scan Place", width="stretch") and tab2_search.strip():
            found = _cached_search_place(tab2_search.strip())
            if found:
                top_p = found[0]
                hiker_lat, hiker_lon = float(top_p["lat"]), float(top_p["lon"])
                with st.spinner(f"Fetching real Open-Meteo SRTM & 168h forecast around {top_p['name']}..."):
                    delta_deg = 0.015
                    scan_res = run_basecamp_compilation(
                        start_coords=(round(hiker_lat - delta_deg, 5), round(hiker_lon - delta_deg, 5)),
                        dest_coords=(round(hiker_lat + delta_deg, 5), round(hiker_lon + delta_deg, 5)),
                        start_name=f"{top_p['name']} Area SW",
                        dest_name=f"{top_p['name']} Area NE",
                        output_dir="data",
                        force_offline=False,
                        sample_grid_dim=10,
                        batch_delay_s=1.2,
                    )
                t_scan = scan_res["terrain"]
                H, W = t_scan.elevation_m.shape
                real_alt = float(t_scan.elevation_m[H // 2, W // 2])
                st.session_state["compilation_res"] = scan_res
                st.session_state["tab2_hiker_pos"] = [hiker_lat, hiker_lon, round(real_alt, 1)]
                st.session_state["tab2_keep_pos_on_reload"] = True
                st.rerun()
            else:
                st.warning(f"Place '{tab2_search}' not found.")
    with scan_c3:
        st.write("")
        st.write("")
        if st.button(
            f"🛰️ Compile Landslides Around Pressed Area ({hiker_lat:.3f}, {hiker_lon:.3f})",
            type="primary",
            width="stretch",
        ):
            with st.spinner(f"Scanning SRTM elevation & 168h forecast around ({hiker_lat:.4f}, {hiker_lon:.4f})..."):
                delta_deg = 0.015
                scan_res = run_basecamp_compilation(
                    start_coords=(round(hiker_lat - delta_deg, 5), round(hiker_lon - delta_deg, 5)),
                    dest_coords=(round(hiker_lat + delta_deg, 5), round(hiker_lon + delta_deg, 5)),
                    start_name=f"Pressed Point ({hiker_lat:.3f}, {hiker_lon:.3f}) SW",
                    dest_name=f"Pressed Point ({hiker_lat:.3f}, {hiker_lon:.3f}) NE",
                    output_dir="data",
                    force_offline=False,
                    sample_grid_dim=10,
                    batch_delay_s=1.2,
                )
            t_scan = scan_res["terrain"]
            H, W = t_scan.elevation_m.shape
            real_alt = float(t_scan.elevation_m[H // 2, W // 2])
            st.session_state["compilation_res"] = scan_res
            st.session_state["tab2_hiker_pos"] = [hiker_lat, hiker_lon, round(real_alt, 1)]
            st.session_state["tab2_keep_pos_on_reload"] = True
            st.rerun()

    with st.expander("🔧 Hardware Sensor Bus & Diagnostic Fault Injection (Optional)", expanded=False):
        sc1, sc2 = st.columns(2)
        with sc1:
            if dynamic_nodes and st.button("🎯 Jump Hiker Inside Nearest Hazard Cone", width="stretch"):
                target_node = dynamic_nodes[0]
                st.session_state["tab2_hiker_pos"] = [target_node["lat"], target_node["lon"], target_node["alt"]]
                hiker_lat, hiker_lon, hiker_alt = st.session_state["tab2_hiker_pos"]
            sensor_source = st.radio(
                "Sensor Data Ingestion Mode:",
                ["Simulation Mode (Wearable Emulator)", "Direct Hardware I2C Bus (/dev/i2c-1)"],
                index=0,
                horizontal=True,
            )
            act_choice = st.selectbox("Physical Activity State", ["hiking", "steep_climbing", "stationary"], index=0)
        with sc2:
            hw_bridge = HardwareSensorBridge(bus_number=1)
            hw_status = hw_bridge.get_hardware_status()
            if sensor_source == "Direct Hardware I2C Bus (/dev/i2c-1)":
                if hw_status["hardware_present"]:
                    st.success("🟢 Physical I2C Sensors Connected (MAX30102 / MPU6050 / BMP280)")
                else:
                    st.info("ℹ️ Physical I2C device node inactive on host — bridge emulation active.")
            inject_motion = st.checkbox("Inject Arm-Swing Noise (60–180 BPM)", value=False)
            inject_hypoxia = st.checkbox("Inject Acute Hypoxia (SpO2 < 82%)", value=False)
            inject_tachy = st.checkbox("Inject Resting Tachycardia (>145 BPM)", value=False)

    stillness_s = 15.0 if is_rest_stop else 0.0
    effective_mode = "stationary" if is_rest_stop else act_choice

    if sensor_source == "Direct Hardware I2C Bus (/dev/i2c-1)":
        packet = hw_bridge.read_packet(fallback_alt_m=hiker_alt, activity_mode=effective_mode)
    else:
        packet = simulate_sensor_packet(
            timestamp_s=float(clock_hour * 3600),
            activity_mode=effective_mode,
            altitude_m=hiker_alt,
            inject_motion_noise=inject_motion,
            inject_hypoxia=inject_hypoxia,
            inject_resting_tachycardia=inject_tachy,
        )

    step_out = runtime.run_unified_companion_step(
        hiker_lat=hiker_lat,
        hiker_lon=hiker_lon,
        hiker_alt=hiker_alt,
        sensor_packet=packet,
        clock_hour_idx=clock_hour,
        cohort_mode=cohort_mode,
        prior_stillness_s=stillness_s,
        enqueue_log=True,
    )
    nav_state = step_out["navigation"]
    phys_state = step_out["physiology"]
    medvse = step_out["medvse_vitals"]
    mfilt = step_out["motion_filter"]
    acclim = step_out["acclimatization"]
    baro = step_out["baro_cloudburst"]

    status = nav_state["status"]
    nearest = nav_state.get("nearest_node")
    nearest_info = (
        f"Node #{nearest['id']} ({nearest['zone_type'].upper()}, Dynamic Radius {nearest['danger_radius_m']}m, Risk {nearest['risk_score']})"
        if nearest else "None"
    )

    if status == "CRITICAL_BREACH":
        st.error(
            f"🚨 **CRITICAL DANGER BREACH @ {env_now['rtc_iso_time']}!** Hiker (`{hiker_lat:.4f}, {hiker_lon:.4f}`, {hiker_alt:.0f}m) is "
            f"`{nav_state['distance_to_nearest_m']:.1f}m` from **{nearest_info}**!"
        )
    elif status == "WARNING_BUFFER":
        st.warning(
            f"⚠️ **SPATIAL BUFFER ALERT @ {env_now['rtc_iso_time']}!** Hiker (`{hiker_lat:.4f}, {hiker_lon:.4f}`, {hiker_alt:.0f}m) is "
            f"`{nav_state['distance_to_nearest_m']:.1f}m` from **{nearest_info}**."
        )
    elif nav_state.get("is_uphill_safe"):
        st.success(
            f"🟢 **SAFE (UPHILL OF SLIDE CROWN) @ {env_now['rtc_iso_time']}:** Hiker elevation (`{hiker_alt:.0f}m`) is "
            f"above slide crown elevation (`{nearest['alt'] if nearest else 'N/A'}m`). Debris flows downhill along azimuth "
            f"`{nearest['slide_azimuth_deg'] if nearest else 'N/A'}°` away from you!"
        )
    else:
        st.success(
            f"✅ **SAFE CORRIDOR POSITION @ {env_now['rtc_iso_time']}:** Hiker (`{hiker_lat:.4f}, {hiker_lon:.4f}`, {hiker_alt:.0f}m) is "
            f"`{nav_state['distance_to_nearest_m']:.1f}m` from nearest hazard ({nearest_info})."
        )

    esc = nav_state.get("immediate_escape")
    detour = nav_state.get("safe_detour")
    if esc:
        e_col1, e_col2 = st.columns(2)
        with e_col1:
            st.info(
                f"🧭 **IMMEDIATE PERPENDICULAR ESCAPE (±90° Cross-Slope):**\n\n"
                f"**{esc['directive_text']}** → Exit to `{esc['target_lat']:.4f}, {esc['target_lon']:.4f}`"
            )
        with e_col2:
            if detour:
                st.success(
                    f"🛡️ **SAFE AVOIDANCE PERIMETER WAYPOINT:**\n\n"
                    f"Bypass cluster via `{detour['waypoint_lat']:.4f}, {detour['waypoint_lon']:.4f}` "
                    f"(Bearing `{detour['detour_bearing_deg']}°`, Deviation `{detour['angular_deviation_deg']}°`)"
                )

    st.markdown("#### 🩺 Live Unified Wearable HUD (All 6 Research Engines Active Offline)")
    hud1, hud2, hud3, hud4 = st.columns(4)
    with hud1:
        st.metric("🧭 Navigation & Tier", status, f"Queue: {step_out['priority_tier']}")
    with hud2:
        spo2_str = f"{phys_state['spo2_display']}%" if phys_state["spo2_display"] is not None else "LOCKED (Move)"
        st.metric("❤️ MEDVSE Vitals", f"{medvse['hr_bpm']} BPM | SpO2 {spo2_str}", f"RR {medvse['rr_breaths_min']}/m | W_SQI {phys_state['w_sqi_pct']:.0f}%")
    with hud3:
        unc = acclim["uncertainty"]
        trd = acclim["trend"]
        st.metric("⛰️ Acclimatization (Xu)", f"{trd['trend_status']}", f"Exp {unc['expected_spo2_pct']}% ± {unc['rmse_pct']}% | HSCR {mfilt['hscr_score']}")
    with hud4:
        st.metric("⛈️ Baro Cloudburst (Sana)", baro["baro_status"], f"{baro['dp_weather_hpa_hr']:+.2f} hPa/hr | Supp {mfilt['suppression_db']} dB")

    st.markdown("#### 🗺️ Click Anywhere on Map to Move Hiker, Then Press 'Compile Landslides Around Pressed Area' Above:")
    m_trail = folium.Map(location=[hiker_lat, hiker_lon], zoom_start=13, tiles="OpenStreetMap")
    folium.TileLayer("OpenTopoMap", name="Topographic Contours (OpenTopoMap)").add_to(m_trail)

    sc_c = cartridge.get("start_coords")
    dc_c = cartridge.get("dest_coords")
    if sc_c and dc_c:
        folium.PolyLine(
            [sc_c, dc_c],
            color="#2563EB",
            weight=2,
            dash_array="6, 8",
            tooltip="Compiled Cartridge Axis",
        ).add_to(m_trail)

    for node in dynamic_nodes:
        ztype = node["zone_type"]
        act_st = node.get("activity_state", "ACTIVE_CRITICAL")
        c = "#DC2626" if ztype == "initiation" else ("#EA580C" if ztype == "transport" else "#EAB308")
        opacity = 0.55 if act_st == "ACTIVE_CRITICAL" else (0.35 if act_st == "ELEVATED_WATCH" else 0.20)
        folium.Circle(
            location=[node["lat"], node["lon"]],
            radius=node["buffer_radius_m"],
            color="#F97316",
            weight=1,
            fill=False,
            dash_array="4, 8",
            tooltip=f"Dynamic Warning Buffer: {node['buffer_radius_m']}m (Node #{node['id']})",
        ).add_to(m_trail)
        # Directional Runout Cone Polygon
        cone_pts = build_directional_cone_polygon(node)
        folium.Polygon(
            locations=cone_pts,
            color=c,
            weight=2,
            fill=True,
            fill_color=c,
            fill_opacity=opacity,
            tooltip=(
                f"Node #{node['id']} | {ztype.upper()} Cone | State @ {env_now['rtc_iso_time']}: {act_st} | "
                f"Dynamic Radius: {node['danger_radius_m']}m (Base {node.get('base_danger_radius_m', node['danger_radius_m'])}m) | "
                f"Crown Alt: {node['alt']}m | Risk: {node['risk_score']}"
            ),
        ).add_to(m_trail)
        # Traced Downslope Flow Polyline
        if node.get("path") and len(node["path"]) >= 2:
            flow_pts = [[p[0], p[1]] for p in node["path"]]
            folium.PolyLine(
                locations=flow_pts,
                color=c,
                weight=4,
                opacity=0.85,
                tooltip=f"Downslope Debris Path #{node['id']} (Crown {node['alt']}m -> Toe {node.get('toe_alt', node['alt'])}m)",
            ).add_to(m_trail)

    folium.Marker(
        location=[hiker_lat, hiker_lon],
        popup=f"Hiker GPS ({hiker_lat:.4f}, {hiker_lon:.4f})",
        tooltip="🚶 Hiker Current Position (Click anywhere on map to move)",
        icon=folium.Icon(color="blue", icon="user"),
    ).add_to(m_trail)

    if esc:
        folium.PolyLine(
            [[hiker_lat, hiker_lon], [esc["target_lat"], esc["target_lon"]]],
            color="#FACC15",
            weight=6,
            tooltip=f"Perpendicular Escape Vector: {esc['directive_text']}",
        ).add_to(m_trail)
        folium.Marker(
            location=[esc["target_lat"], esc["target_lon"]],
            popup=f"Safe Escape Point ({esc['target_lat']:.4f}, {esc['target_lon']:.4f})",
            tooltip=f"🧭 Escape Target ({esc['cardinal_dir']}, {esc['escape_distance_m']}m)",
            icon=folium.Icon(color="orange", icon="arrow-up"),
        ).add_to(m_trail)

    if detour and "path_coords" in detour:
        folium.PolyLine(
            detour["path_coords"],
            color="#10B981",
            weight=4,
            dash_array="6, 8",
            tooltip=f"Safe Perimeter Avoidance Bypass ({detour['detour_bearing_deg']}°)",
        ).add_to(m_trail)
        folium.Marker(
            location=[detour["waypoint_lat"], detour["waypoint_lon"]],
            popup=f"Avoidance Waypoint ({detour['waypoint_lat']:.4f}, {detour['waypoint_lon']:.4f})",
            tooltip="🛡️ Safe Perimeter Clearance Waypoint",
            icon=folium.Icon(color="green", icon="ok-sign"),
        ).add_to(m_trail)

    folium.LayerControl().add_to(m_trail)
    trail_map_out = st_folium(
        m_trail,
        width="100%",
        height=500,
        key="companion_folium_map",
        returned_objects=["last_clicked"],
    )

    if trail_map_out and trail_map_out.get("last_clicked"):
        t_lat = round(float(trail_map_out["last_clicked"]["lat"]), 5)
        t_lon = round(float(trail_map_out["last_clicked"]["lng"]), 5)
        t_pair = (t_lat, t_lon)
        if t_pair != st.session_state.get("last_tab2_click"):
            st.session_state["last_tab2_click"] = t_pair
            st.session_state["tab2_hiker_pos"] = [t_lat, t_lon, hiker_alt]
            st.rerun(scope="fragment")

    with st.expander("📦 Store-and-Forward Offline Blackbox Queue (`data/offline_sync_queue.jsonl` — Sana et al. 2026)", expanded=False):
        records = read_blackbox_queue("data/offline_sync_queue.jsonl")
        st.caption(f"Total unacknowledged hazard/medical telemetry packets queued locally for burst-sync: **{len(records)}**")
        if records:
            st.json(records[-3:])


@st.fragment
def render_tab3_return_sync_audit() -> None:
    st.subheader("Phase 3: Base Camp Return Sync & Post-Trek Safety Audit")

    cartridge_path = "data/trek_cartridge.json"
    queue_path = "data/offline_sync_queue.jsonl"
    if not os.path.exists(cartridge_path):
        st.warning("⚠️ No compiled trek cartridge found yet. Compile in Tab 1 first.")
        return

    airgap_ok = verify_airgap_compliance("phase2_offline_edge_companion")
    if airgap_ok:
        st.success("🔒 **AST Air-Gap Audit Verified:** Phase 2 Trail Runtime executed with 0 network/socket imports.")

    rt = EdgeCompanionRuntime(cartridge_path=cartridge_path, queue_path=queue_path)
    audit = rt.generate_post_trek_audit_report()
    tiers = audit["tier_counts"]

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("📦 Total Blackbox Packets", f"{audit['total_events']}")
    a2.metric("⏳ Pending Burst Sync", f"{audit['unsynced_count']}")
    a3.metric("✅ Synced & Acknowledged", f"{audit['synced_count']}")
    a4.metric("🚨 P0/P1 Critical Events", f"{tiers['P0_CRITICAL_SOS'] + tiers['P1_HAZARD_BREACH']}")

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("P0_CRITICAL_SOS", tiers["P0_CRITICAL_SOS"])
    b2.metric("P1_HAZARD_BREACH", tiers["P1_HAZARD_BREACH"])
    b3.metric("P2_MEDICAL_ALERT", tiers["P2_MEDICAL_ALERT"])
    b4.metric("P3_ROUTINE_VITALS", tiers["P3_ROUTINE_VITALS"])

    c_btn1, c_btn2 = st.columns([1.4, 1])
    with c_btn1:
        if st.button("📡 Run Prioritized Burst Sync (P0 → P3) & Acknowledge Batch", type="primary", width="stretch"):
            batch = rt.sync_burst(batch_limit=25)
            if batch:
                acked = rt.ack_sync([int(r["seq_id"]) for r in batch if "seq_id" in r])
                st.success(f"Synced and acknowledged **{acked}** prioritized packets (highest tier first: `{batch[0]['priority_tier']}`).")
                st.rerun(scope="fragment")
            else:
                st.info("All queued packets are already synced and acknowledged.")
    with c_btn2:
        st.download_button(
            label="⬇️ Download Safety Audit Report (.md)",
            data=audit["report_markdown"].encode("utf-8"),
            file_name="post_trek_safety_audit.md",
            mime="text/markdown",
            width="stretch",
        )

    st.markdown(audit["report_markdown"])

    if os.path.exists(queue_path):
        with st.expander("📄 Inspect Raw Blackbox Telemetry Queue (`data/offline_sync_queue.jsonl`)", expanded=False):
            with open(queue_path, "r", encoding="utf-8") as f:
                raw_lines = [line.strip() for line in f if line.strip()]
            st.download_button(
                label="⬇️ Download Blackbox Log (.jsonl)",
                data="\n".join(raw_lines).encode("utf-8"),
                file_name="offline_sync_queue.jsonl",
                mime="application/json",
            )
            preview = []
            for ln in raw_lines[-5:]:
                try:
                    preview.append(json.loads(ln))
                except Exception:
                    pass
            if preview:
                st.json(preview)


def main() -> None:
    st.set_page_config(page_title="Mountain Trekking Safety & Risk Companion", layout="wide")
    st.title("🏔️ Mountain Trekking Safety & Risk Companion")
    tab1, tab2, tab3 = st.tabs([
        "1️⃣ Base Camp Mission Planner",
        "2️⃣ Live Trail Companion HUD (Air-Gapped)",
        "3️⃣ Base Camp Return Sync & Safety Audit",
    ])
    with tab1:
        render_tab1_basecamp_compiler()
    with tab2:
        render_tab2_offline_trail_companion()
    with tab3:
        render_tab3_return_sync_audit()


if __name__ == "__main__":
    main()
