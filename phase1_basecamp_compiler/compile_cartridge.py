from __future__ import annotations
import argparse
import os
from typing import Callable
from phase1_basecamp_compiler.terrain_ingestor import (
    ingest_trek_corridor,
    PRESET_TREK_ROUTES,
    search_place_coordinates,
)
from phase1_basecamp_compiler.tflite_hazard_model import (
    generate_calibrated_training_data,
    build_and_train_hazard_model,
    export_quantized_tflite,
    predict_initiation_grid,
)
from phase1_basecamp_compiler.runout_compiler import (
    propagate_all_runouts,
    compress_sparse_danger_nodes,
    generate_safe_trail_waypoints,
    export_trek_cartridge,
)

DEFAULT_START_COORDS: tuple[float, float] = (10.2145, 77.4935)
DEFAULT_DEST_COORDS: tuple[float, float] = (10.1850, 77.4620)


def run_basecamp_compilation(
    start_coords: tuple[float, float] = DEFAULT_START_COORDS,
    dest_coords: tuple[float, float] = DEFAULT_DEST_COORDS,
    start_name: str = "Trek Start",
    dest_name: str = "Trek Destination",
    output_dir: str = "data",
    force_offline: bool = False,
    sample_grid_dim: int = 10,
    batch_delay_s: float = 1.2,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    cartridge_path = os.path.join(output_dir, "trek_cartridge.json")
    model_path = os.path.join(output_dir, "landslide_init.tflite")

    terrain = ingest_trek_corridor(
        start_coords,
        dest_coords,
        buffer_km=2.0,
        force_offline_fallback=force_offline,
        target_grid_size=50,
        sample_grid_dim=sample_grid_dim,
        batch_delay_s=batch_delay_s,
        progress_callback=progress_callback,
    )

    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        model_size = os.path.getsize(model_path)
    else:
        X_train, y_train = generate_calibrated_training_data(n_samples=3000)
        model = build_and_train_hazard_model(X_train, y_train, epochs=10)
        model_size = export_quantized_tflite(model, model_path)

    p_init_grid = predict_initiation_grid(model_path, terrain)

    raw_nodes = propagate_all_runouts(terrain, p_init_grid, threshold=0.75)
    if not raw_nodes:
        raw_nodes = propagate_all_runouts(terrain, p_init_grid, threshold=0.55)
    if not raw_nodes:
        raw_nodes = propagate_all_runouts(terrain, p_init_grid, threshold=0.35)
    danger_nodes = compress_sparse_danger_nodes(raw_nodes)

    trail_waypoints = generate_safe_trail_waypoints(start_coords, dest_coords, terrain)

    baseline_params = {
        "resting_hr_bpm": 68.0,
        "max_hr_bpm": 178.0,
        "fitness_level": "intermediate",
        "baseline_spo2_sea_level": 98.0,
        "start_name": start_name,
        "dest_name": dest_name,
        "source_label": terrain.source_label,
        "corridor_length_km": terrain.corridor_length_km,
        "antecedent_7d_rain_mm": terrain.antecedent_7d_rain_mm,
        "forecast_7d_rain_mm": terrain.forecast_7d_rain_mm,
        "live_rain_min_mm": terrain.live_rain_min_mm,
        "live_rain_max_mm": terrain.live_rain_max_mm,
    }

    cartridge_meta = export_trek_cartridge(
        output_path=cartridge_path,
        start_coords=start_coords,
        dest_coords=dest_coords,
        danger_nodes=danger_nodes,
        trail_waypoints=trail_waypoints,
        baseline_params=baseline_params,
        hourly_timeline=terrain.hourly_timeline,
    )

    return {
        "terrain": terrain,
        "p_init_grid": p_init_grid,
        "model_size_bytes": model_size,
        "cartridge_meta": cartridge_meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Base Camp Trek Compiler (Real Open-Meteo SRTM + 168h Forecast)")
    parser.add_argument("--offline-demo", action="store_true", help="Offline unit test fixture mode")
    parser.add_argument("--grid-dim", type=int, default=None, help="DEM sample grid dimension (e.g. 10, 20, 30, 50)")
    parser.add_argument("--place", default=None, help="Preset trek place name or search keyword (e.g. Kodaikanal, Ooty, Munnar)")
    parser.add_argument("--start-lat", type=float, default=None, help="Start latitude")
    parser.add_argument("--start-lon", type=float, default=None, help="Start longitude")
    parser.add_argument("--end-lat", type=float, default=None, help="Destination latitude")
    parser.add_argument("--end-lon", type=float, default=None, help="Destination longitude")
    parser.add_argument("--density", choices=["rough", "standard", "medium", "high", "ultra"], default="standard", help="Corridor grid density")
    parser.add_argument("--outdir", default="data", help="Output directory")
    args = parser.parse_args()

    density_dim_map = {"rough": 8, "standard": 10, "medium": 20, "high": 30, "ultra": 50}
    dim = args.grid_dim if args.grid_dim else density_dim_map[args.density]

    start_c = DEFAULT_START_COORDS
    dest_c = DEFAULT_DEST_COORDS
    start_n = "Trek Start"
    dest_n = "Trek Destination"

    if args.place:
        matched = False
        for preset_name, data in PRESET_TREK_ROUTES.items():
            if args.place.lower() in preset_name.lower():
                start_c = data["start_coords"]
                dest_c = data["dest_coords"]
                start_n = data["start_name"]
                dest_n = data["dest_name"]
                matched = True
                break
        if not matched:
            results = search_place_coordinates(args.place)
            if results:
                r = results[0]
                start_c = (r["lat"], r["lon"])
                dest_c = (r["lat"] - 0.03, r["lon"] + 0.03)
                start_n = r["name"]
                dest_n = f"{r['name']} Ridge"

    if args.start_lat is not None and args.start_lon is not None:
        start_c = (args.start_lat, args.start_lon)
    if args.end_lat is not None and args.end_lon is not None:
        dest_c = (args.end_lat, args.end_lon)

    res = run_basecamp_compilation(
        start_coords=start_c,
        dest_coords=dest_c,
        start_name=start_n,
        dest_name=dest_n,
        output_dir=args.outdir,
        force_offline=args.offline_demo,
        sample_grid_dim=dim,
    )
    print(f"Compilation complete. Source: {res['terrain'].source_label}, Cartridge size: {res['cartridge_meta']['size_bytes']} bytes")


if __name__ == "__main__":
    main()
