from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import json
import os
import time
import numpy as np
import requests
from scipy.ndimage import zoom

OPEN_METEO_GEOCODING_URL: str = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_ELEVATION_URL: str = "https://api.open-meteo.com/v1/elevation"
OPEN_METEO_FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"
EARTH_RADIUS_M: float = 6_371_000.0
METERS_PER_DEG_LAT: float = 111_320.0
DEFAULT_CORRIDOR_BUFFER_KM: float = 2.0
DEFAULT_GRID_SPACING_M: float = 90.0
BATCH_SIZE_COORDS: int = 100
CACHE_FILE_PATH: str = "data/open_meteo_cache.json"

PRESET_TREK_ROUTES: dict[str, dict] = {
    "Kodaikanal (Dolphin's Nose to Vellagavi, Palani Hills)": {
        "start_name": "Dolphin's Nose (Kodaikanal)",
        "dest_name": "Vellagavi Village",
        "start_coords": (10.2145, 77.4935),
        "dest_coords": (10.1850, 77.4620),
    },
    "Ooty Nilgiris (Doddabetta Peak to Avalanche Valley)": {
        "start_name": "Doddabetta Peak (Ooty)",
        "dest_name": "Avalanche Valley",
        "start_coords": (11.4007, 76.7358),
        "dest_coords": (11.3180, 76.6250),
    },
    "Ooty to Kodaikanal Regional Ghat Corridor": {
        "start_name": "Ooty Town Center",
        "dest_name": "Kodaikanal Lake",
        "start_coords": (11.4102, 76.6950),
        "dest_coords": (10.2381, 77.4892),
    },
    "Munnar Western Ghats (Anamudi to Top Station)": {
        "start_name": "Anamudi Base",
        "dest_name": "Top Station Munnar",
        "start_coords": (10.1697, 77.0609),
        "dest_coords": (10.1245, 77.2458),
    },
    "Uttarkashi–Gangotri Highway (Swain et al. 2026 Baseline)": {
        "start_name": "Uttarkashi Highway Corridor",
        "dest_name": "Gangotri Valley Base",
        "start_coords": (30.9960, 78.9390),
        "dest_coords": (31.0280, 78.9850),
    },
    "Mont Blanc Alps (Chamonix to Courmayeur, France/Italy)": {
        "start_name": "Chamonix-Mont-Blanc",
        "dest_name": "Courmayeur Valley",
        "start_coords": (45.9237, 6.8694),
        "dest_coords": (45.7969, 6.9664),
    },
    "Mount Whitney Trail (Sierra Nevada, California, USA)": {
        "start_name": "Whitney Portal",
        "dest_name": "Mount Whitney Summit",
        "start_coords": (36.5870, -118.2394),
        "dest_coords": (36.5785, -118.2920),
    },
    "Annapurna Sanctuary (Pokhara to ABC, Nepal Himalayas)": {
        "start_name": "Machapuchare Base Camp",
        "dest_name": "Annapurna Base Camp",
        "start_coords": (28.5000, 83.8700),
        "dest_coords": (28.5300, 83.8780),
    },
    "Mount Fuji Yoshida Route (Yamanashi, Japan)": {
        "start_name": "Fuji 5th Station",
        "dest_name": "Mount Fuji Crater",
        "start_coords": (35.3942, 138.7450),
        "dest_coords": (35.3606, 138.7274),
    },
    "Inca Trail to Machu Picchu (Andes, Peru)": {
        "start_name": "KM 82 Piscacucho",
        "dest_name": "Machu Picchu Sun Gate",
        "start_coords": (-13.2500, -72.4167),
        "dest_coords": (-13.1631, -72.5450),
    },
}

OFFLINE_PLACES_DB: list[dict] = [
    {"name": "Kodaikanal", "admin1": "Tamil Nadu", "country": "India", "lat": 10.2381, "lon": 77.4892, "elevation": 2133.0, "display_name": "Kodaikanal, Tamil Nadu, India"},
    {"name": "Dolphin's Nose", "admin1": "Tamil Nadu", "country": "India", "lat": 10.2145, "lon": 77.4935, "elevation": 2040.0, "display_name": "Dolphin's Nose, Kodaikanal, India"},
    {"name": "Vellagavi", "admin1": "Tamil Nadu", "country": "India", "lat": 10.1850, "lon": 77.4620, "elevation": 1420.0, "display_name": "Vellagavi, Palani Hills, India"},
    {"name": "Ooty", "admin1": "Tamil Nadu", "country": "India", "lat": 11.4102, "lon": 76.6950, "elevation": 2240.0, "display_name": "Udhagamandalam (Ooty), Tamil Nadu, India"},
    {"name": "Coonoor", "admin1": "Tamil Nadu", "country": "India", "lat": 11.3530, "lon": 76.7959, "elevation": 1850.0, "display_name": "Coonoor, Nilgiris, Tamil Nadu, India"},
    {"name": "Doddabetta Peak", "admin1": "Tamil Nadu", "country": "India", "lat": 11.4007, "lon": 76.7358, "elevation": 2637.0, "display_name": "Doddabetta Peak, Nilgiris, India"},
    {"name": "Avalanche Valley", "admin1": "Tamil Nadu", "country": "India", "lat": 11.3180, "lon": 76.6250, "elevation": 1980.0, "display_name": "Avalanche Valley, Nilgiris, India"},
    {"name": "Munnar", "admin1": "Kerala", "country": "India", "lat": 10.0889, "lon": 77.0595, "elevation": 1532.0, "display_name": "Munnar, Idukki, Kerala, India"},
    {"name": "Top Station", "admin1": "Kerala", "country": "India", "lat": 10.1245, "lon": 77.2458, "elevation": 1880.0, "display_name": "Top Station, Western Ghats, India"},
    {"name": "Manali", "admin1": "Himachal Pradesh", "country": "India", "lat": 32.2432, "lon": 77.1892, "elevation": 2050.0, "display_name": "Manali, Himachal Pradesh, India"},
    {"name": "Uttarkashi", "admin1": "Uttarakhand", "country": "India", "lat": 30.7268, "lon": 78.4354, "elevation": 1158.0, "display_name": "Uttarkashi, Uttarakhand, India"},
    {"name": "Gangotri", "admin1": "Uttarakhand", "country": "India", "lat": 30.9947, "lon": 78.9398, "elevation": 3100.0, "display_name": "Gangotri, Uttarakhand, India"},
    {"name": "Kedarnath", "admin1": "Uttarakhand", "country": "India", "lat": 30.7352, "lon": 79.0669, "elevation": 3584.0, "display_name": "Kedarnath, Uttarakhand, India"},
    {"name": "Chamonix", "admin1": "Haute-Savoie", "country": "France", "lat": 45.9237, "lon": 6.8694, "elevation": 1035.0, "display_name": "Chamonix-Mont-Blanc, France"},
    {"name": "Mount Whitney", "admin1": "California", "country": "United States", "lat": 36.5785, "lon": -118.2920, "elevation": 4421.0, "display_name": "Mount Whitney, California, USA"},
    {"name": "Mount Fuji", "admin1": "Shizuoka", "country": "Japan", "lat": 35.3606, "lon": 138.7274, "elevation": 3776.0, "display_name": "Mount Fuji, Honshu, Japan"},
    {"name": "Annapurna", "admin1": "Gandaki", "country": "Nepal", "lat": 28.5960, "lon": 83.8203, "elevation": 4130.0, "display_name": "Annapurna Sanctuary, Nepal"},
    {"name": "Machu Picchu", "admin1": "Cusco", "country": "Peru", "lat": -13.1631, "lon": -72.5450, "elevation": 2430.0, "display_name": "Machu Picchu, Cusco, Peru"},
    {"name": "Mount Kilimanjaro", "admin1": "Kilimanjaro", "country": "Tanzania", "lat": -3.0674, "lon": 37.3556, "elevation": 5895.0, "display_name": "Mount Kilimanjaro, Tanzania"},
]


@dataclass
class TerrainGrid:
    lat_grid: np.ndarray
    lon_grid: np.ndarray
    x_m_grid: np.ndarray
    y_m_grid: np.ndarray
    elevation_m: np.ndarray
    slope_deg: np.ndarray
    aspect_deg: np.ndarray
    curvature: np.ndarray
    cumulative_rain_mm: np.ndarray
    resolution_m: float
    is_fallback: bool
    source_label: str = "OPEN_METEO_LIVE_SRTM_AND_HOURLY_FORECAST"
    antecedent_7d_rain_mm: float = 0.0
    forecast_7d_rain_mm: float = 0.0
    live_rain_min_mm: float = 0.0
    live_rain_max_mm: float = 0.0
    corridor_length_km: float = 0.0
    hourly_timeline: list[dict] = field(default_factory=list)


def _load_api_cache() -> dict:
    if os.path.exists(CACHE_FILE_PATH):
        try:
            with open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_api_cache(cache: dict) -> None:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(CACHE_FILE_PATH)), exist_ok=True)
        with open(CACHE_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass


def search_place_coordinates(query: str, count: int = 5, timeout_s: float = 6.0) -> list[dict]:
    query_clean = query.strip()
    if not query_clean:
        return []
    try:
        resp = requests.get(
            OPEN_METEO_GEOCODING_URL,
            params={"name": query_clean, "count": count, "language": "en", "format": "json"},
            timeout=timeout_s,
        )
        if resp.status_code == 200:
            data = resp.json().get("results", [])
            results = []
            for item in data:
                name = item.get("name", "")
                admin = item.get("admin1", "")
                country = item.get("country", "")
                disp = f"{name}, {admin}, {country}".strip(", ")
                results.append({
                    "name": name,
                    "admin1": admin,
                    "country": country,
                    "lat": round(float(item["latitude"]), 5),
                    "lon": round(float(item["longitude"]), 5),
                    "elevation": float(item.get("elevation", 0.0)),
                    "display_name": disp,
                })
            if results:
                return results
    except Exception:
        pass

    q_lower = query_clean.lower()
    matches = [p for p in OFFLINE_PLACES_DB if q_lower in p["name"].lower() or q_lower in p["display_name"].lower()]
    return matches[:count]


def build_corridor_grid(
    start_coords: tuple[float, float],
    dest_coords: tuple[float, float],
    buffer_km: float = DEFAULT_CORRIDOR_BUFFER_KM,
    resolution_m: float = DEFAULT_GRID_SPACING_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    min_lat = min(start_coords[0], dest_coords[0])
    max_lat = max(start_coords[0], dest_coords[0])
    min_lon = min(start_coords[1], dest_coords[1])
    max_lon = max(start_coords[1], dest_coords[1])

    mean_lat = (min_lat + max_lat) / 2.0
    lat_buffer_deg = (buffer_km * 1000.0) / METERS_PER_DEG_LAT
    meters_per_deg_lon = METERS_PER_DEG_LAT * np.cos(np.radians(mean_lat))
    lon_buffer_deg = (buffer_km * 1000.0) / max(meters_per_deg_lon, 1.0)

    start_lat = min_lat - lat_buffer_deg
    end_lat = max_lat + lat_buffer_deg
    start_lon = min_lon - lon_buffer_deg
    end_lon = max_lon + lon_buffer_deg

    dlat = resolution_m / METERS_PER_DEG_LAT
    dlon = resolution_m / max(meters_per_deg_lon, 1.0)

    lats = np.arange(start_lat, end_lat + dlat * 0.5, dlat)
    lons = np.arange(start_lon, end_lon + dlon * 0.5, dlon)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    x_m_grid = (lon_grid - start_lon) * meters_per_deg_lon
    y_m_grid = (lat_grid - start_lat) * METERS_PER_DEG_LAT
    return lat_grid, lon_grid, x_m_grid, y_m_grid


def fetch_open_meteo_elevation_batch(
    lats: list[float],
    lons: list[float],
    timeout_s: float = 15.0,
    batch_delay_s: float = 1.2,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[float]:
    elevations: list[float] = []
    total_batches = max(1, (len(lats) + BATCH_SIZE_COORDS - 1) // BATCH_SIZE_COORDS)

    for b_idx, i in enumerate(range(0, len(lats), BATCH_SIZE_COORDS), start=1):
        if b_idx > 1 and batch_delay_s > 0:
            time.sleep(batch_delay_s)

        if progress_callback is not None:
            progress_callback(b_idx, total_batches, f"Fetching real SRTM DEM batch {b_idx}/{total_batches} ({len(elevations)}/{len(lats)} pts)...")

        b_lats = lats[i : i + BATCH_SIZE_COORDS]
        b_lons = lons[i : i + BATCH_SIZE_COORDS]
        lat_str = ",".join(f"{x:.4f}" for x in b_lats)
        lon_str = ",".join(f"{x:.4f}" for x in b_lons)
        params = {"latitude": lat_str, "longitude": lon_str}

        last_err = None
        for attempt in range(6):
            resp = requests.get(OPEN_METEO_ELEVATION_URL, params=params, timeout=timeout_s)
            if resp.status_code == 200:
                data = resp.json()
                elev = data.get("elevation")
                if elev is None:
                    raise RuntimeError("No elevation in response")
                if isinstance(elev, list):
                    elevations.extend([float(e) if e is not None else 0.0 for e in elev])
                else:
                    elevations.append(float(elev))
                last_err = None
                break
            elif resp.status_code == 429:
                wait_s = 3.0 * (attempt + 1)
                if progress_callback is not None:
                    progress_callback(b_idx, total_batches, f"Server rate-limit pause: waiting {wait_s:.0f}s before batch {b_idx}/{total_batches}...")
                last_err = "Elevation API rate limit (429)"
                time.sleep(wait_s)
            else:
                raise RuntimeError(f"Elevation API HTTP {resp.status_code}")
        if last_err is not None:
            raise RuntimeError(last_err)
    return elevations


def fetch_open_meteo_rainfall_batch(
    lats: list[float],
    lons: list[float],
    forecast_days: int = 7,
    timeout_s: float = 15.0,
) -> list[float]:
    rainfalls: list[float] = []
    for i in range(0, len(lats), BATCH_SIZE_COORDS):
        b_lats = lats[i : i + BATCH_SIZE_COORDS]
        b_lons = lons[i : i + BATCH_SIZE_COORDS]
        lat_str = ",".join(f"{x:.4f}" for x in b_lats)
        lon_str = ",".join(f"{x:.4f}" for x in b_lons)
        params = {
            "latitude": lat_str,
            "longitude": lon_str,
            "daily": "precipitation_sum",
            "forecast_days": forecast_days,
            "timezone": "auto",
        }
        last_err = None
        for attempt in range(5):
            resp = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=timeout_s)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    for item in data:
                        precip = item.get("daily", {}).get("precipitation_sum", [0.0])
                        rainfalls.append(float(np.sum([p for p in precip if p is not None])))
                else:
                    precip = data.get("daily", {}).get("precipitation_sum", [0.0])
                    rainfalls.append(float(np.sum([p for p in precip if p is not None])))
                last_err = None
                break
            elif resp.status_code == 429:
                last_err = "Forecast API rate limit (429)"
                time.sleep(2.5 * (attempt + 1))
            else:
                raise RuntimeError(f"Forecast API HTTP {resp.status_code}")
        if last_err is not None:
            raise RuntimeError(last_err)
    return rainfalls


def fetch_open_meteo_hourly_timeline(
    lat: float,
    lon: float,
    past_days: int = 7,
    forecast_days: int = 7,
    timeout_s: float = 15.0,
) -> dict:
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "hourly": "precipitation,rain,soil_moisture_0_to_7cm,temperature_2m,weather_code",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "auto",
    }
    last_err = None
    data = None
    for attempt in range(5):
        resp = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=timeout_s)
        if resp.status_code == 200:
            data = resp.json()
            last_err = None
            break
        elif resp.status_code == 429:
            last_err = "Hourly Forecast API rate limit (429)"
            time.sleep(2.5 * (attempt + 1))
        else:
            raise RuntimeError(f"Hourly Forecast API HTTP {resp.status_code}")
    if last_err is not None or data is None:
        raise RuntimeError(last_err or "Empty forecast response")

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    precips = [float(p) if p is not None else 0.0 for p in hourly.get("precipitation", [])]
    soil_m = [float(s) if s is not None else 0.25 for s in hourly.get("soil_moisture_0_to_7cm", [])]
    temps = [float(t) if t is not None else 15.0 for t in hourly.get("temperature_2m", [])]
    wcodes = [int(w) if w is not None else 0 for w in hourly.get("weather_code", [])]

    split_idx = past_days * 24
    past_precips = precips[:split_idx]
    fc_times = times[split_idx:]
    fc_precips = precips[split_idx:]
    fc_soil = soil_m[split_idx:]
    fc_temps = temps[split_idx:]
    fc_wcodes = wcodes[split_idx:]

    antecedent_7d_mm = float(np.sum(past_precips))
    forecast_7d_mm = float(np.sum(fc_precips))

    # Compute exponentially-decayed rolling pore-water rainfall memory (tau = 168h) across past + forecast
    tau_hours = 168.0
    rolling_rain = 0.0
    decay = float(np.exp(-1.0 / tau_hours))
    for p_val in past_precips:
        rolling_rain = rolling_rain * decay + p_val

    # Also account for baseline soil moisture saturation contribution (m^3/m^3 -> equivalent mm pore water)
    timeline: list[dict] = []
    n_fc = len(fc_times)
    base_rolling = max(rolling_rain, 1.0)

    for h_idx in range(n_fc):
        p_hr = fc_precips[h_idx]
        sm_hr = fc_soil[h_idx] if h_idx < len(fc_soil) else 0.25
        t_hr = fc_temps[h_idx] if h_idx < len(fc_temps) else 15.0
        wc_hr = fc_wcodes[h_idx] if h_idx < len(fc_wcodes) else 0

        rolling_rain = rolling_rain * decay + p_hr
        effective_pore_rain_mm = rolling_rain + (sm_hr * 110.0)

        # Dynamic hazard multiplier wrto time (1.0 at departure baseline, scales with rain + soil moisture + active downpour)
        rain_ratio = effective_pore_rain_mm / max(base_rolling + (fc_soil[0] * 110.0 if fc_soil else 25.0), 10.0)
        intensity_boost = 0.15 * min(p_hr, 15.0)
        hazard_scale = round(float(np.clip(rain_ratio + intensity_boost, 0.70, 2.20)), 2)

        iso_short = fc_times[h_idx].replace("T", " ")
        timeline.append({
            "h": h_idx,
            "t": iso_short,
            "p": round(p_hr, 2),
            "r": round(effective_pore_rain_mm, 1),
            "sm": round(sm_hr, 3),
            "tc": round(t_hr, 1),
            "wc": wc_hr,
            "s": hazard_scale,
        })

    return {
        "antecedent_7d_rain_mm": round(antecedent_7d_mm, 1),
        "forecast_7d_rain_mm": round(forecast_7d_mm, 1),
        "timeline": timeline,
    }


def generate_synthetic_himalayan_dem(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    start_coords: tuple[float, float],
    dest_coords: tuple[float, float],
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mean_lat = float(np.mean(lat_grid))
    m_lat = METERS_PER_DEG_LAT
    m_lon = METERS_PER_DEG_LAT * np.cos(np.radians(mean_lat))

    x = (lon_grid - start_coords[1]) * m_lon
    y = (lat_grid - start_coords[0]) * m_lat

    dx_trail = (dest_coords[1] - start_coords[1]) * m_lon
    dy_trail = (dest_coords[0] - start_coords[0]) * m_lat
    L_sq = dx_trail**2 + dy_trail**2 + 1e-6

    proj = (x * dx_trail + y * dy_trail) / L_sq
    proj = np.clip(proj, -0.2, 1.2)

    cx = proj * dx_trail
    cy = proj * dy_trail
    dist_to_valley = np.hypot(x - cx, y - cy)

    base_elev = 2600.0 + proj * 800.0
    wall_profile = 750.0 / (1.0 + np.exp(-(dist_to_valley - 350.0) / 120.0))
    noise = rng.normal(0, 15.0, size=lat_grid.shape)
    elevation_m = base_elev + wall_profile + noise

    storm_center_dist = np.hypot(x - dx_trail * 0.6, y - dy_trail * 0.6)
    rain_base = 250.0 * np.exp(-(storm_center_dist**2) / (2 * (600.0**2)))
    ridge_orographic = np.clip((elevation_m - 2800.0) / 4.0, 0, 120.0)
    cumulative_rain_mm = rain_base + ridge_orographic + rng.uniform(5.0, 25.0, size=lat_grid.shape)
    return elevation_m.astype(np.float32), cumulative_rain_mm.astype(np.float32)


def compute_topographic_derivatives(
    elevation_m: np.ndarray,
    dx_m: float,
    dy_m: float,
    escarpment_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dz_dy, dz_dx = np.gradient(elevation_m, dy_m, dx_m)
    dz_dx *= escarpment_factor
    dz_dy *= escarpment_factor

    slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
    slope_deg = np.degrees(slope_rad)

    aspect_rad = np.arctan2(-dz_dx, -dz_dy)
    aspect_deg = np.degrees(aspect_rad) % 360.0

    d2z_dy2 = np.gradient(dz_dy, dy_m, axis=0)
    d2z_dx2 = np.gradient(dz_dx, dx_m, axis=1)
    curvature = (d2z_dx2 + d2z_dy2).astype(np.float32)

    return slope_deg.astype(np.float32), aspect_deg.astype(np.float32), curvature


def ingest_trek_corridor(
    start_coords: tuple[float, float],
    dest_coords: tuple[float, float],
    buffer_km: float = DEFAULT_CORRIDOR_BUFFER_KM,
    resolution_m: float = DEFAULT_GRID_SPACING_M,
    force_offline_fallback: bool = False,
    target_grid_size: int = 50,
    sample_grid_dim: int = 10,
    batch_delay_s: float = 1.2,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> TerrainGrid:
    lat_dist_km = abs(dest_coords[0] - start_coords[0]) * 111.32
    mean_lat = (start_coords[0] + dest_coords[0]) / 2.0
    mean_lon = (start_coords[1] + dest_coords[1]) / 2.0
    lon_dist_km = abs(dest_coords[1] - start_coords[1]) * 111.32 * np.cos(np.radians(mean_lat))
    corridor_len_km = round(float(np.hypot(lat_dist_km, lon_dist_km)), 2)

    lat_pad = (buffer_km * 1000.0) / METERS_PER_DEG_LAT
    lon_pad = (buffer_km * 1000.0) / max(METERS_PER_DEG_LAT * np.cos(np.radians(mean_lat)), 1.0)

    min_lat = min(start_coords[0], dest_coords[0]) - lat_pad
    max_lat = max(start_coords[0], dest_coords[0]) + lat_pad
    min_lon = min(start_coords[1], dest_coords[1]) - lon_pad
    max_lon = max(start_coords[1], dest_coords[1]) + lon_pad

    lats_target = np.linspace(min_lat, max_lat, target_grid_size)
    lons_target = np.linspace(min_lon, max_lon, target_grid_size)
    lat_grid, lon_grid = np.meshgrid(lats_target, lons_target, indexing="ij")

    m_lat = METERS_PER_DEG_LAT
    m_lon = METERS_PER_DEG_LAT * np.cos(np.radians(mean_lat))
    x_m_grid = (lon_grid - min_lon) * m_lon
    y_m_grid = (lat_grid - min_lat) * m_lat

    effective_dx = (max_lon - min_lon) * m_lon / max(target_grid_size - 1, 1)
    effective_dy = (max_lat - min_lat) * m_lat / max(target_grid_size - 1, 1)

    if force_offline_fallback:
        elevation_m, cumulative_rain_mm = generate_synthetic_himalayan_dem(
            lat_grid, lon_grid, start_coords, dest_coords
        )
        escarpment_factor = float(np.clip(np.sqrt(effective_dx / 45.0), 1.0, 6.0))
        slope_deg, aspect_deg, curvature = compute_topographic_derivatives(
            elevation_m, effective_dx, effective_dy, escarpment_factor=escarpment_factor
        )
        return TerrainGrid(
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            x_m_grid=x_m_grid,
            y_m_grid=y_m_grid,
            elevation_m=elevation_m,
            slope_deg=slope_deg,
            aspect_deg=aspect_deg,
            curvature=curvature,
            cumulative_rain_mm=cumulative_rain_mm,
            resolution_m=effective_dx,
            is_fallback=True,
            source_label="OFFLINE_TEST_FIXTURE",
            corridor_length_km=corridor_len_km,
        )

    # 100% REAL OPEN-METEO DATA — NO SYNTHETIC FALLBACKS
    cache_key = f"v2_{min_lat:.3f}_{max_lat:.3f}_{min_lon:.3f}_{max_lon:.3f}_{sample_grid_dim}"
    cache = _load_api_cache()

    if cache_key in cache and "hourly_wx" in cache[cache_key]:
        elev_coarse = np.array(cache[cache_key]["elev_coarse"], dtype=np.float32)
        wx_data = cache[cache_key]["hourly_wx"]
    else:
        coarse_lats = np.linspace(min_lat, max_lat, sample_grid_dim)
        coarse_lons = np.linspace(min_lon, max_lon, sample_grid_dim)
        c_lat_g, c_lon_g = np.meshgrid(coarse_lats, coarse_lons, indexing="ij")

        elev_coarse_flat = fetch_open_meteo_elevation_batch(
            c_lat_g.flatten().tolist(),
            c_lon_g.flatten().tolist(),
            batch_delay_s=batch_delay_s,
            progress_callback=progress_callback,
        )
        elev_coarse = np.array(elev_coarse_flat, dtype=np.float32).reshape(sample_grid_dim, sample_grid_dim)

        if progress_callback is not None:
            progress_callback(1, 1, "Fetching real 14-day hourly weather & soil moisture timeline...")
        wx_data = fetch_open_meteo_hourly_timeline(mean_lat, mean_lon)

        cache[cache_key] = {
            "elev_coarse": elev_coarse.tolist(),
            "hourly_wx": wx_data,
        }
        _save_api_cache(cache)

    zoom_factor = target_grid_size / sample_grid_dim
    elevation_m = zoom(elev_coarse, zoom_factor, order=3).astype(np.float32)[:target_grid_size, :target_grid_size]

    timeline = wx_data["timeline"]
    antecedent_mm = float(wx_data["antecedent_7d_rain_mm"])
    forecast_mm = float(wx_data["forecast_7d_rain_mm"])
    peak_effective_rain_mm = max((h["r"] for h in timeline), default=antecedent_mm + forecast_mm)

    elev_norm = (elevation_m - np.min(elevation_m)) / max(float(np.ptp(elevation_m)), 1.0)
    # Real effective pore-water rainfall across corridor topography (orographic distribution of real rain)
    cumulative_rain_mm = (peak_effective_rain_mm * (0.85 + 0.35 * elev_norm)).astype(np.float32)

    escarpment_factor = float(np.clip(np.sqrt(effective_dx / 45.0), 1.0, 6.0))
    slope_deg, aspect_deg, curvature = compute_topographic_derivatives(
        elevation_m, effective_dx, effective_dy, escarpment_factor=escarpment_factor
    )

    return TerrainGrid(
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        x_m_grid=x_m_grid,
        y_m_grid=y_m_grid,
        elevation_m=elevation_m,
        slope_deg=slope_deg,
        aspect_deg=aspect_deg,
        curvature=curvature,
        cumulative_rain_mm=cumulative_rain_mm,
        resolution_m=effective_dx,
        is_fallback=False,
        source_label="OPEN_METEO_LIVE_SRTM_AND_HOURLY_FORECAST",
        antecedent_7d_rain_mm=antecedent_mm,
        forecast_7d_rain_mm=forecast_mm,
        live_rain_min_mm=round(float(np.min(cumulative_rain_mm)), 1),
        live_rain_max_mm=round(float(np.max(cumulative_rain_mm)), 1),
        corridor_length_km=corridor_len_km,
        hourly_timeline=timeline,
    )
