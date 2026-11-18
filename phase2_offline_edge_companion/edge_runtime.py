from __future__ import annotations
import ast
import os
from phase2_offline_edge_companion.offline_evasion_nav import (
    load_trek_cartridge,
    evaluate_hiker_hazard_state,
)
from phase2_offline_edge_companion.physiological_companion import (
    process_physiological_step,
    evaluate_imu_spo2_gate,
    compute_w_sqi,
    evaluate_activity_aware_exertion,
)
from phase2_offline_edge_companion.motion_artifact_filter import (
    compute_imu_hscr,
    cancel_motion_artifact_cross_film,
)
from phase2_offline_edge_companion.ppg_vital_tflite import (
    infer_vitals_from_ppg_window,
)
from phase2_offline_edge_companion.altitude_acclimatization import (
    compute_altitude_spo2_uncertainty,
    evaluate_longitudinal_spo2_trend,
)
from phase2_offline_edge_companion.blackbox_sync_manager import (
    detect_barometric_cloudburst_anomaly,
    enqueue_prioritized_telemetry,
    prepare_burst_sync_batch,
    acknowledge_synced_packets,
)

FORBIDDEN_NETWORK_MODULES: tuple[str, ...] = (
    "requests",
    "urllib",
    "http",
    "socket",
    "aiohttp",
    "httpx",
)


def verify_airgap_compliance(package_dir: str) -> bool:
    for root, _, files in os.walk(package_dir):
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8") as src:
                tree = ast.parse(src.read(), filename=path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if top in FORBIDDEN_NETWORK_MODULES:
                            raise RuntimeError(f"Air-gap violation in {path}: imports {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        top = node.module.split(".")[0]
                        if top in FORBIDDEN_NETWORK_MODULES:
                            raise RuntimeError(f"Air-gap violation in {path}: imports from {node.module}")
    return True


class EdgeCompanionRuntime:
    def __init__(
        self,
        cartridge_path: str,
        queue_path: str = "data/offline_sync_queue.jsonl",
        exertion_tflite_path: str | None = None,
    ) -> None:
        self.cartridge_path = cartridge_path
        self.queue_path = queue_path
        self.exertion_tflite_path = exertion_tflite_path
        self.cartridge = load_trek_cartridge(cartridge_path)
        self.stillness_s = 0.0
        self.clock_hour_idx = 0
        self.spo2_history: list[dict] = []
        self.pressure_history_hpa: list[tuple[float, float]] = []
        self.altitude_history_m: list[tuple[float, float]] = []

    def step(
        self,
        hiker_lat: float,
        hiker_lon: float,
        hiker_alt: float,
        sensor_packet: dict,
        clock_hour_idx: int | None = None,
    ) -> dict:
        if clock_hour_idx is not None:
            self.clock_hour_idx = clock_hour_idx

        nav_state = evaluate_hiker_hazard_state(
            hiker_lat=hiker_lat,
            hiker_lon=hiker_lon,
            hiker_alt=hiker_alt,
            cartridge=self.cartridge,
            clock_hour_idx=self.clock_hour_idx,
        )

        in_hazard = nav_state["status"] in ("CRITICAL_BREACH", "WARNING_BUFFER")
        baseline = self.cartridge.get("baseline_params", {"resting_hr_bpm": 68.0})

        physio_state = process_physiological_step(
            sensor_packet=sensor_packet,
            prior_stillness_s=self.stillness_s,
            in_hazard_zone=in_hazard,
            baseline_params=baseline,
            queue_path=self.queue_path,
            tflite_model_path=self.exertion_tflite_path,
        )

        self.stillness_s = float(physio_state["spo2_gate"]["stillness_duration_s"])

        return {
            "navigation": nav_state,
            "physiology": physio_state,
            "clock_hour_idx": self.clock_hour_idx,
        }

    def evaluate_motion_regime(self, imu_accel_mag, fs_hz: float = 25.0) -> dict:
        return compute_imu_hscr(imu_accel_mag, fs_hz=fs_hz)

    def filter_motion_artifact(self, ppg_waveform, imu_accel_xyz, fs_hz: float = 25.0) -> dict:
        return cancel_motion_artifact_cross_film(ppg_waveform, imu_accel_xyz, fs_hz=fs_hz)

    def infer_vital_signs(self, ppg_window, fs_hz: float = 25.0) -> dict:
        return infer_vitals_from_ppg_window(ppg_window, tflite_model_path=self.exertion_tflite_path, fs_hz=fs_hz)

    def check_altitude_acclimatization(self, altitude_m: float, cohort_mode: str = "migrant_acute") -> dict:
        return compute_altitude_spo2_uncertainty(altitude_m, cohort_mode=cohort_mode)

    def evaluate_spo2_trend(self, spo2_history: list[dict], current_spo2_pct: float, current_alt_m: float, cohort_mode: str = "migrant_acute") -> dict:
        return evaluate_longitudinal_spo2_trend(spo2_history, current_spo2_pct, current_alt_m, cohort_mode=cohort_mode)

    def check_cloudburst_anomaly(self, pressure_history_hpa: list, gps_altitude_history_m: list) -> dict:
        return detect_barometric_cloudburst_anomaly(pressure_history_hpa, gps_altitude_history_m)

    def sync_burst(self, batch_limit: int = 20) -> list[dict]:
        return prepare_burst_sync_batch(self.queue_path, batch_limit=batch_limit)

    def ack_sync(self, ack_seq_ids: list[int]) -> int:
        return acknowledge_synced_packets(self.queue_path, ack_seq_ids=ack_seq_ids)

    def run_unified_companion_step(
        self,
        hiker_lat: float,
        hiker_lon: float,
        hiker_alt: float,
        sensor_packet: dict,
        clock_hour_idx: int | None = None,
        cohort_mode: str = "migrant_acute",
        prior_stillness_s: float | None = None,
        enqueue_log: bool = True,
    ) -> dict:
        import numpy as np

        if clock_hour_idx is not None:
            self.clock_hour_idx = clock_hour_idx
        if prior_stillness_s is not None:
            self.stillness_s = float(prior_stillness_s)

        ts = float(sensor_packet.get("timestamp_s", self.clock_hour_idx * 3600.0))
        bmp = sensor_packet.get("bmp280", {})
        mpu = sensor_packet.get("mpu6050", {})
        max30102 = sensor_packet.get("max30102", {})

        pressure_hpa = float(bmp.get("pressure_hpa", 1013.25 * ((1.0 - 2.25577e-5 * hiker_alt) ** 5.25588)))

        self.pressure_history_hpa.append((ts, pressure_hpa))
        self.altitude_history_m.append((ts, hiker_alt))
        if len(self.pressure_history_hpa) > 24:
            self.pressure_history_hpa = self.pressure_history_hpa[-24:]
            self.altitude_history_m = self.altitude_history_m[-24:]

        baro_state = detect_barometric_cloudburst_anomaly(self.pressure_history_hpa, self.altitude_history_m)

        nav_state = evaluate_hiker_hazard_state(
            hiker_lat=hiker_lat,
            hiker_lon=hiker_lon,
            hiker_alt=hiker_alt,
            cartridge=self.cartridge,
            clock_hour_idx=self.clock_hour_idx,
        )

        in_hazard = nav_state["status"] in ("CRITICAL_BREACH", "WARNING_BUFFER")
        baseline = self.cartridge.get("baseline_params", {"resting_hr_bpm": 68.0})

        imu_accel = np.asarray(mpu.get("accel_xyz", np.ones((125, 3))), dtype=np.float32)
        imu_mag = np.asarray(mpu.get("accel_mag", np.ones(125)), dtype=np.float32)
        ppg_raw = np.asarray(max30102.get("ppg_waveform", np.sin(np.linspace(0, 20, 125))), dtype=np.float32)

        gate_info = evaluate_imu_spo2_gate(imu_accel, self.stillness_s)
        w_sqi = compute_w_sqi(ppg_raw, imu_mag)

        imu_effort = float(mpu.get("motion_variance", 0.01))
        ascent_rate = float(bmp.get("ascent_rate_m_min", 0.0))
        hr_bpm = float(max30102.get("heart_rate_bpm", 72.0))

        exertion_info = evaluate_activity_aware_exertion(
            observed_hr_bpm=hr_bpm,
            w_sqi_pct=w_sqi,
            imu_effort=imu_effort,
            ascent_rate_m_min=ascent_rate,
            altitude_m=hiker_alt,
            in_hazard_zone=in_hazard,
            baseline_params=baseline,
            tflite_model_path=self.exertion_tflite_path,
        )

        self.stillness_s = float(gate_info["stillness_duration_s"])
        spo2_display = max30102.get("spo2_pct") if gate_info["spo2_unlocked"] else None

        physio_state = {
            "timestamp_s": ts,
            "spo2_gate": gate_info,
            "w_sqi_pct": w_sqi,
            "exertion": exertion_info,
            "spo2_display": spo2_display,
            "hr_display": hr_bpm,
        }

        motion_info = compute_imu_hscr(imu_mag)
        cleaned = cancel_motion_artifact_cross_film(ppg_raw, imu_mag)
        medvse = infer_vitals_from_ppg_window(cleaned["cleaned_ppg"], tflite_model_path=self.exertion_tflite_path)

        spo2_val = float(spo2_display if spo2_display is not None else medvse["spo2_pct"])
        self.spo2_history.append({"timestamp_s": ts, "alt_m": hiker_alt, "spo2_pct": spo2_val})
        if len(self.spo2_history) > 48:
            self.spo2_history = self.spo2_history[-48:]

        alt_uncert = compute_altitude_spo2_uncertainty(hiker_alt, cohort_mode=cohort_mode)
        spo2_trend = evaluate_longitudinal_spo2_trend(
            self.spo2_history,
            current_spo2_pct=spo2_val,
            current_alt_m=hiker_alt,
            cohort_mode=cohort_mode,
        )

        if nav_state["status"] == "CRITICAL_BREACH" and spo2_trend["trend_status"] == "ACUTE_HYPOXIA_ALERT":
            tier = "P0_CRITICAL_SOS"
        elif nav_state["status"] == "CRITICAL_BREACH":
            tier = "P1_HAZARD_BREACH"
        elif spo2_trend["trend_status"] == "ACUTE_HYPOXIA_ALERT" or exertion_info["requires_medical_alert"] or baro_state["cloudburst_detected"]:
            tier = "P2_MEDICAL_ALERT"
        else:
            tier = "P3_ROUTINE_VITALS"

        queued_event = None
        if enqueue_log:
            queued_event = enqueue_prioritized_telemetry(
                self.queue_path,
                packet={
                    "timestamp_s": ts,
                    "lat": hiker_lat,
                    "lon": hiker_lon,
                    "alt": hiker_alt,
                    "nav_status": nav_state["status"],
                    "hr_bpm": round(float(medvse["hr_bpm"]), 1),
                    "spo2_pct": round(spo2_val, 1),
                    "rr_breaths_min": round(float(medvse["rr_breaths_min"]), 1),
                    "baro_status": baro_state["baro_status"],
                },
                priority_tier=tier,
            )

        return {
            "navigation": nav_state,
            "physiology": physio_state,
            "motion_filter": {
                "hscr_score": motion_info["hscr_score"],
                "motion_regime": motion_info["motion_regime"],
                "suppression_db": cleaned["suppression_db"],
            },
            "medvse_vitals": medvse,
            "acclimatization": {
                "uncertainty": alt_uncert,
                "trend": spo2_trend,
            },
            "baro_cloudburst": baro_state,
            "queued_event": queued_event,
            "priority_tier": tier,
            "clock_hour_idx": self.clock_hour_idx,
        }

    def generate_post_trek_audit_report(self) -> dict:
        import json
        tier_counts = {
            "P0_CRITICAL_SOS": 0,
            "P1_HAZARD_BREACH": 0,
            "P2_MEDICAL_ALERT": 0,
            "P3_ROUTINE_VITALS": 0,
        }
        total = 0
        synced = 0
        unsynced = 0
        if os.path.exists(self.queue_path):
            with open(self.queue_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    total += 1
                    if rec.get("synced"):
                        synced += 1
                    else:
                        unsynced += 1
                    t = rec.get("priority_tier", "P3_ROUTINE_VITALS")
                    if t in tier_counts:
                        tier_counts[t] += 1
                    else:
                        tier_counts["P3_ROUTINE_VITALS"] += 1

        md = (
            f"### Post-Trek Blackbox Safety & Telemetry Audit\n"
            f"- **Total Recorded Packets:** `{total}` (`{synced}` synced, `{unsynced}` pending)\n"
            f"- **P0 Critical SOS Events:** `{tier_counts['P0_CRITICAL_SOS']}`\n"
            f"- **P1 Landslide Hazard Breaches:** `{tier_counts['P1_HAZARD_BREACH']}`\n"
            f"- **P2 Medical / Cloudburst Alerts:** `{tier_counts['P2_MEDICAL_ALERT']}`\n"
            f"- **P3 Routine Vital Logs:** `{tier_counts['P3_ROUTINE_VITALS']}`\n"
        )
        return {
            "total_events": total,
            "synced_count": synced,
            "unsynced_count": unsynced,
            "tier_counts": tier_counts,
            "report_markdown": md,
        }
