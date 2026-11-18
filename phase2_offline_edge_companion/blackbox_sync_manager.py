from __future__ import annotations
import json
import os

CLOUDBURST_PRESSURE_DROP_HPA_HR: float = -2.5
PRIORITY_TIERS: tuple[str, ...] = (
    "P0_CRITICAL_SOS",
    "P1_HAZARD_BREACH",
    "P2_MEDICAL_ALERT",
    "P3_ROUTINE_VITALS",
)
DEFAULT_MAX_QUEUE_BYTES: int = 512_000


def detect_barometric_cloudburst_anomaly(
    pressure_history_hpa: list[tuple[float, float]],
    gps_altitude_history_m: list[tuple[float, float]],
) -> dict:
    if len(pressure_history_hpa) < 2 or len(gps_altitude_history_m) < 2:
        return {
            "dp_weather_hpa_hr": 0.0,
            "cloudburst_detected": False,
            "baro_status": "BARO_STABLE",
            "local_hazard_multiplier": 1.0,
        }

    t0, p0 = pressure_history_hpa[0]
    t1, p1 = pressure_history_hpa[-1]
    dt_hours = max(1e-4, (t1 - t0) / 3600.0)

    alt0 = gps_altitude_history_m[0][1]
    alt1 = gps_altitude_history_m[-1][1]
    dalt = alt1 - alt0

    dp_obs = p1 - p0
    # Elevation pressure change: ~ -0.11 hPa per meter of ascent
    dp_elev = -0.11 * dalt
    dp_wx = dp_obs - dp_elev
    dp_wx_hpa_hr = dp_wx / dt_hours

    cloudburst = bool(dp_wx_hpa_hr <= CLOUDBURST_PRESSURE_DROP_HPA_HR)
    status = "CLOUDBURST_BARO_DROP" if cloudburst else "BARO_STABLE"
    multiplier = 1.35 if cloudburst else 1.0

    return {
        "dp_weather_hpa_hr": round(float(dp_wx_hpa_hr), 2),
        "cloudburst_detected": cloudburst,
        "baro_status": status,
        "local_hazard_multiplier": multiplier,
    }


def _read_records(queue_path: str) -> list[dict]:
    if not os.path.exists(queue_path):
        return []
    records = []
    with open(queue_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    if "seq_id" not in rec:
                        rec = {
                            "seq_id": idx,
                            "priority_tier": "P2_MEDICAL_ALERT" if rec.get("exertion", {}).get("requires_medical_alert") else "P3_ROUTINE_VITALS",
                            "synced": bool(rec.get("synced", False)),
                            "payload": rec,
                        }
                    records.append(rec)
                except Exception:
                    pass
    return records


def _write_records(queue_path: str, records: list[dict]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(queue_path)), exist_ok=True)
    with open(queue_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    return os.path.getsize(queue_path)


def enqueue_prioritized_telemetry(
    queue_path: str,
    packet: dict,
    priority_tier: str = "P2_MEDICAL_ALERT",
    max_queue_bytes: int = DEFAULT_MAX_QUEUE_BYTES,
) -> dict:
    if priority_tier not in PRIORITY_TIERS:
        priority_tier = "P2_MEDICAL_ALERT"

    records = _read_records(queue_path)
    seq_id = max((r.get("seq_id", 0) for r in records), default=0) + 1

    entry = {
        "seq_id": seq_id,
        "priority_tier": priority_tier,
        "synced": False,
        "payload": packet,
    }
    records.append(entry)

    # Prune oldest lower-priority synced or P3 records if exceeding byte limit
    while records:
        test_size = sum(len(json.dumps(r, separators=(",", ":")).encode("utf-8")) + 1 for r in records)
        if test_size <= max_queue_bytes or len(records) <= 1:
            break
        pruned = False
        # 1. Prune already synced
        for i, r in enumerate(records):
            if r.get("synced", False):
                records.pop(i)
                pruned = True
                break
        # 2. Prune oldest P3
        if not pruned:
            for i, r in enumerate(records):
                if r.get("priority_tier") == "P3_ROUTINE_VITALS":
                    records.pop(i)
                    pruned = True
                    break
        # 3. Prune oldest entry
        if not pruned:
            records.pop(0)

    size_bytes = _write_records(queue_path, records)
    return {
        "seq_id": seq_id,
        "queued_count": len(records),
        "size_bytes": size_bytes,
    }


def prepare_burst_sync_batch(
    queue_path: str,
    batch_limit: int = 20,
) -> list[dict]:
    records = _read_records(queue_path)
    unsynced = [r for r in records if not r.get("synced", False)]

    tier_order = {t: idx for idx, t in enumerate(PRIORITY_TIERS)}
    unsynced.sort(key=lambda r: (tier_order.get(r.get("priority_tier", "P3_ROUTINE_VITALS"), 99), r.get("seq_id", 0)))
    return unsynced[:batch_limit]


def acknowledge_synced_packets(
    queue_path: str,
    ack_seq_ids: list[int],
) -> int:
    records = _read_records(queue_path)
    ack_set = set(ack_seq_ids)
    acked_count = 0

    for r in records:
        if r.get("seq_id") in ack_set and not r.get("synced", False):
            r["synced"] = True
            acked_count += 1

    if acked_count > 0:
        _write_records(queue_path, records)

    return acked_count
