"""Inference-rate sweep (replaces the old camera-FPS sweep, audit #5).

Why not a camera-FPS sweep: production never processes every camera frame.
The capture thread runs at the camera's rate, but the inference thread takes
the newest frame whenever the previous detection + pose cycle finishes
(~2.8 cycles/s on the target CPU, config.py). Camera fps above the inference
rate does not change what the model sees, so the quantity that actually
governs fall detection is the effective INFERENCE rate.

The numbers come straight from ur_fall.py (each UR Fall sequence replayed by
time at every rate, tracker reset per run, production fall alert logic), so
the sweep and the dataset result can never disagree. Every rate is checked to
sample a different frame set (the old sweep had 20 and 25 fps identical).

Crowd counting has no temporal state (plain predict per cycle), so per-frame
counting accuracy does not depend on rate; only the alert onset does:
CROWD_CONFIRM_CYCLES consecutive cycles, i.e. CROWD_CONFIRM_CYCLES / r seconds.
"""
from __future__ import annotations

import config


def run(ur_fall_result: dict) -> dict:
    if not ur_fall_result or not str(ur_fall_result.get("status", "")).startswith(("OK", "PARTIAL")):
        return {"status": "SKIPPED: UR Fall result unavailable"}
    points = []
    for label, r in ur_fall_result["by_rate"].items():
        points.append({
            "rate": label, "rate_hz": r["rate_hz"],
            "recall": r["recall"], "precision_event_level": r["precision_event_level"],
            "adl_sequences_with_false_alarm": r["adl_sequences_with_false_alarm"],
            "false_alarms_per_hour_of_adl_footage": r["false_alarms_per_hour_of_adl_footage"],
            "latency_s_from_fall_onset": r["latency_s_from_fall_onset"],
            "n_fall_sequences": r["n_fall_sequences"], "n_adl_sequences": r["n_adl_sequences"],
            "status_quality": r["status_quality"],
            "crowd_alert_onset_s": (config.CROWD_CONFIRM_CYCLES / r["rate_hz"]) if r["rate_hz"] else None,
        })
    return {
        "status": "OK",
        "source": "UR Fall (30 fps), replayed by time at each effective inference rate",
        "every_rate_uses_distinct_frames": ur_fall_result.get("every_rate_uses_distinct_frames"),
        "production_rate": ur_fall_result.get("production_rate_key"),
        "measured_production_cycle": ur_fall_result.get("measured_production_cycle"),
        "camera_fps_note": ("A camera only needs a frame rate at or above the inference rate it "
                            "feeds; production samples the newest frame each cycle, so extra "
                            "camera fps is never seen by the model."),
        "points": points,
    }
