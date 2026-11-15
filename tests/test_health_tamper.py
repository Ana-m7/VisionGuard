"""Tests for camera health monitoring and tamper detection.

No camera, model or real database: frames are synthetic numpy arrays, the
FrameStore is driven directly, and the API tests run against a throwaway
SQLite file with the session check overridden.
"""
import asyncio
import queue

import numpy as np
import pytest

import config
from detector import engine
from detector.engine import FrameStore, TamperDetector, tamper_event


def _drain(q):
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


# ── FrameStore health ────────────────────────────────────────────
def test_camera_never_seen_is_offline():
    fs = FrameStore()
    status = fs.camera_status()
    assert set(status) == set(config.VIDEO_SOURCES)
    for s in status.values():
        assert s == {"online": False, "last_seen": None, "seconds_since": None,
                     "tamper": None}
    assert sorted(fs.stale_cameras()) == sorted(config.VIDEO_SOURCES)


def test_publishing_a_frame_marks_camera_online():
    fs = FrameStore()
    cam = next(iter(config.VIDEO_SOURCES))
    fs.set(cam, b"jpeg")
    s = fs.camera_status()[cam]
    assert s["online"] and s["seconds_since"] < 1
    assert cam not in fs.stale_cameras()
    assert fs.get(cam) == b"jpeg"          # existing behaviour unchanged


def test_tamper_state_is_reported_until_cleared():
    fs = FrameStore()
    cam = next(iter(config.VIDEO_SOURCES))
    fs.set_tamper(cam, "covered")
    assert fs.camera_status()[cam]["tamper"] == "covered"
    fs.set_tamper(cam, None)
    assert fs.camera_status()[cam]["tamper"] is None


def test_camera_goes_stale_after_timeout(monkeypatch):
    fs = FrameStore()
    cam = next(iter(config.VIDEO_SOURCES))
    fs.set(cam, b"jpeg")
    monkeypatch.setattr(config, "CAMERA_OFFLINE_TIMEOUT", 0)
    assert fs.camera_status()[cam]["online"] is False
    assert cam in fs.stale_cameras()


# ── Tamper detection ─────────────────────────────────────────────
H, W = 270, 480
rng = np.random.default_rng(0)


def dark():
    return np.full((H, W, 3), 5, np.uint8)


def scene():
    return rng.integers(40, 220, (H, W, 3), dtype=np.uint8)


def test_covered_fires_only_after_confirm_frames():
    t = TamperDetector()
    for _ in range(config.TAMPER_COVERED_CONFIRM_FRAMES - 1):
        assert t.check(dark()) is None
    assert t.check(dark()) == "covered"
    assert t.check(dark()) == "covered"    # stays covered


def test_bright_but_featureless_frame_counts_as_covered():
    # a hand over a webcam: auto-exposure lifts it far above the blackout
    # threshold, but the picture has almost no contrast
    t = TamperDetector()
    hand = rng.integers(118, 126, (H, W, 3), dtype=np.uint8)
    results = [t.check(hand.copy()) for _ in range(config.TAMPER_FROZEN_CONFIRM_FRAMES + 5)]
    assert t.brightness > config.TAMPER_BLACKOUT_THRESHOLD
    assert t.contrast < config.TAMPER_UNIFORM_STD
    assert "frozen" not in results and results[-1] == "covered"


def test_uncovering_clears_and_resets_streak():
    t = TamperDetector()
    for _ in range(config.TAMPER_COVERED_CONFIRM_FRAMES):
        t.check(dark())
    assert t.check(scene()) is None
    # a short dark blip afterwards must not re-fire
    for _ in range(config.TAMPER_COVERED_CONFIRM_FRAMES - 1):
        assert t.check(dark()) is None


def test_identical_frames_are_frozen_after_confirm_frames():
    t = TamperDetector()
    still = scene()
    assert t.check(still) is None          # first frame has nothing to compare
    for _ in range(config.TAMPER_FROZEN_CONFIRM_FRAMES - 1):
        assert t.check(still.copy()) is None
    assert t.check(still.copy()) == "frozen"


def test_sensor_noise_below_floor_still_counts_as_frozen():
    t = TamperDetector()
    base = scene().astype(np.int16)
    results = []
    for _ in range(config.TAMPER_FROZEN_CONFIRM_FRAMES + 1):
        noisy = base + rng.integers(-3, 4, base.shape)
        results.append(t.check(np.clip(noisy, 0, 255).astype(np.uint8)))
    assert results[-1] == "frozen"


def test_live_changing_feed_is_never_tampered():
    t = TamperDetector()
    for _ in range(3 * config.TAMPER_FROZEN_CONFIRM_FRAMES):
        assert t.check(scene()) is None


def test_covered_takes_precedence_over_frozen():
    t = TamperDetector()
    results = [t.check(dark()) for _ in range(3 * config.TAMPER_FROZEN_CONFIRM_FRAMES)]
    assert "frozen" not in results and results[-1] == "covered"


def test_reset_clears_all_streaks():
    t = TamperDetector()
    for _ in range(config.TAMPER_COVERED_CONFIRM_FRAMES):
        t.check(dark())
    t.reset()
    assert t.check(dark()) is None


def test_tamper_event_shapes():
    ev = tamper_event(4, "frozen")
    assert ev["type"] == "camera_tamper" and ev["is_alert"] == 1
    assert ev["tamper_type"] == "frozen" and "FROZEN FEED" in ev["message"]
    cleared = tamper_event(4, None)
    assert cleared["is_alert"] == 0 and cleared["tamper_type"] is None


# ── Alert log lines ──────────────────────────────────────────────
@pytest.fixture
def vg_log():
    """Collect formatted lines from the visionguard loggers (they do not
    propagate to the root logger, so pytest's caplog cannot see them)."""
    import logging

    class Collector(logging.Handler):
        def __init__(self):
            super().__init__(logging.DEBUG)
            self.lines = []

        def emit(self, record):
            self.lines.append(record.getMessage())

    h = Collector()
    root = logging.getLogger("visionguard")
    old_level = root.level
    root.addHandler(h)
    root.setLevel(logging.DEBUG)
    yield h.lines
    root.removeHandler(h)
    root.setLevel(old_level)


def test_alert_line_names_action_type_camera_and_id(vg_log):
    import logging
    from log_setup import log_alert
    log_alert(logging.getLogger("visionguard.test"), logging.WARNING,
              "ALERT", "fall", 4, "torso 80 deg off vertical", 12)
    line = vg_log[-1]
    assert line.startswith("ALERT") and "FALL DETECTED" in line
    assert "CAM-04 (Webcam)" in line and "#12" in line and "torso 80" in line


def test_tamper_lines_say_what_fired_and_what_cleared(vg_log):
    tamper_event(4, "covered", 30, 4)
    tamper_event(4, None, 120, 60, previous="covered")
    alert, cleared = vg_log[-2:]
    assert alert.startswith("ALERT") and "CAMERA TAMPERING" in alert
    assert "lens covered" in alert and "contrast 4" in alert
    assert cleared.startswith("CLEARED") and "was lens covered" in cleared


# ── Health monitor task ──────────────────────────────────────────
class FakeStore:
    def __init__(self, stale):
        self.stale = stale

    def stale_cameras(self):
        return list(self.stale)


def test_monitor_reports_offline_and_online_once_each(monkeypatch):
    from backend import app as app_module
    monkeypatch.setattr(config, "CAMERA_OFFLINE_TIMEOUT", 0.05)
    monkeypatch.setattr(config, "CAMERA_MONITOR_INTERVAL", 0.05)
    store = FakeStore([4])
    monkeypatch.setattr(app_module, "frame_store", store)
    _drain(engine.alert_queue)

    seen = {}

    async def scenario():
        task = asyncio.create_task(app_module.camera_monitor())
        await asyncio.sleep(0.4)           # several sweeps while offline
        seen["during"] = set(app_module.reported_offline)
        store.stale = []
        await asyncio.sleep(0.4)           # several sweeps while online
        seen["after"] = set(app_module.reported_offline)
        task.cancel()

    asyncio.run(scenario())
    events = [(e["type"], e["cam_id"]) for e in _drain(engine.alert_queue)]
    assert events == [("camera_offline", 4), ("camera_online", 4)]
    # the state /stats exposes for dashboards that missed the push
    assert seen == {"during": {4}, "after": set()}


# ── API ──────────────────────────────────────────────────────────
@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from backend import app as app_module
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "test.db"))
    engine.init_db()
    store = FrameStore()
    monkeypatch.setattr(app_module, "frame_store", store)
    app_module.app.dependency_overrides[app_module.current_user] = (
        lambda: {"id": 1, "username": "t", "role": "admin"})
    # No `with`: the lifespan (camera threads, background tasks) never starts
    yield TestClient(app_module.app), store
    app_module.app.dependency_overrides.clear()


def test_cameras_status_endpoint(client):
    http, store = client
    cam = next(iter(config.VIDEO_SOURCES))
    store.set(cam, b"jpeg")
    rows = {r["cam_id"]: r for r in http.get("/cameras/status").json()}
    assert set(rows) == set(config.VIDEO_SOURCES)
    assert rows[cam]["online"] is True and rows[cam]["label"]
    others = [c for c in rows if c != cam]
    assert all(rows[c]["online"] is False for c in others)


def test_stats_counts_only_online_cameras(client):
    http, store = client
    cam = next(iter(config.VIDEO_SOURCES))
    store.set(cam, b"jpeg")
    stats = http.get("/stats").json()
    assert stats["cameras_live"] == 1
    assert stats["cameras_total"] == len(config.VIDEO_SOURCES)
    assert str(cam) in stats["camera_status"]


def test_acknowledgement_is_logged_with_alert_and_user(client, vg_log):
    import sqlite3
    http, _ = client
    conn = sqlite3.connect(config.DB_PATH)
    ids = []
    for _ in range(3):   # one incident that left three alert rows
        cur = conn.execute(
            "INSERT INTO crowd_events (cam_id, timestamp, person_count, "
            "max_zone_density, is_alert, frame_path) VALUES (4, '2026-01-01 10:00:00', 3, 2, 1, '')")
        ids.append(cur.lastrowid)
    conn.commit()
    conn.close()
    assert http.post(f"/alerts/ack?type=gathering&id={ids[-1]}").json()["acknowledged"] == 3
    line = next(l for l in vg_log if l.startswith("ACK"))
    assert "CROWD GATHERING" in line and "CAM-04" in line and f"#{ids[-1]}" in line
    assert "by t" in line and "plus 2 earlier" in line


def test_stats_reports_tamper_and_offline_alert_state(client, monkeypatch):
    from backend import app as app_module
    http, store = client
    cam = next(iter(config.VIDEO_SOURCES))
    store.set_tamper(cam, "frozen")
    monkeypatch.setattr(app_module, "reported_offline", {cam})
    s = http.get("/stats").json()["camera_status"][str(cam)]
    assert s["tamper"] == "frozen" and s["offline_alert"] is True
