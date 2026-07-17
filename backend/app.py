"""VisionGuard API server.

Single process: the FastAPI lifespan handler initialises the database and
starts the detection engine's camera threads. Camera threads place encoded
JPEG frames into a shared FrameStore; the MJPEG endpoint serves those bytes
directly — frames are encoded exactly once in the whole pipeline.
"""
import os
import sys
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from detector.engine import FrameStore, LiveStats, init_db, start_engine

frame_store = FrameStore()
live_stats = LiveStats()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    start_engine(frame_store, live_stats)
    yield


app = FastAPI(title="VisionGuard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local demo only — do not expose publicly
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def recent_cutoff():
    """Timestamp string N seconds ago, matching the detector's format."""
    cutoff = datetime.now() - timedelta(seconds=config.RECENT_WINDOW_SECONDS)
    return cutoff.strftime("%Y-%m-%d %H:%M:%S")


# ================================================================
# VIDEO STREAMING
# ================================================================
def generate_stream(cam_id):
    while True:
        jpeg = frame_store.get(cam_id)
        if jpeg is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        time.sleep(config.STREAM_INTERVAL)


@app.get("/video_feed/{cam_id}")
def video_feed(cam_id: int):
    if cam_id not in config.VIDEO_SOURCES:
        raise HTTPException(status_code=404, detail="Unknown camera")
    return StreamingResponse(
        generate_stream(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ================================================================
# STATS
# ================================================================
@app.get("/stats")
def get_stats():
    conn = get_db()
    cur = conn.cursor()
    cutoff = recent_cutoff()
    cur.execute(
        "SELECT COUNT(*) FROM crowd_events WHERE is_alert = 1 AND timestamp >= ?",
        (cutoff,))
    alert_count = cur.fetchone()[0]
    cur.execute(
        "SELECT COUNT(*) FROM fall_events WHERE is_fall = 1 AND timestamp >= ?",
        (cutoff,))
    alert_count += cur.fetchone()[0]
    conn.close()

    live = live_stats.snapshot()
    return {
        "alert_count": alert_count,
        "frames_processed": live["frames_processed"],
        "persons_now": live["persons_now"],
        "cameras_live": len(live["cameras"]),
        "cameras_total": len(config.VIDEO_SOURCES),
    }


# ================================================================
# ALERTS
# ================================================================
@app.get("/latest")
def get_latest_crowd():
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """SELECT id, cam_id, timestamp, person_count, max_zone_density, is_alert
           FROM crowd_events
           WHERE is_alert = 1 AND timestamp >= ?
           ORDER BY id DESC LIMIT 1""",
        (recent_cutoff(),))
    row = cur.fetchone()
    conn.close()
    if not row:
        return {"id": 0, "cam_id": None, "timestamp": None,
                "person_count": 0, "zone_density": 0, "is_alert": 0}
    return {"id": row["id"], "cam_id": row["cam_id"], "timestamp": row["timestamp"],
            "person_count": row["person_count"], "zone_density": row["max_zone_density"],
            "is_alert": row["is_alert"]}


@app.get("/fall/latest")
def get_latest_fall():
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """SELECT id, cam_id, timestamp, is_fall, ratio
           FROM fall_events
           WHERE is_fall = 1 AND timestamp >= ?
           ORDER BY id DESC LIMIT 1""",
        (recent_cutoff(),))
    row = cur.fetchone()
    conn.close()
    if not row:
        return {"id": 0, "cam_id": None, "is_fall": False,
                "timestamp": None, "ratio": None}
    return {"id": row["id"], "cam_id": row["cam_id"], "is_fall": bool(row["is_fall"]),
            "timestamp": row["timestamp"], "ratio": row["ratio"]}


@app.get("/alerts/recent")
def get_recent_alerts(limit: int = 50):
    """Recent alerts from both detectors, newest first — used by the
    dashboard to rebuild its event log after a page load."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """SELECT * FROM (
               SELECT 'gathering' AS type, id, cam_id, timestamp,
                      person_count, max_zone_density AS zone_density, NULL AS ratio
               FROM crowd_events WHERE is_alert = 1
               UNION ALL
               SELECT 'fall' AS type, id, cam_id, timestamp,
                      NULL AS person_count, NULL AS zone_density, ratio
               FROM fall_events WHERE is_fall = 1
           )
           ORDER BY timestamp DESC LIMIT ?""",
        (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


@app.post("/alerts/clear")
def clear_alerts():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM crowd_events")
    cur.execute("DELETE FROM fall_events")
    cur.execute("DELETE FROM sqlite_sequence WHERE name IN ('crowd_events', 'fall_events')")
    conn.commit()
    conn.close()
    return {"ok": True}


if __name__ == "__main__":
    # No auto-reload: the detector threads live in this process, and a
    # reload supervisor would restart them on every file change.
    uvicorn.run(app, host=config.HOST, port=config.PORT)
