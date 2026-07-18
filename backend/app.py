"""VisionGuard API server.

Single process: the FastAPI lifespan handler initialises the database and
starts the detection engine's camera threads. Camera threads place encoded
JPEG frames into a shared FrameStore; the MJPEG endpoint serves those bytes
directly — frames are encoded exactly once in the whole pipeline.
"""
import asyncio
import json
import os
import queue
import sys
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import csv
import io

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from detector.engine import FrameStore, LiveStats, alert_queue, init_db, start_engine

frame_store = FrameStore()
live_stats = LiveStats()

# ================================================================
# WEBSOCKET ALERT PUSH
# ================================================================
ws_clients = set()


async def broadcast_alerts():
    """Drain the engine's alert queue and push each event to every
    connected dashboard. The 1s get-timeout keeps the executor thread
    from blocking forever on shutdown."""
    loop = asyncio.get_running_loop()
    while True:
        try:
            event = await loop.run_in_executor(
                None, lambda: alert_queue.get(timeout=1))
        except queue.Empty:
            continue
        dead = set()
        for ws in ws_clients:
            try:
                await ws.send_json(event)
            except Exception:
                dead.add(ws)
        ws_clients.difference_update(dead)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    start_engine(frame_store, live_stats)
    broadcaster = asyncio.create_task(broadcast_alerts())
    yield
    broadcaster.cancel()


app = FastAPI(title="VisionGuard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local demo only — do not expose publicly
    allow_methods=["*"],
    allow_headers=["*"],
)

# Alert evidence snapshots, addressed by filename from the DB rows
os.makedirs(config.EVIDENCE_DIR, exist_ok=True)
app.mount("/evidence", StaticFiles(directory=config.EVIDENCE_DIR), name="evidence")


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


@app.websocket("/ws/alerts")
async def ws_alerts(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    try:
        while True:  # we never expect client messages; this detects disconnects
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        ws_clients.discard(ws)


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
        "detection_fps": live["detection_fps"],
        "avg_latency_ms": live["avg_latency_ms"],
    }


# ================================================================
# ALERTS
# ================================================================
@app.get("/latest")
def get_latest_crowd():
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """SELECT id, cam_id, timestamp, person_count, max_zone_density,
                  is_alert, frame_path, acknowledged_at
           FROM crowd_events
           WHERE is_alert = 1 AND timestamp >= ?
           ORDER BY id DESC LIMIT 1""",
        (recent_cutoff(),))
    row = cur.fetchone()
    conn.close()
    if not row:
        return {"id": 0, "cam_id": None, "timestamp": None,
                "person_count": 0, "zone_density": 0, "is_alert": 0,
                "frame_path": None, "acknowledged_at": None}
    return {"id": row["id"], "cam_id": row["cam_id"], "timestamp": row["timestamp"],
            "person_count": row["person_count"], "zone_density": row["max_zone_density"],
            "is_alert": row["is_alert"], "frame_path": row["frame_path"],
            "acknowledged_at": row["acknowledged_at"]}


@app.get("/fall/latest")
def get_latest_fall():
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """SELECT id, cam_id, timestamp, is_fall, ratio, angle,
                  frame_path, acknowledged_at
           FROM fall_events
           WHERE is_fall = 1 AND timestamp >= ?
           ORDER BY id DESC LIMIT 1""",
        (recent_cutoff(),))
    row = cur.fetchone()
    conn.close()
    if not row:
        return {"id": 0, "cam_id": None, "is_fall": False,
                "timestamp": None, "ratio": None, "angle": None,
                "frame_path": None, "acknowledged_at": None}
    return {"id": row["id"], "cam_id": row["cam_id"], "is_fall": bool(row["is_fall"]),
            "timestamp": row["timestamp"], "ratio": row["ratio"], "angle": row["angle"],
            "frame_path": row["frame_path"], "acknowledged_at": row["acknowledged_at"]}


@app.get("/alerts/recent")
def get_recent_alerts(limit: int = 50):
    """Recent alerts from both detectors, newest first — used by the
    dashboard to rebuild its event log after a page load."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """SELECT * FROM (
               SELECT 'gathering' AS type, id, cam_id, timestamp,
                      person_count, max_zone_density AS zone_density,
                      NULL AS ratio, NULL AS angle,
                      frame_path, acknowledged_at
               FROM crowd_events WHERE is_alert = 1
               UNION ALL
               SELECT 'fall' AS type, id, cam_id, timestamp,
                      NULL AS person_count, NULL AS zone_density, ratio, angle,
                      frame_path, acknowledged_at
               FROM fall_events WHERE is_fall = 1
           )
           ORDER BY timestamp DESC LIMIT ?""",
        (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ================================================================
# ANALYTICS
# ================================================================
@app.get("/analytics/occupancy")
def analytics_occupancy(minutes: int = 30):
    """Average persons in view per camera, bucketed per minute."""
    cutoff = (datetime.now() - timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """SELECT substr(timestamp, 1, 16) AS minute, cam_id,
                  AVG(person_count) AS avg_persons, MAX(max_density) AS max_density
           FROM occupancy_samples
           WHERE timestamp >= ?
           GROUP BY minute, cam_id
           ORDER BY minute""",
        (cutoff,))
    rows = [{"minute": r["minute"], "cam_id": r["cam_id"],
             "avg_persons": round(r["avg_persons"], 1),
             "max_density": r["max_density"]} for r in cur.fetchall()]
    conn.close()
    return rows


@app.get("/analytics/alerts")
def analytics_alerts(hours: int = 12):
    """Alert counts per hour for both detectors."""
    cutoff = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db()
    cur = conn.cursor()
    buckets = {}
    cur.execute(
        """SELECT substr(timestamp, 1, 13) AS hour, COUNT(*) AS n
           FROM crowd_events WHERE is_alert = 1 AND timestamp >= ?
           GROUP BY hour""",
        (cutoff,))
    for r in cur.fetchall():
        buckets.setdefault(r["hour"], {"gathering": 0, "fall": 0})["gathering"] = r["n"]
    cur.execute(
        """SELECT substr(timestamp, 1, 13) AS hour, COUNT(*) AS n
           FROM fall_events WHERE is_fall = 1 AND timestamp >= ?
           GROUP BY hour""",
        (cutoff,))
    for r in cur.fetchall():
        buckets.setdefault(r["hour"], {"gathering": 0, "fall": 0})["fall"] = r["n"]
    conn.close()
    return [{"hour": h, **counts} for h, counts in sorted(buckets.items())]


@app.get("/analytics/zones")
def analytics_zones(minutes: int = 30):
    """Summed per-zone person counts over the window — heatmap data."""
    cutoff = (datetime.now() - timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT zones FROM occupancy_samples WHERE timestamp >= ?", (cutoff,))
    total = [[0] * config.GRID_COLS for _ in range(config.GRID_ROWS)]
    for (zones_json,) in cur.fetchall():
        try:
            zones = json.loads(zones_json)
        except (TypeError, ValueError):
            continue
        for r in range(config.GRID_ROWS):
            for c in range(config.GRID_COLS):
                total[r][c] += zones[r][c]
    conn.close()
    return {"rows": config.GRID_ROWS, "cols": config.GRID_COLS,
            "total": total, "max": max(v for row in total for v in row)}


@app.post("/alerts/ack")
def acknowledge_alert(type: str, id: int):
    """Persist an operator acknowledgement on an alert row, so the
    audit trail survives page reloads and restarts."""
    table = {"gathering": "crowd_events", "fall": "fall_events"}.get(type)
    if not table:
        raise HTTPException(status_code=400, detail="Unknown alert type")
    when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db()
    cur = conn.cursor()
    cur.execute(f"UPDATE {table} SET acknowledged_at = ? WHERE id = ?", (when, id))
    updated = cur.rowcount
    conn.commit()
    conn.close()
    if not updated:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"ok": True, "acknowledged_at": when}


@app.get("/export/alerts.csv")
def export_alerts_csv():
    """Full alert history (both detectors) as a CSV download."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """SELECT * FROM (
               SELECT 'gathering' AS type, id, cam_id, timestamp,
                      person_count, max_zone_density AS zone_density,
                      NULL AS ratio, NULL AS angle,
                      frame_path, acknowledged_at
               FROM crowd_events WHERE is_alert = 1
               UNION ALL
               SELECT 'fall' AS type, id, cam_id, timestamp,
                      NULL AS person_count, NULL AS zone_density, ratio, angle,
                      frame_path, acknowledged_at
               FROM fall_events WHERE is_fall = 1
           )
           ORDER BY timestamp""")
    rows = cur.fetchall()
    conn.close()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["type", "id", "cam_id", "timestamp", "person_count",
                     "zone_density", "ratio", "angle", "frame_path",
                     "acknowledged_at"])
    for r in rows:
        writer.writerow([r["type"], r["id"], r["cam_id"], r["timestamp"],
                         r["person_count"], r["zone_density"], r["ratio"],
                         r["angle"], r["frame_path"], r["acknowledged_at"]])
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="visionguard_alerts.csv"'},
    )


@app.post("/alerts/clear")
def clear_alerts():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM crowd_events")
    cur.execute("DELETE FROM fall_events")
    cur.execute("DELETE FROM occupancy_samples")
    cur.execute("DELETE FROM sqlite_sequence WHERE name IN "
                "('crowd_events', 'fall_events', 'occupancy_samples')")
    conn.commit()
    conn.close()
    return {"ok": True}


if __name__ == "__main__":
    # No auto-reload: the detector threads live in this process, and a
    # reload supervisor would restart them on every file change.
    uvicorn.run(app, host=config.HOST, port=config.PORT)
