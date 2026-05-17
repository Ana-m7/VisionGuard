from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import sqlite3
import os
import cv2
import numpy as np
import threading
import time
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.path.join(os.path.dirname(__file__), "../crowd_events.db")
RECENT_WINDOW_SECONDS = 60
STREAM_INTERVAL = 0.05  # seconds (~20 FPS)


@app.on_event("startup")
def clear_db_on_startup():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # Drop and recreate so schema changes (e.g. new columns) always take effect.
    cur.execute("DROP TABLE IF EXISTS crowd_events")
    cur.execute("DROP TABLE IF EXISTS fall_events")
    cur.execute("""
        CREATE TABLE crowd_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cam_id INTEGER,
            timestamp TEXT,
            person_count INTEGER,
            max_zone_density INTEGER,
            is_alert INTEGER,
            frame_path TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE fall_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cam_id INTEGER,
            timestamp TEXT,
            is_fall INTEGER
        )
    """)
    conn.commit()
    conn.close()
    print("DB reset on startup.")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ================================================================
# FRAME STORE — receives frames from main_detect.py
# ================================================================
latest_frames = {}
frame_lock = threading.Lock()


@app.post("/push_frame/{cam_id}")
async def push_frame(cam_id: int, frame: UploadFile = File(...)):
    data = await frame.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        with frame_lock:
            latest_frames[cam_id] = img
    return {"ok": True}


def generate_stream(cam_id):
    while True:
        with frame_lock:
            frame = latest_frames.get(cam_id)
        if frame is None:
            time.sleep(0.05)
            continue
        _, buffer = cv2.imencode(
            '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75]
        )
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + buffer.tobytes()
            + b'\r\n'
        )
        time.sleep(STREAM_INTERVAL)


@app.get("/video_feed/{cam_id}")
def video_feed(cam_id: int):
    with frame_lock:
        has_feed = cam_id in latest_frames
    if not has_feed:
        raise HTTPException(status_code=404, detail="No stream for this camera")
    return StreamingResponse(
        generate_stream(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ================================================================
# STATS
# ================================================================
@app.get("/stats")
def get_stats():
    conn = get_db()
    cur = conn.cursor()

    # FIX: Use a consistent recent window for BOTH detectors
    # so the count reflects "active/recent" alerts, not all-time
    WINDOW = f"datetime('now', 'localtime', '-{RECENT_WINDOW_SECONDS} seconds')"

    cur.execute(f"""
        SELECT COUNT(*)
        FROM crowd_events
        WHERE is_alert = 1
        AND timestamp >= {WINDOW}
    """)
    alert_count = cur.fetchone()[0]

    # FIX: fall_events also filtered to same window — was counting ALL TIME before
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='fall_events'"
    )
    if cur.fetchone():
        cur.execute(f"""
            SELECT COUNT(*) FROM fall_events
            WHERE is_fall = 1
            AND timestamp >= {WINDOW}
        """)
        alert_count += cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM crowd_events")
    frame_count = cur.fetchone()[0]

    cur.execute("SELECT COALESCE(SUM(person_count), 0) FROM crowd_events")
    person_count = cur.fetchone()[0]

    conn.close()
    return {
        "alert_count": alert_count,
        "frame_count": frame_count,
        "person_count": person_count
    }


# ================================================================
# CROWD ENDPOINTS
# ================================================================
@app.get("/latest")
def get_latest():
    conn = get_db()
    cur = conn.cursor()
    window = f"datetime('now', 'localtime', '-{RECENT_WINDOW_SECONDS} seconds')"
    cur.execute(f"""
        SELECT id, cam_id, timestamp, person_count, max_zone_density, is_alert
        FROM crowd_events
        WHERE is_alert = 1
        AND timestamp >= {window}
        ORDER BY id DESC LIMIT 1
    """)
    row = cur.fetchone()
    conn.close()
    if not row:
        return {
            "id": 0, "cam_id": None, "timestamp": None,
            "person_count": 0, "zone_density": 0, "is_alert": 0
        }
    return {
        "id": row[0], "cam_id": row[1], "timestamp": row[2],
        "person_count": row[3], "zone_density": row[4], "is_alert": row[5]
    }


@app.get("/alerts/recent")
def get_recent_alerts():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, timestamp, person_count, max_zone_density, is_alert
        FROM crowd_events
        WHERE is_alert = 1
        ORDER BY id DESC LIMIT 50
    """)
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "id": r[0], "timestamp": r[1],
            "person_count": r[2], "zone_density": r[3], "is_alert": r[4]
        }
        for r in rows
    ]


# ================================================================
# FALL ENDPOINT
# ================================================================
@app.get("/fall/latest")
def get_latest_fall():
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='fall_events'"
    )
    if not cur.fetchone():
        conn.close()
        return {"id": 0, "is_fall": False, "timestamp": None}
    window = f"datetime('now', 'localtime', '-{RECENT_WINDOW_SECONDS} seconds')"
    cur.execute(f"""
        SELECT id, cam_id, timestamp, is_fall
        FROM fall_events
        WHERE is_fall = 1
        AND timestamp >= {window}
        ORDER BY id DESC LIMIT 1
    """)
    row = cur.fetchone()
    conn.close()
    if not row:
        return {"id": 0, "cam_id": None, "is_fall": False, "timestamp": None}
    return {"id": row[0], "cam_id": row[1], "is_fall": bool(row[3]), "timestamp": row[2]}


# ================================================================
# MAINTENANCE
# ================================================================
@app.post("/alerts/clear")
def clear_alerts():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM crowd_events")
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='fall_events'"
    )
    if cur.fetchone():
        cur.execute("DELETE FROM fall_events")
    cur.execute("DELETE FROM sqlite_sequence WHERE name IN ('crowd_events', 'fall_events')")
    conn.commit()
    conn.close()
    return {"ok": True}

if __name__ == "__main__":
    # reload=True with reload_includes scoped to .py only — prevents uvicorn's
    # StatReload from restarting the server when crowd_events.db or evidence
    # frames are written (which would kill all active MJPEG streams).
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_includes=["*.py"],
    )