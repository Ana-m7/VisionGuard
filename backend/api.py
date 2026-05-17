from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import sqlite3
import os
import cv2
import numpy as np
import threading
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.path.join(os.path.dirname(__file__), "../crowd_events.db")

def get_db():
    return sqlite3.connect(DB_PATH)

# ── FRAME STORE ───────────────────────────────────────────────────
latest_frames = {}
frame_lock = threading.Lock()

@app.post("/push_frame/{cam_id}")
async def push_frame(cam_id: int, frame: UploadFile = File(...)):
    data = await frame.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
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
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            buffer.tobytes() +
            b'\r\n'
        )
        time.sleep(0.04)

@app.get("/video_feed/{cam_id}")
def video_feed(cam_id: int):
    return StreamingResponse(
        generate_stream(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ── DB ENDPOINTS ──────────────────────────────────────────────────
@app.get("/stats")
def get_stats():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM crowd_events WHERE is_alert = 1")
    alert_count = cur.fetchone()[0]
    # add fall alerts to count
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fall_events'")
    if cur.fetchone():
        cur.execute("SELECT COUNT(*) FROM fall_events WHERE is_fall = 1")
        alert_count += cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM crowd_events")
    frame_count = cur.fetchone()[0]
    conn.close()
    return {"alert_count": alert_count, "frame_count": frame_count}

@app.get("/latest")
def get_latest():
    conn = get_db()
    cur = conn.cursor()
    # Return the latest ALERT row (is_alert=1) with its unique ID
    cur.execute("""
        SELECT id, timestamp, person_count, max_zone_density, is_alert
        FROM crowd_events
        WHERE is_alert = 1
        ORDER BY id DESC LIMIT 1
    """)
    row = cur.fetchone()
    conn.close()
    if not row:
        return {"id": 0, "timestamp": None, "person_count": 0, "zone_density": 0, "is_alert": 0}
    return {"id": row[0], "timestamp": row[1], "person_count": row[2], "zone_density": row[3], "is_alert": row[4]}

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
        {"id": r[0], "timestamp": r[1], "person_count": r[2], "zone_density": r[3], "is_alert": r[4]}
        for r in rows
    ]

@app.get("/fall/latest")
def get_latest_fall():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fall_events'")
    if not cur.fetchone():
        conn.close()
        return {"id": 0, "is_fall": False, "timestamp": None}
    cur.execute("SELECT id, timestamp, is_fall FROM fall_events WHERE is_fall = 1 ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    if not row:
        return {"id": 0, "is_fall": False, "timestamp": None}
    return {"id": row[0], "is_fall": bool(row[2]), "timestamp": row[1]}