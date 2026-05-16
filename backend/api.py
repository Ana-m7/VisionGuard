from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import os

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

@app.get("/stats")
def get_stats():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM crowd_events WHERE is_alert = 1")
    alert_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM crowd_events")
    frame_count = cur.fetchone()[0]
    cur.execute("SELECT COALESCE(SUM(person_count), 0) FROM crowd_events")
    person_count = cur.fetchone()[0]
    conn.close()
    return {"alert_count": alert_count, "frame_count": frame_count, "person_count": person_count}

@app.get("/latest")
def get_latest():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT timestamp, person_count, max_zone_density, is_alert FROM crowd_events ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    if not row:
        return {"timestamp": None, "person_count": 0, "zone_density": 0, "is_alert": 0}
    return {"timestamp": row[0], "person_count": row[1], "zone_density": row[2], "is_alert": row[3]}

@app.get("/alerts/recent")
def get_recent_alerts():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT timestamp, person_count, max_zone_density, is_alert
        FROM crowd_events
        ORDER BY id DESC LIMIT 10
    """)
    rows = cur.fetchall()
    conn.close()
    return [
        {"timestamp": r[0], "person_count": r[1], "zone_density": r[2], "is_alert": r[3]}
        for r in rows
    ]

@app.get("/fall/latest")
def get_latest_fall():
    conn = get_db()
    cur = conn.cursor()
    # check if fall_events table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fall_events'")
    if not cur.fetchone():
        conn.close()
        return {"is_fall": False, "timestamp": None}
    cur.execute("SELECT timestamp, is_fall FROM fall_events ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    if not row:
        return {"is_fall": False, "timestamp": None}
    return {"is_fall": bool(row[1]), "timestamp": row[0]}