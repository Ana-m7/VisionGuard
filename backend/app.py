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

from fastapi import (Depends, FastAPI, HTTPException, Request, Response as FResponse,
                     WebSocket, WebSocketDisconnect)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from detector.engine import FrameStore, LiveStats, alert_queue, init_db, start_engine
from backend import auth
from backend.escalation import escalation_loop

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
    auth.init_auth_db()
    auth.purge_expired_sessions()
    if auth.user_count() == 0:
        print("No accounts yet — open the dashboard to register the administrator.")
    start_engine(frame_store, live_stats)
    broadcaster = asyncio.create_task(broadcast_alerts())
    # The escalation sweep pushes its events onto the same queue the camera
    # threads use, so escalations reach dashboards over the existing socket.
    escalator = asyncio.create_task(escalation_loop(alert_queue.put))
    yield
    broadcaster.cancel()
    escalator.cancel()


app = FastAPI(title="VisionGuard", lifespan=lifespan)

# Same-origin only: the dashboard is served by this process, and session
# cookies are not sent cross-origin with credentials under a wildcard anyway.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"http://{config.HOST}:{config.PORT}",
                   f"http://localhost:{config.PORT}"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================
# AUTH DEPENDENCIES
# ================================================================
def current_user(request: Request):
    """Resolve the session cookie, or 401. Cookies (not bearer tokens) because
    the MJPEG feeds are <img> tags, which cannot send an Authorization header."""
    user = auth.user_for_token(request.cookies.get(config.SESSION_COOKIE))
    if not user:
        raise HTTPException(status_code=401, detail="Not signed in")
    return user


def require_admin(user=Depends(current_user)):
    if user["role"] != auth.ROLE_ADMIN:
        raise HTTPException(status_code=403, detail="Administrator access required")
    return user

os.makedirs(config.EVIDENCE_DIR, exist_ok=True)


@app.get("/evidence/{filename}")
def evidence(filename: str, user=Depends(current_user)):
    """Alert evidence snapshots, addressed by filename from the DB rows.
    Served through a route rather than a StaticFiles mount so the session
    check applies — a mount would have left every snapshot public."""
    # basename() so a crafted "../.." filename cannot escape the folder
    path = os.path.join(config.EVIDENCE_DIR, os.path.basename(filename))
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="No such snapshot")
    return FileResponse(path, media_type="image/jpeg")

# The dashboard is served from this same process, so the whole demo is one
# URL (http://127.0.0.1:8000) with no separate static server.
FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")


NO_CACHE = {"Cache-Control": "no-store, must-revalidate"}


@app.get("/")
def dashboard(request: Request):
    # no-store: the dashboard is edited between runs, and a cached copy would
    # silently show stale camera labels and config after a restart.
    if not auth.user_for_token(request.cookies.get(config.SESSION_COOKIE)):
        return RedirectResponse("/login", status_code=302)
    return FileResponse(os.path.join(FRONTEND_DIR, "dashboard.html"),
                        headers=NO_CACHE)


@app.get("/login")
def login_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "login.html"),
                        headers=NO_CACHE)


# ================================================================
# AUTH
# ================================================================
def _set_session_cookie(response, token):
    response.set_cookie(
        config.SESSION_COOKIE, token,
        httponly=True,           # unreadable from JavaScript
        samesite="lax",
        max_age=config.SESSION_TTL_HOURS * 3600,
        path="/",
    )


@app.get("/auth/status")
def auth_status(request: Request):
    """Drives the login page: before any account exists it offers the one-time
    administrator registration, afterwards it offers sign-in."""
    user = auth.user_for_token(request.cookies.get(config.SESSION_COOKIE))
    return {"needs_setup": auth.user_count() == 0,
            "signed_in": bool(user), "user": user}


@app.post("/auth/register")
def auth_register(payload: dict, response: FResponse):
    """First-run only: creates the administrator. Once any account exists this
    route is closed, so it can never be used to grant yourself admin later."""
    if auth.user_count() > 0:
        raise HTTPException(status_code=403,
                            detail="Setup already complete — ask an administrator for an account")
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    uid = auth.create_user(username, password, auth.ROLE_ADMIN,
                           (payload.get("full_name") or "").strip() or None,
                           (payload.get("phone") or "").strip() or None,
                           (payload.get("telegram_chat_id") or "").strip() or None)
    _set_session_cookie(response, auth.create_session(uid))
    return {"ok": True, "role": auth.ROLE_ADMIN}


@app.post("/auth/login")
def auth_login(payload: dict, response: FResponse):
    user = auth.get_user_by_username((payload.get("username") or "").strip())
    if not user or not auth.verify_password(
            payload.get("password") or "", user["password_hash"], user["salt"]):
        # One message for both cases — revealing which half was wrong tells an
        # attacker which usernames exist.
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    _set_session_cookie(response, auth.create_session(user["id"]))
    return {"ok": True, "role": user["role"], "full_name": user["full_name"]}


@app.post("/auth/logout")
def auth_logout(request: Request, response: FResponse):
    auth.destroy_session(request.cookies.get(config.SESSION_COOKIE))
    response.delete_cookie(config.SESSION_COOKIE, path="/")
    return {"ok": True}


@app.get("/auth/me")
def auth_me(user=Depends(current_user)):
    return {**user, "role_label": auth.ROLE_LABELS.get(user["role"], user["role"])}


# ================================================================
# USER MANAGEMENT (administrator only)
# ================================================================
@app.get("/users")
def get_users(user=Depends(require_admin)):
    return auth.list_users()


@app.post("/users")
def add_user(payload: dict, user=Depends(require_admin)):
    """Create a guard (or another administrator). The phone number is what the
    escalation ladder notifies, so it is required."""
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    role = payload.get("role") or auth.ROLE_GUARD
    phone = (payload.get("phone") or "").strip()
    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    if role not in (auth.ROLE_ADMIN, auth.ROLE_GUARD):
        raise HTTPException(status_code=400, detail="Unknown role")
    chat_id = (payload.get("telegram_chat_id") or "").strip()
    if phone and not phone.startswith("+"):
        raise HTTPException(
            status_code=400,
            detail="Phone must be in international format, e.g. +919876543210")
    if chat_id and not chat_id.lstrip("-").isdigit():
        raise HTTPException(
            status_code=400, detail="Telegram chat ID must be numeric")
    # Reachable on at least one transport, or escalation can never notify them
    if not phone and not chat_id:
        raise HTTPException(
            status_code=400,
            detail="Give a Telegram chat ID or a phone number so alerts can reach them")
    try:
        uid = auth.create_user(username, password, role,
                               (payload.get("full_name") or "").strip() or None,
                               phone or None, chat_id or None)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True, "id": uid}


@app.delete("/users/{user_id}")
def remove_user(user_id: int, user=Depends(require_admin)):
    if user_id == user["id"]:
        raise HTTPException(status_code=400, detail="You cannot remove your own account")
    if not auth.delete_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return {"ok": True}


def get_db():
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
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
def video_feed(cam_id: int, user=Depends(current_user)):
    if cam_id not in config.VIDEO_SOURCES:
        raise HTTPException(status_code=404, detail="Unknown camera")
    return StreamingResponse(
        generate_stream(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws/alerts")
async def ws_alerts(ws: WebSocket):
    # The handshake carries cookies like any request, so the session is
    # checked before the socket is accepted — an unauthenticated client
    # never joins the alert broadcast.
    if not auth.user_for_token(ws.cookies.get(config.SESSION_COOKIE)):
        await ws.close(code=1008)
        return
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
def get_stats(user=Depends(current_user)):
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
def get_latest_crowd(user=Depends(current_user)):
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
def get_latest_fall(user=Depends(current_user)):
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
def get_recent_alerts(limit: int = 50, user=Depends(current_user)):
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
def analytics_occupancy(minutes: int = 30, user=Depends(current_user)):
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
def analytics_alerts(hours: int = 12, user=Depends(current_user)):
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
def analytics_zones(minutes: int = 30, cam_id: int = None,
                    user=Depends(current_user)):
    """Summed per-zone person counts over the window — movement-hotspot data.

    Without cam_id this sums every camera together, which mixes unrelated
    scenes. Pass a cam_id to see where people cluster in one camera's frame,
    which is the view an operator can actually act on.
    """
    cutoff = (datetime.now() - timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db()
    cur = conn.cursor()
    if cam_id is not None:
        cur.execute(
            "SELECT zones FROM occupancy_samples WHERE timestamp >= ? AND cam_id = ?",
            (cutoff, cam_id))
    else:
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
    # Cameras that have produced at least one sample in the window, so the
    # dashboard can offer only cameras that actually have data.
    cams = [r[0] for r in cur.execute(
        "SELECT DISTINCT cam_id FROM occupancy_samples WHERE timestamp >= ? ORDER BY cam_id",
        (cutoff,)).fetchall()]
    conn.close()
    return {"rows": config.GRID_ROWS, "cols": config.GRID_COLS, "cam_id": cam_id,
            "cameras": cams, "total": total,
            "max": max(v for row in total for v in row)}


@app.get("/analytics/by_camera")
def analytics_by_camera(hours: int = 24, user=Depends(current_user)):
    """Alert counts per camera, split by type, over the window.

    Answers the question a supervisor acts on — *which location generates the
    incidents* — rather than which hour of the day they happened in.
    """
    cutoff = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db()
    cur = conn.cursor()
    per_cam = {}
    for cam in config.VIDEO_SOURCES:
        per_cam[cam] = {"cam_id": cam, "crowd": 0, "fall": 0}
    cur.execute(
        """SELECT cam_id, COUNT(*) n FROM crowd_events
           WHERE is_alert = 1 AND timestamp >= ? GROUP BY cam_id""", (cutoff,))
    for r in cur.fetchall():
        per_cam.setdefault(r["cam_id"], {"cam_id": r["cam_id"], "crowd": 0, "fall": 0})
        per_cam[r["cam_id"]]["crowd"] = r["n"]
    cur.execute(
        """SELECT cam_id, COUNT(*) n FROM fall_events
           WHERE is_fall = 1 AND timestamp >= ? GROUP BY cam_id""", (cutoff,))
    for r in cur.fetchall():
        per_cam.setdefault(r["cam_id"], {"cam_id": r["cam_id"], "crowd": 0, "fall": 0})
        per_cam[r["cam_id"]]["fall"] = r["n"]
    conn.close()
    rows = sorted(per_cam.values(),
                  key=lambda d: -(d["crowd"] + d["fall"]))
    return {"hours": hours, "cameras": rows}


ALERT_TABLES = {"gathering": ("crowd_events", "is_alert"),
                ("fall"): ("fall_events", "is_fall")}


@app.post("/alerts/ack")
def acknowledge_alert(type: str, id: int, user=Depends(current_user)):
    """Persist an operator acknowledgement, so the audit trail survives page
    reloads and restarts — and so the alert stops climbing the escalation
    ladder.

    This acknowledges the named alert *and every earlier unacknowledged alert
    of the same type on the same camera*. A continuing situation writes a
    fresh row each time the detector flips state, so a single incident can
    leave a dozen rows behind. Acknowledging only the one the operator
    happened to click would leave the rest escalating and the notifications
    arriving long after the situation was handled.
    """
    entry = ALERT_TABLES.get(type)
    if not entry:
        raise HTTPException(status_code=400, detail="Unknown alert type")
    table, flag = entry
    when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db()
    cur = conn.cursor()
    row = cur.execute(f"SELECT cam_id FROM {table} WHERE id = ?", (id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Alert not found")
    cur.execute(
        f"""UPDATE {table} SET acknowledged_at = ?
            WHERE {flag} = 1 AND acknowledged_at IS NULL
              AND cam_id IS ? AND id <= ?""",
        (when, row["cam_id"], id))
    updated = cur.rowcount
    conn.commit()
    conn.close()
    return {"ok": True, "acknowledged_at": when, "acknowledged": updated}


@app.post("/alerts/ack_all")
def acknowledge_all(user=Depends(current_user)):
    """Acknowledge every outstanding alert — the 'Clear Alerts' control.

    Clearing used to be cosmetic: it reset the dashboard's own state and left
    the rows unacknowledged, so the escalation ladder kept climbing and
    notifications kept arriving for alerts the operator had visibly dismissed.
    """
    when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db()
    cur = conn.cursor()
    total = 0
    for table, flag in ALERT_TABLES.values():
        cur.execute(
            f"""UPDATE {table} SET acknowledged_at = ?
                WHERE {flag} = 1 AND acknowledged_at IS NULL""", (when,))
        total += cur.rowcount
    conn.commit()
    conn.close()
    return {"ok": True, "acknowledged": total, "acknowledged_at": when}


@app.get("/export/alerts.csv")
def export_alerts_csv(user=Depends(current_user)):
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
def clear_alerts(user=Depends(require_admin)):
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
    # Pop the dashboard open in the default browser once the server is up.
    # threading.Timer fires after uvicorn.run() has already taken over the
    # main thread, so the delay just gives the socket time to start listening.
    import threading
    import webbrowser

    url = f"http://{config.HOST}:{config.PORT}"
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"Opening {url} in your browser…")

    # No auto-reload: the detector threads live in this process, and a
    # reload supervisor would restart them on every file change.
    uvicorn.run(app, host=config.HOST, port=config.PORT)
