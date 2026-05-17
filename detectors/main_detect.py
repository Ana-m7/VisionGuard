import cv2
import sqlite3
import os
import requests
import threading
import time
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict

# ================================================================
# VIDEO SOURCES
# ================================================================
VIDEO_SOURCES = {
    1: "../videos/video1.mp4",   # CAM-01 → crowd detection
    3: "../videos/video2.mp4",     # CAM-03 → fall detection
}

# ================================================================
# CONFIG
# ================================================================
CROWD_THRESHOLD   = 4
GRID_ROWS         = 3
GRID_COLS         = 3

ASPECT_RATIO_FALL = 1.3

DETECT_EVERY      = 3
PUSH_INTERVAL     = 0.05  # seconds (~20 FPS) for MJPEG smoothness

FRAME_SIZE        = (480, 270)

EVIDENCE_DIR      = "../evidence_frames"

DB_PATH           = "../crowd_events.db"

API_BASE          = "http://127.0.0.1:8000"

CAM_CROWD = 1
CAM_FALL  = 3

# ================================================================
# COOLDOWN CONFIG
# How many seconds must pass before the same detector can insert
# a new DB row (and therefore trigger a new frontend alert) again.
# This stops "same ongoing event → hundreds of rows → popup spam".
# ================================================================
CROWD_ALERT_COOLDOWN   = 30   # seconds
FALL_ALERT_COOLDOWN    = 20   # seconds
FALL_EVIDENCE_COOLDOWN = 15   # seconds between saving evidence frames

# ================================================================
# DB SETUP
# ================================================================
os.makedirs(EVIDENCE_DIR, exist_ok=True)

conn    = sqlite3.connect(DB_PATH, check_same_thread=False)
db_lock = threading.Lock()
cur     = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS crowd_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        person_count INTEGER,
        max_zone_density INTEGER,
        is_alert INTEGER,
        frame_path TEXT
    )
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS fall_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        is_fall INTEGER
    )
""")

conn.commit()


def db_insert(query, params):
    with db_lock:
        cur.execute(query, params)
        conn.commit()


# ================================================================
# FRAME PUSH SYSTEM
# ================================================================
_frames     = {}
_frame_lock = threading.Lock()


def _push_worker(cam_id):
    while True:
        with _frame_lock:
            frame = _frames.get(cam_id)

        if frame is not None:
            try:
                _, buf = cv2.imencode(
                    '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40]
                )
                requests.post(
                    f"{API_BASE}/push_frame/{cam_id}",
                    files={'frame': ('f.jpg', buf.tobytes(), 'image/jpeg')},
                    timeout=1
                )
            except Exception as e:
                print(f"Push error CAM-{cam_id}: {e}")

        time.sleep(PUSH_INTERVAL)


for cam_id in VIDEO_SOURCES.keys():
    threading.Thread(
        target=_push_worker,
        args=(cam_id,),
        daemon=True
    ).start()


def push_frame(cam_id, frame):
    with _frame_lock:
        _frames[cam_id] = frame.copy()


# ================================================================
# LOAD MODELS
# ================================================================
print("Loading models...")
crowd_model = YOLO("../yolov8n.pt")
fall_model  = YOLO("../yolov8n-pose.pt")
print("Models loaded.")


# ================================================================
# CROWD HELPERS
# ================================================================
def check_crowd_density(boxes, frame_shape):
    h, w = frame_shape[:2]
    cell_h = h // GRID_ROWS
    cell_w = w // GRID_COLS
    zone_counts = defaultdict(int)

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        col = min(cx // cell_w, GRID_COLS - 1)
        row = min(cy // cell_h, GRID_ROWS - 1)
        zone_counts[(row, col)] += 1

    max_density = max(zone_counts.values(), default=0)
    is_alert    = max_density >= CROWD_THRESHOLD
    return is_alert, max_density, zone_counts


def draw_grid(frame, zone_counts):
    h, w = frame.shape[:2]
    cell_h = h // GRID_ROWS
    cell_w = w // GRID_COLS

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            x1 = c * cell_w
            y1 = r * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h
            count = zone_counts.get((r, c), 0)

            if count >= CROWD_THRESHOLD:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(
                    frame, str(count), (x1 + 10, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    return frame


# ================================================================
# CROWD DETECTOR
# ================================================================
def run_crowd_detection(
    frame, cam_id, last_state, frame_num, timestamp, last_alert_time
):
    results      = crowd_model(frame, classes=[0], verbose=False)[0]
    person_count = len(results.boxes)

    is_alert, max_density, zone_counts = check_crowd_density(
        results.boxes, frame.shape
    )

    annotated = draw_grid(frame.copy(), zone_counts)
    annotated = results.plot(img=annotated)

    now = time.time()

    if is_alert and not last_state:
        # Log once per new crowd event (transition to alert)
        if now - last_alert_time >= CROWD_ALERT_COOLDOWN:
            last_alert_time = now
            frame_path = f"{EVIDENCE_DIR}/crowd_cam{cam_id}_{frame_num}.jpg"
            cv2.imwrite(frame_path, annotated)
            print(f"[{timestamp}] CAM-{cam_id:02d} CROWD ALERT — {person_count} persons")
            db_insert(
                """
                INSERT INTO crowd_events
                (timestamp, person_count, max_zone_density, is_alert, frame_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                (timestamp, person_count, max_density, 1, frame_path)
            )

    elif (not is_alert) and last_state:
        # Alert just cleared — log once
        print(f"[{timestamp}] CAM-{cam_id:02d} Crowd cleared")
        db_insert(
            """
            INSERT INTO crowd_events
            (timestamp, person_count, max_zone_density, is_alert, frame_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            (timestamp, person_count, max_density, 0, "")
        )

    color = (0, 0, 255) if is_alert else (0, 255, 0)
    cv2.putText(
        annotated,
        f"Persons:{person_count} | Density:{max_density}",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
    )

    annotated = cv2.resize(annotated, FRAME_SIZE)
    push_frame(cam_id, annotated)
    return is_alert, last_alert_time, annotated


# ================================================================
# FALL DETECTOR
# ================================================================
def run_fall_detection(
    frame, cam_id, last_state, timestamp,
    last_alert_time, last_evidence_time
):
    results = fall_model(frame, classes=[0], verbose=False)[0]

    fall_detected = False
    draw_frame    = frame.copy()

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        w = x2 - x1
        h = y2 - y1

        if h == 0:
            continue

        ratio = w / h

        if ratio > ASPECT_RATIO_FALL:
            fall_detected = True
            cv2.rectangle(
                draw_frame,
                (int(x1), int(y1)), (int(x2), int(y2)),
                (0, 0, 255), 3
            )
            cv2.putText(
                draw_frame,
                f"FALL ({ratio:.2f})",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
        else:
            cv2.rectangle(
                draw_frame,
                (int(x1), int(y1)), (int(x2), int(y2)),
                (0, 255, 0), 1
            )

    now = time.time()

    if fall_detected and not last_state:
        # Log once per new fall event (transition to alert)
        if now - last_alert_time >= FALL_ALERT_COOLDOWN:
            last_alert_time = now

            if now - last_evidence_time >= FALL_EVIDENCE_COOLDOWN:
                safe_ts    = timestamp.replace(':', '-')
                frame_path = f"{EVIDENCE_DIR}/fall_cam{cam_id}_{safe_ts}.jpg"
                cv2.imwrite(frame_path, draw_frame)
                last_evidence_time = now

            db_insert(
                "INSERT INTO fall_events (timestamp, is_fall) VALUES (?, ?)",
                (timestamp, 1)
            )
            print(f"[{timestamp}] CAM-{cam_id:02d} Fall DETECTED")

    elif (not fall_detected) and last_state:
        # Fall just cleared — log once
        db_insert(
            "INSERT INTO fall_events (timestamp, is_fall) VALUES (?, ?)",
            (timestamp, 0)
        )
        print(f"[{timestamp}] CAM-{cam_id:02d} Fall CLEAR")

    annotated = cv2.resize(draw_frame, FRAME_SIZE)
    color = (0, 0, 255) if fall_detected else (0, 255, 0)
    cv2.putText(
        annotated,
        "!! FALL DETECTED !!" if fall_detected else "OK",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3
    )

    push_frame(cam_id, annotated)
    return fall_detected, last_alert_time, last_evidence_time, annotated


# ================================================================
# CAMERA THREAD
# ================================================================
def camera_thread(cam_id, video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR opening video: {video_path}")
        return

    src_fps     = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_delay = 1.0 / src_fps

    frame_num         = 0
    last_crowd_state  = None
    last_fall_state   = False
    last_annotated    = None

    # Cooldown timestamps (epoch seconds) — NOT reset on video loop so
    # a looping video doesn't immediately re-fire the same alert.
    last_crowd_alert_time   = 0.0
    last_fall_alert_time    = 0.0
    last_fall_evidence_time = 0.0

    print(f"CAM-{cam_id:02d} started: {video_path}  ({src_fps:.1f} fps)")

    while True:
        t_start = time.time()

        ret, frame = cap.read()

        if not ret:
            print(f"Restarting CAM-{cam_id}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_num        = 0
            last_crowd_state = None
            last_fall_state  = False
            # cooldown timers intentionally NOT reset here
            continue

        frame_num += 1

        if frame_num % DETECT_EVERY != 0:
            if last_annotated is not None:
                push_frame(cam_id, last_annotated)
            else:
                push_frame(cam_id, cv2.resize(frame, FRAME_SIZE))
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if cam_id == CAM_FALL:
                last_fall_state, last_fall_alert_time, last_fall_evidence_time, last_annotated = \
                    run_fall_detection(
                        frame, cam_id, last_fall_state, timestamp,
                        last_fall_alert_time, last_fall_evidence_time
                    )
            else:
                last_crowd_state, last_crowd_alert_time, last_annotated = \
                    run_crowd_detection(
                        frame, cam_id, last_crowd_state,
                        frame_num, timestamp, last_crowd_alert_time
                    )

        elapsed   = time.time() - t_start
        sleep_for = frame_delay - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

    cap.release()


# ================================================================
# START THREADS
# ================================================================
threads = []

for cam_id, video_path in VIDEO_SOURCES.items():
    t = threading.Thread(
        target=camera_thread,
        args=(cam_id, video_path),
        daemon=True
    )
    t.start()
    threads.append(t)

print(f"All {len(VIDEO_SOURCES)} cameras started.")

try:
    for t in threads:
        t.join()
except KeyboardInterrupt:
    print("Shutting down...")
    conn.close()