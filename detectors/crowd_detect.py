import cv2
import sqlite3
import os
import requests
import threading
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict

# --- CONFIG ---
VIDEO_PATH = "../videos/crowd2.mp4"
CROWD_THRESHOLD = 2
GRID_ROWS = 3
GRID_COLS = 3
EVIDENCE_DIR = "evidence_frames"
DB_PATH = "../crowd_events.db"
API_URL = "http://localhost:8000/push_frame/1"

os.makedirs(EVIDENCE_DIR, exist_ok=True)

# --- DB SETUP ---
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
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
conn.commit()

# --- NON-BLOCKING FRAME PUSH ---
_push_lock = threading.Lock()
_latest_frame = [None]

def _push_worker():
    while True:
        with _push_lock:
            frame = _latest_frame[0]
        if frame is not None:
            try:
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                requests.post(API_URL, files={'frame': ('f.jpg', buf.tobytes(), 'image/jpeg')}, timeout=1)
            except:
                pass
        threading.Event().wait(0.04)  # ~25fps cap

push_thread = threading.Thread(target=_push_worker, daemon=True)
push_thread.start()

def push_frame(frame):
    with _push_lock:
        _latest_frame[0] = frame.copy()

# --- DETECTION HELPERS ---
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
    return max_density >= CROWD_THRESHOLD, max_density, zone_counts

def draw_grid(frame, zone_counts):
    h, w = frame.shape[:2]
    cell_h = h // GRID_ROWS
    cell_w = w // GRID_COLS
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            x1, y1 = c * cell_w, r * cell_h
            x2, y2 = x1 + cell_w, y1 + cell_h
            count = zone_counts.get((r, c), 0)
            if count >= CROWD_THRESHOLD:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(frame, str(count), (x1+10, y1+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 1)
                if count > 0:
                    cv2.putText(frame, str(count), (x1+10, y1+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    return frame

# --- MODEL ---
model = YOLO("../yolov8n.pt")
cap = cv2.VideoCapture(VIDEO_PATH)
frame_num = 0
last_alert_state = None
DETECT_EVERY = 3  # run YOLO every N frames

print("CrowdGuard running — streaming to dashboard")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_num = 0
        continue

    frame_num += 1

    if frame_num % DETECT_EVERY != 0:
        # push raw frame without running YOLO — keeps video smooth
        push_frame(cv2.resize(frame, (1280, 720)))
        continue

    results = model(frame, classes=[0], verbose=False)[0]
    person_count = len(results.boxes)
    is_alert, max_density, zone_counts = check_crowd_density(results.boxes, frame.shape)

    # only write to DB when state changes
    if is_alert != last_alert_state:
        last_alert_state = is_alert
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        frame_path = ""
        if is_alert:
            frame_path = f"{EVIDENCE_DIR}/crowd_{frame_num}.jpg"
            cv2.imwrite(frame_path, frame)
            print(f"[{timestamp}] CROWD ALERT — {person_count} persons, density {max_density}")
        else:
            print(f"[{timestamp}] Crowd cleared")
        cur.execute(
            "INSERT INTO crowd_events (timestamp, person_count, max_zone_density, is_alert, frame_path) VALUES (?,?,?,?,?)",
            (timestamp, person_count, max_density, int(is_alert), frame_path)
        )
        conn.commit()

    frame = draw_grid(frame, zone_counts)
    annotated = results.plot(img=frame)
    color = (0, 0, 255) if is_alert else (0, 255, 0)
    status = "!! CROWD ALERT !!" if is_alert else "OK"
    cv2.putText(annotated, f"Persons: {person_count} | Density: {max_density} | {status}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    push_frame(cv2.resize(annotated, (1280, 720)))

cap.release()
conn.close()
print("Done.")