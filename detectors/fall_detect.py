import cv2
import sqlite3
import requests
import threading
from datetime import datetime
from ultralytics import YOLO

# --- CONFIG ---
VIDEO_PATH = "../videos/fall.mp4"
ASPECT_RATIO_THRESHOLD = 1.3
DB_PATH = "../crowd_events.db"
API_URL = "http://localhost:8000/push_frame/3"

# --- DB SETUP ---
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("""
    CREATE TABLE IF NOT EXISTS fall_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        is_fall INTEGER
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
        threading.Event().wait(0.04)

push_thread = threading.Thread(target=_push_worker, daemon=True)
push_thread.start()

def push_frame(frame):
    with _push_lock:
        _latest_frame[0] = frame.copy()

# --- MODEL ---
model = YOLO("../yolov8n-pose.pt")
cap = cv2.VideoCapture(VIDEO_PATH)
last_fall_state = False
DETECT_EVERY = 3

print("Fall detection running — streaming to dashboard")

frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_num = 0
        continue

    frame_num += 1

    if frame_num % DETECT_EVERY != 0:
        push_frame(cv2.resize(frame, (1280, 720)))
        continue

    results = model(frame, classes=[0], verbose=False)[0]
    fall_detected = False

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        width = x2 - x1
        height = y2 - y1
        if height == 0:
            continue
        ratio = width / height
        if ratio > ASPECT_RATIO_THRESHOLD:
            fall_detected = True
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            cv2.putText(frame, f"FALL (ratio:{ratio:.2f})",
                        (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 1)

    if fall_detected != last_fall_state:
        last_fall_state = fall_detected
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur.execute(
            "INSERT INTO fall_events (timestamp, is_fall) VALUES (?, ?)",
            (timestamp, int(fall_detected))
        )
        conn.commit()
        print(f"[{timestamp}] Fall -> {'FALL' if fall_detected else 'CLEAR'}")

    annotated = cv2.resize(frame, (1280, 720))
    status = "!! FALL DETECTED !!" if fall_detected else "OK"
    color = (0, 0, 255) if fall_detected else (0, 255, 0)
    cv2.putText(annotated, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    push_frame(annotated)

cap.release()
conn.close()
print("Done.")