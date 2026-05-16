# detect.py
import cv2
import sqlite3
import time
import os
from datetime import datetime
from ultralytics import YOLO

# --- CONFIG ---
VIDEO_PATH = "sample_video.mp4"   # swap to 0 for webcam
CROWD_THRESHOLD = 10               # tune this based on your video
EVIDENCE_DIR = "evidence_frames"
DB_PATH = "crowd_events.db"

os.makedirs(EVIDENCE_DIR, exist_ok=True)

# --- DB SETUP ---
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("""
    CREATE TABLE IF NOT EXISTS crowd_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        person_count INTEGER,
        is_alert INTEGER,
        frame_path TEXT
    )
""")
conn.commit()

# --- MODEL ---
model = YOLO("yolov8n.pt")  # nano = fast, good enough for counting

cap = cv2.VideoCapture(VIDEO_PATH)
frame_num = 0
log_every = 15  # log every N frames, not every single one

print("Starting detection... Press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    if frame_num % log_every != 0:
        continue

    # run inference, only look at 'person' class (class 0)
    results = model(frame, classes=[0], verbose=False)[0]
    person_count = len(results.boxes)
    is_alert = person_count >= CROWD_THRESHOLD
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # save frame if alert
    frame_path = ""
    if is_alert:
        frame_path = f"{EVIDENCE_DIR}/frame_{frame_num}.jpg"
        cv2.imwrite(frame_path, frame)

    # log to db
    cur.execute(
        "INSERT INTO crowd_events (timestamp, person_count, is_alert, frame_path) VALUES (?, ?, ?, ?)",
        (timestamp, person_count, int(is_alert), frame_path)
    )
    conn.commit()

    # draw on frame
    color = (0, 0, 255) if is_alert else (0, 255, 0)
    label = f"Persons: {person_count} {'⚠ CROWD ALERT' if is_alert else ''}"
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    cv2.imshow("CrowdGuard Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
conn.close()
cv2.destroyAllWindows()
print(f"Done. Events logged to {DB_PATH}")