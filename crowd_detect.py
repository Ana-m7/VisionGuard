import cv2
import sqlite3
import os
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict

# --- CONFIG ---
VIDEO_PATH = "test2.mp4"   # change to your video filename
CROWD_THRESHOLD = 7                # people per zone to trigger alert
GRID_ROWS = 3
GRID_COLS = 3
EVIDENCE_DIR = "evidence_frames"
DB_PATH = "crowd_events.db"
LOG_EVERY = 15                     # process every N frames

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

# --- DENSITY DETECTION ---
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
    is_crowd = max_density >= CROWD_THRESHOLD
    return is_crowd, max_density, zone_counts

# --- DRAW GRID OVERLAY ---
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

            # red if crowded, faint white otherwise
            if count >= CROWD_THRESHOLD:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(frame, str(count), (x1 + 10, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                if count > 0:
                    cv2.putText(frame, str(count), (x1 + 10, y1 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return frame

# --- MODEL ---
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(VIDEO_PATH)
frame_num = 0

print("Starting CrowdGuard... Press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    if frame_num % LOG_EVERY != 0:
        continue

    results = model(frame, classes=[0], verbose=False)[0]
    person_count = len(results.boxes)
    is_alert, max_density, zone_counts = check_crowd_density(results.boxes, frame.shape)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # save evidence frame on alert
    frame_path = ""
    if is_alert:
        frame_path = f"{EVIDENCE_DIR}/frame_{frame_num}.jpg"
        cv2.imwrite(frame_path, frame)

    # log to db
    cur.execute(
        "INSERT INTO crowd_events (timestamp, person_count, max_zone_density, is_alert, frame_path) VALUES (?, ?, ?, ?, ?)",
        (timestamp, person_count, max_density, int(is_alert), frame_path)
    )
    conn.commit()

    # draw grid + annotations
    frame = draw_grid(frame, zone_counts)
    annotated = results.plot(img=frame)

    color = (0, 0, 255) if is_alert else (0, 255, 0)
    status = "⚠ CROWD ALERT" if is_alert else "OK"
    cv2.putText(annotated, f"Persons: {person_count} | Zone density: {max_density} | {status}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    annotated = cv2.resize(annotated, (1280, 720))

    cv2.imshow("CrowdGuard", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
conn.close()
cv2.destroyAllWindows()
print(f"Done. Events logged to {DB_PATH}")