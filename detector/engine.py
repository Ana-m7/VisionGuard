"""VisionGuard detection engine.

Runs one thread per camera. Each thread owns its own YOLO model instance
(ultralytics inference is not thread-safe across threads) and its own
SQLite connection. A single yolov8n-pose model serves both detectors:
its person boxes feed crowd-density counting and its box geometry feeds
fall detection.

Annotated frames are JPEG-encoded exactly once and placed in the shared
FrameStore that the backend serves as MJPEG. No HTTP hop, no re-encoding.
"""
import os
import sqlite3
import threading
import time
from collections import defaultdict
from datetime import datetime

import cv2
from ultralytics import YOLO

import config


# ================================================================
# SHARED STATE — written by camera threads, read by the API
# ================================================================
class FrameStore:
    """Latest encoded JPEG per camera."""

    def __init__(self):
        self._frames = {}
        self._lock = threading.Lock()

    def set(self, cam_id, jpeg_bytes):
        with self._lock:
            self._frames[cam_id] = jpeg_bytes

    def get(self, cam_id):
        with self._lock:
            return self._frames.get(cam_id)


class LiveStats:
    """Real-time counters the /stats endpoint reports."""

    def __init__(self):
        self._per_cam = {}
        self._frames_processed = 0
        self._lock = threading.Lock()

    def update(self, cam_id, persons, max_density):
        with self._lock:
            self._per_cam[cam_id] = {"persons": persons, "max_density": max_density}
            self._frames_processed += 1

    def snapshot(self):
        with self._lock:
            return {
                "frames_processed": self._frames_processed,
                "persons_now": sum(c["persons"] for c in self._per_cam.values()),
                "cameras": sorted(self._per_cam.keys()),
            }


# ================================================================
# DATABASE
# ================================================================
def init_db():
    """Drop and recreate both tables. Called once at startup so schema
    changes always take effect and stale rows never confuse the
    frontend's ID-based alert deduplication."""
    os.makedirs(config.EVIDENCE_DIR, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    cur = conn.cursor()
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
            is_fall INTEGER,
            ratio REAL
        )
    """)
    conn.commit()
    conn.close()


# ================================================================
# CROWD LOGIC
# ================================================================
def zone_densities(boxes, frame_shape):
    h, w = frame_shape[:2]
    cell_h = h // config.GRID_ROWS
    cell_w = w // config.GRID_COLS
    counts = defaultdict(int)
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        col = min(int((x1 + x2) / 2) // cell_w, config.GRID_COLS - 1)
        row = min(int((y1 + y2) / 2) // cell_h, config.GRID_ROWS - 1)
        counts[(row, col)] += 1
    max_density = max(counts.values(), default=0)
    return counts, max_density


def draw_grid(frame, counts):
    h, w = frame.shape[:2]
    cell_h = h // config.GRID_ROWS
    cell_w = w // config.GRID_COLS
    for r in range(config.GRID_ROWS):
        for c in range(config.GRID_COLS):
            x1, y1 = c * cell_w, r * cell_h
            x2, y2 = x1 + cell_w, y1 + cell_h
            count = counts.get((r, c), 0)
            if count >= config.CROWD_THRESHOLD:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(frame, str(count), (x1 + 10, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
    return frame


# ================================================================
# FALL LOGIC
# ================================================================
def find_falls(boxes):
    """Return (fall_detected, worst_ratio) using box aspect ratio.
    A person wider than tall beyond ASPECT_RATIO_FALL counts as fallen."""
    worst = 0.0
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        h = y2 - y1
        if h <= 0:
            continue
        ratio = (x2 - x1) / h
        if ratio > config.ASPECT_RATIO_FALL:
            worst = max(worst, ratio)
    return worst > 0, worst


def draw_fall_boxes(frame, boxes):
    for box in boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        h = y2 - y1
        ratio = (x2 - x1) / h if h > 0 else 0
        if ratio > config.ASPECT_RATIO_FALL:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"FALL ({ratio:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame


# ================================================================
# CAMERA THREAD
# ================================================================
def camera_thread(cam_id, source, frame_store, live_stats):
    model = YOLO(config.POSE_MODEL_PATH)  # own instance: thread safety
    conn = sqlite3.connect(config.DB_PATH)
    cur = conn.cursor()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR opening camera source: {source}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_delay = 1.0 / src_fps
    is_file = isinstance(source, str) and os.path.isfile(source)

    frame_num = 0
    crowd_state = None   # None = neutral until first detection
    fall_state = False

    print(f"CAM-{cam_id:02d} started: {source} ({src_fps:.1f} fps)")

    while True:
        t_start = time.time()
        ret, frame = cap.read()

        if not ret:
            if not is_file:
                print(f"CAM-{cam_id:02d} stream lost, retrying...")
                time.sleep(1)
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video files
            frame_num = 0
            crowd_state = None
            fall_state = False
            continue

        frame_num += 1
        if frame_num % config.DETECT_EVERY == 0:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result = model(frame, classes=[0], verbose=False)[0]
            boxes = result.boxes
            person_count = len(boxes)

            counts, max_density = zone_densities(boxes, frame.shape)
            crowd_alert = max_density >= config.CROWD_THRESHOLD
            fall_alert, fall_ratio = find_falls(boxes)

            # Compose overlays: grid → pose skeletons/boxes → fall boxes
            annotated = draw_grid(frame.copy(), counts)
            annotated = result.plot(img=annotated)
            annotated = draw_fall_boxes(annotated, boxes)

            annotated = cv2.resize(annotated, config.FRAME_SIZE)
            crowd_color = (0, 0, 255) if crowd_alert else (0, 255, 0)
            fall_color = (0, 0, 255) if fall_alert else (0, 255, 0)
            cv2.putText(annotated, f"Persons:{person_count} | Density:{max_density}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, crowd_color, 2)
            cv2.putText(annotated, "!! FALL DETECTED !!" if fall_alert else "OK",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, fall_color, 2)

            # State transitions → DB rows + evidence frames
            safe_ts = timestamp.replace(":", "-").replace(" ", "_")
            if crowd_alert != crowd_state:
                frame_path = ""
                if crowd_alert:
                    frame_path = os.path.join(
                        config.EVIDENCE_DIR, f"crowd_cam{cam_id}_{safe_ts}.jpg")
                    cv2.imwrite(frame_path, annotated)
                    print(f"[{timestamp}] CAM-{cam_id:02d} CROWD ALERT — {person_count} persons")
                elif crowd_state is not None:
                    print(f"[{timestamp}] CAM-{cam_id:02d} crowd cleared")
                if crowd_state is not None or crowd_alert:
                    cur.execute(
                        """INSERT INTO crowd_events
                           (cam_id, timestamp, person_count, max_zone_density, is_alert, frame_path)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (cam_id, timestamp, person_count, max_density,
                         int(crowd_alert), frame_path))
                    conn.commit()
                crowd_state = crowd_alert

            if fall_alert != fall_state:
                if fall_alert:
                    frame_path = os.path.join(
                        config.EVIDENCE_DIR, f"fall_cam{cam_id}_{safe_ts}.jpg")
                    cv2.imwrite(frame_path, annotated)
                    print(f"[{timestamp}] CAM-{cam_id:02d} FALL DETECTED (ratio {fall_ratio:.2f})")
                else:
                    print(f"[{timestamp}] CAM-{cam_id:02d} fall cleared")
                cur.execute(
                    "INSERT INTO fall_events (cam_id, timestamp, is_fall, ratio) VALUES (?, ?, ?, ?)",
                    (cam_id, timestamp, int(fall_alert), round(fall_ratio, 2)))
                conn.commit()
                fall_state = fall_alert

            # Encode once; backend serves these bytes directly as MJPEG
            ok, buf = cv2.imencode(
                ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
            if ok:
                frame_store.set(cam_id, buf.tobytes())
            live_stats.update(cam_id, person_count, max_density)

        sleep_for = frame_delay - (time.time() - t_start)
        if sleep_for > 0:
            time.sleep(sleep_for)


# ================================================================
# ENTRY POINT — called from the backend's lifespan handler
# ================================================================
def start_engine(frame_store, live_stats):
    threads = []
    for cam_id, source in config.VIDEO_SOURCES.items():
        t = threading.Thread(
            target=camera_thread,
            args=(cam_id, source, frame_store, live_stats),
            daemon=True,
        )
        t.start()
        threads.append(t)
    print(f"Detection engine started with {len(threads)} camera(s).")
    return threads
