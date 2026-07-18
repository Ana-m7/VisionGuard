"""VisionGuard detection engine.

Runs one thread per camera. Each thread owns its own YOLO model instance
(ultralytics inference is not thread-safe across threads) and its own
SQLite connection. A single yolov8n-pose model serves both detectors:
its person boxes feed crowd-density counting and its box geometry feeds
fall detection.

Annotated frames are JPEG-encoded exactly once and placed in the shared
FrameStore that the backend serves as MJPEG. No HTTP hop, no re-encoding.
"""
import json
import math
import os
import queue
import sqlite3
import threading
import time
from collections import defaultdict, deque
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

    FPS_WINDOW_SECONDS = 10

    def __init__(self):
        self._per_cam = {}
        self._frames_processed = 0
        self._cycles = deque(maxlen=500)  # timestamps of recent detection cycles
        self._lock = threading.Lock()

    def update(self, cam_id, persons, max_density, latency_ms):
        with self._lock:
            self._per_cam[cam_id] = {"persons": persons,
                                     "max_density": max_density,
                                     "latency_ms": latency_ms}
            self._frames_processed += 1
            self._cycles.append(time.time())

    def snapshot(self):
        with self._lock:
            cutoff = time.time() - self.FPS_WINDOW_SECONDS
            recent = sum(1 for t in self._cycles if t >= cutoff)
            latencies = [c["latency_ms"] for c in self._per_cam.values()]
            return {
                "frames_processed": self._frames_processed,
                "persons_now": sum(c["persons"] for c in self._per_cam.values()),
                "cameras": sorted(self._per_cam.keys()),
                "detection_fps": round(recent / self.FPS_WINDOW_SECONDS, 1),
                "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else None,
            }


# Alert events (state transitions) for real-time push to the dashboard.
# Camera threads put dicts here; the backend's broadcaster consumes them.
alert_queue = queue.Queue()


# ================================================================
# DATABASE
# ================================================================
def _ensure_column(cur, table, column, decl):
    """Add a column to an existing table if it's missing — lets the
    schema evolve without wiping the persistent history."""
    cols = [row[1] for row in cur.execute(f"PRAGMA table_info({table})")]
    if column not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def init_db():
    """Create tables if missing. History persists across restarts so the
    analytics endpoints have data to aggregate; set RESET_DB_ON_START in
    config.py to wipe everything on startup instead."""
    os.makedirs(config.EVIDENCE_DIR, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    cur = conn.cursor()
    if config.RESET_DB_ON_START:
        cur.execute("DROP TABLE IF EXISTS crowd_events")
        cur.execute("DROP TABLE IF EXISTS fall_events")
        cur.execute("DROP TABLE IF EXISTS occupancy_samples")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS crowd_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cam_id INTEGER,
            timestamp TEXT,
            person_count INTEGER,
            max_zone_density INTEGER,
            is_alert INTEGER,
            frame_path TEXT,
            acknowledged_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fall_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cam_id INTEGER,
            timestamp TEXT,
            is_fall INTEGER,
            ratio REAL,
            angle REAL,
            frame_path TEXT,
            acknowledged_at TEXT
        )
    """)
    # migrations for databases created before these columns existed
    _ensure_column(cur, "crowd_events", "acknowledged_at", "TEXT")
    _ensure_column(cur, "fall_events", "frame_path", "TEXT")
    _ensure_column(cur, "fall_events", "acknowledged_at", "TEXT")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS occupancy_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cam_id INTEGER,
            timestamp TEXT,
            person_count INTEGER,
            max_density INTEGER,
            zones TEXT
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
# FALL LOGIC — pose keypoints first, box aspect ratio as fallback
# ================================================================
# COCO keypoint indices from yolov8n-pose
KP_NOSE = 0
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_ANKLE, KP_R_ANKLE = 15, 16


def _midpoint(kxy, kconf, idx_a, idx_b):
    """Mean of the joints in the pair that clear the confidence bar,
    or None if neither does."""
    pts = [kxy[i] for i in (idx_a, idx_b) if kconf[i] >= config.KEYPOINT_CONF_MIN]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def classify_persons(result):
    """Classify every detected person as fallen or upright.

    A fall claim requires skeletal evidence — no usable shoulder/hip
    keypoints (upper body only, e.g. seated at a desk) means never
    fallen. Signals:
      1. Torso angle — the mid-hip → mid-shoulder vector vs. vertical.
         A standing person is near 0°, a person on the ground near 90°.
      2. Head level — a head at or below hip height is a fall even when
         the projected torso angle is small (falling toward/away from
         the camera foreshortens the torso).
      3. Legs-upright veto — ankles extending well below the hips mean
         the person is standing or bending over, not on the ground, so
         signals 1-2 are overridden (bending tilts the torso past the
         threshold, but only a fall takes the legs out from under you).

    Returns a list of dicts:
      {key, box, fallen, angle (deg or None), ratio}
    where key is the tracker ID when available (stable across frames).
    """
    persons = []
    boxes = result.boxes
    kpts = result.keypoints
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        h = y2 - y1
        ratio = (x2 - x1) / h if h > 0 else 0.0

        angle = None
        head_down = False
        legs_upright = False
        if kpts is not None and kpts.conf is not None:
            kxy = kpts.xy[i].tolist()
            kconf = kpts.conf[i].tolist()
            mid_sh = _midpoint(kxy, kconf, KP_L_SHOULDER, KP_R_SHOULDER)
            mid_hip = _midpoint(kxy, kconf, KP_L_HIP, KP_R_HIP)
            mid_ankle = _midpoint(kxy, kconf, KP_L_ANKLE, KP_R_ANKLE)
            if mid_sh and mid_hip:
                dx = mid_sh[0] - mid_hip[0]
                dy = mid_sh[1] - mid_hip[1]
                if dx or dy:
                    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
                if mid_ankle:
                    torso_len = math.hypot(dx, dy)
                    legs_upright = (mid_ankle[1] - mid_hip[1]
                                    > config.FALL_LEG_VETO_RATIO * torso_len)
            if mid_hip and kconf[KP_NOSE] >= config.KEYPOINT_CONF_MIN:
                # head fails to rise meaningfully above the hips
                head_down = (mid_hip[1] - kxy[KP_NOSE][1]) < 0.15 * h

        fallen = (angle is not None
                  and (angle > config.FALL_TORSO_ANGLE_DEG or head_down)
                  and not legs_upright)

        key = int(boxes.id[i]) if boxes.id is not None else f"det{i}"
        persons.append({
            "key": key,
            "box": (int(x1), int(y1), int(x2), int(y2)),
            "fallen": fallen,
            "angle": angle,
            "ratio": ratio,
        })
    return persons


def confirm_falls(persons, streaks):
    """Require FALL_CONFIRM_FRAMES consecutive fallen detections per person
    (keyed by tracker ID) before confirming, so a single misclassified
    frame never fires an alert. Returns the confirmed subset."""
    seen = set()
    confirmed = []
    for p in persons:
        seen.add(p["key"])
        if p["fallen"]:
            streaks[p["key"]] = streaks.get(p["key"], 0) + 1
            if streaks[p["key"]] >= config.FALL_CONFIRM_FRAMES:
                confirmed.append(p)
        else:
            streaks.pop(p["key"], None)
    for key in list(streaks):  # drop people who left the frame
        if key not in seen:
            del streaks[key]
    return confirmed


def draw_fall_overlays(frame, confirmed):
    for p in confirmed:
        x1, y1, x2, y2 = p["box"]
        label = (f"FALL {p['angle']:.0f}deg" if p["angle"] is not None
                 else f"FALL ratio {p['ratio']:.2f}")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, label, (x1, y1 - 10),
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
    fall_streaks = {}    # tracker ID → consecutive fallen detections
    last_sample = 0.0    # last occupancy-sample write time

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
            fall_streaks = {}
            continue

        frame_num += 1
        if frame_num % config.DETECT_EVERY == 0:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # track() keeps per-person IDs stable across frames (ByteTrack),
            # which lets fall confirmation follow individuals, not frames
            t_infer = time.perf_counter()
            result = model.track(frame, classes=[0], persist=True,
                                 conf=config.DETECTION_CONF_MIN, verbose=False)[0]
            latency_ms = (time.perf_counter() - t_infer) * 1000
            boxes = result.boxes
            person_count = len(boxes)

            counts, max_density = zone_densities(boxes, frame.shape)
            crowd_alert = person_count >= config.CROWD_MIN_PERSONS

            persons = classify_persons(result)
            confirmed = confirm_falls(persons, fall_streaks)
            fall_alert = len(confirmed) > 0
            fall_angle = next((p["angle"] for p in confirmed if p["angle"] is not None), None)
            fall_ratio = max((p["ratio"] for p in confirmed), default=0.0)

            # Compose overlays: grid → pose skeletons/boxes → fall boxes
            annotated = draw_grid(frame.copy(), counts)
            annotated = result.plot(img=annotated)
            annotated = draw_fall_overlays(annotated, confirmed)

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
                evidence = ""  # DB stores just the filename; the API
                               # serves it at /evidence/{filename}
                if crowd_alert:
                    evidence = f"crowd_cam{cam_id}_{safe_ts}.jpg"
                    cv2.imwrite(os.path.join(config.EVIDENCE_DIR, evidence), annotated)
                    print(f"[{timestamp}] CAM-{cam_id:02d} CROWD ALERT — {person_count} persons")
                elif crowd_state is not None:
                    print(f"[{timestamp}] CAM-{cam_id:02d} crowd cleared")
                if crowd_state is not None or crowd_alert:
                    cur.execute(
                        """INSERT INTO crowd_events
                           (cam_id, timestamp, person_count, max_zone_density, is_alert, frame_path)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (cam_id, timestamp, person_count, max_density,
                         int(crowd_alert), evidence))
                    conn.commit()
                    alert_queue.put({
                        "type": "gathering", "id": cur.lastrowid,
                        "cam_id": cam_id, "timestamp": timestamp,
                        "is_alert": int(crowd_alert),
                        "person_count": person_count,
                        "zone_density": max_density,
                        "frame_path": evidence,
                    })
                crowd_state = crowd_alert

            if fall_alert != fall_state:
                evidence = ""
                if fall_alert:
                    evidence = f"fall_cam{cam_id}_{safe_ts}.jpg"
                    cv2.imwrite(os.path.join(config.EVIDENCE_DIR, evidence), annotated)
                    detail = (f"torso {fall_angle:.0f} deg" if fall_angle is not None
                              else f"ratio {fall_ratio:.2f}")
                    print(f"[{timestamp}] CAM-{cam_id:02d} FALL DETECTED ({detail})")
                else:
                    print(f"[{timestamp}] CAM-{cam_id:02d} fall cleared")
                cur.execute(
                    """INSERT INTO fall_events
                       (cam_id, timestamp, is_fall, ratio, angle, frame_path)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (cam_id, timestamp, int(fall_alert), round(fall_ratio, 2),
                     round(fall_angle, 1) if fall_angle is not None else None,
                     evidence))
                conn.commit()
                alert_queue.put({
                    "type": "fall", "id": cur.lastrowid,
                    "cam_id": cam_id, "timestamp": timestamp,
                    "is_alert": int(fall_alert),
                    "ratio": round(fall_ratio, 2),
                    "angle": round(fall_angle, 1) if fall_angle is not None else None,
                    "frame_path": evidence,
                })
                fall_state = fall_alert

            # Periodic occupancy sample — the raw material for the
            # analytics endpoints (trends + zone heatmap)
            if time.time() - last_sample >= config.OCCUPANCY_SAMPLE_SECONDS:
                last_sample = time.time()
                zones_json = json.dumps(
                    [[counts.get((r, c), 0) for c in range(config.GRID_COLS)]
                     for r in range(config.GRID_ROWS)])
                cur.execute(
                    """INSERT INTO occupancy_samples
                       (cam_id, timestamp, person_count, max_density, zones)
                       VALUES (?, ?, ?, ?, ?)""",
                    (cam_id, timestamp, person_count, max_density, zones_json))
                conn.commit()

            # Encode once; backend serves these bytes directly as MJPEG
            ok, buf = cv2.imencode(
                ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
            if ok:
                frame_store.set(cam_id, buf.tobytes())
            live_stats.update(cam_id, person_count, max_density, latency_ms)

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
