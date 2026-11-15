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
import logging
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
from log_setup import log_alert

logger = logging.getLogger("visionguard.engine")


# ================================================================
# SHARED STATE — written by camera threads, read by the API
# ================================================================
class FrameStore:
    """Latest encoded JPEG per camera."""

    def __init__(self):
        self._frames = {}
        self._last_seen = {}   # cam_id → time.time() of the last published frame
        self._tamper = {}      # cam_id → current tamper type ("covered"/"frozen")
        self._lock = threading.Lock()

    def set(self, cam_id, jpeg_bytes):
        with self._lock:
            self._frames[cam_id] = jpeg_bytes
            self._last_seen[cam_id] = time.time()

    def get(self, cam_id):
        with self._lock:
            return self._frames.get(cam_id)

    def set_tamper(self, cam_id, tamper_type):
        """Current tamper state, so a dashboard that missed the WebSocket
        push (opened later, reconnecting) still learns of it from /stats."""
        with self._lock:
            self._tamper[cam_id] = tamper_type

    def camera_status(self):
        """Per configured camera: online flag, last frame time, seconds since,
        current tamper type. A camera that has never published is offline."""
        now = time.time()
        with self._lock:
            result = {}
            for cam_id in config.VIDEO_SOURCES:
                last = self._last_seen.get(cam_id)
                result[cam_id] = {
                    "online": (last is not None
                               and now - last < config.CAMERA_OFFLINE_TIMEOUT),
                    "last_seen": round(last, 3) if last is not None else None,
                    "seconds_since": round(now - last, 1) if last is not None else None,
                    "tamper": self._tamper.get(cam_id),
                }
            return result

    def stale_cameras(self):
        """Configured cameras with no frame within CAMERA_OFFLINE_TIMEOUT —
        derived from camera_status() so the two can never disagree."""
        return [cam_id for cam_id, s in self.camera_status().items()
                if not s["online"]]


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
# TAMPER DETECTION — covered lens and frozen/replayed feed
# ================================================================
TAMPER_LABELS = {"covered": "COVERED LENS", "frozen": "FROZEN FEED"}


class TamperDetector:
    """Per-camera tamper check, run by the capture thread on every raw frame.

    Covered lens: mean brightness below TAMPER_BLACKOUT_THRESHOLD, or contrast
    (grey std dev) below TAMPER_UNIFORM_STD — a hand over a webcam is not dark
    once auto-exposure compensates, but it is featureless — for
    TAMPER_COVERED_CONFIRM_FRAMES frames in a row.
    Frozen feed: fewer than TAMPER_FROZEN_CHANGE_RATIO of pixels change by more
    than the noise floor between consecutive frames, for
    TAMPER_FROZEN_CONFIRM_FRAMES frames in a row. Covered takes precedence —
    a black frame is also a still one.

    Works on a TAMPER_SAMPLE_SIZE nearest-neighbour sample, so the per-frame
    cost stays well under a millisecond whatever the source resolution.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._prev_gray = None
        self._covered_streak = 0
        self._frozen_streak = 0
        self.brightness = self.contrast = 0.0   # last frame's stats, for logs

    def check(self, frame):
        """Returns "covered", "frozen", or None for this frame."""
        small = cv2.resize(frame, config.TAMPER_SAMPLE_SIZE,
                           interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        mean, std = cv2.meanStdDev(gray)        # one pass for both stats
        self.brightness, self.contrast = float(mean[0, 0]), float(std[0, 0])
        if (self.brightness < config.TAMPER_BLACKOUT_THRESHOLD
                or self.contrast < config.TAMPER_UNIFORM_STD):
            self._covered_streak += 1
            self._frozen_streak = 0
            self._prev_gray = gray   # valid reference once the lens is uncovered
            if self._covered_streak >= config.TAMPER_COVERED_CONFIRM_FRAMES:
                return "covered"
            return None
        self._covered_streak = 0

        if self._prev_gray is not None:
            diff = cv2.absdiff(gray, self._prev_gray)
            _, changed = cv2.threshold(
                diff, config.TAMPER_FROZEN_PIXEL_DIFF, 255, cv2.THRESH_BINARY)
            if cv2.countNonZero(changed) < config.TAMPER_FROZEN_CHANGE_RATIO * gray.size:
                self._frozen_streak += 1
            else:
                self._frozen_streak = 0
        self._prev_gray = gray       # next frame compares against this one

        if self._frozen_streak >= config.TAMPER_FROZEN_CONFIRM_FRAMES:
            return "frozen"
        return None


TAMPER_DETAILS = {"covered": "lens covered or blocked",
                  "frozen": "feed frozen or replayed"}


def tamper_event(cam_id, tamper_type, brightness=None, contrast=None, previous=None):
    """WebSocket event for a tamper state transition (None = cleared).
    previous is the tamper type being cleared, so the log says what ended.
    brightness/contrast are the triggering frame's stats, logged so the
    thresholds can be tuned against a real camera."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stats = (f" (brightness {brightness:.0f}, contrast {contrast:.0f})"
             if brightness is not None else "")
    if tamper_type:
        label = TAMPER_LABELS[tamper_type]
        log_alert(logger, logging.WARNING, "ALERT", "tamper", cam_id,
                  TAMPER_DETAILS[tamper_type] + stats)
        message = f"CAM-{cam_id:02d} — {label} detected"
    else:
        was = f"was {TAMPER_DETAILS[previous]}" if previous else "condition ended"
        log_alert(logger, logging.INFO, "CLEARED", "tamper", cam_id, was + stats)
        message = f"CAM-{cam_id:02d} — tamper condition cleared"
    return {"type": "camera_tamper", "tamper_type": tamper_type,
            "cam_id": cam_id, "timestamp": ts,
            "is_alert": int(tamper_type is not None), "message": message}


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
    # WAL lets the API read while camera threads write, and lets the camera
    # threads write without blocking each other — so a busy camera never
    # stalls another camera's alert row.
    cur.execute("PRAGMA journal_mode=WAL")
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
    for table in ("crowd_events", "fall_events"):
        _ensure_column(cur, table, "escalation_level", "INTEGER DEFAULT 0")
        _ensure_column(cur, table, "escalated_at", "TEXT")
        # Retire every pre-existing alert to the top of the ladder. Without
        # this, restarting the server would find yesterday's unacknowledged
        # alerts still "pending" and fire a burst of notifications for them.
        cur.execute(f"UPDATE {table} SET escalation_level = 2 "
                    f"WHERE escalation_level IS NULL OR escalation_level < 2")
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
                # muted so the grid reads as a reference overlay, not as data
                cv2.rectangle(frame, (x1, y1), (x2, y2), (130, 130, 130), 1)
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


# ================================================================
# ON-FRAME STATUS OVERLAYS
# ================================================================
HUD_CLEAR = (110, 200, 110)   # BGR — green
HUD_ALERT = (60, 60, 255)     # BGR — red
HUD_TEXT = (235, 235, 235)
HUD_H = 34                    # status strip height, px
STAMP_H = 22                  # evidence caption band height, px


def _dim_band(frame, y1, y2, strength):
    """Darken a horizontal band in place so text stays readable over any
    footage. Pixels outside the band are mathematically unchanged."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y1), (frame.shape[1], y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, strength, frame, 1 - strength, 0, frame)


def draw_hud(frame, cam_id, person_count, max_density, crowd_alert, fall_alert):
    """Per-detector status strip.

    Each detector gets its own labelled line, always present. The CROWD line
    never says anything about falls and the FALL line never says anything
    about crowding, so a snapshot can't be misread as the wrong alert type.
    (The old HUD printed a bare "OK" / "!! FALL DETECTED !!" banner that read
    as a verdict on the whole frame, which made crowd evidence snapshots look
    like fall alerts.)
    """
    # No camera label here: the dashboard tile header already names the
    # camera, and the evidence caption carries it for standalone snapshots.
    _dim_band(frame, 0, HUD_H, 0.45)
    noun = "person" if person_count == 1 else "persons"
    cv2.putText(frame, f"CROWD  {person_count} {noun}  peak zone {max_density}",
                (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                HUD_ALERT if crowd_alert else HUD_CLEAR, 1, cv2.LINE_AA)
    cv2.putText(frame, "FALL   DETECTED" if fall_alert else "FALL   none",
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                HUD_ALERT if fall_alert else HUD_CLEAR, 1, cv2.LINE_AA)
    return frame


def stamp_evidence(frame, caption):
    """Copy of the frame captioned with the alert it is evidence *for*.

    A crowd snapshot routinely contains a fall in progress (and vice versa),
    so without this the picture alone cannot say which alert it belongs to.
    Runs once per state transition, not per frame.
    """
    out = frame.copy()
    h = out.shape[0]
    _dim_band(out, h - STAMP_H, h, 0.65)
    cv2.putText(out, caption, (8, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                HUD_ALERT, 1, cv2.LINE_AA)
    return out


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
# CAPTURE / INFERENCE HANDOFF
# ================================================================
# How far the display may be held back to stay in step with inference.
# Caps both memory and the worst-case delay the operator sees.
MAX_DISPLAY_LAG_FRAMES = 30
DISPLAY_LAG_SMOOTHING = 0.2   # EMA factor; keeps the delay from jittering
class LatestFrame:
    """Single-slot mailbox: the capture thread drops the newest raw frame
    here and the inference thread picks it up. Old frames are overwritten
    rather than queued, so inference always works on current reality and
    never builds a backlog it has to catch up on."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0        # bumped per frame; lets the reader skip repeats
        self._generation = 0  # bumped when a video file loops

    def put(self, frame):
        with self._lock:
            self._frame = frame
            self._seq += 1
            return self._seq

    def get(self):
        with self._lock:
            return self._frame, self._seq, self._generation

    def new_generation(self):
        """Video file wrapped around — tell the inference side to reset the
        alert state machine so the first detection of the new pass alerts."""
        with self._lock:
            self._generation += 1


class Overlay:
    """Latest inference output for one camera. Written by that camera's
    inference thread, read by its capture thread every frame. Values are
    replaced wholesale, never mutated in place, so a reader always sees a
    consistent set."""

    def __init__(self):
        self._lock = threading.Lock()
        self._d = {"result": None, "counts": {}, "confirmed": [],
                   "person_count": 0, "max_density": 0,
                   "crowd_alert": False, "fall_alert": False,
                   "seq": 0}   # which captured frame these boxes describe

    def set(self, **kw):
        with self._lock:
            self._d = kw

    def get(self):
        with self._lock:
            return self._d


def annotate(frame, cam_id, ov):
    """Draw one camera's overlay onto a freshly captured frame and scale it
    to the streamed size. Runs on every captured frame (~5 ms), which is what
    keeps the feed smooth while inference plods along in its own thread."""
    out = draw_grid(frame.copy(), ov["counts"])
    if ov["result"] is not None:
        # conf=False keeps per-person labels to "id:3 person" — the
        # confidence score is noise on a 480px-wide feed.
        out = ov["result"].plot(img=out, conf=False)
    out = draw_fall_overlays(out, ov["confirmed"])
    out = cv2.resize(out, config.FRAME_SIZE)
    draw_hud(out, cam_id, ov["person_count"], ov["max_density"],
             ov["crowd_alert"], ov["fall_alert"])
    return out


# ================================================================
# CAPTURE THREAD — reads, draws, encodes, streams. Never blocks on the model.
# ================================================================
def capture_thread(cam_id, source, frame_store, latest, overlay):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("CAM-%02d failed to open source: %s", cam_id, source)
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_delay = 1.0 / src_fps
    is_file = isinstance(source, str) and os.path.isfile(source)
    # Recent frames, so the display can be held back to match the overlay
    history = deque(maxlen=MAX_DISPLAY_LAG_FRAMES + 1)
    display_lag = 0.0
    tamper = TamperDetector()
    tamper_state = None
    logger.info("CAM-%02d started: %s (%.1f fps)", cam_id, source, src_fps)

    while True:
        t_start = time.time()
        ret, frame = cap.read()

        if not ret:
            if not is_file:
                # Live cameras need a moment to warm up: the first read after
                # opening often fails before the driver delivers frames.
                time.sleep(0.5)
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video files
            latest.new_generation()
            history.clear()                      # don't show the previous pass
            display_lag = 0.0
            tamper.reset()                       # no diff across the loop seam
            continue

        # Tamper check on the raw frame, before it is handed off. Alerts only
        # on a state change, like the crowd and fall detectors.
        tamper_type = tamper.check(frame)
        if tamper_type != tamper_state:
            previous, tamper_state = tamper_state, tamper_type
            frame_store.set_tamper(cam_id, tamper_type)
            alert_queue.put(tamper_event(cam_id, tamper_type, tamper.brightness,
                                         tamper.contrast, previous))

        seq = latest.put(frame)                 # hand off for inference
        history.append((seq, frame))

        # Show the frame the current boxes were actually computed from, not
        # the newest one. Inference lands a few frames behind, so drawing its
        # output on live pixels makes the overlay visibly trail the action.
        # Holding the display back by that same lag re-aligns them: the video
        # still advances every frame (so it stays smooth) but is delayed by
        # one inference cycle, which is imperceptible on a monitoring feed.
        ov = overlay.get()
        lag = seq - ov["seq"] if ov["seq"] else 0
        display_lag += DISPLAY_LAG_SMOOTHING * (lag - display_lag)
        want = seq - min(int(display_lag), len(history) - 1)
        shown = next((f for s, f in reversed(history) if s <= want), frame)

        annotated = annotate(shown, cam_id, ov)
        ok, buf = cv2.imencode(
            ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        if ok:
            frame_store.set(cam_id, buf.tobytes())

        # Hold true source pace — the browser sees the video at real speed.
        sleep_for = frame_delay - (time.time() - t_start)
        if sleep_for > 0:
            time.sleep(sleep_for)


# ================================================================
# INFERENCE THREAD — one per camera. Owns the model, the DB and the alert
# state machine. Runs as fast as it can without ever stalling the video.
# ================================================================
def inference_thread(cam_id, latest, overlay, live_stats):
    model = YOLO(config.POSE_MODEL_PATH)  # own instance: thread safety
    conn = sqlite3.connect(config.DB_PATH, timeout=30)  # wait out brief write locks
    cur = conn.cursor()

    crowd_state = None   # None = neutral until first detection
    fall_state = False
    fall_streaks = {}    # tracker ID → consecutive fallen detections
    last_sample = 0.0    # last occupancy-sample write time
    last_seq = -1
    last_generation = 0
    next_infer = 0.0

    while True:
        frame, seq, generation = latest.get()
        if frame is None or seq == last_seq or time.time() < next_infer:
            time.sleep(0.005)       # nothing new yet, or throttled
            continue
        last_seq = seq
        next_infer = time.time() + config.DETECT_INTERVAL

        if generation != last_generation:   # video file looped
            last_generation = generation
            # Tracker IDs restart with the file, so per-person fall streaks
            # are meaningless across the seam. The crowd/fall alert states are
            # deliberately NOT reset: a looping file is one continuous scene,
            # and resetting them manufactured a brand new "incident" on every
            # lap — a fresh alert every few seconds, each escalating on its
            # own timer, which no operator could ever keep acknowledged.
            fall_streaks = {}

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
        # Hysteresis: it takes CROWD_MIN_PERSONS to raise an alert, but the
        # count has to fall CROWD_CLEAR_MARGIN below that to clear it. A scene
        # sitting on the threshold would otherwise flap in and out of alert
        # every few seconds and log each flap as a fresh incident.
        if crowd_state:
            crowd_alert = person_count >= (config.CROWD_MIN_PERSONS
                                           - config.CROWD_CLEAR_MARGIN)
        else:
            crowd_alert = person_count >= config.CROWD_MIN_PERSONS

        persons = classify_persons(result)
        confirmed = confirm_falls(persons, fall_streaks)
        fall_alert = len(confirmed) > 0
        fall_angle = next((p["angle"] for p in confirmed if p["angle"] is not None), None)
        fall_ratio = max((p["ratio"] for p in confirmed), default=0.0)

        # Publish for the capture thread to draw on every subsequent frame
        overlay.set(result=result, counts=counts, confirmed=confirmed,
                    person_count=person_count, max_density=max_density,
                    crowd_alert=crowd_alert, fall_alert=fall_alert,
                    seq=seq)
        live_stats.update(cam_id, person_count, max_density, latency_ms)

        # Evidence snapshots are rendered on demand, only when an alert fires
        annotated = None

        # State transitions → DB rows + evidence frames
        safe_ts = timestamp.replace(":", "-").replace(" ", "_")
        if crowd_alert != crowd_state:
            evidence = ""  # DB stores just the filename; the API
                           # serves it at /evidence/{filename}
            if crowd_alert:
                evidence = f"crowd_cam{cam_id}_{safe_ts}.jpg"
                if annotated is None:
                    annotated = annotate(frame, cam_id, overlay.get())
                cv2.imwrite(
                    os.path.join(config.EVIDENCE_DIR, evidence),
                    stamp_evidence(annotated,
                                   f"EVIDENCE  CROWD GATHERING  CAM-{cam_id:02d}  {timestamp}"))
            if crowd_state is not None or crowd_alert:
                cur.execute(
                    """INSERT INTO crowd_events
                       (cam_id, timestamp, person_count, max_zone_density, is_alert, frame_path)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (cam_id, timestamp, person_count, max_density,
                     int(crowd_alert), evidence))
                conn.commit()
                # logged after the insert so the line carries the row id
                people = f"{person_count} person{'' if person_count == 1 else 's'} in view"
                if crowd_alert:
                    log_alert(logger, logging.WARNING, "ALERT", "gathering", cam_id,
                              f"{people}, peak zone {max_density}", cur.lastrowid)
                else:
                    log_alert(logger, logging.INFO, "CLEARED", "gathering", cam_id,
                              people, cur.lastrowid)
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
                if annotated is None:
                    annotated = annotate(frame, cam_id, overlay.get())
                cv2.imwrite(
                    os.path.join(config.EVIDENCE_DIR, evidence),
                    stamp_evidence(annotated,
                                   f"EVIDENCE  FALL DETECTED  CAM-{cam_id:02d}  {timestamp}"))
            cur.execute(
                """INSERT INTO fall_events
                   (cam_id, timestamp, is_fall, ratio, angle, frame_path)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (cam_id, timestamp, int(fall_alert), round(fall_ratio, 2),
                 round(fall_angle, 1) if fall_angle is not None else None,
                 evidence))
            conn.commit()
            if fall_alert:
                detail = (f"torso {fall_angle:.0f} deg off vertical"
                          if fall_angle is not None else f"box ratio {fall_ratio:.2f}")
                log_alert(logger, logging.WARNING, "ALERT", "fall", cam_id,
                          detail, cur.lastrowid)
            else:
                log_alert(logger, logging.INFO, "CLEARED", "fall", cam_id,
                          "no one detected as down any more", cur.lastrowid)
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


# ================================================================
# ENTRY POINT — called from the backend's lifespan handler
# ================================================================
def start_engine(frame_store, live_stats):
    """Two threads per camera: one captures/draws/streams at the source frame
    rate, one runs the model. They share a single-slot frame mailbox and an
    overlay, so slow inference degrades detection freshness — never video
    smoothness."""
    threads = []
    for cam_id, source in config.VIDEO_SOURCES.items():
        latest, overlay = LatestFrame(), Overlay()
        for target, args in (
            (capture_thread, (cam_id, source, frame_store, latest, overlay)),
            (inference_thread, (cam_id, latest, overlay, live_stats)),
        ):
            t = threading.Thread(target=target, args=args, daemon=True)
            t.start()
            threads.append(t)
    logger.info(
        "Detection engine started — %d camera(s), %d threads",
        len(config.VIDEO_SOURCES), len(threads),
    )
    return threads
