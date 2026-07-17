"""Central configuration for VisionGuard.

Every tunable lives here so the detector, backend, and docs stay in sync.
Paths are resolved relative to this file, so the app runs from any CWD.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Cameras ──────────────────────────────────────────────────────
# cam_id → video source. An int (e.g. 0) selects a local webcam,
# a string path selects a video file, an rtsp:// URL selects an IP camera.
VIDEO_SOURCES = {
    1: os.path.join(ROOT, "videos", "video1.mp4"),
    2: os.path.join(ROOT, "videos", "video2.mp4"),
}

CAMERA_LABELS = {
    1: "Main Entrance",
    2: "Corridor B",
}

# ── Detection ────────────────────────────────────────────────────
# Single yolov8n-pose model serves both detectors: its person boxes feed
# crowd counting and its geometry feeds fall detection. Benchmarked against
# running yolov8n + yolov8n-pose together: person counts agree within ±1 on
# 92-100% of frames while inference cost is cut roughly in half.
POSE_MODEL_PATH = os.path.join(ROOT, "models", "yolov8n-pose.pt")

CROWD_THRESHOLD = 3       # persons in one grid cell to trigger a crowd alert
GRID_ROWS = 3
GRID_COLS = 4
ASPECT_RATIO_FALL = 1.3   # box w/h ratio above which a person counts as fallen
DETECT_EVERY = 3          # run inference on every Nth frame

# ── Frames / streaming ───────────────────────────────────────────
FRAME_SIZE = (480, 270)   # resolution pushed to the dashboard
JPEG_QUALITY = 75
STREAM_INTERVAL = 0.05    # seconds between MJPEG frames sent to the browser

# ── Storage ──────────────────────────────────────────────────────
EVIDENCE_DIR = os.path.join(ROOT, "evidence_frames")
DB_PATH = os.path.join(ROOT, "crowd_events.db")

# ── API ──────────────────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 8000
RECENT_WINDOW_SECONDS = 60  # how far back /stats and /latest look
