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
    3: "Webcam",
}

# Local webcam as CAM-03. If no webcam is present the camera thread
# exits cleanly and the dashboard shows its placeholder — the rest of
# the system is unaffected.
ENABLE_WEBCAM = True
if ENABLE_WEBCAM:
    VIDEO_SOURCES[3] = 0  # cv2 device index

# ── Detection ────────────────────────────────────────────────────
# Single yolov8n-pose model serves both detectors: its person boxes feed
# crowd counting and its geometry feeds fall detection. Benchmarked against
# running yolov8n + yolov8n-pose together: person counts agree within ±1 on
# 92-100% of frames while inference cost is cut roughly in half.
POSE_MODEL_PATH = os.path.join(ROOT, "models", "yolov8n-pose.pt")

CROWD_MIN_PERSONS = 3     # total persons in frame to trigger a crowd alert
CROWD_THRESHOLD = 3       # persons in one grid cell to highlight it red
GRID_ROWS = 3
GRID_COLS = 4
DETECT_EVERY = 3          # run inference on every Nth frame

DETECTION_CONF_MIN = 0.45   # min box confidence to count as a person at all —
                            # filters phantom detections on live cameras

# Fall detection requires skeletal evidence — a person with no usable
# shoulder/hip keypoints (e.g. upper body only, seated at a desk) is
# never classified as fallen. Signals, checked per person:
#   1. torso angle (mid-hip → mid-shoulder vs. vertical) beyond threshold
#   2. head at or below hip level (catches falls toward/away from the
#      camera, where the projected torso angle stays small)
#   3. legs-upright veto: a person whose ankles extend well below their
#      hips is standing/bending, not fallen — bending to pick something
#      up tilts the torso past the threshold but the legs stay vertical
FALL_TORSO_ANGLE_DEG = 65   # torso this far off vertical counts as fallen
KEYPOINT_CONF_MIN = 0.3     # min keypoint confidence to trust a joint
FALL_LEG_VETO_RATIO = 0.5   # ankles lower than hips by > this × torso
                            # length means upright legs → never a fall
FALL_CONFIRM_FRAMES = 4     # consecutive detection cycles (same person)
                            # before a fall alert fires (~0.4 s of activity)

# ── Frames / streaming ───────────────────────────────────────────
FRAME_SIZE = (480, 270)   # resolution pushed to the dashboard
JPEG_QUALITY = 75
STREAM_INTERVAL = 0.05    # seconds between MJPEG frames sent to the browser

# ── Storage ──────────────────────────────────────────────────────
EVIDENCE_DIR = os.path.join(ROOT, "evidence_frames")
DB_PATH = os.path.join(ROOT, "crowd_events.db")
RESET_DB_ON_START = False       # True wipes all history on every start
OCCUPANCY_SAMPLE_SECONDS = 10   # how often each camera logs an occupancy sample

# ── API ──────────────────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 8000
RECENT_WINDOW_SECONDS = 60  # how far back /stats and /latest look
