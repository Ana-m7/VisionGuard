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
    2: os.path.join(ROOT, "videos", "video2.mp4"),
    # video2.mp4 is the clip that exercises both detectors. Two other clips
    # were evaluated and set aside, and are not shipped in the repository:
    # video3.mp4 contains no fall at all, and video4.mp4 films a walkway from
    # an upper floor where nobody is large enough to detect — 0 of 184
    # sampled frames reached the crowd threshold. Drop a file into videos/
    # and add it here (plus ACTIVE_CAMS in frontend/dashboard.html) to
    # bring another camera online.
}

CAMERA_LABELS = {
    1: "Main Entrance",
    2: "Corridor B",
    3: "Front Porch",
    4: "Webcam",
}

# Local webcam as CAM-04, alongside the file-backed cameras. If no webcam is
# present the camera thread exits cleanly and the dashboard shows its
# placeholder — the rest of the system is unaffected.
ENABLE_WEBCAM = True
if ENABLE_WEBCAM:
    VIDEO_SOURCES[4] = 0  # cv2 device index

# ── Detection ────────────────────────────────────────────────────
# Single yolov8n-pose model serves both detectors: its person boxes feed
# crowd counting and its geometry feeds fall detection. Benchmarked against
# running yolov8n + yolov8n-pose together: person counts agree within ±1 on
# 92-100% of frames while inference cost is cut roughly in half.
POSE_MODEL_PATH = os.path.join(ROOT, "models", "yolov8n-pose.pt")

CROWD_MIN_PERSONS = 4     # total persons in frame to trigger a crowd alert
CROWD_CLEAR_MARGIN = 2    # hysteresis: once alerting, the count must fall to
                          # CROWD_MIN_PERSONS - this before the alert clears.
                          # Without it a scene hovering on the threshold
                          # (3-4-3-4 people) flaps in and out of alert every
                          # few seconds, filling the log and the escalation
                          # ladder with hundreds of duplicate incidents.
CROWD_THRESHOLD = 3       # persons in one grid cell to highlight it red
GRID_ROWS = 3
GRID_COLS = 4
# Minimum seconds between inference passes per camera. Capture and inference
# run in separate threads, so this throttles only how often detections
# refresh — it never affects video smoothness. Raise it to free CPU on a
# slower machine, lower it (0 = flat out) for fresher boxes.
DETECT_INTERVAL = 0.12

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
FALL_CONFIRM_FRAMES = 2     # consecutive detection cycles (same person)
                            # before a fall alert fires.
                            # Was 4 when inference ran on every 3rd frame.
                            # Now that inference runs in its own thread it
                            # samples the newest frame at its own pace, so a
                            # short fall yields far fewer consecutive cycles:
                            # a real fall in the test footage spans only ~5
                            # sampled frames, and 4-in-a-row was no longer
                            # reachable with several cameras running.

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

# ── Accounts and sessions ────────────────────────────────────────
# Two roles: "admin" (creates users, clears history) and "guard" (watches
# feeds, acknowledges alerts). The first run has no accounts at all — the
# login page offers a one-time administrator registration, and once an admin
# exists that route is closed and only an admin can create further users.
SESSION_COOKIE = "visionguard_session"
SESSION_TTL_HOURS = 12
PBKDF2_ROUNDS = 200_000

# Wipes users + sessions on startup so the first-run administrator
# registration can be demonstrated again. Leave False in normal use.
RESET_USERS_ON_START = False

# ── Escalation ───────────────────────────────────────────────────
# An unacknowledged alert climbs a ladder: after the first delay the
# assigned guard is notified, after the second the administrator is.
# Timers are per alert and independent of every other alert.
ESCALATE_GUARD_SECONDS = 15
ESCALATE_ADMIN_SECONDS = 30
ESCALATION_SCAN_SECONDS = 3   # how often the background scanner sweeps

# When True, a crowd alert escalates even if the crowd has since dispersed —
# the operator is still told "there was an unhandled gathering at CAM-02".
# When False, a crowd that cleared before anyone was notified is treated as
# a non-event and suppressed. Falls always escalate regardless (a fall is
# over in a second by nature). Set True for looping demo footage, whose
# crowd scenes are too short to ever reach the escalation delay while still
# active.
ESCALATE_ENDED_CROWDS = True
# After an operator acknowledges an alert, further alerts of that same type
# on that same camera stay silent for this long. A continuing situation keeps
# producing alerts — a looping demo clip re-raises one every few seconds, and
# a real crowd that never disperses does the same — and without a suppression
# window the operator would be notified again moments after saying "I have
# this". The alerts are still recorded; only the notifications are held back.
ACK_SUPPRESS_SECONDS = 120

# At most one notification per camera, per alert type, per rung in this
# window. A repeating situation — a fall clip on a loop, or someone who keeps
# re-entering frame — raises a fresh alert every few seconds, and each one is
# genuinely an alert. Sending every one turns a phone into a nuisance and
# trains the operator to ignore it, so repeats inside the window are grouped:
# they are still recorded and still escalate, they just do not re-notify.
NOTIFY_GROUP_SECONDS = 90

ESCALATE_MAX_AGE_SECONDS = 300  # never escalate an alert older than this.
                          # Stops a backlog of stale unacknowledged alerts —
                          # e.g. after leaving the system running unattended —
                          # from firing a burst of notifications at once.

# ── Notifications (Twilio WhatsApp sandbox) ──────────────────────
# Real SMS to Indian numbers needs TRAI DLT registration (paid, 2-4 weeks),
# so the free WhatsApp sandbox is the working transport. Credentials come
# from the environment — never commit them. With any of these unset the
# notifier falls back to logging the message, so a missing key or a network
# failure can never break a running demo.
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_FROM = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+17372508034")

# ── Notifications (Telegram) ─────────────────────────────────────
# Preferred transport. Twilio's free tier rejects custom message bodies
# (error 21654 — trial accounts may only send Twilio's own templates) and
# its Content API needs a paid plan, so the alert text could not be our own.
# Telegram delivers arbitrary text to a real phone for free with no
# templates, approval or per-message quota.
#
# Recipients are chat IDs, not phone numbers: a person messages the bot once
# and the app learns their chat ID from getUpdates (see tools/telegram_ids.py).
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# Which transport to prefer when several are configured.
NOTIFY_TRANSPORT = os.environ.get("NOTIFY_TRANSPORT", "auto")  # auto|telegram|whatsapp|log

# Hard ceiling on outbound messages per hour. Twilio trial accounts get a
# small fixed allowance (100 messages), and looping demo footage can raise
# an alert every few seconds — without this, an afternoon of testing with
# nobody acknowledging would burn the entire quota. Sends past the cap are
# logged instead of sent.
NOTIFY_MAX_PER_HOUR = 20
