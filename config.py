"""Central configuration for VisionGuard.

Every tunable lives here so the detector, backend, and docs stay in sync.
Paths are resolved relative to this file, so the app runs from any CWD.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv(path):
    """Read KEY=VALUE lines from a local .env file into the environment, so
    secrets such as TELEGRAM_BOT_TOKEN need not be exported before every run.
    The file is gitignored. A variable already set in the shell wins."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export "):].strip()
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


_load_dotenv(os.path.join(ROOT, ".env"))

# ── Cameras ──────────────────────────────────────────────────────
# cam_id → video source. An int (e.g. 0) selects a local webcam,
# a string path selects a video file, an rtsp:// URL selects an IP camera.
VIDEO_SOURCES = {
    # Video file disconnected — live webcam only. To reconnect, uncomment
    # this line and add 2 back to ACTIVE_CAMS in frontend/dashboard.html:
    # 2: os.path.join(ROOT, "videos", "video2.mp4"),
    # video2.mp4 is the clip that exercises both detectors. Two other clips
    # were evaluated and set aside, and are not shipped in the repository:
    # video3.mp4 contains no fall at all, and video4.mp4 films a walkway from
    # an upper floor where nobody is large enough to detect — 0 of 184
    # sampled frames reached the crowd threshold. Drop a file into videos/
    # and add it here (plus ACTIVE_CAMS in frontend/dashboard.html) to
    # bring another camera online.

    # The only wired camera: overhead courtyard footage with a fall. Video
    # frames only; OpenCV never decodes the audio track.
    1: os.path.join(ROOT, "videos", "videofall06.mp4"),

    # Unwired. To reconnect a camera, uncomment its line and add its id to
    # ACTIVE_CAMS in frontend/dashboard.html.
    # 2: os.path.join(ROOT, "videos", "video3.mp4"),  # YouTube test clip
    #                                    (https://youtu.be/8DJi4PrdFCg), not shipped
    # 4: os.path.join(ROOT, "videos", "fall.mp4"),
}

CAMERA_LABELS = {
    1: "Courtyard",
    2: "Corridor B",
    3: "Front Porch",
    4: "Webcam",
}

# Local webcam as CAM-04, alongside the file-backed cameras. If no webcam is
# present the camera thread exits cleanly and the dashboard shows its
# placeholder — the rest of the system is unaffected.
ENABLE_WEBCAM = False
if ENABLE_WEBCAM:
    VIDEO_SOURCES[4] = 0  # cv2 device index (also add 4 to ACTIVE_CAMS)

# ── Detection ────────────────────────────────────────────────────
# TWO models per camera, not one — a deliberate deviation from this
# project's original single-model design (see README "Design decision: one
# model, not two"). That design assumed the pose model's own person boxes
# count crowds about as well as a dedicated detection model, which
# benchmarking on normal-angle footage supported. It breaks down on this
# demo's steep top-down CCTV footage with occluded people.
#
# DETECTION_MODEL_PATH → crowd counting (zone_densities, person_count)
# POSE_MODEL_PATH      → fall detection (classify_persons, keypoints) —
#                         needs a pose model, since plain detection has no
#                         keypoints at all.
#
# A fixed per-camera crop+zoom preprocessing step (run the single pose
# model on just a cropped/upscaled region) was also tried and measured
# even better accuracy at no speed cost — but it only sees inside the
# cropped rectangle. Anyone outside it (e.g. a person near the frame edge)
# is completely invisible to the detector, which is not an acceptable
# blind spot for a fall-detection system. Reverted in favour of this
# two-model, full-frame approach: slower, but nobody in the frame goes
# unseen.
#
# Crowd model is nano, not small: on a sample frame with 6 real people,
# yolov8s (small) reported 8 boxes, but two pairs of those boxes heavily
# overlap the same real person (confirmed visually) — it was
# double-counting, not finding 2 more real people, and it was live-observed
# over-counting a 5-person scene as 8. yolov8n (nano) found exactly 6 with
# no overlapping duplicates, so it is both faster AND more correct here —
# the harder pose/keypoint task is where the bigger model's extra capacity
# actually helps (yolov8s-pose: 5/6 found vs yolov8n-pose: 2/6).
#
# Measured on an i5-1135G7 (CPU only, no CUDA): ~250ms (nano detect) +
# ~270ms (small pose) per frame run sequentially, so about 2.8 detection
# cycles/sec per camera. Capture and inference are decoupled threads, so
# the live video stays perfectly smooth regardless — only the
# detection/alert refresh rate is affected.
DETECTION_MODEL_PATH = os.path.join(ROOT, "models", "yolov8n.pt")
POSE_MODEL_PATH = os.path.join(ROOT, "models", "yolov8s-pose.pt")

CROWD_MIN_PERSONS = 4    # total persons in frame to trigger a crowd alert
CROWD_CLEAR_MARGIN = 2   # hysteresis: once alerting, the count must fall to
                          # CROWD_MIN_PERSONS - this before the alert clears.
                          # Without it a scene hovering on the threshold
                          # (3-4-3-4 people) flaps in and out of alert every
                          # few seconds, filling the log and the escalation
                          # ladder with hundreds of duplicate incidents.
CROWD_CONFIRM_CYCLES = 3  # consecutive detection cycles at or above
                          # CROWD_MIN_PERSONS before a crowd alert is raised
                          # (~1s), so a one-frame miscount never raises one
CROWD_THRESHOLD = 3       # persons in one grid cell to highlight it red
GRID_ROWS = 3
GRID_COLS = 4
# Minimum seconds between inference passes per camera. Capture and inference
# run in separate threads, so this throttles only how often detections
# refresh — it never affects video smoothness. Raise it to free CPU on a
# slower machine, lower it (0 = flat out) for fresher boxes.
DETECT_INTERVAL = 0.12

DETECTION_CONF_MIN = 0.25   # min box confidence to count as a person at all —
                            # filters phantom detections on live cameras.
                            # Lowered from 0.45 for the top-down CCTV demo
                            # footage, where partially occluded people in a
                            # crowd score lower confidence than the
                            # upright, unoccluded COCO training images this
                            # model was trained on.

FALL_DETECTION_CONF_MIN = 0.1  # separate, lower confidence floor used only
                            # for the pose model's own "is this a person"
                            # box score — crowd counting keeps the stricter
                            # DETECTION_CONF_MIN above. A person lying flat
                            # doesn't look like a "person" box to a model
                            # trained mostly on upright people: measured on
                            # test footage at only ~0.13 box confidence,
                            # despite the model finding their shoulder/hip/
                            # ankle keypoints at 0.8-0.95 confidence — the
                            # skeleton is read fine, the box score just
                            # undersells it. Safe to lower only for fall
                            # detection because a false "person" here still
                            # has to clear keypoint confidence, torso angle,
                            # and the leg veto before it is ever called a
                            # fall.
FALL_TRACKER_CONFIG = os.path.join(ROOT, "detector", "fall_tracker.yaml")
                            # ByteTrack's stock config refuses to ever start
                            # a track below 0.25 confidence, which would
                            # silently undo FALL_DETECTION_CONF_MIN above —
                            # a person lying flat the whole time never gets
                            # a track ID and is invisible to fall detection.
                            # See detector/fall_tracker.yaml.

NESTED_BOX_CONTAINMENT = 0.9  # a person box at least this much inside a more
                            # confident person box is a duplicate (typically
                            # the head detected again inside the body box)
                            # and is dropped. Measured on video3.mp4: one
                            # visitor was counted as two in about 1 frame
                            # in 3, turning 2 people into a "crowd" of 3.

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
FALL_TORSO_FORESHORTEN_RATIO = 0.25  # torso shorter than this × box height
                            # means the shoulder/hip span has collapsed in
                            # the projection (lying down under a steep
                            # overhead camera foreshortens the torso this
                            # way — measured ~0.20 for a real fall in
                            # top-down test footage vs. ~0.29-0.34 for
                            # standing/bending). The leg-veto ratio assumes
                            # torso_len is a sane reference scale; once it
                            # collapses that assumption breaks; a person
                            # lying flat can have their ankle project well
                            # below their hip purely from camera geometry,
                            # not from standing. Below this ratio the veto
                            # is skipped and the torso-angle/head-level
                            # signals decide alone.
FALL_CONFIRM_FRAMES = 2     # consecutive detection cycles (same person)
                            # before a fall alert fires.
                            # Was 4 when inference ran on every 3rd frame.
                            # Now that inference runs in its own thread it
                            # samples the newest frame at its own pace, so a
                            # short fall yields far fewer consecutive cycles:
                            # a real fall in the test footage spans only ~5
                            # sampled frames, and 4-in-a-row was no longer
                            # reachable with several cameras running.
FALL_CLEAR_SECONDS = 2.0    # a fall alert clears only after nobody has been
                            # seen down for this long. Without it, one cycle
                            # where the pose model misses the person ended the
                            # alert and raised a fresh one seconds later — the
                            # same fall logged as several incidents.

# ── Frames / streaming ───────────────────────────────────────────
FRAME_SIZE = (480, 270)   # resolution pushed to the dashboard
JPEG_QUALITY = 75
STREAM_INTERVAL = 0.05    # seconds between MJPEG frames sent to the browser

# ── Camera health monitoring ─────────────────────────────────────
CAMERA_OFFLINE_TIMEOUT = 10    # seconds of silence → camera offline
CAMERA_MONITOR_INTERVAL = 5    # how often the health monitor runs

# Tamper detection runs on every captured frame, so it works on a small
# nearest-neighbour sample of the frame rather than the full image. Nearest
# sampling keeps each sampled pixel's real value (no averaging), so the
# brightness and per-pixel-difference thresholds below mean the same thing
# they would at full resolution, at a fraction of the cost.
TAMPER_SAMPLE_SIZE = (320, 180)     # (w, h) of the analysed sample
TAMPER_BLACKOUT_THRESHOLD = 15      # mean pixel brightness below this → covered
TAMPER_UNIFORM_STD = 20             # contrast (grey std dev) below this → covered.
                                    # A webcam's auto-exposure brightens a hand
                                    # over the lens well above the blackout
                                    # threshold, but the image stays featureless.
                                    # Normal scene on the project webcam: 61-65.
TAMPER_COVERED_CONFIRM_FRAMES = 10  # consecutive dark frames before alert (~400ms)
TAMPER_FROZEN_PIXEL_DIFF = 15       # per-pixel diff below this = sensor noise
TAMPER_FROZEN_CHANGE_RATIO = 0.001  # fraction of pixels that must differ
TAMPER_FROZEN_CONFIRM_FRAMES = 1440 # consecutive frozen reads before alert
                                    # (~60s at 24 fps). Was 15 (~0.6s), which
                                    # flagged any momentarily still scene — a
                                    # person lying motionless after a fall in
                                    # an empty courtyard read as a frozen feed
                                    # and hid the fall alert. A genuinely
                                    # frozen feed stays frozen.

# ── Logging ──────────────────────────────────────────────────────
LOG_DIR = os.path.join(ROOT, "logs")
LOG_LEVEL = "INFO"               # DEBUG | INFO | WARNING | ERROR
LOG_MAX_BYTES = 5 * 1024 * 1024  # rotate at 5 MB
LOG_BACKUP_COUNT = 3             # keep 3 rotated files → 15 MB ceiling

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
