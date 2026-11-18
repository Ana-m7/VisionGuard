# VisionGuard

Real-time video surveillance analysis system using computer vision to detect crowd formation, person falls and camera tampering. Annotated video feeds stream live to a browser dashboard that raises alerts the instant an event happens, and alerts nobody acknowledges escalate to a guard's and then an administrator's phone.

Built with **YOLOv8**, **OpenCV**, **FastAPI**, **SQLite** and vanilla JavaScript.

## Architecture

Everything runs in a **single process**: the FastAPI server starts the detection engine's camera threads at startup, and they communicate through shared memory. There is no network hop between detector and API, and every frame is JPEG encoded exactly once.

```
┌──────────────────────────────────────────────────────────────────┐
│  backend/app.py  (FastAPI, port 8000)                            │
│                                                                  │
│  lifespan startup ──► init_db() ──► detector/engine.py           │
│                                                                  │
│   ┌────────────────────────────────────────────────────────┐     │
│   │  detector/engine.py : two threads per camera           │     │
│   │                                                        │     │
│   │  capture thread   read ► tamper check ► draw overlay   │     │
│   │                   ► JPEG encode ► FrameStore           │     │
│   │                                                        │     │
│   │  inference thread yolov8n       ► crowd counting       │     │
│   │                   yolov8s-pose  ► fall detection       │     │
│   │                   state change  ► SQLite + alert queue │     │
│   └────────────────────────────────────────────────────────┘     │
│                                                                  │
│  FrameStore bytes  → GET /video_feed/{cam_id}  (MJPEG)           │
│  Alert queue       → WS /ws/alerts  (instant push)               │
│  Camera monitor    → offline / back-online events                │
│  Escalation loop   → Telegram / WhatsApp / log notifications     │
│  LiveStats + DB    → GET /stats, /latest, /alerts/recent, ...    │
└───────────────────────────────┬──────────────────────────────────┘
                                │ HTTP + WebSocket (session cookie)
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  frontend/dashboard.html  (served by the API at /)               │
│                                                                  │
│  Live MJPEG feeds with tracking IDs and pose skeletons           │
│  Alerts over WebSocket, polling kept as fallback                 │
│  Deduplicates by DB row ID so no alert ever fires twice          │
│  Alert popup shows the evidence snapshot of the moment           │
│  Event log, analytics panels, user management, EN / HI, themes   │
└──────────────────────────────────────────────────────────────────┘
```

### Design decision: two models per camera

Each camera runs **two models in sequence** on the same frame:

| Model | Used for | Why this size |
|-------|----------|---------------|
| `yolov8n.pt` (detection) | Crowd counting and zone density | On a sample frame with 6 real people, `yolov8s` reported 8 boxes (two people double-counted), while `yolov8n` found exactly 6. Nano is both faster and more correct here. |
| `yolov8s-pose.pt` (pose) | Fall detection (keypoints) | Keypoints are the harder task, and the larger model earns its cost: `yolov8s-pose` found 5 of 6 people where `yolov8n-pose` found 2. |

An earlier version used a single `yolov8n-pose` model for both jobs. On normal-angle footage its person boxes counted crowds within one person of a detection model on 92 to 100 percent of frames, at roughly half the inference cost. That stopped holding on steep top-down CCTV footage with occluded people, which is what the current demo footage is, so crowd counting moved to a dedicated detector.

A crop-and-zoom preprocessing step (running the pose model on an upscaled region of interest) was also tried. It measured better accuracy at no speed cost, but anyone outside the cropped rectangle became invisible to the detector, which is not an acceptable blind spot for fall detection. It was reverted in favour of full-frame inference.

Cost: about 250 ms (detection) + 270 ms (pose) per cycle on an Intel i5-1135G7 CPU, so about 2.8 detection cycles per second per camera. Video smoothness is unaffected, because capture and inference run in separate threads (see below).

### Design decision: single process, frames encoded once

Earlier versions ran the detector as a separate process that JPEG encoded frames at quality 40, POSTed them to the backend over localhost HTTP, decoded them, then encoded them again at quality 75 for MJPEG streaming: four codec passes and a network hop per frame. Now camera threads encode each annotated frame **once** and place the bytes in a shared `FrameStore`; the MJPEG endpoint serves those bytes directly. Higher quality, lower latency, one command to run.

### Design decision: capture decoupled from inference

Earlier versions ran capture and inference on one thread per camera and only published a frame to the browser after an inference pass. The whole loop stalled for the duration of the model call, so a 24 fps source arrived as roughly **4 fps** of very choppy video.

Each camera now runs **two threads**. The capture thread reads, draws the latest overlay, encodes and publishes at the source frame rate; the inference thread pulls the newest frame from a single-slot mailbox and runs the models at their own pace. Measured on the project footage: **4 fps to 21 fps per camera**, two cameras at once, with no change to detection quality.

Because the mailbox holds only the newest frame, a slow model drops stale frames instead of building a backlog, so the feed can never drift behind real time.

One consequence needed handling: drawing boxes computed from an older frame onto live pixels makes the overlay visibly trail the action. The capture thread therefore keeps a short ring buffer and **displays the frame the current boxes were actually computed from**. The video still advances every frame, so it stays smooth, but it is held back by one inference cycle — imperceptible on a monitoring feed, and the skeletons sit on the bodies.

### Alert flow

1. The inference thread picks up the newest captured frame
2. On a state transition (clear to alert), a row is written to SQLite with `cam_id`, the real detection data and the evidence snapshot filename, and the event is pushed onto an internal queue
3. An async broadcaster sends the event to every dashboard connected to `WS /ws/alerts`, so alerts reach the browser **instantly**
4. The frontend also polls `/latest` and `/fall/latest` every 3 seconds as a fallback; both paths feed one handler that deduplicates by row ID, so whichever arrives first wins and nothing fires twice
5. The alert fires: camera border flashes, badge appears, popup opens with the evidence snapshot
6. If nobody acknowledges it, the escalation loop notifies a guard and then an administrator (see [Escalation ladder](#escalation-ladder))

## Project Structure

```
VisionGuard/
├── config.py               All tunables: cameras, thresholds, paths, ports, escalation
├── log_setup.py            Console + rotating file logging, structured alert lines
├── backend/
│   ├── app.py              FastAPI server: starts the engine, serves the dashboard,
│   │                       MJPEG, REST, WebSocket, camera health monitor
│   ├── auth.py             Accounts, PBKDF2 passwords, cookie sessions
│   ├── escalation.py       Unacknowledged-alert ladder: guard then administrator
│   └── notify.py           Telegram / WhatsApp delivery with a log fallback
├── detector/
│   └── engine.py           Detection engine: capture + inference threads, crowd,
│                           fall and tamper detection, DB writes, FrameStore
├── frontend/
│   ├── dashboard.html      Browser dashboard: single HTML file, no build step
│   └── login.html          Sign-in / first-run administrator registration
├── models/
│   ├── yolov8n.pt          Detection model, used for crowd counting
│   ├── yolov8s-pose.pt     Pose model, used for fall detection
│   ├── yolov8s.pt          Kept for benchmarking (not used at runtime)
│   └── yolov8n-pose.pt     Kept for benchmarking (not used at runtime)
├── videos/                 Source footage for file-backed cameras (see Cameras)
├── tools/
│   └── telegram_ids.py     Prints the chat IDs of everyone who messaged the bot
├── tests/                  Unit and API tests (pytest)
├── model_testing_suite/    Accuracy evaluation on public datasets (see below)
├── evidence_frames/        Auto created, JPEG snapshots of alert moments
├── logs/                   Auto created, rotating log files
├── crowd_events.db         Auto created, SQLite database (persists across restarts)
└── requirements.txt
```

## Detection

### Crowd detection

* The detection model finds every person in the frame (boxes below `DETECTION_CONF_MIN`, currently **0.25**, are discarded). The floor was lowered from 0.45 because partially occluded people in top-down footage score lower confidence than the upright, unoccluded people the model was trained on.
* **Duplicate boxes are removed.** YOLO sometimes reports part of a person (typically the head) as a second, lower-confidence person inside the first person's box. Standard NMS keeps it, because a small box inside a large one has low overlap, so one person was counted as two. A box at least 90 percent inside a more confident box (`NESTED_BOX_CONTAINMENT`) is dropped. Two real people overlapping in a crowd each extend outside the other's box, so both are kept.
* Counting uses **plain detection, not tracking**. At about three inference cycles per second, people walking across the frame move too far between cycles for the tracker to confirm them, and unconfirmed people are left out of its output: on the street demo clip, tracking reported 4 to 5 people where 6 to 9 were in view.
* If **4 or more** persons (`CROWD_MIN_PERSONS`) are in view for **3 consecutive detection cycles** (`CROWD_CONFIRM_CYCLES`, about one second), a crowd alert is triggered, so a single miscounted frame never raises one. Clearing requires the count to fall to **2** (`CROWD_CLEAR_MARGIN`): without that hysteresis a scene hovering on the threshold flaps in and out of alert every few seconds and logs each flap as a fresh incident.
* Independently, the frame is divided into a 3×4 grid. Each person is assigned to the cell containing their box centre, cells with `CROWD_THRESHOLD` (3) or more persons are highlighted red, and the zone counts feed the movement-hotspots panel.

### Fall detection

Based on pose keypoints, per person:

* **Torso angle:** the vector from mid-hip to mid-shoulder is measured against vertical. A standing person is near 0°, a person on the ground near 90°. Beyond **65°** counts as fallen.
* **Head level:** a head at or below hip height counts as fallen even when the projected torso angle is small. This catches falls toward or away from the camera, where perspective foreshortens the torso.
* **Legs-upright veto:** bending to pick something up tilts the torso past the threshold, but the ankles stay far below the hips. When the legs are clearly extended beneath a person, the fall signals are overridden.
* **Foreshortening exception:** under a steep overhead camera, a person lying flat projects a very short torso (measured about 0.20 of box height for a real fall, against 0.29 to 0.34 for standing or bending). Once the torso is shorter than `FALL_TORSO_FORESHORTEN_RATIO` of the box height it is no longer a trustworthy scale for the leg check, so the veto is skipped and torso angle and head level decide alone.
* **Skeletal evidence is required.** A person whose shoulders or hips are not visible (for example upper body only, seated at a desk) is never classified as fallen. A wide bounding box alone is not treated as a fall, since partial views routinely produce wide boxes.
* **Per-person confirmation:** ByteTrack assigns each person a persistent ID, and a fall alert only fires after the *same person* is classified as fallen for `FALL_CONFIRM_FRAMES` (2) consecutive detection cycles, so a single misclassified frame never triggers an alert. The count is deliberately low because inference samples the newest frame rather than every Nth one, and a real fall spans well under a second.
* **Clearing hysteresis:** a fall alert clears only after nobody has been seen down for `FALL_CLEAR_SECONDS` (2 s). Without it, one cycle in which the pose model missed the person ended the alert and raised a new one seconds later, logging the same fall as several incidents.
* The triggering torso angle is stored with the event.

### Person tracking

The pose model runs through `model.track()` (ByteTrack) rather than plain inference, so every person keeps a stable ID across frames, used to anchor fall confirmation to individuals.

### Tamper detection

The capture thread checks every raw frame, on a small 320×180 sample so the cost stays well under a millisecond:

* **Covered lens:** mean brightness below `TAMPER_BLACKOUT_THRESHOLD`, or contrast (greyscale standard deviation) below `TAMPER_UNIFORM_STD`, for 10 consecutive frames. The contrast test matters because a webcam's auto-exposure brightens a hand over the lens well above the blackout threshold, but the image stays featureless.
* **Frozen feed:** fewer than 0.1 percent of pixels change by more than sensor noise between consecutive frames, for 1,440 consecutive frames (about 60 seconds at 24 fps). Catches a stalled stream or a replayed still image. The window is deliberately long: at 15 frames (0.6 s), a person lying motionless after a fall in an otherwise empty scene read as a frozen feed and hid the fall alert.

Covered takes precedence, since a black frame is also a still one. Tamper alerts fire and clear on state change, like the other detectors.

### Camera health

A background monitor reports a camera as **offline** once it has sent no frame for `CAMERA_OFFLINE_TIMEOUT` (10 s), and as **back online** when frames resume, each on state change only. The first check waits one full timeout so cameras still starting up are not reported. Current offline and tamper state is also included in `/stats`, so a dashboard that missed the push still catches up.

### State transition alerts

No cooldown timers. Each inference thread tracks its previous detection state, and a DB row is written **only when state changes**: clear to alert writes an alert row (plus an evidence frame), alert to clear writes a clear row.

A looping video file is treated as one continuous scene rather than a new one each lap. Only the per-person fall streaks reset at the seam, because tracker IDs restart with the file. Resetting the alert state instead manufactured a fresh incident every lap — a new alert every few seconds, each escalating on its own timer, which no operator could keep acknowledged.

### Evidence gallery

A snapshot JPEG is saved to `evidence_frames/` at the moment every alert triggers, captioned with the alert it is evidence for, and its filename is stored on the alert row. The API serves it at `/evidence/{filename}` (session required), the alert popup shows it, and clicking any event log entry reopens it.

### Operator acknowledgements

Acknowledging an alert persists a timestamp on the alert row (`POST /alerts/ack`). Acknowledgements survive page reloads and server restarts, are included in the CSV export, and **stop the alert climbing the escalation ladder** — the ladder only considers rows where `acknowledged_at IS NULL`.

Acknowledging one alert also acknowledges every earlier unacknowledged alert of the same type on the same camera. A continuing situation writes a fresh row each time the detector flips state, so one incident can leave a dozen rows behind; acknowledging only the row the operator happened to click would leave the rest escalating. **Clear Alerts** (`POST /alerts/ack_all`) acknowledges everything outstanding.

## Accounts, roles and escalation

### Roles

| Role | Can do |
|------|--------|
| **Administrator** | Everything below, plus create/remove users and clear history. Receives the second escalation rung. |
| **Security Guard** | View feeds, acknowledge alerts, export CSV. Receives the first escalation rung. |

The first run has **no accounts**. The login page offers a one-time administrator registration; once any account exists that route is closed (403) and only an administrator can create further users. Set `RESET_USERS_ON_START = True` in `config.py` to wipe accounts and see that first-run screen again — it leaves alert history untouched.

Sessions are opaque random tokens stored in SQLite and delivered in an **HttpOnly cookie**, not a bearer token. That is forced by the design: the camera feeds are plain `<img src="/video_feed/2">` tags, and an image tag cannot attach an `Authorization` header — only a cookie rides along automatically. Passwords use PBKDF2-HMAC-SHA256 (200,000 rounds, per-user salt) from the standard library, so there is no bcrypt or argon2 dependency.

Every data route is gated, including the MJPEG streams, the evidence snapshots and the WebSocket handshake.

### Escalation ladder

An alert nobody acknowledges escalates on its own:

```
alert fires
   ── 15s unacknowledged ──►  notify security guards
   ── 30s unacknowledged ──►  notify administrators
```

Each alert row carries its own `escalation_level` (0 → 1 → 2) and `escalated_at`, so **every alert climbs on its own clock** — two alerts raised seconds apart escalate independently, and acknowledging one never affects another. Acknowledging stops the climb at any point. The scanner claims a row by writing the new level before sending, so overlapping sweeps can never send the same notification twice.

On startup every pre-existing alert is retired to the top of the ladder, so restarting the server never fires a burst of notifications for yesterday's unacknowledged alerts. Alerts older than `ESCALATE_MAX_AGE_SECONDS` never escalate, for the same reason.

**Conditions versus incidents.** Escalation runs 15 to 30 seconds after an alert, by which time the situation may have ended — so whether an ended situation is still worth notifying depends on what kind of thing it was.

*A fall is an incident.* It is over in about a second by its nature — the person is on the ground, or they got up. Suppressing falls for having ended would suppress every fall the system will ever see, so falls always escalate.

*A crowd gathering is a condition.* Whether a crowd that has since dispersed still warrants a notification is a judgement call, exposed as `ESCALATE_ENDED_CROWDS`. When `False`, a crowd that cleared before anyone was notified is treated as a non-event. When `True` (the default), the operator is still told "there was an unhandled gathering at CAM-02" — necessary for looping demo footage, whose crowd scenes are too short to reach the escalation delay while still active.

**Repeats are grouped.** At most one notification per camera, per alert type, per rung, per `NOTIFY_GROUP_SECONDS` (default 90). Every repeat is still recorded and escalated, but sending all of them turns a phone into a nuisance and trains the operator to ignore it.

**Acknowledgement suppression.** Once an operator acknowledges an alert, further alerts *of that type on that camera* stay silent for `ACK_SUPPRESS_SECONDS` (default 120). Other cameras and the other detector are unaffected, and the alerts are still recorded in full — only the notifications are held back.

### Notifications

Escalations are delivered over the **Telegram Bot API**, called over plain HTTP from the standard library — no SDK to install.

**Why Telegram and not SMS or WhatsApp.** Both were tried first and both are blocked on a free tier:

* **SMS to Indian numbers** requires TRAI **DLT registration** — a paid registry with 2 to 4 week sender ID provisioning that needs a registered business entity. True of every provider, not just Twilio.
* **WhatsApp via Twilio** delivers, but a trial account may only send Twilio's own template messages (error `21654`), and the Content API needed to author a custom template returns `401 — upgrade required`. A fall alert would have arrived worded as an appointment reminder.
* **Telegram** delivers arbitrary text to a real phone, free, with no templates, approval flow or per-message quota.

The transport sits behind a `Notifier` interface, so this is a swap of one class rather than a change to the escalation logic. `TwilioWhatsAppNotifier` is still in the tree and starts working the moment a paid account is configured.

Put the bot token in a `.env` file in the project root (gitignored, read automatically at startup):

```
TELEGRAM_BOT_TOKEN=123456:ABC-your-bot-token
```

Or set it in the environment before starting the server:

```bash
# Git Bash / macOS / Linux
export TELEGRAM_BOT_TOKEN="123456:ABC-your-bot-token"
```

```powershell
# PowerShell
$env:TELEGRAM_BOT_TOKEN = "123456:ABC-your-bot-token"
```

On startup the log should show `Notifications: Telegram (capped at 20/hour)`. If it says `log only`, the token was not set in that shell.

Recipients are addressed by **chat ID**, not phone number. A Telegram bot cannot open a conversation, so each person must message the bot once; that first message makes their chat ID visible:

```bash
python tools/telegram_ids.py
```

Paste the printed ID into the user's *Telegram chat ID* field in User Management. A user is reachable if they have a chat ID (Telegram) or a phone number (WhatsApp) — whichever the active transport needs.

`NOTIFY_TRANSPORT` pins a transport (`telegram`, `whatsapp`, `log`) or picks the best configured one on `auto`. With nothing configured every message is logged instead of sent, so the escalation ladder is fully demonstrable with no accounts at all. `NOTIFY_MAX_PER_HOUR` caps real sends per rolling hour. Delivery failures are logged and never raised, so a missing key or a dropped network connection cannot take the system down mid incident.

## Logging

Every module logs under the `visionguard` namespace to the console and to a rotating file in `logs/` (rolls at 5 MB, keeps 3 backups). Alert lifecycle events use one structured, greppable line format naming the action, alert type, camera and row ID:

```
ALERT     CROWD GATHERING  CAM-04 (Webcam) #57 — 2 persons in view, peak zone 2
ACK       CROWD GATHERING  CAM-04 (Webcam) #57 — by admin
CLEARED   CROWD GATHERING  CAM-04 (Webcam) #58 — 1 person in view
```

Type names match the dashboard, so `grep "FALL DETECTED"` or `grep ESCALATED` pulls a whole incident trail, and `#id` is the alert's database row — the same ID as its dashboard entry and CSV export row.

## Database and Analytics

SQLite file at `crowd_events.db`, in WAL mode so the API can read while camera threads write. **History persists across restarts** so the analytics endpoints have data to aggregate (set `RESET_DB_ON_START = True` in `config.py` to wipe on startup instead).

| Table | Columns |
|-------|---------|
| `crowd_events` | `id, cam_id, timestamp, person_count, max_zone_density, is_alert, frame_path, acknowledged_at, escalation_level, escalated_at` |
| `fall_events` | `id, cam_id, timestamp, is_fall, ratio, angle, frame_path, acknowledged_at, escalation_level, escalated_at` |
| `occupancy_samples` | `id, cam_id, timestamp, person_count, max_density, zones` |
| `users` | `id, username, password_hash, salt, role, full_name, phone, telegram_chat_id, created_at` |
| `sessions` | `token, user_id, created_at, expires_at` |

Schema changes are applied to existing databases via lightweight `ALTER TABLE` migrations at startup, so history is never wiped for a schema change.

Every 10 s each camera logs how many people it sees and a JSON 3×4 grid of zone counts to `occupancy_samples`. This powers the dashboard's analytics panels, each written to answer a question an operator or supervisor acts on:

* **Occupancy** — people in view over time, per camera, with the crowd-alert threshold drawn as a dashed line. A line climbing toward it is a cue to check that feed *before* the alert fires.
* **Alerts by location** — a horizontal bar per camera, split crowd / fall, over 24 h. Answers *where do the incidents happen*.
* **Movement hotspots** — the 3×4 zone grid for one selected camera, peak cell outlined.

## Setup and Running

### Prerequisites

* Python 3.9+ (the code avoids newer syntax such as `X | None`)
* The source footage for each configured camera (see [Cameras](#cameras))

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
python backend/app.py
```

The dashboard opens automatically in your default browser at **`http://127.0.0.1:8000`** once the server is up. The API process serves the dashboard itself, so there is no separate static server to start. On the first run you will be asked to create the administrator account.

> Auto reload is intentionally disabled: the detection threads live inside the server process, and a reload supervisor would restart them on every file change.

## API Reference

All endpoints are served at `http://127.0.0.1:8000`. Everything except the login page and the `/auth/status|register|login` routes requires a session.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | The dashboard. Redirects to `/login` when signed out. |
| `GET` | `/login` | Sign-in page (first run: administrator registration). |
| `GET` | `/auth/status` | Whether setup is needed and who is signed in. |
| `POST` | `/auth/register` | First-run administrator creation. 403 once any account exists. |
| `POST` | `/auth/login` | Sign in; sets the session cookie. |
| `POST` | `/auth/logout` | Destroy the session. |
| `GET` | `/auth/me` | The signed-in user. |
| `GET` | `/users` | List accounts (administrator only). |
| `POST` | `/users` | Create a guard or administrator (administrator only). |
| `DELETE` | `/users/{id}` | Remove an account and its sessions (administrator only). |
| `GET` | `/video_feed/{cam_id}` | MJPEG stream for camera `cam_id`. 404 for unknown cameras. |
| `WS` | `/ws/alerts` | Pushes every event as JSON the instant it happens: `gathering`, `fall`, `camera_tamper`, `camera_offline`, `camera_online`, `escalation`. |
| `GET` | `/stats` | Alert count (last 60 s), detection FPS, average inference latency, persons in view, cameras live and total, per-camera health and tamper state. |
| `GET` | `/cameras/status` | Per-camera online flag, last frame time, seconds since, tamper state. |
| `GET` | `/evidence/{filename}` | Evidence snapshot JPEG for an alert. |
| `GET` | `/latest` | Most recent crowd alert row within the last 60 seconds. |
| `GET` | `/fall/latest` | Most recent fall alert row within the last 60 seconds. |
| `GET` | `/alerts/recent?limit=N` | Recent alerts from both detectors, newest first (default 50). |
| `POST` | `/alerts/ack?type=&id=` | Acknowledge an alert, and every earlier unacknowledged alert of the same type on the same camera. Stops their escalation. |
| `POST` | `/alerts/ack_all` | Acknowledge every outstanding alert — what the dashboard's Clear Alerts control calls. |
| `GET` | `/export/alerts.csv` | Full alert history (both detectors) as a CSV download. |
| `GET` | `/analytics/occupancy?minutes=N` | Average persons per minute per camera (default 30 min). |
| `GET` | `/analytics/by_camera?hours=N` | Alert counts per camera, split by type, busiest first (default 24 h). |
| `GET` | `/analytics/zones?minutes=N&cam_id=C` | Summed zone person counts; `cam_id` scopes it to one camera (default 30 min, all cameras). |
| `GET` | `/analytics/alerts?hours=N` | Alert counts per hour for both detectors (default 12 h). Retained; superseded on the dashboard by `/analytics/by_camera`. |
| `POST` | `/alerts/clear` | Delete all alert and occupancy rows and reset ID counters (administrator only). |

## Configuration

Everything lives in **`config.py`**. The main settings:

| Constant | Current value | Description |
|----------|---------------|-------------|
| `VIDEO_SOURCES` | CAM-01 file | cam_id to source. Int = webcam index, path = video file, `rtsp://` URL = IP camera |
| `CAMERA_LABELS` | Courtyard, Corridor B, Front Porch, Webcam | Display names for CAM-01 to CAM-04 |
| `ENABLE_WEBCAM` | `False` | `True` adds the local webcam as CAM-04 |
| `DETECTION_MODEL_PATH` | `models/yolov8n.pt` | Model used for crowd counting |
| `POSE_MODEL_PATH` | `models/yolov8s-pose.pt` | Model used for fall detection |
| `DETECTION_CONF_MIN` | `0.25` | Minimum box confidence to count as a person at all |
| `NESTED_BOX_CONTAINMENT` | `0.9` | A person box this much inside a more confident one is a duplicate and dropped |
| `CROWD_MIN_PERSONS` | `4` | Persons in frame to trigger a crowd alert |
| `CROWD_CONFIRM_CYCLES` | `3` | Consecutive detection cycles at or above the threshold before a crowd alert fires |
| `CROWD_CLEAR_MARGIN` | `2` | Hysteresis: the count must fall this far below the trigger before the alert clears |
| `CROWD_THRESHOLD` | `3` | Persons per grid cell to highlight the cell red |
| `GRID_ROWS`, `GRID_COLS` | `3`, `4` | Density grid dimensions |
| `DETECT_INTERVAL` | `0.12` | Minimum seconds between inference passes per camera. Throttles detection freshness only, never video smoothness |
| `FALL_TORSO_ANGLE_DEG` | `65` | Torso angle off vertical beyond which a person counts as fallen |
| `KEYPOINT_CONF_MIN` | `0.3` | Minimum keypoint confidence to trust a joint |
| `FALL_LEG_VETO_RATIO` | `0.5` | Ankles this far below hips (times torso length) means upright legs, never a fall |
| `FALL_TORSO_FORESHORTEN_RATIO` | `0.25` | Torso shorter than this fraction of box height skips the leg veto (steep overhead cameras) |
| `FALL_CONFIRM_FRAMES` | `2` | Consecutive fallen detections (same person) before an alert fires |
| `FALL_CLEAR_SECONDS` | `2.0` | A fall alert clears only after nobody has been seen down for this long |
| `FRAME_SIZE` | `480×270` | Streamed frame resolution |
| `JPEG_QUALITY` | `75` | Encoding quality for streamed frames |
| `STREAM_INTERVAL` | `0.05` | Seconds between MJPEG frames sent to the browser |
| `CAMERA_OFFLINE_TIMEOUT` | `10` | Seconds without a frame before a camera is reported offline |
| `TAMPER_*` | see file | Covered-lens and frozen-feed thresholds and confirmation frame counts |
| `LOG_LEVEL`, `LOG_MAX_BYTES`, `LOG_BACKUP_COUNT` | `INFO`, 5 MB, 3 | Logging level and file rotation |
| `RECENT_WINDOW_SECONDS` | `60` | How far back `/stats` and `/latest` look |
| `RESET_DB_ON_START` | `False` | `True` wipes all history on every start |
| `OCCUPANCY_SAMPLE_SECONDS` | `10` | How often each camera logs an occupancy sample |
| `SESSION_TTL_HOURS` | `12` | How long a sign-in lasts |
| `RESET_USERS_ON_START` | `False` | `True` wipes accounts so first-run setup can be re-demonstrated |
| `ESCALATE_GUARD_SECONDS` | `15` | Unacknowledged time before guards are notified |
| `ESCALATE_ADMIN_SECONDS` | `30` | Unacknowledged time before administrators are notified |
| `ESCALATION_SCAN_SECONDS` | `3` | How often the escalation sweep runs |
| `ESCALATE_ENDED_CROWDS` | `True` | Escalate a crowd alert even if the crowd has since dispersed |
| `ESCALATE_MAX_AGE_SECONDS` | `300` | Alerts older than this never escalate, so a backlog cannot fire a burst |
| `ACK_SUPPRESS_SECONDS` | `120` | After an acknowledgement, further alerts of that type on that camera do not notify for this long |
| `NOTIFY_GROUP_SECONDS` | `90` | At most one notification per camera, alert type and rung in this window |
| `NOTIFY_MAX_PER_HOUR` | `20` | Ceiling on real notifications per rolling hour; excess is logged |
| `NOTIFY_TRANSPORT` | `auto` | `telegram`, `whatsapp`, `log`, or `auto` to pick the best configured |
| `HOST`, `PORT` | `127.0.0.1:8000` | API bind address |

Frontend constants (`POLL_INTERVAL_MS`, `ALERT_DEDUP_MS`, `MODAL_COOLDOWN_MS`, `STREAM_RETRY_MS`, `ACTIVE_CAMS`, `CROWD_ALERT_AT`) are at the top of the `<script>` block in `dashboard.html`. `ACTIVE_CAMS` must match `VIDEO_SOURCES`, and `CROWD_ALERT_AT` must match `CROWD_MIN_PERSONS`.

### Cameras

The dashboard has four camera tiles. The current configuration uses one of them:

| ID | Label | Source | Notes |
|----|-------|--------|-------|
| CAM-01 | Courtyard | `videos/videofall06.mp4` (looped) | Overhead courtyard footage containing a fall. Only video frames are read; the audio track is never decoded. |

Every other camera is unwired. `videos/video1.mp4`, `videos/video2.mp4` and `videos/fall.mp4` are also shipped and can be assigned to any camera; set `ENABLE_WEBCAM = True` to add the local webcam as CAM-04. A camera whose source cannot be opened logs an error and shows as offline on the dashboard; the rest of the system is unaffected.

### Adding a camera

1. Add it to `VIDEO_SOURCES` (and `CAMERA_LABELS`) in `config.py`: a file path, webcam index, or `rtsp://` URL
2. Add its `cam_id` to `ACTIVE_CAMS` in `frontend/dashboard.html`
3. The dashboard has tiles for CAM-01 to CAM-04; a fifth camera also needs a `CAMS` entry and a camera block in the dashboard HTML

## Tests

```bash
python -m pytest tests/ -v
```

39 tests, running in about 15 seconds with no model inference:

* **Detection logic** (`tests/test_engine.py`) — duplicate-box removal, zone assignment, torso-angle fall classification, the legs-upright veto, head-level falls, occlusion handling, tracker-keyed per-person confirmation streaks
* **Health, tamper and API** (`tests/test_health_tamper.py`) — online/offline/stale camera state, covered and frozen tamper detection including the bright-but-featureless case, the camera monitor's offline/online events, structured log lines, and the `/cameras/status`, `/stats` and acknowledgement endpoints

## Model Evaluation

`model_testing_suite/` measures the production models and detection functions against public, hand-labelled datasets, without changing any VisionGuard code. `model_testing_suite/README.txt` explains every test in plain language; results and charts are in `model_testing_suite/model_eval_results/`.

```bash
python model_testing_suite/test_models.py
```

| Dataset | What it tests | Result |
|---------|---------------|--------|
| CrowdHuman (80 images) | Person detection under occlusion | Precision 0.66, recall 0.57, F1 0.61, mAP50 0.53 |
| MOT20-01 (80 frames) | Very dense crowd | Precision 0.99, recall 0.24; count MAE 28 |
| Mall (150 frames) | Crowd counting | Count MAE 18.9, RMSE 20.1 |
| Le2i (30 clips) | Fall vs. daily activity | Precision 0.89, recall 0.67, F1 0.76; 8% false-positive rate on daily-activity clips |
| UR Fall (4 sequences) | Fall detection with true onset | F1 0.80; 1 false positive in 2 daily-activity sequences |
| Synthetic tamper (6 cases) | Covered and frozen detection | All cases correct |
| CAUCAFall, PETS2009, UHCTD | — | Partial or skipped (dataset access limits, documented in the results) |

Minimum camera requirements, from the frame-rate, resolution and person-size sweeps:

* About **15 fps** for crowd detection and **20 fps** for fall detection
* About **320×240** resolution
* A person must be **more than 15 percent of frame height** to be detected reliably; below 8 percent they are essentially never detected
