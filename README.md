# VisionGuard

Real time video surveillance analysis system using computer vision and ML to detect crowd formation and person falls. Annotated video feeds stream live to a browser dashboard that raises visual alerts the instant an event happens.

Built with **YOLOv8**, **OpenCV**, **FastAPI**, and vanilla JavaScript.

## Architecture

Everything runs in a **single process**: the FastAPI server starts the detection engine's camera threads at startup, and they communicate through shared memory. There is no network hop between detector and API, and every frame is JPEG encoded exactly once.

```
┌─────────────────────────────────────────────────────────────────┐
│  backend/app.py  (FastAPI, port 8000)                           │
│                                                                 │
│  lifespan startup ──► init_db() ──► detector/engine.py          │
│                                                                 │
│   ┌──────────────────────────────────────────────┐              │
│   │  detector/engine.py : two threads per camera │              │
│   │                                              │              │
│   │  CAM-01 (video1.mp4) ──► pose model          │              │
│   │  CAM-02 (video2.mp4) ──► pose model          │              │
│   │  CAM-03 (webcam)     ──► pose model          │              │
│   │                                              │              │
│   │  ONE pose model per camera serves BOTH       │              │
│   │  detectors: person boxes → crowd density,    │              │
│   │  keypoints → fall detection.                 │              │
│   │  Capture thread streams at source fps;       │              │
│   │  inference runs in its own thread.           │              │
│   │  Annotated frame → JPEG (encoded once) ──►   │              │
│   │  FrameStore held in memory.                  │              │
│   │  State transitions → SQLite (with cam_id).   │              │
│   └──────────────────────────────────────────────┘              │
│                                                                 │
│  FrameStore bytes  → GET /video_feed/{cam_id}  (MJPEG)          │
│  Alert queue       → WS /ws/alerts  (instant push)              │
│  LiveStats + DB    → GET /stats, /latest, /fall/latest,         │
│                      GET /alerts/recent, /analytics/*           │
└───────────────────────────────┬─────────────────────────────────┘
                                │ HTTP + WebSocket
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  frontend/dashboard.html  (served by the API at /)              │
│                                                                 │
│  Live MJPEG feeds with tracking IDs and pose skeletons          │
│  Receives alerts over WebSocket, polling kept as fallback       │
│  Deduplicates by DB row ID so no alert ever fires twice         │
│  Alert popup shows the evidence snapshot of the moment          │
│  Event log, analytics charts, stat cards, dark/light theme      │
└─────────────────────────────────────────────────────────────────┘
```

### Design decision: one model, not two

Earlier versions ran two models per frame: one for person detection and a separate pose model for falls. Benchmarking both on the project footage showed the pose model counts people as well as the detection model on 92 to 100 percent of frames (within one person), while running one model instead of two **cuts inference cost roughly in half** (~107 ms vs ~207 ms per frame on CPU). The pose model's boxes now feed crowd counting and its keypoints feed fall detection.

### Design decision: single process, frames encoded once

Earlier versions ran the detector as a separate process that JPEG encoded frames at quality 40, POSTed them to the backend over localhost HTTP, decoded them, then encoded them again at quality 75 for MJPEG streaming: four codec passes and a network hop per frame. Now camera threads encode each annotated frame **once** and place the bytes in a shared `FrameStore`; the MJPEG endpoint serves those bytes directly. Higher quality, lower latency, one command to run.

### Design decision: capture decoupled from inference

Earlier versions ran capture and inference on one thread per camera and only published a frame to the browser after an inference pass. The whole loop stalled for the duration of the model call, so a 24 fps source arrived as roughly **4 fps** of very choppy video.

Each camera now runs **two threads**. The capture thread reads, draws the latest overlay, encodes and publishes at the source frame rate; the inference thread pulls the newest frame from a single slot mailbox and runs the model at its own pace. Detections are at most one inference cycle stale — invisible at walking speed — and the operator sees smooth video no matter how slow the model is. Measured on the project footage: **4 fps to 21 fps per camera**, two cameras at once, with no change to detection quality.

Because the mailbox holds only the newest frame, a slow model drops stale frames instead of building a backlog, so the feed can never drift behind real time.

One consequence needed handling: drawing boxes computed from an older frame onto live pixels makes the overlay visibly trail the action. The capture thread therefore keeps a short ring buffer and **displays the frame the current boxes were actually computed from**. The video still advances every frame, so it stays smooth, but it is held back by one inference cycle — imperceptible on a monitoring feed, and the skeletons now sit on the bodies.

### Alert flow

1. The inference thread picks up the newest captured frame
2. On a state transition (clear to alert), a row is written to SQLite with `cam_id`, the real detection data, and the evidence snapshot filename, and the event is pushed onto an internal queue
3. An async broadcaster sends the event to every dashboard connected to `WS /ws/alerts`, so alerts reach the browser **instantly**
4. The frontend also polls `/latest` and `/fall/latest` every 3 seconds as a fallback; both paths feed one handler that deduplicates by row ID, so whichever arrives first wins and nothing fires twice
5. The alert fires: camera border flashes, badge appears, popup opens with the evidence snapshot
6. Row IDs persist to `localStorage` so a page reload does not replay old alerts; the event log is rebuilt from `/alerts/recent`

## Project Structure

```
VisionGuard/
├── config.py               All tunables: cameras, thresholds, paths, ports
├── backend/
│   ├── app.py              FastAPI server: starts the engine, serves the dashboard + MJPEG + REST + WebSocket
│   ├── auth.py             Accounts, PBKDF2 passwords, cookie sessions
│   ├── escalation.py       Unacknowledged-alert ladder: guard then administrator
│   └── notify.py           Telegram / WhatsApp delivery with a log fallback
├── detector/
│   └── engine.py           Detection engine: inference, DB writes, FrameStore
├── frontend/
│   ├── dashboard.html      Browser dashboard: single HTML file, no build step
│   └── login.html          Sign-in / first-run administrator registration
├── models/
│   ├── yolov8n.pt          Detection model (kept for benchmarking)
│   └── yolov8n-pose.pt     Pose model, the one the engine uses
├── videos/
│   ├── video1.mp4          Source footage for CAM-01
│   ├── video2.mp4          Source footage for CAM-02
│   └── video3.mp4          Source footage for CAM-03
├── tools/
│   └── telegram_ids.py     Prints the chat IDs of everyone who messaged the bot
├── tests/                  Unit tests for the detection logic
├── evidence_frames/        Auto created, JPEG snapshots of alert moments
├── crowd_events.db         Auto created, SQLite database (persists across restarts)
└── requirements.txt
```

## Detection

### Crowd detection

* The pose model detects all persons in the frame
* If **4 or more persons** are in view (`CROWD_MIN_PERSONS`), a crowd alert is triggered. Clearing needs the count to fall to **2** (`CROWD_CLEAR_MARGIN`): without that hysteresis a scene hovering on the threshold flaps in and out of alert every few seconds and logs each flap as a fresh incident
* Independently, the frame is divided into a 3×4 grid; each person is assigned to the cell containing their bounding box centre, cells with 3 or more persons are highlighted red, and the zone counts feed the movement-hotspots panel

### Fall detection

Based on pose keypoints, per person. Detections below the confidence floor (0.45) are discarded before any logic runs, which stops phantom "persons" (furniture, shadows) on live cameras from ever raising alerts.

* The **torso vector** (mid hip to mid shoulder from pose keypoints) is measured against vertical: a standing person is near 0°, a person on the ground near 90°. Beyond **65°** counts as fallen (the margin keeps ordinary leaning below the threshold).
* **Head level check:** a head at or below hip height counts as fallen even when the projected torso angle is small. This catches falls toward or away from the camera, where perspective foreshortens the torso.
* **Legs upright veto:** bending to pick something up tilts the torso past the threshold, but the ankles stay far below the hips. When the legs are clearly extended beneath a person, fall signals are overridden. Only a body that is actually down (ankles near hip level) can alert.
* **Skeletal evidence is required.** A person whose shoulders or hips are not visible (for example upper body only, seated at a desk) is never classified as fallen. A wide bounding box alone is not treated as a fall, since partial views routinely produce wide boxes.
* **Per person confirmation:** ByteTrack assigns each person a persistent ID, and a fall alert only fires after the *same person* is classified as fallen for `FALL_CONFIRM_FRAMES` consecutive detection cycles, so a single misclassified frame never triggers an alert. The count is deliberately low because inference runs on its own thread and samples the newest frame rather than every Nth one — a real fall spans well under a second, so demanding many consecutive cycles would miss it.
* The triggering torso angle is stored with the event.

### Person tracking

Detection runs through `model.track()` (ByteTrack) rather than plain inference, so every person keeps a stable ID across frames, visible on the feed and used to anchor fall confirmation to individuals.

### State transition alerts

No cooldown timers. Each camera thread tracks its previous detection state, and a DB row is written **only when state changes**: clear to alert writes an alert row (plus an evidence frame), alert to clear writes a clear row.

A looping video file is treated as one continuous scene rather than a new one each lap. Only the per person fall streaks reset at the seam, because tracker IDs restart with the file. Resetting the alert state instead manufactured a fresh incident every lap — a new alert every few seconds, each escalating on its own timer, which no operator could keep acknowledged.

### Evidence gallery

A snapshot JPEG is saved to `evidence_frames/` at the moment every alert triggers, and its filename is stored on the alert row. The API serves the folder at `/evidence/{filename}`, the alert popup shows the snapshot, and clicking any event log entry reopens it. Every alert is auditable back to the exact annotated frame that caused it.

### Operator acknowledgements

Acknowledging an alert persists a timestamp on the alert row (`POST /alerts/ack`). Acknowledgements survive page reloads and server restarts, are included in the CSV export, and **stop the alert climbing the escalation ladder** — the ladder only considers rows where `acknowledged_at IS NULL`.

Acknowledging one alert also acknowledges every earlier unacknowledged alert of the same type on the same camera. A continuing situation writes a fresh row each time the detector flips state, so one incident can leave a dozen rows behind; acknowledging only the row the operator happened to click would leave the rest escalating and notifications arriving long after the situation was handled. **Clear Alerts** (`POST /alerts/ack_all`) acknowledges everything outstanding.

## Accounts, roles and escalation

### Roles

| Role | Can do |
|------|--------|
| **Administrator** | Everything below, plus create/remove users and clear history. Receives the second escalation rung. |
| **Security Guard** | View feeds, acknowledge alerts, export CSV. Receives the first escalation rung. |

The first run has **no accounts**. The login page offers a one-time administrator registration; once any account exists that route is closed (403) and only an administrator can create further users. Set `RESET_USERS_ON_START = True` in `config.py` to wipe accounts and see that first-run screen again — it leaves alert history untouched.

Sessions are opaque random tokens stored in SQLite and delivered in an **HttpOnly cookie**, not a bearer token. That is forced by the design: the camera feeds are plain `<img src="/video_feed/3">` tags, and an image tag cannot attach an `Authorization` header — only a cookie rides along automatically. Passwords use PBKDF2-HMAC-SHA256 (200,000 rounds, per-user salt) from the standard library, so there is no bcrypt or argon2 dependency.

Every data route is gated, including the MJPEG streams, the evidence snapshots and the WebSocket handshake.

### Escalation ladder

An alert nobody acknowledges escalates on its own:

```
alert fires
   ── 15s unacknowledged ──►  notify security guards
   ── 30s unacknowledged ──►  notify administrators
```

Each alert row carries its own `escalation_level` (0 → 1 → 2) and `escalated_at`, so **every alert climbs on its own clock** — two alerts raised seconds apart escalate independently, and acknowledging one never affects another. Acknowledging stops the climb at any point. The scanner claims a row by writing the new level before sending, so overlapping sweeps can never send the same notification twice.

On startup every pre-existing alert is retired to the top of the ladder, so restarting the server never fires a burst of notifications for yesterday's unacknowledged alerts.

**Conditions versus incidents.** Escalation runs 15 to 30 seconds after an alert, by which time the situation may have ended — so whether an ended situation is still worth notifying depends on what kind of thing it was.

*A fall is an incident.* It is over in about a second by its nature — the person is on the ground, or they got up. Suppressing falls for having ended would suppress every fall the system will ever see, so falls always escalate.

*A crowd gathering is a condition.* Whether a crowd that has since dispersed still warrants a notification is a judgement call, exposed as `ESCALATE_ENDED_CROWDS`. When `False`, a crowd that cleared before anyone was notified is treated as a non-event. When `True` (the default), the operator is still told "there was an unhandled gathering at CAM-02" — necessary for looping demo footage, whose crowd scenes are too short to reach the escalation delay while still active.

**Repeats are grouped.** At most one notification per camera, per alert type, per rung, per `NOTIFY_GROUP_SECONDS` (default 90). A repeating situation raises a genuine alert each time, and every one is recorded and escalated — but sending all of them turns a phone into a nuisance and trains the operator to ignore it.

**Acknowledgement suppression.** A situation that continues keeps producing alerts — a crowd that never disperses, or a looping demo clip that re-raises one every few seconds. Once an operator acknowledges an alert, further alerts *of that type on that camera* stay silent for `ACK_SUPPRESS_SECONDS` (default 120). Other cameras and the other detector are unaffected, and the alerts are still recorded in full — only the notifications are held back. Without this the operator is notified again moments after confirming they are on it, which is exactly the noise that trains people to ignore alarms.

### Notifications

Escalations are delivered over the **Telegram Bot API**, called over plain HTTP from the standard library — no SDK to install.

**Why Telegram and not SMS or WhatsApp.** Both were tried first and both are blocked on a free tier:

* **SMS to Indian numbers** requires TRAI **DLT registration** — a paid registry with 2 to 4 week sender ID provisioning that needs a registered business entity. True of every provider, not just Twilio.
* **WhatsApp via Twilio** delivers, but a trial account may only send Twilio's own template messages (error `21654`), and the Content API needed to author a custom template returns `401 — upgrade required`. A fall alert would have arrived worded as an appointment reminder.
* **Telegram** delivers arbitrary text to a real phone, free, with no templates, approval flow or per-message quota.

The transport sits behind a `Notifier` interface, so this is a swap of one class rather than a change to the escalation logic. `TwilioWhatsAppNotifier` is still in the tree and starts working the moment a paid account is configured.

```powershell
$env:TELEGRAM_BOT_TOKEN = "123456:ABC-your-bot-token"
```

Recipients are addressed by **chat ID**, not phone number. A Telegram bot cannot open a conversation, so each person must message the bot once; that first message makes their chat ID visible:

```bash
python tools/telegram_ids.py
```

Paste the printed ID into the user's *Telegram chat ID* field in User Management. A user is reachable if they have a chat ID (Telegram) or a phone number (WhatsApp) — whichever the active transport needs.

`NOTIFY_TRANSPORT` pins a transport (`telegram`, `whatsapp`, `log`) or picks the best configured one on `auto`. With nothing configured every message is logged instead of sent, so the escalation ladder is fully demonstrable with no accounts at all. `NOTIFY_MAX_PER_HOUR` caps real sends per rolling hour. Delivery failures are logged and never raised, so a missing key or a dropped network connection cannot take the system down mid incident.

## Database and Analytics

SQLite file at `crowd_events.db`. **History persists across restarts** so the analytics endpoints have data to aggregate (set `RESET_DB_ON_START = True` in `config.py` to wipe on startup instead).

**`crowd_events`**: `id, cam_id, timestamp, person_count, max_zone_density, is_alert, frame_path, acknowledged_at`

**`fall_events`**: `id, cam_id, timestamp, is_fall, ratio, angle, frame_path, acknowledged_at`

Schema changes are applied to existing databases via lightweight `ALTER TABLE` migrations at startup, so history is never wiped for a schema change.

**`occupancy_samples`**: `id, cam_id, timestamp, person_count, max_density, zones`. Every 10 s each camera logs how many people it sees and a JSON 3×4 grid of zone counts. This powers the dashboard's analytics panels, each written to answer a question an operator or supervisor acts on:

* **Occupancy** — people in view over time, per camera, with the crowd-alert threshold drawn as a dashed line. A line climbing toward it is a cue to check that feed *before* the alert fires.
* **Alerts by location** — a horizontal bar per camera, split crowd / fall, over 24 h. Answers *where do the incidents happen* — the input to deciding where to post a guard or improve a location — rather than which hour of the day they fell in.
* **Movement hotspots** — the 3×4 zone grid for one selected camera, peak cell outlined. Summed across cameras this mixes unrelated scenes; per camera it shows the traffic pattern in that specific view.

## Setup and Running

### Prerequisites

* Python 3.9+

### Install

```bash
pip install -r requirements.txt
```

### Run

Start everything (backend plus detection engine):

```bash
python backend/app.py
```

The dashboard opens automatically in your default browser at **`http://127.0.0.1:8000`** once the server is up. If it does not, open that URL yourself — the API process serves the dashboard itself, so there is no separate static server to start.

That is it: one command.

> Auto reload is intentionally disabled: the detection threads live inside the server process, and a reload supervisor would restart them on every file change.

## API Reference

All endpoints are served at `http://127.0.0.1:8000`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | The dashboard (`frontend/dashboard.html`). |
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
| `GET` | `/stats` | Alert count (last 60s), detection FPS, average inference latency, persons in view, cameras live and total. |
| `GET` | `/evidence/{filename}` | Evidence snapshot JPEG for an alert. |
| `POST` | `/alerts/ack?type=&id=` | Acknowledge an alert, and every earlier unacknowledged alert of the same type on the same camera. Stops their escalation. |
| `POST` | `/alerts/ack_all` | Acknowledge every outstanding alert — what the dashboard's Clear Alerts control calls. |
| `GET` | `/export/alerts.csv` | Full alert history (both detectors) as a CSV download. |
| `GET` | `/latest` | Most recent crowd alert row within the last 60 seconds. |
| `GET` | `/fall/latest` | Most recent fall alert row within the last 60 seconds. |
| `GET` | `/alerts/recent?limit=N` | Recent alerts from both detectors, newest first (default 50). |
| `WS` | `/ws/alerts` | WebSocket that pushes every alert state transition as JSON the instant it happens. |
| `GET` | `/analytics/occupancy?minutes=N` | Average persons per minute per camera (default 30 min). |
| `GET` | `/analytics/by_camera?hours=N` | Alert counts per camera, split by type, sorted busiest first (default 24 h). |
| `GET` | `/analytics/zones?minutes=N&cam_id=C` | Summed zone person counts; `cam_id` scopes it to one camera (default 30 min, all cameras). |
| `GET` | `/analytics/alerts?hours=N` | Alert counts per hour for both detectors (default 12 h). Retained; superseded on the dashboard by `/analytics/by_camera`. |
| `POST` | `/alerts/clear` | Delete all rows from all tables and reset ID counters. |

## Configuration

Everything lives in **`config.py`**:

| Constant | Default | Description |
|----------|---------|-------------|
| `VIDEO_SOURCES` | 2 videos + webcam | cam_id to source. Int = webcam index, path = video file, `rtsp://` URL = IP camera |
| `CAMERA_LABELS` | Entrance, Corridor, Webcam | Display names |
| `CROWD_MIN_PERSONS` | `4` | Total persons in frame to trigger a crowd alert |
| `CROWD_CLEAR_MARGIN` | `2` | Hysteresis: the count must fall this far below the trigger before the alert clears |
| `CROWD_THRESHOLD` | `3` | Persons per grid cell to highlight the cell red |
| `GRID_ROWS`, `GRID_COLS` | `3`, `4` | Density grid dimensions |
| `DETECTION_CONF_MIN` | `0.45` | Minimum box confidence to count as a person at all |
| `FALL_TORSO_ANGLE_DEG` | `65` | Torso angle off vertical beyond which a person counts as fallen |
| `KEYPOINT_CONF_MIN` | `0.3` | Minimum keypoint confidence to trust a joint |
| `FALL_LEG_VETO_RATIO` | `0.5` | Ankles this far below hips (times torso length) means upright legs, never a fall |
| `FALL_CONFIRM_FRAMES` | `2` | Consecutive fallen detections (same person) before an alert fires |
| `DETECT_INTERVAL` | `0.12` | Minimum seconds between inference passes per camera. Throttles detection freshness only — never video smoothness |
| `FRAME_SIZE` | `480×270` | Streamed frame resolution |
| `JPEG_QUALITY` | `75` | Encoding quality for streamed frames |
| `STREAM_INTERVAL` | `0.05s` | Delay between MJPEG frames sent to the browser |
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

Frontend constants (`POLL_INTERVAL_MS`, `ALERT_DEDUP_MS`, `MODAL_COOLDOWN_MS`, `STREAM_RETRY_MS`, `ACTIVE_CAMS`) are at the top of the `<script>` block in `dashboard.html`.

### Cameras

| ID | Label | Source | Notes |
|----|-------|--------|-------|
| CAM-01 | Main Entrance | `videos/video1.mp4` (looped) | |
| CAM-02 | Corridor B | `videos/video2.mp4` (looped) | |
| CAM-03 | Webcam | local webcam (`cv2` device 0) | Set `ENABLE_WEBCAM = False` in `config.py` to disable. If no webcam is present the camera thread exits cleanly and the dashboard shows a placeholder. |

### Adding a camera

1. Add it to `VIDEO_SOURCES` (and `CAMERA_LABELS`) in `config.py`: a file path, webcam index, or `rtsp://` URL
2. Add its `cam_id` to `ACTIVE_CAMS` and `CAMS` in the frontend
3. Copy a camera block in the dashboard HTML

## Tests

Pure logic unit tests cover zone assignment, torso angle fall classification, the legs upright veto, occlusion handling, and the per person confirmation streaks. No model inference is needed, so they run in seconds:

```bash
python -m pytest tests/ -v
```

## Known Limitations

* **Sessions are not encrypted in transit**: the demo runs over plain HTTP on localhost. Serving this beyond a trusted machine needs HTTPS and the `Secure` cookie flag.
* **Telegram requires the recipient to message the bot first**: bots cannot open a conversation, so a new guard must press Start before alerts can reach them.
* **Falls with feet pointing at the camera** can be delayed by the legs upright veto, since the feet appear below the hips in the image. A second camera angle is the standard answer.
