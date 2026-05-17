# VisionGuard

Real-time video surveillance analysis system using computer vision and ML to detect crowd formation and person falls. Annotated video feeds stream live to a browser dashboard that fires alerts with audio and visual indicators.

Built with **YOLOv8**, **OpenCV**, **FastAPI**, and vanilla JavaScript.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  detector/main_detect.py                                        │
│                                                                 │
│  CAM-01 (video1.mp4) ──► YOLOv8n + YOLOv8n-pose               │
│  CAM-02 (video2.mp4) ──► YOLOv8n + YOLOv8n-pose               │
│                                                                 │
│  Both cameras run BOTH detectors every detection cycle.         │
│  Crowd overlay is drawn first; fall boxes are composited on     │
│  top. A single resized frame is pushed per cycle.               │
│  Each camera runs in its own thread.                            │
│  Every 3rd frame is run through the models.                     │
│  Annotated frames are pushed to the backend at ~20 FPS.         │
│  Detection events (alerts + clears) are written to SQLite       │
│  with cam_id so the frontend knows which camera triggered.      │
└───────────────┬────────────────────────┬────────────────────────┘
                │ HTTP POST /push_frame  │ SQLite write
                ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  backend/app.py  (FastAPI, port 8000)                           │
│                                                                 │
│  On startup: drops and recreates DB tables (fresh schema)       │
│  Stores latest frame per camera in memory                       │
│  Serves MJPEG streams  → GET /video_feed/{cam_id}              │
│  Serves alert state    → GET /latest, /fall/latest, /stats     │
│  Maintenance           → POST /alerts/clear                     │
└───────────────────────────────┬─────────────────────────────────┘
                                │ HTTP (polling + MJPEG)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  frontend/VG_FE_Basic.html  (open in browser)                   │
│                                                                 │
│  Live MJPEG feeds for CAM-01 and CAM-02 (2/2 active)           │
│  Polls /latest + /fall/latest every 3s for new alert rows       │
│  Deduplicates by DB row ID — never fires the same alert twice   │
│  3s dedup window suppresses back-to-back rows for same event    │
│  Uses cam_id from DB row to flash/badge the correct camera      │
│  Alert triggers: flash overlay, audio beep, modal popup         │
│  Event log, stat cards, dark/light theme                        │
│  Sidebar is sticky and scrolls independently — camera height    │
│  is fixed and never grows with the event log                    │
└─────────────────────────────────────────────────────────────────┘
```

### Alert flow

1. Detector runs both crowd and fall inference on every camera every 3rd frame
2. On a state transition (clear → alert), a row is written to SQLite with `cam_id`
3. Frontend polls `/latest` and `/fall/latest` every 3 seconds
4. If the returned row ID is new **and** more than 3 seconds have passed since the last alert of that type, `triggerAlert(cam_id, type)` fires
5. The correct camera's border flashes, badge appears, audio beep plays, modal opens after 700 ms
6. Row ID is saved to `localStorage` so reloading the page doesn't re-fire the same alert

### Detection compositing

For each detection frame, the pipeline is:

1. **Crowd model** runs on the original clean frame → draws 3×4 grid + person bounding boxes → returns annotated frame at original resolution (no push yet)
2. **Fall model** runs inference on the same original frame (clean, for accuracy) → draws fall bounding boxes on top of the crowd-annotated frame
3. The combined frame is resized to 480×270 and pushed to the backend once

This means both overlays are always visible simultaneously on every camera feed.

### Video streaming

The detector pushes annotated JPEG frames to `POST /push_frame/{cam_id}` at ~20 FPS. The backend holds the latest frame per camera in memory and serves it as a continuous MJPEG multipart stream at `GET /video_feed/{cam_id}`. The browser renders this inside an `<img>` tag — no WebSocket or WebRTC needed.

If the stream drops (e.g. backend restarts), the frontend shows a placeholder and automatically retries the connection every 3 seconds.

### DB reset on startup

Every time the backend starts, it **drops and recreates** both tables. This guarantees:
- A clean slate — no stale alerts from a previous session confuse the frontend's ID-based deduplication
- Schema changes (new columns, etc.) always take effect without a manual migration

---

## Project Structure

```
VisionGuard/
├── .vscode/
│   └── settings.json       VS Code Live Server config (excludes .db + evidence_frames from watch)
├── backend/
│   └── app.py              FastAPI server — frame store, MJPEG streaming, REST API
├── detector/
│   └── main_detect.py      Detection engine — YOLOv8 inference, DB writes, frame push
├── frontend/
│   └── VG_FE_Basic.html    Browser dashboard — single HTML file, no build step
├── models/
│   ├── yolov8n.pt          YOLOv8 nano detection model (~6 MB)
│   └── yolov8n-pose.pt     YOLOv8 nano pose estimation model (~6 MB)
├── videos/
│   ├── video1.mp4          Source footage for CAM-01
│   └── video2.mp4          Source footage for CAM-02
├── evidence_frames/        Auto-created — saved JPEG snapshots of alert moments
├── crowd_events.db         Auto-created — SQLite database (reset on every backend start)
└── requirements.txt
```

---

## Detection

### Crowd detection (both cameras)

- YOLOv8n detects all persons (class 0) in the frame
- Frame is divided into a 3×4 grid
- Each person is assigned to the grid cell containing their bounding box centre
- If any cell contains **≥ 3 persons**, an alert is triggered
- Alert fires only on the **transition** from clear → alert (state-change detection)
- When the crowd clears, state resets — the next crowd event always triggers a fresh alert
- The annotated frame shows the grid overlay with per-cell person counts; high-density cells are highlighted red

### Fall detection (both cameras)

- YOLOv8n-pose detects person bounding boxes
- If a bounding box has **width/height ratio > 1.3** the person is considered horizontal (fallen)
- Alert fires only on the clear → fall transition
- When the fall clears, state resets
- Annotated frame shows green boxes for upright persons, red boxes with ratio label for fallen

### State-transition detection

No cooldown timers are used. Instead, the detector tracks the previous detection state per camera:

- `last_crowd_state` starts as `None` (neutral) and becomes `True`/`False` after the first detection
- `last_fall_state` starts as `False`
- A DB row is written **only when state changes**: clear→alert writes `is_alert=1`, alert→clear writes `is_alert=0`
- On video loop restart both states reset to their initial values, so the first detection on the new loop always writes a fresh row

### Frontend deduplication

The frontend uses two layers of deduplication:

1. **Row ID check** — `latest.id !== lastCrowdId` / `latestFall.id !== lastFallId`. Each unique DB row fires at most one alert. IDs are persisted to `localStorage` so a page reload doesn't re-fire old alerts.
2. **3-second window** — if a new row arrives within 3 seconds of the previous alert of the same type, it is suppressed. This catches rapid back-to-back rows that represent the same physical event.

### Evidence frames

Saved to `evidence_frames/` automatically:
- `crowd_cam{id}_{frame_num}.jpg` — frame at the moment a crowd alert triggers
- `fall_cam{id}_{timestamp}.jpg` — frame at the moment a fall is detected

---

## Database

SQLite file at `crowd_events.db`. Dropped and recreated fresh every time the backend starts.

**`crowd_events`**

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| cam_id | INTEGER | Camera that triggered the alert |
| timestamp | TEXT | `YYYY-MM-DD HH:MM:SS` |
| person_count | INTEGER | Total persons in frame |
| max_zone_density | INTEGER | Highest per-cell count |
| is_alert | INTEGER | 1 = alert, 0 = cleared |
| frame_path | TEXT | Path to evidence frame |

**`fall_events`**

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| cam_id | INTEGER | Camera that triggered the fall |
| timestamp | TEXT | `YYYY-MM-DD HH:MM:SS` |
| is_fall | INTEGER | 1 = fall detected, 0 = cleared |

---

## Setup

### Prerequisites

- Python 3.9+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` installs: `fastapi`, `uvicorn`, `ultralytics` (YOLOv8), `opencv-python`, `numpy`, `torch`, `torchvision`, `python-multipart`

### First-time model download

The model files in `models/` are committed to the repo. If they are missing for any reason, YOLOv8 will auto-download them from Ultralytics on first run.

---

## Running

Three separate terminal sessions are needed. Run all commands from the `VisionGuard/` root.

**Terminal 1 — Start the backend**

```bash
python backend/app.py
```

The backend drops and recreates the SQLite tables on every startup, so there is no need to manually clear the DB between runs.

> **Why not `uvicorn ... --reload`?**
> Uvicorn's StatReload watches the entire project directory. Every time the detector writes an alert to `crowd_events.db` or saves an evidence frame, StatReload detects the file change and restarts the server — killing all active MJPEG streams. Running via `python backend/app.py` scopes reload to `*.py` files only.

**Terminal 2 — Start the detector**

```bash
python detector/main_detect.py
```

All paths inside the detector are resolved relative to the script's own location, so this works from any working directory.

**Terminal 3 — Open the dashboard**

Open `frontend/VG_FE_Basic.html` in your browser. If using **VS Code Live Server** (Go Live), the included `.vscode/settings.json` configures it to ignore `*.db`, `evidence_frames/`, and `__pycache__/` — without this, every DB write from the detector triggers a browser refresh and closes any open alert modals.

### Startup order

Start the **backend first**, then the **detector**. The detector pushes frames to the backend immediately on launch, so the backend must be ready to receive them. The frontend can be opened at any time.

---

## API Reference

All endpoints are served at `http://127.0.0.1:8000`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/video_feed/{cam_id}` | MJPEG stream for camera `cam_id`. Returns 404 if no frames have been received for that cam yet. |
| `POST` | `/push_frame/{cam_id}` | Receive a JPEG frame from the detector (multipart form, field name `frame`). |
| `GET` | `/stats` | Combined alert count (crowd + fall, last 60s), crowd event row count, total person count. |
| `GET` | `/latest` | Most recent crowd alert row within the last 60 seconds. Includes `cam_id`. |
| `GET` | `/fall/latest` | Most recent fall alert row within the last 60 seconds. Includes `cam_id`. |
| `GET` | `/alerts/recent` | Last 50 crowd alert rows (all time). |
| `POST` | `/alerts/clear` | Delete all rows from both tables and reset auto-increment counters. |

---

## Configuration

Key constants are defined at the top of each file.

**`detector/main_detect.py`**

| Constant | Default | Description |
|----------|---------|-------------|
| `CROWD_THRESHOLD` | `3` | Persons per grid cell to trigger a crowd alert |
| `GRID_ROWS` / `GRID_COLS` | `3` / `4` | Density grid dimensions |
| `ASPECT_RATIO_FALL` | `1.3` | Bounding box w/h ratio above which a person is considered fallen |
| `DETECT_EVERY` | `3` | Run inference on every Nth frame (1 = every frame) |
| `PUSH_INTERVAL` | `0.05s` | How often the push-worker thread sends the latest frame to the backend |
| `FRAME_SIZE` | `480×270` | Resolution frames are resized to before pushing |
| `API_BASE` | `http://127.0.0.1:8000` | Backend URL |

**`backend/app.py`**

| Constant | Default | Description |
|----------|---------|-------------|
| `RECENT_WINDOW_SECONDS` | `60` | How far back `/stats`, `/latest`, `/fall/latest` look |
| `STREAM_INTERVAL` | `0.05s` | Delay between MJPEG frames sent to the browser |

**`frontend/VG_FE_Basic.html`**

| Constant | Default | Description |
|----------|---------|-------------|
| `POLL_INTERVAL_MS` | `3000` | How often to poll the backend for new alerts |
| `ALERT_DEDUP_MS` | `3000` | Minimum gap between alerts of the same type (suppresses rapid duplicate rows) |
| `MODAL_COOLDOWN_MS` | `30000` | Minimum gap between modal popups for the same alert type (session-only) |
| `STREAM_RETRY_MS` | `3000` | How long to wait before retrying a dropped MJPEG stream |
| `ACTIVE_CAMS` | `{1, 2}` | Which camera IDs have live feeds |

---

## Cameras

Both cameras run both crowd density detection and fall detection. The dashboard shows them side by side at full width.

| ID | Label | Location | Feed | Detection |
|----|-------|----------|------|-----------|
| CAM-01 | Main Entrance | Main entrance | Live MJPEG | Crowd density + Fall (YOLOv8n + YOLOv8n-pose) |
| CAM-02 | Corridor B | Corridor B | Live MJPEG | Crowd density + Fall (YOLOv8n + YOLOv8n-pose) |

Adding a new camera requires:
1. Adding it to `VIDEO_SOURCES` in `detector/main_detect.py`
2. Adding its `cam_id` to `ACTIVE_CAMS` in the frontend
3. Adding the HTML camera block to the camera grid in `VG_FE_Basic.html`

---

## Known Limitations

- **Video sources are local files** — to use live cameras, replace the paths in `VIDEO_SOURCES` with RTSP stream URLs (`cv2.VideoCapture("rtsp://...")`).
- **Fall detection uses aspect ratio only** — a person lying down is detected, but so is a person crouching sideways or a wide object. Pose keypoint validation would improve accuracy.
- **No authentication** — the API accepts requests from any origin. Do not expose port 8000 to a public network.
- **DB is wiped on every backend start** — this is intentional (clean slate, fresh IDs). If you need persistence across restarts, remove the `DROP TABLE` statements from the startup handler.
- **Single backend process** — all camera threads share one FastAPI process. For more than ~4 cameras, consider running separate detector and backend instances.
- **`on_event` deprecation** — FastAPI has deprecated `@app.on_event("startup")` in favour of lifespan handlers. It works correctly but will print a deprecation warning on startup.
