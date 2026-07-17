# VisionGuard

Real-time video surveillance analysis system using computer vision and ML to detect crowd formation and person falls. Annotated video feeds stream live to a browser dashboard that fires alerts with audio and visual indicators.

Built with **YOLOv8**, **OpenCV**, **FastAPI**, and vanilla JavaScript.

---

## Architecture

Everything runs in a **single process**: the FastAPI server starts the detection engine's camera threads at startup, and they communicate through shared memory — no network hop between detector and API, and every frame is JPEG-encoded exactly once.

```
┌─────────────────────────────────────────────────────────────────┐
│  backend/app.py  (FastAPI, port 8000)                           │
│                                                                 │
│  lifespan startup ──► init_db() ──► detector/engine.py          │
│                                                                 │
│   ┌──────────────────────────────────────────────┐              │
│   │  detector/engine.py — one thread per camera  │              │
│   │                                              │              │
│   │  CAM-01 (video1.mp4) ──► YOLOv8n-pose        │              │
│   │  CAM-02 (video2.mp4) ──► YOLOv8n-pose        │              │
│   │                                              │              │
│   │  ONE pose model per camera serves BOTH       │              │
│   │  detectors: person boxes → crowd density,    │              │
│   │  box geometry → fall detection.              │              │
│   │  Inference on every 3rd frame.               │              │
│   │  Annotated frame → JPEG (encoded once) ──►   │              │
│   │  in-memory FrameStore.                       │              │
│   │  State transitions → SQLite (with cam_id).   │              │
│   └──────────────────────────────────────────────┘              │
│                                                                 │
│  FrameStore bytes  → GET /video_feed/{cam_id}  (MJPEG)          │
│  LiveStats + DB    → GET /stats, /latest, /fall/latest,         │
│                      GET /alerts/recent                         │
│  Maintenance       → POST /alerts/clear                         │
└───────────────────────────────┬─────────────────────────────────┘
                                │ HTTP (polling + MJPEG)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  frontend/VG_FE_Basic.html  (open in browser)                   │
│                                                                 │
│  Live MJPEG feeds for CAM-01 and CAM-02                         │
│  Polls /latest + /fall/latest every 3s for new alert rows       │
│  Rebuilds event log from /alerts/recent on page load            │
│  Deduplicates by DB row ID — never fires the same alert twice   │
│  Uses cam_id from DB row to flash/badge the correct camera      │
│  Alert triggers: flash overlay, audio beep, modal popup         │
│  Real detection data in alerts (person count, zone density,     │
│  fall aspect ratio) — live stat cards, dark/light theme         │
└─────────────────────────────────────────────────────────────────┘
```

### Design decision: one model, not two

Earlier versions ran two models per frame — YOLOv8n for person detection and YOLOv8n-pose for falls. Benchmarking both on the project footage showed the pose model's person counts agree with the detection model's on 92–100% of frames (within ±1 person), while running one model instead of two **cuts inference cost roughly in half** (~107 ms vs ~207 ms per frame on CPU). The pose model's boxes now feed crowd counting and its geometry feeds fall detection.

### Design decision: single process, zero-copy frame path

Earlier versions ran the detector as a separate process that JPEG-encoded frames at quality 40, POSTed them to the backend over localhost HTTP, decoded them, then re-encoded at quality 75 for MJPEG streaming — four codec passes and a network hop per frame. Now camera threads encode each annotated frame **once** and place the bytes in a shared in-memory `FrameStore`; the MJPEG endpoint serves those bytes directly. Higher quality, lower latency, one command to run.

### Alert flow

1. Camera thread runs inference every 3rd frame
2. On a state transition (clear → alert), a row is written to SQLite with `cam_id` and real detection data (person count + zone density, or fall aspect ratio)
3. Frontend polls `/latest` and `/fall/latest` every 3 seconds
4. If the returned row ID is new **and** outside the 3-second dedup window, the alert fires: camera border flashes, badge appears, audio beep plays, modal opens with the real detection data
5. Row IDs persist to `localStorage` so a page reload doesn't re-fire old alerts; the event log is rebuilt from `/alerts/recent`

---

## Project Structure

```
VisionGuard/
├── .vscode/
│   └── settings.json       VS Code Live Server config (excludes .db + evidence_frames from watch)
├── config.py               ALL tunables — cameras, thresholds, paths, ports
├── backend/
│   └── app.py              FastAPI server — starts the engine, serves MJPEG + REST API
├── detector/
│   └── engine.py           Detection engine — YOLOv8n-pose inference, DB writes, FrameStore
├── frontend/
│   └── VG_FE_Basic.html    Browser dashboard — single HTML file, no build step
├── models/
│   ├── yolov8n.pt          YOLOv8 nano detection model (kept for benchmarking)
│   └── yolov8n-pose.pt     YOLOv8 nano pose model — the one the engine uses
├── videos/
│   ├── video1.mp4          Source footage for CAM-01
│   └── video2.mp4          Source footage for CAM-02
├── evidence_frames/        Auto-created — saved JPEG snapshots of alert moments
├── crowd_events.db         Auto-created — SQLite database (reset on every start)
└── requirements.txt
```

---

## Detection

### Crowd detection

- YOLOv8n-pose detects all persons in the frame
- Frame is divided into a 3×4 grid; each person is assigned to the cell containing their bounding-box centre
- If any cell contains **≥ 3 persons**, an alert is triggered
- The annotated frame shows the grid with per-cell counts; high-density cells are highlighted red

### Fall detection

- If a person's bounding box has **width/height ratio > 1.3** they are considered horizontal (fallen)
- The triggering aspect ratio is stored with the event and shown in the alert
- Pose keypoints are drawn on the feed; keypoint-based fall classification (torso angle) is the next planned upgrade

### State-transition alerts

No cooldown timers. Each camera thread tracks its previous detection state, and a DB row is written **only when state changes**: clear→alert writes an alert row (plus an evidence frame), alert→clear writes a clear row. Crowd state starts neutral (`None`) so the first detection after startup or a video loop always writes a fresh row.

### Evidence frames

Saved to `evidence_frames/` at the moment an alert triggers:
`crowd_cam{id}_{timestamp}.jpg` · `fall_cam{id}_{timestamp}.jpg`

---

## Database

SQLite file at `crowd_events.db`, dropped and recreated on every start (clean slate — fresh IDs keep the frontend's dedup logic simple; a persistent history/analytics layer is planned).

**`crowd_events`** — `id, cam_id, timestamp, person_count, max_zone_density, is_alert, frame_path`

**`fall_events`** — `id, cam_id, timestamp, is_fall, ratio`

---

## Setup & Running

### Prerequisites

- Python 3.9+

### Install

```bash
pip install -r requirements.txt
```

### Run

**Terminal — start everything (backend + detection engine):**

```bash
python backend/app.py
```

**Browser — open the dashboard:**

Open `frontend/VG_FE_Basic.html` in your browser. If using **VS Code Live Server** (Go Live), the included `.vscode/settings.json` makes it ignore `*.db`, `evidence_frames/`, and `__pycache__/` so DB writes don't trigger page refreshes.

That's it — one command, one browser tab.

> Auto-reload is intentionally disabled: the detection threads live inside the server process, and a reload supervisor would restart them on every file change.

---

## API Reference

All endpoints are served at `http://127.0.0.1:8000`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/video_feed/{cam_id}` | MJPEG stream for camera `cam_id`. 404 for unknown cameras. |
| `GET` | `/stats` | Alert count (last 60s), frames analyzed, persons currently in view, cameras live/total. |
| `GET` | `/latest` | Most recent crowd alert row within the last 60 seconds. |
| `GET` | `/fall/latest` | Most recent fall alert row within the last 60 seconds (includes aspect ratio). |
| `GET` | `/alerts/recent?limit=N` | Recent alerts from both detectors, newest first (default 50). |
| `POST` | `/alerts/clear` | Delete all rows from both tables and reset auto-increment counters. |

---

## Configuration

Everything lives in **`config.py`**:

| Constant | Default | Description |
|----------|---------|-------------|
| `VIDEO_SOURCES` | 2 video files | cam_id → source. Int = webcam index, path = video file, `rtsp://` URL = IP camera |
| `CAMERA_LABELS` | Entrance/Corridor | Display names |
| `CROWD_THRESHOLD` | `3` | Persons per grid cell to trigger a crowd alert |
| `GRID_ROWS` / `GRID_COLS` | `3` / `4` | Density grid dimensions |
| `ASPECT_RATIO_FALL` | `1.3` | Box w/h ratio above which a person is considered fallen |
| `DETECT_EVERY` | `3` | Run inference on every Nth frame |
| `FRAME_SIZE` | `480×270` | Streamed frame resolution |
| `JPEG_QUALITY` | `75` | Encoding quality for streamed frames |
| `STREAM_INTERVAL` | `0.05s` | Delay between MJPEG frames sent to the browser |
| `RECENT_WINDOW_SECONDS` | `60` | How far back `/stats` and `/latest` look |
| `HOST` / `PORT` | `127.0.0.1:8000` | API bind address |

Frontend constants (`POLL_INTERVAL_MS`, `ALERT_DEDUP_MS`, `MODAL_COOLDOWN_MS`, `STREAM_RETRY_MS`, `ACTIVE_CAMS`) are at the top of the `<script>` block in `VG_FE_Basic.html`.

### Adding a camera

1. Add it to `VIDEO_SOURCES` (and `CAMERA_LABELS`) in `config.py`
2. Add its `cam_id` to `ACTIVE_CAMS` and `CAMS` in the frontend
3. Copy a camera block in the dashboard HTML

---

## Known Limitations

- **Fall detection uses box aspect ratio only** — pose keypoints are drawn but not yet used for classification. Keypoint-based fall logic (torso angle, head/hip height) is the next planned upgrade.
- **No authentication** — the API accepts requests from any origin. Do not expose port 8000 to a public network.
- **DB is wiped on every start** — intentional for now (clean slate, fresh IDs). A persistent history + analytics layer is planned.
- **Polling, not push** — the frontend polls every 3s; WebSocket alerts are planned.
