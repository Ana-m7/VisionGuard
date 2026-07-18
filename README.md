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
│   │  detector/engine.py : one thread per camera  │              │
│   │                                              │              │
│   │  CAM-01 (video1.mp4) ──► pose model          │              │
│   │  CAM-02 (video2.mp4) ──► pose model          │              │
│   │  CAM-03 (webcam)     ──► pose model          │              │
│   │                                              │              │
│   │  ONE pose model per camera serves BOTH       │              │
│   │  detectors: person boxes → crowd density,    │              │
│   │  keypoints → fall detection.                 │              │
│   │  Inference on every 3rd frame.               │              │
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
│  frontend/VG_FE_Basic.html  (open in browser)                   │
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

### Alert flow

1. Camera thread runs inference every 3rd frame
2. On a state transition (clear to alert), a row is written to SQLite with `cam_id`, the real detection data, and the evidence snapshot filename, and the event is pushed onto an internal queue
3. An async broadcaster sends the event to every dashboard connected to `WS /ws/alerts`, so alerts reach the browser **instantly**
4. The frontend also polls `/latest` and `/fall/latest` every 3 seconds as a fallback; both paths feed one handler that deduplicates by row ID, so whichever arrives first wins and nothing fires twice
5. The alert fires: camera border flashes, badge appears, popup opens with the evidence snapshot
6. Row IDs persist to `localStorage` so a page reload does not replay old alerts; the event log is rebuilt from `/alerts/recent`

## Project Structure

```
VisionGuard/
├── .vscode/
│   └── settings.json       VS Code Live Server config (excludes .db + evidence_frames from watch)
├── config.py               All tunables: cameras, thresholds, paths, ports
├── backend/
│   └── app.py              FastAPI server: starts the engine, serves MJPEG + REST + WebSocket
├── detector/
│   └── engine.py           Detection engine: inference, DB writes, FrameStore
├── frontend/
│   └── VG_FE_Basic.html    Browser dashboard: single HTML file, no build step
├── models/
│   ├── yolov8n.pt          Detection model (kept for benchmarking)
│   └── yolov8n-pose.pt     Pose model, the one the engine uses
├── videos/
│   ├── video1.mp4          Source footage for CAM-01
│   └── video2.mp4          Source footage for CAM-02
├── tests/                  Unit tests for the detection logic
├── evidence_frames/        Auto created, JPEG snapshots of alert moments
├── crowd_events.db         Auto created, SQLite database (persists across restarts)
└── requirements.txt
```

## Detection

### Crowd detection

* The pose model detects all persons in the frame
* If **3 or more persons** are in view (`CROWD_MIN_PERSONS`), a crowd alert is triggered
* Independently, the frame is divided into a 3×4 grid; each person is assigned to the cell containing their bounding box centre, cells with 3 or more persons are highlighted red, and the zone counts feed the analytics heatmap

### Fall detection

Based on pose keypoints, per person. Detections below the confidence floor (0.45) are discarded before any logic runs, which stops phantom "persons" (furniture, shadows) on live cameras from ever raising alerts.

* The **torso vector** (mid hip to mid shoulder from pose keypoints) is measured against vertical: a standing person is near 0°, a person on the ground near 90°. Beyond **65°** counts as fallen (the margin keeps ordinary leaning below the threshold).
* **Head level check:** a head at or below hip height counts as fallen even when the projected torso angle is small. This catches falls toward or away from the camera, where perspective foreshortens the torso.
* **Legs upright veto:** bending to pick something up tilts the torso past the threshold, but the ankles stay far below the hips. When the legs are clearly extended beneath a person, fall signals are overridden. Only a body that is actually down (ankles near hip level) can alert.
* **Skeletal evidence is required.** A person whose shoulders or hips are not visible (for example upper body only, seated at a desk) is never classified as fallen. A wide bounding box alone is not treated as a fall, since partial views routinely produce wide boxes.
* **Per person confirmation:** ByteTrack assigns each person a persistent ID, and a fall alert only fires after the *same person* is classified as fallen for 4 consecutive detection cycles (about 0.4 s), so brief misclassifications never trigger an alert.
* The triggering torso angle is stored with the event.

### Person tracking

Detection runs through `model.track()` (ByteTrack) rather than plain inference, so every person keeps a stable ID across frames, visible on the feed and used to anchor fall confirmation to individuals.

### State transition alerts

No cooldown timers. Each camera thread tracks its previous detection state, and a DB row is written **only when state changes**: clear to alert writes an alert row (plus an evidence frame), alert to clear writes a clear row. Crowd state starts neutral so the first detection after startup or a video loop always writes a fresh row.

### Evidence gallery

A snapshot JPEG is saved to `evidence_frames/` at the moment every alert triggers, and its filename is stored on the alert row. The API serves the folder at `/evidence/{filename}`, the alert popup shows the snapshot, and clicking any event log entry reopens it. Every alert is auditable back to the exact annotated frame that caused it.

### Operator acknowledgements

Acknowledging an alert in the popup persists a timestamp on the alert row (`POST /alerts/ack`). Acknowledgements survive page reloads and server restarts, and are included in the CSV export, so the event log doubles as an incident audit trail.

## Database and Analytics

SQLite file at `crowd_events.db`. **History persists across restarts** so the analytics endpoints have data to aggregate (set `RESET_DB_ON_START = True` in `config.py` to wipe on startup instead).

**`crowd_events`**: `id, cam_id, timestamp, person_count, max_zone_density, is_alert, frame_path, acknowledged_at`

**`fall_events`**: `id, cam_id, timestamp, is_fall, ratio, angle, frame_path, acknowledged_at`

Schema changes are applied to existing databases via lightweight `ALTER TABLE` migrations at startup, so history is never wiped for a schema change.

**`occupancy_samples`**: `id, cam_id, timestamp, person_count, max_density, zones`. Every 10 s each camera logs how many people it sees and a JSON 3×4 grid of zone counts. This powers the dashboard's analytics panels:

* **Occupancy trend**: average persons per minute, per camera (last 30 min)
* **Alerts per hour**: stacked gathering and fall counts (last 12 h)
* **Zone density heatmap**: where people actually accumulate, summed over the window

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

Then open `frontend/VG_FE_Basic.html` in your browser. If using **VS Code Live Server** (Go Live), the included `.vscode/settings.json` makes it ignore `*.db`, `evidence_frames/`, and `__pycache__/` so DB writes do not trigger page refreshes.

That is it: one command, one browser tab.

> Auto reload is intentionally disabled: the detection threads live inside the server process, and a reload supervisor would restart them on every file change.

## API Reference

All endpoints are served at `http://127.0.0.1:8000`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/video_feed/{cam_id}` | MJPEG stream for camera `cam_id`. 404 for unknown cameras. |
| `GET` | `/stats` | Alert count (last 60s), detection FPS, average inference latency, persons in view, cameras live and total. |
| `GET` | `/evidence/{filename}` | Evidence snapshot JPEG for an alert. |
| `POST` | `/alerts/ack?type=&id=` | Persist an operator acknowledgement on an alert row. |
| `GET` | `/export/alerts.csv` | Full alert history (both detectors) as a CSV download. |
| `GET` | `/latest` | Most recent crowd alert row within the last 60 seconds. |
| `GET` | `/fall/latest` | Most recent fall alert row within the last 60 seconds. |
| `GET` | `/alerts/recent?limit=N` | Recent alerts from both detectors, newest first (default 50). |
| `WS` | `/ws/alerts` | WebSocket that pushes every alert state transition as JSON the instant it happens. |
| `GET` | `/analytics/occupancy?minutes=N` | Average persons per minute per camera (default 30 min). |
| `GET` | `/analytics/alerts?hours=N` | Alert counts per hour for both detectors (default 12 h). |
| `GET` | `/analytics/zones?minutes=N` | Summed zone person counts for the heatmap (default 30 min). |
| `POST` | `/alerts/clear` | Delete all rows from all tables and reset ID counters. |

## Configuration

Everything lives in **`config.py`**:

| Constant | Default | Description |
|----------|---------|-------------|
| `VIDEO_SOURCES` | 2 videos + webcam | cam_id to source. Int = webcam index, path = video file, `rtsp://` URL = IP camera |
| `CAMERA_LABELS` | Entrance, Corridor, Webcam | Display names |
| `CROWD_MIN_PERSONS` | `3` | Total persons in frame to trigger a crowd alert |
| `CROWD_THRESHOLD` | `3` | Persons per grid cell to highlight the cell red |
| `GRID_ROWS`, `GRID_COLS` | `3`, `4` | Density grid dimensions |
| `DETECTION_CONF_MIN` | `0.45` | Minimum box confidence to count as a person at all |
| `FALL_TORSO_ANGLE_DEG` | `65` | Torso angle off vertical beyond which a person counts as fallen |
| `KEYPOINT_CONF_MIN` | `0.3` | Minimum keypoint confidence to trust a joint |
| `FALL_LEG_VETO_RATIO` | `0.5` | Ankles this far below hips (times torso length) means upright legs, never a fall |
| `FALL_CONFIRM_FRAMES` | `4` | Consecutive fallen detections (same person) before an alert fires |
| `DETECT_EVERY` | `3` | Run inference on every Nth frame |
| `FRAME_SIZE` | `480×270` | Streamed frame resolution |
| `JPEG_QUALITY` | `75` | Encoding quality for streamed frames |
| `STREAM_INTERVAL` | `0.05s` | Delay between MJPEG frames sent to the browser |
| `RECENT_WINDOW_SECONDS` | `60` | How far back `/stats` and `/latest` look |
| `RESET_DB_ON_START` | `False` | `True` wipes all history on every start |
| `OCCUPANCY_SAMPLE_SECONDS` | `10` | How often each camera logs an occupancy sample |
| `HOST`, `PORT` | `127.0.0.1:8000` | API bind address |

Frontend constants (`POLL_INTERVAL_MS`, `ALERT_DEDUP_MS`, `MODAL_COOLDOWN_MS`, `STREAM_RETRY_MS`, `ACTIVE_CAMS`) are at the top of the `<script>` block in `VG_FE_Basic.html`.

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

* **No authentication**: the API accepts requests from any origin. Do not expose port 8000 to a public network.
* **Falls with feet pointing at the camera** can be delayed by the legs upright veto, since the feet appear below the hips in the image. A second camera angle is the standard answer.
