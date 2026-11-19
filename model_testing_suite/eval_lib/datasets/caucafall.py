"""CAUCAFall (Mendeley) — person-detection accuracy under varied lighting
(CC BY 4.0, varied lighting/occlusion/angles as advertised).

RESEARCH NOTE (verified live during this build): data.mendeley.com's public
file-listing API (``/api/datasets/7w7fccy7ky/files``) is real and gives
direct per-file ``download_url``s with NO Google-Drive-style confirmation
needed -- but it reports ``content-range: items 0-99/40209`` and ignores
every pagination mechanism tried (``Range`` header with/without '=',
``page``/``offset``/``skip``/``limit``/JSON:API ``page[offset]`` query
params) -- it always serves the same first 100 files regardless. Those 100
files are all labelled YOLO bounding boxes (class 0 = person) for ~50
frames of a single subject's first activity block, with NO folder/activity
NAME exposed by the API (only an opaque ``folder_id`` UUID), so which
activity is a "fall" vs an ADL cannot be determined from this endpoint.

Consequently this module reports a genuine, honestly-scoped metric --
person-detection Precision/Recall/F1 against the YOLO ground-truth boxes on
the ~50 frames that ARE reachable -- and explicitly marks fall/no-fall
event-level accuracy as unavailable, rather than guessing which activity a
clip belongs to. "Download All Files" on the dataset page presumably uses a
different, browser-authenticated bulk-export mechanism not reachable
anonymously; that is the documented manual fallback.
"""
from __future__ import annotations

import os

from .. import http_utils as hu

FILES_API = "https://data.mendeley.com/api/datasets/7w7fccy7ky/files"
DATASET_PAGE = "https://data.mendeley.com/datasets/7w7fccy7ky/4"


def run(args) -> dict:
    import cv2
    import requests
    from .. import common
    from .. import model_runner as mr
    from .crowdhuman import parse_yolo_person_labels

    root = os.path.join(args.data_root, "caucafall")
    os.makedirs(root, exist_ok=True)
    result = {"name": "CAUCAFall", "status": None, "licence_note": "CC BY 4.0"}

    try:
        sess = requests.Session()
        sess.headers.update({"User-Agent": hu.USER_AGENT})
        r = sess.get(FILES_API, timeout=30)
        r.raise_for_status()
        entries = r.json()
    except Exception as e:  # noqa: BLE001
        entries = []
        result["api_note"] = f"Mendeley file-list API request failed ({e}); using cached files only"

    pairs = {}
    for e in entries:
        stem, ext = os.path.splitext(e["filename"])
        pairs.setdefault(stem, {})[ext] = e
    if not pairs:  # offline: fall back to previously downloaded png/txt pairs
        for f in sorted(os.listdir(root)):
            stem, ext = os.path.splitext(f)
            if ext in (".png", ".txt"):
                pairs.setdefault(stem, {})[ext] = None
        pairs = {k: v for k, v in sorted(pairs.items())}

    frames_pred, frames_gt = [], []
    failed_downloads, dropped = [], {}
    limit = min(len(pairs), 50)
    for stem, files in list(pairs.items())[:limit]:
        if ".png" not in files or ".txt" not in files:
            continue
        img_path = os.path.join(root, stem + ".png")
        lbl_path = os.path.join(root, stem + ".txt")
        ok = True
        for ext, path in ((".png", img_path), (".txt", lbl_path)):
            if not os.path.isfile(path) and files[ext] is not None:
                r = hu.simple_download(files[ext]["content_details"]["download_url"], path, timeout=30)
                ok &= r.ok
        if not ok or not os.path.isfile(lbl_path):
            failed_downloads.append(stem)
            continue
        frame = cv2.imread(img_path)
        if frame is None:
            failed_downloads.append(stem)
            continue
        h, w = frame.shape[:2]
        with open(lbl_path, encoding="utf-8") as f:
            gt_boxes, d = parse_yolo_person_labels(f.readlines(), w, h)
        for k, v in d.items():
            dropped[k] = dropped.get(k, 0) + v
        det_result, _ = mr.detect_persons(frame)
        frames_pred.append(mr.boxes_xyxy_conf(det_result))
        frames_gt.append(gt_boxes)

    if not frames_gt:
        result["status"] = "SKIPPED: no usable frame/label pairs downloaded"
        return result

    det = common.evaluate_detection(frames_pred, frames_gt, seed=args.seed)
    det.pop("_per_frame")

    result.update({
        "status_quality": "pilot",
        "detection_production": det,
        "failed_downloads": failed_downloads,
        "non_person_label_boxes_dropped_by_class": dropped,
        "status": ("PARTIAL: Mendeley's file-listing API has no working pagination "
                   "(see module docstring) -- only ~50 frames of ONE activity from "
                   "ONE subject were reachable, with no activity-name metadata, so "
                   "fall-vs-ADL event-level accuracy could NOT be computed. Reporting "
                   "person-detection accuracy on the frames that were reachable instead. "
                   f"Manual full access: {DATASET_PAGE} -> 'Download All Files'."),
        "n_frames": len(frames_gt),
        "fall_event_accuracy": "UNAVAILABLE (see status)",
        "limitation": "one subject, one activity block, 1-2 people per frame: not generalisable",
    })
    return result
