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


def _yolo_to_xyxy(line: str, img_w: int, img_h: int):
    parts = line.split()
    if len(parts) < 5:
        return None
    _, cx, cy, w, h = parts[:5]
    cx, cy, w, h = float(cx) * img_w, float(cy) * img_h, float(w) * img_w, float(h) * img_h
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def run(args) -> dict:
    import cv2
    import requests
    from .. import metrics as m
    from .. import model_runner as mr

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
        result["status"] = f"SKIPPED: Mendeley file-list API request failed ({e})"
        return result

    pairs = {}
    for e in entries:
        stem, ext = os.path.splitext(e["filename"])
        pairs.setdefault(stem, {})[ext] = e

    mr.warmup()
    frames_pred, frames_gt = [], []
    n_downloaded = 0
    limit = min(len(pairs), 50)
    for stem, files in list(pairs.items())[:limit]:
        if ".png" not in files or ".txt" not in files:
            continue
        img_path = os.path.join(root, stem + ".png")
        lbl_path = os.path.join(root, stem + ".txt")
        if not os.path.isfile(img_path):
            hu.simple_download(files[".png"]["content_details"]["download_url"], img_path, timeout=30)
        if not os.path.isfile(lbl_path):
            hu.simple_download(files[".txt"]["content_details"]["download_url"], lbl_path, timeout=30)
        n_downloaded += 1
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        gt_boxes = []
        with open(lbl_path) as f:
            for line in f:
                box = _yolo_to_xyxy(line, w, h)
                if box:
                    gt_boxes.append(box)
        det_result, _ = mr.detect_persons(frame, persist=False)
        pred_boxes = [(x1, y1, x2, y2, c) for x1, y1, x2, y2, c in mr.boxes_xyxy_conf(det_result)]
        frames_pred.append(pred_boxes)
        frames_gt.append(gt_boxes)

    if not frames_gt:
        result["status"] = "SKIPPED: no usable frame/label pairs downloaded"
        return result

    tp = fp = fn = 0
    for pred, gt in zip(frames_pred, frames_gt):
        t, f_, n_, _ = m.match_boxes(pred, gt, iou_thresh=0.5)
        tp += t; fp += f_; fn += n_
    precision, recall, f1 = m.precision_recall_f1(tp, fp, fn)

    result.update({
        "status": ("PARTIAL: Mendeley's file-listing API has no working pagination "
                   "(see module docstring) -- only ~50 frames of ONE activity from "
                   "ONE subject were reachable, with no activity-name metadata, so "
                   "fall-vs-ADL event-level accuracy could NOT be computed. Reporting "
                   "person-detection accuracy on the frames that were reachable instead. "
                   f"Manual full access: {DATASET_PAGE} -> 'Download All Files'."),
        "n_frames": len(frames_gt),
        "detection_precision": precision, "detection_recall": recall, "detection_f1": f1,
        "fall_event_accuracy": "UNAVAILABLE (see status)",
    })
    return result
