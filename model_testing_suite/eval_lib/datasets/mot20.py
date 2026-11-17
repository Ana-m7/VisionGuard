"""MOT20-01 ("crowded indoor train station") — detection, tracking, and
crowd/zone-density hysteresis evaluation.

Verified live: motchallenge.net serves MOT20.zip (4.68GB) with
``Accept-Ranges: bytes``, so only MOT20-01's images are range-extracted
(the smallest, fully-labelled train sequence) — the 4.68GB archive is never
downloaded in full by default.
"""
from __future__ import annotations

import os
import zipfile
from collections import defaultdict

from .. import http_utils as hu

LABELS_URL = "https://motchallenge.net/data/MOT20Labels.zip"
IMAGES_URL = "https://motchallenge.net/data/MOT20.zip"
SEQ = "MOT20-01"


def _load_gt(gt_path):
    """frame -> [(gt_id, x1, y1, x2, y2), ...], pedestrian class only (class 1),
    conf flag == 1 (per MOTChallenge format: frame,id,l,t,w,h,conf,class,vis)."""
    by_frame = defaultdict(list)
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            frame, gid, l, t, w, h, conf, cls = parts[:8]
            if int(conf) != 1 or int(float(cls)) != 1:
                continue
            frame, gid = int(frame), int(gid)
            l, t, w, h = float(l), float(t), float(w), float(h)
            by_frame[frame].append((gid, l, t, l + w, t + h))
    return by_frame


def run(args) -> dict:
    import cv2
    from .. import metrics as m
    from .. import model_runner as mr
    import config as vg_config

    root = os.path.join(args.data_root, "mot20")
    img_dir = os.path.join(root, SEQ, "img1")
    gt_path = os.path.join(root, "MOT20Labels", "train", SEQ, "gt", "gt.txt")
    result = {"name": "MOT20-01", "status": None}

    r1 = hu.simple_download(LABELS_URL, os.path.join(root, "MOT20Labels.zip"), timeout=120)
    if not r1.ok:
        result["status"] = f"SKIPPED: {r1.message}"
        return result
    if not os.path.isfile(gt_path):
        with zipfile.ZipFile(os.path.join(root, "MOT20Labels.zip")) as zf:
            zf.extractall(root)

    n_frames = args.mot20_frames if not args.full else 429
    have = len([f for f in os.listdir(img_dir) if f.endswith(".jpg")]) if os.path.isdir(img_dir) else 0
    if have < n_frames:
        try:
            rf = hu.RemoteZipFile(IMAGES_URL, timeout=60)
            with zipfile.ZipFile(rf) as zf:
                frame_names = sorted(n for n in zf.namelist()
                                     if n.startswith(f"MOT20/train/{SEQ}/img1/")
                                     and n.endswith(".jpg"))[:n_frames]
                os.makedirs(img_dir, exist_ok=True)
                for n in frame_names:
                    data = zf.read(n)
                    with open(os.path.join(img_dir, os.path.basename(n)), "wb") as f:
                        f.write(data)
            hu.log(f"MOT20-01: range-extracted {len(frame_names)} frames from the "
                   f"4.68GB MOT20.zip (full archive NOT downloaded)")
        except Exception as e:  # noqa: BLE001
            result["status"] = f"SKIPPED: range-extraction of MOT20-01 frames failed ({e})"
            return result

    frame_files = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))[:n_frames]
    if not frame_files:
        result["status"] = "SKIPPED: no MOT20-01 frames available"
        return result

    gt_by_frame = _load_gt(gt_path)
    mr.warmup()

    frames_pred_boxes, frames_gt_boxes = [], []
    frames_pred_tracks, frames_gt_tracks = [], []
    pred_counts, gt_counts = [], []
    crowd_alert_log = []  # (frame_idx, detected_count, gt_count, crowd_alert)
    crowd_state = None

    for idx, fname in enumerate(frame_files, start=1):
        frame = cv2.imread(os.path.join(img_dir, fname))
        if frame is None:
            continue
        gt = gt_by_frame.get(idx, [])
        gt_boxes = [(g[1], g[2], g[3], g[4]) for g in gt]
        gt_tracks = [(g[0], g[1], g[2], g[3], g[4]) for g in gt]

        det_result, _ = mr.detect_persons(frame, persist=True)
        pred_boxes = [(x1, y1, x2, y2, c) for x1, y1, x2, y2, c in mr.boxes_xyxy_conf(det_result)]
        boxes_obj = det_result.boxes
        pred_tracks = []
        if boxes_obj is not None and boxes_obj.id is not None:
            for i in range(len(boxes_obj)):
                tid = int(boxes_obj.id[i])
                x1, y1, x2, y2 = boxes_obj.xyxy[i].tolist()
                pred_tracks.append((tid, x1, y1, x2, y2))

        frames_pred_boxes.append(pred_boxes)
        frames_gt_boxes.append(gt_boxes)
        frames_pred_tracks.append(pred_tracks)
        frames_gt_tracks.append(gt_tracks)
        pred_counts.append(len(pred_boxes))
        gt_counts.append(len(gt_boxes))

        # Replay the exact production hysteresis rule from detector/engine.py.
        if crowd_state:
            crowd_alert = len(pred_boxes) >= (vg_config.CROWD_MIN_PERSONS - vg_config.CROWD_CLEAR_MARGIN)
        else:
            crowd_alert = len(pred_boxes) >= vg_config.CROWD_MIN_PERSONS
        crowd_state = crowd_alert
        gt_crowd_alert = len(gt_boxes) >= vg_config.CROWD_MIN_PERSONS
        crowd_alert_log.append((idx, len(pred_boxes), len(gt_boxes), crowd_alert, gt_crowd_alert))

    tp = fp = fn = 0
    for pred, gt in zip(frames_pred_boxes, frames_gt_boxes):
        t, f_, n_, _ = m.match_boxes(pred, gt, iou_thresh=0.5)
        tp += t; fp += f_; fn += n_
    precision, recall, f1 = m.precision_recall_f1(tp, fp, fn)

    try:
        tracking = m.simple_tracking_quality(frames_pred_tracks, frames_gt_tracks, iou_thresh=0.5)
    except Exception as e:  # noqa: BLE001
        tracking = {"error": str(e)}

    mae, rmse = m.mae_rmse(pred_counts, gt_counts)
    crowd_matches = sum(1 for *_, a, g in crowd_alert_log if a == g)
    crowd_agreement = crowd_matches / len(crowd_alert_log) if crowd_alert_log else None

    result.update({
        "status": "OK",
        "n_frames": len(frame_files),
        "precision": precision, "recall": recall, "f1": f1,
        "tracking": tracking,
        "count_mae": mae, "count_rmse": rmse,
        "crowd_hysteresis_agreement_with_gt_threshold": crowd_agreement,
        "pred_counts": pred_counts, "gt_counts": gt_counts,
        "crowd_alert_log": crowd_alert_log,
    })
    return result
