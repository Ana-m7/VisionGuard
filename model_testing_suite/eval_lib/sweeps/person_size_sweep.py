"""Person-to-frame-size sweep: bucket ground-truth person boxes by height
as a % of frame height, and report detection Recall per bucket. Directly
answers the question config.py's own comments raise about a rejected test
video ("nobody large enough to detect"). Uses MOT20-01 (perspective view,
wide range of person sizes) and CrowdHuman (varied scales); Mall is
excluded here since it ships head POINTS, not boxes, so no per-instance
height is available.
"""
from __future__ import annotations

import os

from .. import http_utils as hu
from .. import metrics as m
from .. import model_runner as mr

BUCKETS = [(0, 2), (2, 4), (4, 8), (8, 15), (15, 200)]  # % of frame height


def _bucket_label(pct):
    for lo, hi in BUCKETS:
        if lo <= pct < hi:
            return f"{lo}-{hi}%" if hi < 200 else f">{lo}%"
    return ">15%"


def _collect_mot20(mot20_dir, gt_path, n_frames=60):
    import cv2
    from ..datasets.mot20 import _load_gt
    img_dir = os.path.join(mot20_dir, "img1")
    if not os.path.isdir(img_dir) or not os.path.isfile(gt_path):
        return []
    frame_files = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))[:n_frames]
    gt_by_frame = _load_gt(gt_path)
    rows = []
    mr.warmup()
    for idx, fname in enumerate(frame_files, start=1):
        frame = cv2.imread(os.path.join(img_dir, fname))
        if frame is None:
            continue
        frame_h = frame.shape[0]
        gt = gt_by_frame.get(idx, [])
        gt_boxes = [(g[1], g[2], g[3], g[4]) for g in gt]
        det_result, _ = mr.detect_persons(frame, persist=True)
        pred_boxes = [(x1, y1, x2, y2, c) for x1, y1, x2, y2, c in mr.boxes_xyxy_conf(det_result)]
        _, _, _, matched = m.match_boxes(pred_boxes, gt_boxes, iou_thresh=0.5)
        matched_gi = {gi for _, gi, _ in matched}
        for gi, box in enumerate(gt_boxes):
            h_pct = (box[3] - box[1]) / frame_h * 100
            rows.append((h_pct, gi in matched_gi))
    return rows


def _collect_crowdhuman(root):
    import cv2
    img_dir = os.path.join(root, "images")
    lbl_dir = os.path.join(root, "labels")
    if not os.path.isdir(img_dir):
        return []
    rows = []
    mr.warmup()
    for fname in sorted(os.listdir(img_dir)):
        if not fname.endswith(".jpg"):
            continue
        stem = os.path.splitext(fname)[0]
        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        if not os.path.isfile(lbl_path):
            continue
        frame = cv2.imread(os.path.join(img_dir, fname))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        gt_boxes = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                _, cx, cy, bw, bh = parts[:5]
                bh_px = float(bh) * h
                gt_boxes.append(bh_px)
        det_result, _ = mr.detect_persons(frame, persist=False)
        pred_boxes = [(x1, y1, x2, y2, c) for x1, y1, x2, y2, c in mr.boxes_xyxy_conf(det_result)]
        # Re-derive full xyxy for matching (re-read label with positions).
        from ..datasets.crowdhuman import _yolo_to_xyxy
        gt_xyxy = []
        with open(lbl_path) as f:
            for line in f:
                box = _yolo_to_xyxy(line, w, h)
                if box:
                    gt_xyxy.append(box)
        _, _, _, matched = m.match_boxes(pred_boxes, gt_xyxy, iou_thresh=0.5)
        matched_gi = {gi for _, gi, _ in matched}
        for gi, box in enumerate(gt_xyxy):
            h_pct = (box[3] - box[1]) / h * 100
            rows.append((h_pct, gi in matched_gi))
    return rows


def run(mot20_dir, mot20_gt_path, crowdhuman_root):
    rows = []
    rows += _collect_mot20(mot20_dir, mot20_gt_path)
    rows += _collect_crowdhuman(crowdhuman_root)
    if not rows:
        return {"status": "SKIPPED: no ground-truth boxes with height info available"}

    buckets = {}
    for h_pct, hit in rows:
        label = _bucket_label(h_pct)
        buckets.setdefault(label, []).append(hit)
    points = []
    for lo, hi in BUCKETS:
        label = f"{lo}-{hi}%" if hi < 200 else f">{lo}%"
        hits = buckets.get(label, [])
        recall = sum(hits) / len(hits) if hits else None
        points.append({"bucket": label, "n_instances": len(hits), "recall": recall})
        hu.log(f"Person-size sweep: bucket {label} -> n={len(hits)} recall={recall}")

    return {"status": "OK", "n_instances_total": len(rows), "points": points}
