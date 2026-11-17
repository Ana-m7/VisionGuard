"""Resolution sweep: downscale real frames to progressively lower
resolutions -- including VisionGuard's own config.FRAME_SIZE -- and re-run
the real detector.

Uses TWO sources, not one, because MOT20-01 alone gave a near-flat, low-
confidence signal: that scene is so densely crowded (>60 people/frame)
that yolov8n is already saturated by occlusion regardless of resolution,
masking any true resolution effect. Mall Dataset (a far less saturated,
corridor-style scene closer to VisionGuard's own camera framing) is added
as a second source with its own count-MAE-based signal, giving the
minimum-resolution derivation something with an actual gradient to read.
"""
from __future__ import annotations

import os

from .. import http_utils as hu
from .. import metrics as m
from .. import model_runner as mr

MOT20_NATIVE_W, MOT20_NATIVE_H = 1920, 1080
MALL_NATIVE_W, MALL_NATIVE_H = 640, 480


def _targets(native_w, native_h):
    import config as vg_config
    cfg_w, cfg_h = vg_config.FRAME_SIZE
    raw = [
        ("native", native_w, native_h),
        ("720p" if native_w >= 1280 else "480p", min(1280, native_w), min(720, native_h)),
        (f"config.FRAME_SIZE ({cfg_w}x{cfg_h})", min(cfg_w, native_w), min(cfg_h, native_h)),
        ("360p", min(640, native_w), min(360, native_h)),
        ("240p", min(320, native_w), min(240, native_h)),
    ]
    # Drop duplicate/non-decreasing entries (e.g. Mall is already 640x480,
    # smaller than "720p") while preserving order and labels.
    seen = set()
    out = []
    for label, w, h in raw:
        key = (w, h)
        if key in seen:
            continue
        seen.add(key)
        out.append((label, w, h))
    return out


def _mot20_points(mot20_dir, gt_path, n_frames=30):
    import cv2
    from ..datasets.mot20 import _load_gt

    img_dir = os.path.join(mot20_dir, "img1")
    if not os.path.isdir(img_dir) or not os.path.isfile(gt_path):
        return None
    frame_files = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))[:n_frames]
    gt_by_frame = _load_gt(gt_path)
    mr.warmup()

    points = []
    for label, tw, th in _targets(MOT20_NATIVE_W, MOT20_NATIVE_H):
        sx, sy = tw / MOT20_NATIVE_W, th / MOT20_NATIVE_H
        tp = fp = fn = 0
        pred_counts, gt_counts = [], []
        for idx, fname in enumerate(frame_files, start=1):
            frame = cv2.imread(os.path.join(img_dir, fname))
            if frame is None:
                continue
            resized = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
            gt = gt_by_frame.get(idx, [])
            gt_boxes = [(g[1] * sx, g[2] * sy, g[3] * sx, g[4] * sy) for g in gt]
            det_result, _ = mr.detect_persons(resized, persist=False)
            pred_boxes = [(x1, y1, x2, y2, c) for x1, y1, x2, y2, c in mr.boxes_xyxy_conf(det_result)]
            t, f_, n_, _ = m.match_boxes(pred_boxes, gt_boxes, iou_thresh=0.5)
            tp += t; fp += f_; fn += n_
            pred_counts.append(len(pred_boxes))
            gt_counts.append(len(gt_boxes))
        precision, recall, f1 = m.precision_recall_f1(tp, fp, fn)
        mae, rmse = m.mae_rmse(pred_counts, gt_counts)
        points.append({"label": label, "width": tw, "height": th,
                       "precision": precision, "recall": recall, "f1": f1,
                       "count_mae": mae, "count_rmse": rmse})
        hu.log(f"Resolution sweep (MOT20-01, saturated/dense scene): {label} "
               f"-> recall={recall:.2f} MAE={mae:.1f}")
    return points


def _mall_points(mall_root, n_frames=60):
    import cv2
    import scipy.io as sio

    extract_dir = os.path.join(mall_root, "mall_dataset")
    gt_mat = os.path.join(extract_dir, "mall_gt.mat")
    frames_dir = os.path.join(extract_dir, "frames")
    if not os.path.isfile(gt_mat) or not os.path.isdir(frames_dir):
        return None
    gt = sio.loadmat(gt_mat)
    gt_counts_all = gt["count"].flatten().tolist()
    n_frames = min(n_frames, len(gt_counts_all))
    mr.warmup()

    points = []
    for label, tw, th in _targets(MALL_NATIVE_W, MALL_NATIVE_H):
        pred_counts, gts = [], []
        for i in range(1, n_frames + 1):
            frame = cv2.imread(os.path.join(frames_dir, f"seq_{i:06d}.jpg"))
            if frame is None:
                continue
            resized = (cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
                      if (tw, th) != (frame.shape[1], frame.shape[0]) else frame)
            det_result, _ = mr.detect_persons(resized, persist=False)
            pred_counts.append(len(mr.boxes_xyxy_conf(det_result)))
            gts.append(gt_counts_all[i - 1])
        mae, rmse = m.mae_rmse(pred_counts, gts)
        mean_gt = sum(gts) / len(gts) if gts else 1
        accuracy_proxy = max(0.0, 1 - mae / mean_gt) if mean_gt else None
        points.append({"label": label, "width": tw, "height": th,
                       "count_mae": mae, "count_rmse": rmse,
                       "accuracy_proxy": accuracy_proxy})
        hu.log(f"Resolution sweep (Mall, less-saturated scene): {label} "
               f"-> MAE={mae:.1f} accuracy_proxy={accuracy_proxy}")
    return points


def run(mot20_dir, mot20_gt_path, mall_root, n_frames=30):
    mot20_points = _mot20_points(mot20_dir, mot20_gt_path, n_frames=n_frames)
    mall_points = _mall_points(mall_root, n_frames=max(n_frames, 60))
    if mot20_points is None and mall_points is None:
        return {"status": "SKIPPED: neither MOT20-01 nor Mall frames/gt available"}
    return {
        "status": "OK",
        "mot20_points": mot20_points,
        "mall_points": mall_points,
        # Kept for backward-compat with any code still reading "points".
        "points": mot20_points or [],
    }
