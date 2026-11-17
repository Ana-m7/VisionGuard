"""Frame-rate sweep: what happens to crowd detection and fall confirmation
as the effective camera FPS drops? Uses the already-downloaded MOT20-01
frames (crowd/detection) and Le2i positive clips (fall), subsampled to
simulate lower FPS, run through the REAL production functions
(classify_persons/confirm_falls, and the actual DETECT_INTERVAL/
FALL_CONFIRM_FRAMES constants from config.py).
"""
from __future__ import annotations

import os

from .. import http_utils as hu
from .. import metrics as m
from .. import model_runner as mr

TARGET_FPS = [5, 10, 15, 20, 25]
NATIVE_FPS = 25


def _crowd_fps_point(frame_files, img_dir, gt_by_frame, target_fps):
    stride = max(1, round(NATIVE_FPS / target_fps))
    indices = list(range(0, len(frame_files), stride))
    import cv2
    tp = fp = fn = 0
    for idx in indices:
        frame_no = idx + 1
        frame = cv2.imread(os.path.join(img_dir, frame_files[idx]))
        if frame is None:
            continue
        gt = gt_by_frame.get(frame_no, [])
        gt_boxes = [(g[1], g[2], g[3], g[4]) for g in gt]
        det_result, _ = mr.detect_persons(frame, persist=True)
        pred_boxes = [(x1, y1, x2, y2, c) for x1, y1, x2, y2, c in mr.boxes_xyxy_conf(det_result)]
        t, f_, n_, _ = m.match_boxes(pred_boxes, gt_boxes, iou_thresh=0.5)
        tp += t; fp += f_; fn += n_
    precision, recall, f1 = m.precision_recall_f1(tp, fp, fn)
    return {"target_fps": target_fps, "n_frames_sampled": len(indices),
            "precision": precision, "recall": recall, "f1": f1}


def _fall_fps_point(positive_clip_dirs, target_fps, clip_native_fps=25):
    import cv2
    import config as vg_config
    stride = max(1, round(clip_native_fps / target_fps))
    confirmed_count = 0
    insufficient_cycles_count = 0
    for clip_dir in positive_clip_dirs:
        frame_files = sorted(f for f in os.listdir(clip_dir) if f.endswith(".jpg"))
        sampled = frame_files[::stride]
        if len(sampled) < vg_config.FALL_CONFIRM_FRAMES:
            insufficient_cycles_count += 1
        streaks = {}
        confirmed_any = False
        for fname in sampled:
            frame = cv2.imread(os.path.join(clip_dir, fname))
            if frame is None:
                continue
            pose_result, _ = mr.pose_infer(frame, persist=True)
            persons = mr.classify_persons(pose_result)
            if mr.confirm_falls(persons, streaks):
                confirmed_any = True
        if confirmed_any:
            confirmed_count += 1
    n = len(positive_clip_dirs)
    recall = confirmed_count / n if n else None
    return {"target_fps": target_fps, "n_positive_clips": n,
            "recall": recall,
            "clips_with_fewer_sampled_frames_than_FALL_CONFIRM_FRAMES": insufficient_cycles_count,
            "max_detection_cycles_available": len(frame_files[::stride]) if positive_clip_dirs else None}


def run(mot20_dir, le2i_clips_dir, le2i_manifest_path, fps_list=None):
    fps_list = fps_list or TARGET_FPS
    result = {"crowd": [], "fall": []}

    img_dir = os.path.join(mot20_dir, "img1")
    gt_path = os.path.join(mot20_dir, "..", "MOT20Labels", "train", "MOT20-01", "gt", "gt.txt")
    gt_path = os.path.normpath(gt_path)
    if os.path.isdir(img_dir) and os.path.isfile(gt_path):
        from ..datasets.mot20 import _load_gt
        frame_files = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
        gt_by_frame = _load_gt(gt_path)
        mr.warmup()
        for fps in fps_list:
            result["crowd"].append(_crowd_fps_point(frame_files, img_dir, gt_by_frame, fps))
        hu.log(f"FPS sweep (crowd/MOT20-01): {len(fps_list)} points computed "
               f"over {len(frame_files)} frames")
    else:
        result["crowd_status"] = "SKIPPED: MOT20-01 frames/gt not available"

    if os.path.isfile(le2i_manifest_path):
        positive_dirs = []
        with open(le2i_manifest_path) as f:
            for line in f:
                cls, clip, _n = line.strip().split("\t")
                if cls in ("Fall", "Lie"):
                    positive_dirs.append(os.path.join(le2i_clips_dir, cls, clip))
        positive_dirs = [d for d in positive_dirs if os.path.isdir(d)]
        mr.warmup()
        for fps in fps_list:
            result["fall"].append(_fall_fps_point(positive_dirs, fps))
        hu.log(f"FPS sweep (fall/Le2i): {len(fps_list)} points computed "
               f"over {len(positive_dirs)} positive clips")
    else:
        result["fall_status"] = "SKIPPED: Le2i manifest not available"

    return result
