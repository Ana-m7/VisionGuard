"""Metric implementations used across the evaluation suite.

Kept deliberately simple and dependency-light (no pycocotools / motmetrics
required — both are optional accelerators, used only if already installed).
Every function here operates on plain Python lists/tuples so it's easy to
unit-sanity-check independently of any dataset-loading code.
"""
from __future__ import annotations

import math
from collections import defaultdict


def iou(box_a, box_b) -> float:
    """box = (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(pred_boxes, gt_boxes, iou_thresh: float = 0.5):
    """Greedy highest-IoU-first matching (standard single-IoU-threshold
    evaluation, same idea pycocotools uses per-threshold).

    pred_boxes / gt_boxes: list of (x1,y1,x2,y2[,conf]).
    Returns (tp, fp, fn, matched_pairs) where matched_pairs is
    [(pred_idx, gt_idx, iou), ...].
    """
    pairs = []
    for pi, p in enumerate(pred_boxes):
        for gi, g in enumerate(gt_boxes):
            v = iou(p[:4], g[:4])
            if v >= iou_thresh:
                pairs.append((v, pi, gi))
    pairs.sort(key=lambda t: -t[0])
    used_p, used_g = set(), set()
    matched = []
    for v, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matched.append((pi, gi, v))
    tp = len(matched)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn, matched


def precision_recall_f1(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def average_precision_at_iou(frames_pred, frames_gt, iou_thresh: float = 0.5):
    """A compact, dependency-free COCO-style AP@iou_thresh for a single
    class ("person"). Ranks all predictions across all frames by
    confidence, sweeps the confidence threshold, and integrates the
    precision/recall curve (11-point interpolation, like the original
    PASCAL VOC / early-COCO AP definition — good enough for an honest
    comparative number, not a drop-in replacement for pycocotools).

    frames_pred: list of per-frame prediction lists [(x1,y1,x2,y2,conf), ...]
    frames_gt:   list of per-frame gt lists [(x1,y1,x2,y2), ...] (same length/order)
    """
    assert len(frames_pred) == len(frames_gt)
    # Flatten predictions with a frame index, sorted by confidence desc.
    flat = []
    for fi, preds in enumerate(frames_pred):
        for p in preds:
            conf = p[4] if len(p) > 4 and p[4] is not None else 1.0
            flat.append((conf, fi, p[:4]))
    flat.sort(key=lambda t: -t[0])

    n_gt_total = sum(len(g) for g in frames_gt)
    if n_gt_total == 0:
        return 0.0, [], []

    matched_gt = [set() for _ in frames_gt]
    tps, fps = [], []
    for conf, fi, pbox in flat:
        gt_list = frames_gt[fi]
        best_iou, best_gi = 0.0, -1
        for gi, gbox in enumerate(gt_list):
            if gi in matched_gt[fi]:
                continue
            v = iou(pbox, gbox)
            if v > best_iou:
                best_iou, best_gi = v, gi
        if best_iou >= iou_thresh and best_gi >= 0:
            matched_gt[fi].add(best_gi)
            tps.append(1)
            fps.append(0)
        else:
            tps.append(0)
            fps.append(1)

    cum_tp, cum_fp = 0, 0
    precisions, recalls = [], []
    for t, f in zip(tps, fps):
        cum_tp += t
        cum_fp += f
        precisions.append(cum_tp / (cum_tp + cum_fp))
        recalls.append(cum_tp / n_gt_total)

    # 11-point interpolation.
    ap = 0.0
    for r_level in [i / 10 for i in range(11)]:
        p_at_r = [p for p, r in zip(precisions, recalls) if r >= r_level]
        ap += (max(p_at_r) if p_at_r else 0.0) / 11.0
    return ap, precisions, recalls


def mae_rmse(pred_counts, gt_counts):
    assert len(pred_counts) == len(gt_counts)
    if not pred_counts:
        return 0.0, 0.0
    errs = [p - g for p, g in zip(pred_counts, gt_counts)]
    mae = sum(abs(e) for e in errs) / len(errs)
    rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
    return mae, rmse


def simple_tracking_quality(frames_pred_tracks, frames_gt_tracks, iou_thresh: float = 0.5):
    """A simplified ID-switch / fragmentation counter (NOT a full MOTA/IDF1
    implementation — used only if the optional ``motmetrics`` package isn't
    installed; the caller records which implementation was used).

    frames_pred_tracks: list of per-frame [(track_id, x1,y1,x2,y2), ...]
    frames_gt_tracks:   list of per-frame [(gt_id, x1,y1,x2,y2), ...]

    For each GT identity, follow which predicted track_id covers it frame by
    frame (via IoU-best-match); every time that predicted id changes while
    the GT identity is continuously visible, that's one ID switch.
    """
    assert len(frames_pred_tracks) == len(frames_gt_tracks)
    gt_last_pred_id = {}
    id_switches = 0
    fragmentations = 0
    gt_was_tracked = {}
    tp_total = fp_total = fn_total = 0

    for pred_frame, gt_frame in zip(frames_pred_tracks, frames_gt_tracks):
        pred_boxes = [(p[1], p[2], p[3], p[4]) for p in pred_frame]
        gt_boxes = [(g[1], g[2], g[3], g[4]) for g in gt_frame]
        _, _, _, matched = match_boxes(
            [b + (None,) for b in pred_boxes], gt_boxes, iou_thresh)
        matched_gt_idx = {gi for _, gi, _ in matched}
        tp_total += len(matched)
        fp_total += len(pred_frame) - len(matched)
        fn_total += len(gt_frame) - len(matched)

        for pi, gi, _ in matched:
            gt_id = gt_frame[gi][0]
            pred_id = pred_frame[pi][0]
            prev = gt_last_pred_id.get(gt_id)
            if prev is not None and prev != pred_id:
                id_switches += 1
            gt_last_pred_id[gt_id] = pred_id
            if gt_was_tracked.get(gt_id) is False:
                fragmentations += 1
            gt_was_tracked[gt_id] = True
        tracked_this_frame = {gt_frame[gi][0] for _, gi, _ in matched}
        for gt_id in gt_was_tracked:
            if gt_id not in tracked_this_frame:
                gt_was_tracked[gt_id] = False

    precision, recall, f1 = precision_recall_f1(tp_total, fp_total, fn_total)
    return {
        "id_switches": id_switches,
        "fragmentations": fragmentations,
        "detection_precision": precision,
        "detection_recall": recall,
        "detection_f1": f1,
        "implementation": "simplified (motmetrics not used)",
    }


def confusion_counts(y_true, y_pred, classes):
    """classes: ordered list of class labels. Returns a dict
    {(true_cls, pred_cls): count} plus per-class P/R/F1."""
    counts = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        counts[(t, p)] += 1
    per_class = {}
    for c in classes:
        tp = counts[(c, c)]
        fp = sum(counts[(o, c)] for o in classes if o != c)
        fn = sum(counts[(c, o)] for o in classes if o != c)
        per_class[c] = dict(zip(("precision", "recall", "f1"), precision_recall_f1(tp, fp, fn)))
    return dict(counts), per_class


def event_level_prf1(gt_intervals, pred_events, tolerance_frames: int = 0):
    """Event-level matching for fall detection: a predicted event (a frame
    index where an alert was confirmed) is a true positive if it falls
    within [start - tolerance, end + tolerance] of some ground-truth
    interval (start, end), each GT interval claimed at most once.

    gt_intervals: [(start_frame, end_frame), ...]
    pred_events:  [frame_index, ...] (confirmed-alert frame indices)
    Returns (precision, recall, f1, matched_latencies) where matched_latencies
    is [pred_frame - gt_start, ...] for matched events (>=0 means the alert
    fired at/after the true onset, as expected).
    """
    claimed = set()
    tp = 0
    latencies = []
    unmatched_preds = []
    for ev in pred_events:
        hit = None
        for gi, (s, e) in enumerate(gt_intervals):
            if gi in claimed:
                continue
            if (s - tolerance_frames) <= ev <= (e + tolerance_frames):
                hit = gi
                break
        if hit is not None:
            claimed.add(hit)
            tp += 1
            latencies.append(ev - gt_intervals[hit][0])
        else:
            unmatched_preds.append(ev)
    fp = len(unmatched_preds)
    fn = len(gt_intervals) - len(claimed)
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    return precision, recall, f1, latencies
