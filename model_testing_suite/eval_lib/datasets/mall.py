"""Mall Dataset (CUHK) — crowd COUNTING, ZONE DENSITY and the crowd ALERT logic.

Fixed indoor camera, 640x480, 2000 consecutive frames with every head
annotated as a point (mall_gt.mat: count + per-frame head locations). The
dataset is sampled at under 2 fps, close to production's ~2.8 inference
cycles/s, so consecutive Mall frames are treated as consecutive detection
cycles for the alert state machine (documented approximation).

1. Counting (#1): production predict + nested-box removal, all 2000 frames
   by default; MAE/RMSE/bias/relative MAE with moving-block bootstrap CIs
   (block = 25 frames, because neighbouring frames are correlated).
2. Zone density (#20): the REAL ``engine.zone_densities`` is called on the
   predicted boxes and on the GT head points (wrapped as zero-size boxes), on
   the production 3x4 grid; per-cell MAE and agreement on the dashboard's
   red-cell flag (count >= CROWD_THRESHOLD). Caveat: production places a
   person by BOX CENTRE, GT by HEAD POINT, so people near a cell border can
   legitimately land in different cells.
3. Alert logic (#9): the full Mall frame never drops below
   CROWD_MIN_PERSONS (GT 13-53 people), so it has no negative frames. The
   threshold-crossing test therefore uses a fixed sub-window of the frame (a
   narrower field of view) chosen ONLY from ground truth as the window whose
   GT count crosses the threshold most often; the detector is run on that
   crop. Predicted and GT count sequences are both replayed through
   ``production_logic.CrowdAlertMachine`` (CROWD_MIN_PERSONS,
   CROWD_CONFIRM_CYCLES and CROWD_CLEAR_MARGIN from config). A naive
   "count >= threshold" rule is reported alongside as a baseline, to show what
   the confirm/hysteresis logic buys.
"""
from __future__ import annotations

import os
import zipfile

import numpy as np

from .. import common
from .. import http_utils as hu
from .. import metrics as m
from .. import production_logic as pl

ZIP_URL = "https://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/mall_dataset.zip"
NATIVE_W, NATIVE_H = 640, 480


class _PointBox:
    """Minimal stand-in for an ultralytics box so GT head points can be fed
    to the real engine.zone_densities()."""

    def __init__(self, x, y):
        self.xyxy = np.array([[x, y, x, y]], float)


def load(root):
    """-> (frames_dir, gt_counts list, gt_points list[np.ndarray Nx2])."""
    import scipy.io as sio
    zip_path = os.path.join(root, "mall_dataset.zip")
    extract_dir = os.path.join(root, "mall_dataset")
    r = hu.simple_download(ZIP_URL, zip_path, timeout=600)
    if not r.ok:
        raise RuntimeError(r.message)
    gt_mat = os.path.join(extract_dir, "mall_gt.mat")
    frames_dir = os.path.join(extract_dir, "frames")
    if not os.path.isfile(gt_mat) or not os.path.isdir(frames_dir):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root)
    g = sio.loadmat(gt_mat)
    counts = g["count"].flatten().astype(int).tolist()
    points = [np.asarray(g["frame"][0, i]["loc"][0, 0], float).reshape(-1, 2)
              for i in range(g["frame"].shape[1])]
    return frames_dir, counts, points


def frame_path(frames_dir, i):
    """i is 1-based."""
    return os.path.join(frames_dir, f"seq_{i:06d}.jpg")


def candidate_crops():
    w2, h2 = NATIVE_W // 2, NATIVE_H // 2
    return {
        "top-left quarter": (0, 0, w2, h2), "top-right quarter": (w2, 0, NATIVE_W, h2),
        "bottom-left quarter": (0, h2, w2, NATIVE_H), "bottom-right quarter": (w2, h2, NATIVE_W, NATIVE_H),
        "left half": (0, 0, w2, NATIVE_H), "right half": (w2, 0, NATIVE_W, NATIVE_H),
        "top half": (0, 0, NATIVE_W, h2), "bottom half": (0, h2, NATIVE_W, NATIVE_H),
    }


def _points_in(points, crop):
    x1, y1, x2, y2 = crop
    if len(points) == 0:
        return 0
    inside = (points[:, 0] >= x1) & (points[:, 0] < x2) & (points[:, 1] >= y1) & (points[:, 1] < y2)
    return int(inside.sum())


def _transitions(alerts):
    return sum(1 for a, b in zip(alerts, alerts[1:]) if a != b)


def choose_crop(points, n_frames):
    best = None
    for name, crop in candidate_crops().items():
        counts = [_points_in(points[i], crop) for i in range(n_frames)]
        t = _transitions(pl.run_crowd_machine(counts))
        if best is None or t > best[2]:
            best = (name, crop, t, counts)
    return best


def _episodes(alerts):
    eps, start = [], None
    for i, a in enumerate(alerts + [False]):
        if a and start is None:
            start = i
        elif not a and start is not None:
            eps.append((start, i - 1))
            start = None
    return eps


def alert_agreement(pred_alerts, gt_alerts):
    agree = sum(1 for a, b in zip(pred_alerts, gt_alerts) if a == b)
    gt_eps, pr_eps = _episodes(gt_alerts), _episodes(pred_alerts)

    def overlaps(a, b):
        return a[0] <= b[1] and b[0] <= a[1]

    matched_gt = [g for g in gt_eps if any(overlaps(g, p) for p in pr_eps)]
    matched_pr = [p for p in pr_eps if any(overlaps(p, g) for g in gt_eps)]
    delays = []
    for g in matched_gt:
        starts = [p[0] for p in pr_eps if overlaps(g, p)]
        delays.append(min(starts) - g[0])
    on_gt = sum(gt_alerts)
    on_both = sum(1 for a, b in zip(pred_alerts, gt_alerts) if a and b)
    on_pred = sum(pred_alerts)
    return {
        "frame_agreement": m.proportion(agree, len(gt_alerts)),
        "frame_recall_alert_on": m.proportion(on_both, on_gt),
        "frame_precision_alert_on": m.proportion(on_both, on_pred),
        "gt_alert_episodes": len(gt_eps), "pred_alert_episodes": len(pr_eps),
        "episode_recall": m.proportion(len(matched_gt), len(gt_eps)),
        "episode_precision": m.proportion(len(matched_pr), len(pr_eps)),
        "onset_delay_cycles_vs_gt_alert": m.summarize_mean(delays),
    }


def synthetic_alert_checks():
    """Known-answer sequences through the production-mirrored state machine."""
    import config
    n, c, k = config.CROWD_MIN_PERSONS, config.CROWD_CONFIRM_CYCLES, config.CROWD_CLEAR_MARGIN
    cases = {
        "sustained crowd raises after CROWD_CONFIRM_CYCLES": ([n] * c, [False] * (c - 1) + [True]),
        "single-cycle spike never raises": ([0, n + 5, 0, 0], [False] * 4),
        "threshold flapping raises once and holds": ([n] * c + [n - 1, n] * 4, [False] * (c - 1) + [True] * (1 + 8)),
        "clears only below CROWD_MIN_PERSONS - CROWD_CLEAR_MARGIN":
            ([n] * c + [n - k, n - k - 1], [False] * (c - 1) + [True, True, False]),
    }
    out = {}
    for name, (seq, expected) in cases.items():
        got = pl.run_crowd_machine(seq)
        out[name] = {"input_counts": seq, "expected": expected, "observed": got,
                     "pass": got == expected}
    return out


def run(args) -> dict:
    import cv2
    from .. import model_runner as mr
    import config

    root = os.path.join(args.data_root, "mall")
    result = {"name": "Mall Dataset", "status": None,
              "licence_note": "CUHK research-use-only licence"}
    try:
        frames_dir, gt_counts_all, gt_points_all = load(root)
    except Exception as e:  # noqa: BLE001
        result["status"] = f"SKIPPED: Mall dataset unavailable ({e})"
        return result
    n = min(args.mall_frames, len(gt_counts_all))
    errors = common.ErrorExamples("mall", args.out_dir)

    pred_counts, gt_counts, lat = [], [], []
    cell_abs_err, cell_n = 0.0, 0
    red = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    maxd_pred, maxd_gt = [], []
    pred_boxes_cache = []
    for i in range(1, n + 1):
        path = frame_path(frames_dir, i)
        frame = cv2.imread(path)
        if frame is None:
            continue
        res, ms = mr.detect_persons(frame)
        lat.append(ms)
        pred = mr.boxes_xyxy_conf(res)
        pred_boxes_cache.append(pred)
        pred_counts.append(len(pred))
        gt_counts.append(gt_counts_all[i - 1])
        pts = gt_points_all[i - 1]
        p_cells, p_max = mr.zone_densities(res.boxes, frame.shape)
        g_cells, g_max = mr.zone_densities([_PointBox(x, y) for x, y in pts], frame.shape)
        maxd_pred.append(p_max)
        maxd_gt.append(g_max)
        p_red, g_red = pl.red_cells(p_cells), pl.red_cells(g_cells)
        for r in range(config.GRID_ROWS):
            for c in range(config.GRID_COLS):
                cell_abs_err += abs(p_cells.get((r, c), 0) - g_cells.get((r, c), 0))
                cell_n += 1
                key = ("tp" if (r, c) in p_red and (r, c) in g_red else
                       "fp" if (r, c) in p_red else "fn" if (r, c) in g_red else "tn")
                red[key] += 1
        diff = len(pred) - gt_counts_all[i - 1]
        errors.add(f"seq_{i:06d}", path, [], pred, max(0, diff), max(0, -diff),
                   gt_points=pts.tolist(), note=f"pred {len(pred)} gt {gt_counts_all[i - 1]}")

    if not gt_counts:
        result["status"] = "SKIPPED: no Mall frames could be read"
        return result

    # ---- full-frame alert behaviour (GT is always a crowd here)
    full_pred_alerts = pl.run_crowd_machine(pred_counts)
    full_gt_alerts = pl.run_crowd_machine(gt_counts)

    # ---- threshold-crossing crop
    crop_name, crop, gt_transitions, crop_gt = choose_crop(gt_points_all, n)
    x1, y1, x2, y2 = crop
    crop_pred = []
    for i in range(1, n + 1):
        frame = cv2.imread(frame_path(frames_dir, i))
        if frame is None:
            crop_pred.append(0)
            continue
        res, _ = mr.detect_persons(frame[y1:y2, x1:x2])
        crop_pred.append(len(res.boxes))
    gt_alerts = pl.run_crowd_machine(crop_gt)
    pred_alerts = pl.run_crowd_machine(crop_pred)
    naive = [c >= config.CROWD_MIN_PERSONS for c in crop_pred]

    def raises(a):
        return sum(1 for p, q in zip([False] + a, a) if q and not p)

    result.update({
        "status": "OK",
        "status_quality": common.quality(len(gt_counts), args.headline_min_n),
        "n_frames": len(gt_counts),
        "frames_used": f"consecutive frames 1..{n}",
        "count_errors": m.count_errors(pred_counts, gt_counts, seed=args.seed, block=25),
        "model_latency_ms": m.summarize_mean(lat, seed=args.seed),
        "zone_density": {
            "grid": f"{config.GRID_ROWS}x{config.GRID_COLS}",
            "per_cell_mae": cell_abs_err / cell_n if cell_n else None,
            "max_zone_density_errors": m.count_errors(maxd_pred, maxd_gt, seed=args.seed, block=25),
            "red_cell_flag": {
                "threshold": f"count >= CROWD_THRESHOLD ({config.CROWD_THRESHOLD})",
                **red,
                "precision": m.proportion(red["tp"], red["tp"] + red["fp"]),
                "recall": m.proportion(red["tp"], red["tp"] + red["fn"]),
                "agreement": m.proportion(red["tp"] + red["tn"], sum(red.values())),
            },
            "caveat": "prediction placed by box centre (production), GT by head point",
        },
        "alert_full_frame": {
            "note": "GT count is >= CROWD_MIN_PERSONS in every frame, so this only measures "
                    "how often the production alert is ON when a crowd is present",
            "gt_alert_on_fraction": m.proportion(sum(full_gt_alerts), len(full_gt_alerts)),
            "pred_alert_on_fraction": m.proportion(sum(full_pred_alerts), len(full_pred_alerts)),
        },
        "alert_threshold_crossing_crop": {
            "crop_name": crop_name, "crop_xyxy": list(crop),
            "selection": "chosen from GT only: the sub-window whose GT count makes the "
                         "production state machine change state most often",
            "gt_state_changes": gt_transitions,
            "production_logic": alert_agreement(pred_alerts, gt_alerts),
            "count_errors_in_crop": m.count_errors(crop_pred, crop_gt, seed=args.seed, block=25),
            "raises": {"gt": raises(gt_alerts), "production_logic_on_predictions": raises(pred_alerts),
                       "naive_threshold_on_predictions (baseline)": raises(naive)},
            "pred_counts": crop_pred, "gt_counts": crop_gt,
            "pred_alerts": [int(a) for a in pred_alerts], "gt_alerts": [int(a) for a in gt_alerts],
        },
        "alert_logic_synthetic": synthetic_alert_checks(),
        "pred_counts": pred_counts, "gt_counts": gt_counts,
        "error_examples": errors.save(),
    })

    if args.baselines:
        nb = min(n, args.baseline_frames)
        b_counts, l_counts = [], []
        for i in range(1, nb + 1):
            frame = cv2.imread(frame_path(frames_dir, i))
            if frame is None:
                continue
            b_res, _ = mr.detect_persons(frame, det_model_path=mr.model_path("yolov8s.pt"))
            b_counts.append(len(b_res.boxes))
            l_res, _ = mr.detect_persons_legacy_tracked(frame, new_sequence=(i == 1))
            l_counts.append(len(l_res.boxes))
        gt_b = gt_counts_all[:len(b_counts)]
        result["baselines"] = {
            "frames": f"consecutive frames 1..{len(b_counts)}",
            "production_yolov8n_on_same_frames": m.count_errors(pred_counts[:len(b_counts)], gt_b,
                                                                seed=args.seed, block=25),
            "yolov8s_detect_same_pipeline (NOT production)": m.count_errors(b_counts, gt_b,
                                                                            seed=args.seed, block=25),
            "legacy_tracked_pipeline_pre_41ed380 (NOT production)": m.count_errors(
                l_counts, gt_b, seed=args.seed, block=25),
        }
    return result
