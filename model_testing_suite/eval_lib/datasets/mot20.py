"""MOT20-01 (crowded indoor train station) — person detection, counting and
multi-object TRACKING.

What changed after the audit:
- counting/detection use the production call (predict + nested-box removal),
  not det_model.track() (#1)
- MOTChallenge distractor classes (2 person-on-vehicle, 7 static person,
  8 distractor, 12 reflection) are ignore regions, not false positives (#18)
- the meaningless "hysteresis agreement" is gone: every MOT20-01 frame has
  36+ people, so there were no negative frames (#9); alert logic is now tested
  on threshold-crossing data in mall.py
- tracking is evaluated on the POSE model's tracker — the only tracker
  production runs — with CLEAR-MOT (MOTA, MOTP, IDSW, Frag) and IDF1, the
  tracker reset at the start of the sequence (#3, #10)
- ground-truth visibility is kept so the person-size sweep can drop heavily
  occluded people (#13)

MOT20-01 runs at 25 fps, 1920x1080, 429 frames (seqinfo.ini). By default all
429 frames are range-extracted (~170 MB, never the whole 5 GB archive).
"""
from __future__ import annotations

import os
import zipfile
from collections import defaultdict

from .. import common
from .. import frames as fr
from .. import http_utils as hu
from .. import metrics as m

LABELS_URL = "https://motchallenge.net/data/MOT20Labels.zip"
IMAGES_URL = "https://motchallenge.net/data/MOT20.zip"
SEQ = "MOT20-01"
PEDESTRIAN = 1
DISTRACTOR_CLASSES = {2, 7, 8, 12}
SEQ_LENGTH = 429


def load_gt(gt_path):
    """frame -> {"peds": [(gid, x1, y1, x2, y2, vis)], "ignore": [(x1, y1, x2, y2)]}
    (MOTChallenge gt.txt: frame,id,left,top,w,h,conf,class,visibility)."""
    by_frame = defaultdict(lambda: {"peds": [], "ignore": []})
    with open(gt_path, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 9:
                continue
            frame, gid = int(p[0]), int(p[1])
            l, t, w, h = (float(v) for v in p[2:6])
            conf, cls, vis = int(float(p[6])), int(float(p[7])), float(p[8])
            box = (l, t, l + w, t + h)
            if cls == PEDESTRIAN and conf == 1:
                by_frame[frame]["peds"].append((gid, *box, vis))
            elif cls in DISTRACTOR_CLASSES:
                by_frame[frame]["ignore"].append(box)
    return by_frame


def ensure_data(root, n_frames):
    img_dir = os.path.join(root, SEQ, "img1")
    gt_path = os.path.join(root, "MOT20Labels", "train", SEQ, "gt", "gt.txt")
    r = hu.simple_download(LABELS_URL, os.path.join(root, "MOT20Labels.zip"), timeout=120)
    if not r.ok:
        raise RuntimeError(r.message)
    if not os.path.isfile(gt_path):
        with zipfile.ZipFile(os.path.join(root, "MOT20Labels.zip")) as zf:
            zf.extractall(root)
    have = set(fr.list_frames(img_dir))
    want = [f"{i:06d}.jpg" for i in range(1, n_frames + 1)]
    missing = [w for w in want if w not in have]
    if missing:
        rf = hu.RemoteZipFile(IMAGES_URL, timeout=120)
        with zipfile.ZipFile(rf) as zf:
            os.makedirs(img_dir, exist_ok=True)
            for i, name in enumerate(missing, 1):
                data = zf.read(f"MOT20/train/{SEQ}/img1/{name}")
                with open(os.path.join(img_dir, name), "wb") as f:
                    f.write(data)
                if i % 50 == 0:
                    hu.log(f"MOT20-01: range-extracted {i}/{len(missing)} frames")
    return img_dir, gt_path


def run(args) -> dict:
    import cv2
    from .. import model_runner as mr

    root = os.path.join(args.data_root, "mot20")
    result = {"name": "MOT20-01", "status": None}
    n_frames = min(args.mot20_frames, SEQ_LENGTH)
    try:
        img_dir, gt_path = ensure_data(root, n_frames)
    except Exception as e:  # noqa: BLE001
        result["status"] = f"SKIPPED: MOT20-01 data unavailable ({e})"
        return result
    gt_by_frame = load_gt(gt_path)
    frame_files = [f for f in fr.list_frames(img_dir) if fr.frame_number(f) <= n_frames]
    errors = common.ErrorExamples("mot20", args.out_dir)

    det_pred, gts, igns = [], [], []
    pred_counts, gt_counts, sizes = [], [], []
    trk_pred, trk_gt = [], []
    first = True
    for fname in frame_files:
        fno = fr.frame_number(fname)
        path = os.path.join(img_dir, fname)
        frame = cv2.imread(path)
        if frame is None:
            continue
        g = gt_by_frame.get(fno, {"peds": [], "ignore": []})
        gt_boxes = [p[1:5] for p in g["peds"]]
        res, _ = mr.detect_persons(frame)
        pred = mr.boxes_xyxy_conf(res)
        det_pred.append(pred)
        gts.append(gt_boxes)
        igns.append(g["ignore"])
        pred_counts.append(len(pred))
        gt_counts.append(len(gt_boxes))
        match = m.match_boxes(pred, gt_boxes, ignore_boxes=g["ignore"], ioa_thresh=None)
        hit = {gi for _, gi, _ in match["matched"]}
        h = frame.shape[0]
        for gi, p in enumerate(g["peds"]):
            sizes.append({"h_pct": (p[4] - p[2]) / h * 100, "hit": gi in hit, "visibility": p[5]})
        errors.add(f"frame{fno:06d}", path, gt_boxes, pred, match["fp"], match["fn"],
                   ignore_boxes=g["ignore"])
        # Tracking: the production pose tracker, reset on the first frame.
        pres, _ = mr.pose_infer(frame, new_sequence=first)
        first = False
        tracks = mr.tracks_xyxy(pres)
        tracks = m.filter_ignored_preds(tracks, gt_boxes, g["ignore"])
        trk_pred.append(tracks)
        trk_gt.append([p[:5] for p in g["peds"]])

    if not gts:
        result["status"] = "SKIPPED: no MOT20-01 frames could be read"
        return result

    det = common.evaluate_detection(det_pred, gts, igns, seed=args.seed, block=25)
    det.pop("_per_frame")
    mot = m.clear_mot(trk_pred, trk_gt)
    ids = m.idf1(trk_pred, trk_gt)
    result.update({
        "status": "OK",
        "status_quality": common.quality(det["n_gt"], args.headline_min_n),
        "n_frames": len(gts),
        "native_fps": 25,
        "gt_definition": "class 1 (pedestrian), conf flag 1; classes 2/7/8/12 are ignore regions",
        "gt_persons_per_frame": {"min": min(gt_counts), "max": max(gt_counts),
                                 "mean": sum(gt_counts) / len(gt_counts)},
        "detection_production": det,
        "count_errors": m.count_errors(pred_counts, gt_counts, seed=args.seed, block=25),
        "tracking_pose_model": {
            **mot, **ids,
            "evaluated": "pose_model.track() — the tracker production actually runs "
                         "(crowd counting is untracked since commit 41ed380)",
            "implementation": "eval_lib/metrics.py clear_mot + idf1 (py-motmetrics conventions, "
                              "IoU>=0.5)",
            "crosscheck_motmetrics": m.crosscheck_motmetrics(trk_pred, trk_gt)
            or "motmetrics not installed (pip install motmetrics) - cross-check skipped",
        },
        "pred_counts": pred_counts, "gt_counts": gt_counts,
        "_size_instances": sizes,
        "error_examples": errors.save(),
    })
    return result
