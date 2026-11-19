"""WHAT-IF configurations (NOT production): which setting change would most
improve person detection / counting, and what would it cost in speed?

Production counts people with yolov8n at the ultralytics default input size
(640 px) and conf >= DETECTION_CONF_MIN. Every frame is shrunk to 640 px, so
small people in a crowd become only a few pixels tall - the person-size sweep
shows recall collapses below ~8% of frame height. This module measures, on the
SAME images, the obvious candidate fixes:

    production            yolov8n, 640 px, conf 0.25        (reference)
    imgsz 960 / 1280      yolov8n, larger input
    conf 0.15             yolov8n, 640 px, lower threshold
    yolov8s @ 960         bigger model + larger input
    pose model as counter yolov8s-pose boxes (what a "pose-only" pipeline
                          would count with)

For each: precision / recall / F1 with 95% CIs (CrowdHuman official val,
same seeded 300 images), Mall counting error (first `baseline_frames` frames,
block bootstrap) and model time per frame on this CPU.

It also times one full production inference cycle (detect + pose) against
two faster variants (pose model only; pose at 480 px), because the UR Fall
sweep shows falls are caught more often when more cycles run per second.
Only the TIME of those variants is measured here, not their fall accuracy.

Nothing in production is changed: every configuration other than
"production" is a separate, clearly-labelled call (see model_runner._size_kw).
"""
from __future__ import annotations

import os
import time

from .. import common
from .. import http_utils as hu
from .. import metrics as m

CONFIGS = [
    # key, label, kwargs for model_runner.detect_persons (model filled at run time)
    ("production", "production: yolov8n, 640 px, conf 0.25", {}),
    ("imgsz_960", "yolov8n, 960 px", {"imgsz": 960}),
    ("imgsz_1280", "yolov8n, 1280 px", {"imgsz": 1280}),
    ("conf_015", "yolov8n, 640 px, conf 0.15", {"conf": 0.15}),
    ("yolov8s_960", "yolov8s, 960 px", {"model": "yolov8s.pt", "imgsz": 960}),
    ("pose_as_counter", "yolov8s-pose boxes as counter, 640 px", {"model": "POSE"}),
]

CYCLE_VARIANTS = [
    ("production_cycle", "production cycle: detect (yolov8n) + pose (yolov8s-pose), 640 px"),
    ("pose_only", "pose model only (its boxes also used for counting)"),
    ("pose_480", "detect 640 px + pose 480 px"),
]


def _call_kwargs(key, cfg_kwargs, config, mr):
    """Every non-production configuration gets its OWN model instance (role),
    so a changed setting such as imgsz can never carry over into the
    production instances (same lesson as the tracker-contamination bug)."""
    kw = dict(cfg_kwargs)
    model = kw.pop("model", None)
    if key != "production":
        kw["role"] = f"whatif_{key}"
    if model == "POSE":
        kw["det_model_path"] = config.POSE_MODEL_PATH
    elif model:
        kw["det_model_path"] = mr.model_path(model)
    return kw


def _time_cycles(frame_paths, variant, mr, config):
    import cv2
    times = []
    for i, p in enumerate(frame_paths):
        frame = cv2.imread(p)
        if frame is None:
            continue
        t0 = time.perf_counter()
        if variant == "production_cycle":
            mr.detect_persons(frame)
            mr.pose_infer(frame, new_sequence=(i == 0))
        elif variant == "pose_only":
            mr.pose_infer(frame, new_sequence=(i == 0))
        elif variant == "pose_480":
            mr.detect_persons(frame)
            mr.pose_infer(frame, new_sequence=(i == 0), imgsz=480, role="whatif_track_480")
        times.append(time.perf_counter() - t0)
    times = times[1:] or times          # drop the cold first cycle
    return m.summarize_mean(times, seed=0)


def run(args) -> dict:
    import config
    import cv2
    from .. import model_runner as mr
    from . import crowdhuman, mall

    result = {"name": "What-if detection configurations (NOT production)",
              "status": None, "status_quality": "what-if (comparison only)",
              "note": "Only the 'production' row describes the live system. The others show "
                      "what a configuration change WOULD do, measured on the same images."}

    # ---------------------------------------------------------- CrowdHuman
    try:
        items = crowdhuman.load_official(os.path.join(args.data_root, "crowdhuman"),
                                         args.crowdhuman_images, args.seed)["items"]
    except Exception as e:  # noqa: BLE001
        items = []
        result["crowdhuman_status"] = f"SKIPPED: {e}"
    ch = {}
    if items:
        frames = [(cv2.imread(p), gt, ign) for p, gt, ign, _ in items]
        frames = [f for f in frames if f[0] is not None]
        for key, label, kw in CONFIGS:
            call = _call_kwargs(key, kw, config, mr)
            preds, lat = [], []
            for img, _, _ in frames:
                res, ms = mr.detect_persons(img, **call)
                preds.append(mr.boxes_xyxy_conf(res))
                lat.append(ms)
            lat = lat[1:] or lat
            det = common.evaluate_detection(preds, [f[1] for f in frames],
                                            [f[2] for f in frames], seed=args.seed)
            det.pop("_per_frame")
            ch[key] = {"label": label, "detection": det,
                       "count_errors_per_image": m.count_errors(
                           [len(p) for p in preds], [len(f[1]) for f in frames], seed=args.seed),
                       "model_ms_per_frame": m.summarize_mean(lat, seed=args.seed)}
            hu.log(f"what-if CrowdHuman {key}: recall {det['recall']['value']:.3f}, "
                      f"precision {det['precision']['value']:.3f}")
        result["crowdhuman"] = {"n_images": len(frames), "configs": ch}

    # ---------------------------------------------------------- Mall
    mall_res = {}
    try:
        frames_dir, counts, _ = mall.load(os.path.join(args.data_root, "mall"))
        nb = min(len(counts), args.baseline_frames)
        paths = [mall.frame_path(frames_dir, i) for i in range(1, nb + 1)]
        imgs = [cv2.imread(p) for p in paths]
        keep = [i for i, im in enumerate(imgs) if im is not None]
        gt = [counts[i] for i in keep]
        for key, label, kw in CONFIGS:
            call = _call_kwargs(key, kw, config, mr)
            pc, lat = [], []
            for i in keep:
                res, ms = mr.detect_persons(imgs[i], **call)
                pc.append(len(res.boxes))
                lat.append(ms)
            lat = lat[1:] or lat
            mall_res[key] = {"label": label,
                             "count_errors": m.count_errors(pc, gt, seed=args.seed, block=25),
                             "model_ms_per_frame": m.summarize_mean(lat, seed=args.seed)}
        result["mall"] = {"frames": f"consecutive frames 1..{len(keep)}", "configs": mall_res}

        # ------------------------------------------------------ cycle timing
        tpaths = paths[:60]
        result["cycle_time"] = {
            "frames": f"Mall frames 1..{len(tpaths)} (multi-person scene)",
            "note": "wall-clock seconds per inference cycle on this CPU; only timing is "
                    "measured for the variants, not their fall-detection accuracy",
            "variants": {k: {"label": lab, "seconds_per_cycle": _time_cycles(tpaths, k, mr, config)}
                         for k, lab in CYCLE_VARIANTS},
        }
    except Exception as e:  # noqa: BLE001
        result["mall_status"] = f"SKIPPED: {e}"

    result["status"] = "OK" if (ch or mall_res) else "SKIPPED: no data"
    return result
