"""Resolution sweep (audit #12).

- Aspect ratio is PRESERVED (the old sweep squashed 4:3 Mall frames to 16:9).
- Widths go BELOW the network input size. The production model letterboxes
  every frame to imgsz=640 on its long side, so any frame wider than 640 px
  reaches the network at the same size, and narrower frames are upscaled
  (with no new detail). Widths >640 are therefore expected to be flat; the
  informative range is 640 and below.
- The old "config.FRAME_SIZE (480x270)" point is gone: FRAME_SIZE is only the
  size of the MJPEG stream sent to the dashboard. Production feeds the
  full-resolution camera frame to the model.
- Two sources: official CrowdHuman val (box recall/precision with ignore
  regions) and Mall (count relative MAE). Both get 95% CIs.
"""
from __future__ import annotations

import os

from .. import common
from .. import http_utils as hu
from .. import metrics as m

WIDTHS = [640, 480, 416, 320, 256, 160]


def _resize_keep_aspect(frame, width):
    import cv2
    h, w = frame.shape[:2]
    if width >= w:
        return frame, 1.0
    s = width / w
    return cv2.resize(frame, (width, max(1, round(h * s))), interpolation=cv2.INTER_AREA), s


def _scale(boxes, s):
    return [tuple(v * s for v in b[:4]) for b in boxes]


def _crowdhuman_points(items, widths):
    import cv2
    from .. import model_runner as mr
    points = []
    for label, width in [("native", None)] + [(f"{w}px wide", w) for w in widths]:
        preds, gts, igns = [], [], []
        for path, gt, ign, _ in items:
            frame = cv2.imread(path)
            if frame is None:
                continue
            img, s = (frame, 1.0) if width is None else _resize_keep_aspect(frame, width)
            res, _ = mr.detect_persons(img)
            preds.append(mr.boxes_xyxy_conf(res))
            gts.append(_scale(gt, s))
            igns.append(_scale(ign, s))
        det = common.evaluate_detection(preds, gts, igns)
        det.pop("_per_frame")
        points.append({"label": label, "width": width, "recall": det["recall"],
                       "precision": det["precision"], "f1": det["f1"], "n_gt": det["n_gt"]})
        hu.log(f"Resolution sweep (CrowdHuman): {label} -> recall={det['recall']['value']:.3f}")
    return points


def _mall_points(frames_dir, gt_counts, frame_ids, widths, seed):
    import cv2
    from .. import model_runner as mr
    points = []
    for label, width in [("native 640x480", None)] + [(f"{w}px wide", w) for w in widths if w < 640]:
        preds, gts = [], []
        for i in frame_ids:
            frame = cv2.imread(os.path.join(frames_dir, f"seq_{i:06d}.jpg"))
            if frame is None:
                continue
            img = frame if width is None else _resize_keep_aspect(frame, width)[0]
            res, _ = mr.detect_persons(img)
            preds.append(len(res.boxes))
            gts.append(gt_counts[i - 1])
        ce = m.count_errors(preds, gts, seed=seed)
        points.append({"label": label, "width": width or 640, "relative_mae": ce["relative_mae"],
                       "mae": ce["mae"], "n_frames": ce["n"]})
        hu.log(f"Resolution sweep (Mall): {label} -> relMAE={ce['relative_mae']['value']:.3f}")
    return points


def run(args) -> dict:
    from ..datasets import crowdhuman, mall
    out = {"status": "OK", "widths": WIDTHS,
           "network_input_note": "production model letterboxes to imgsz=640: frames wider than "
                                 "640 px reach the network at the same size",
           "frame_size_note": "config.FRAME_SIZE (480x270) is the dashboard stream size only; "
                              "the model receives full-resolution frames"}
    try:
        items = crowdhuman.load_official(os.path.join(args.data_root, "crowdhuman"),
                                         args.crowdhuman_images, args.seed)["items"]
        items = items[:args.resolution_images]
        out["crowdhuman_points"] = _crowdhuman_points(items, WIDTHS)
        out["crowdhuman_n_images"] = len(items)
    except Exception as e:  # noqa: BLE001
        out["crowdhuman_status"] = f"SKIPPED: {e}"
    try:
        frames_dir, counts, _ = mall.load(os.path.join(args.data_root, "mall"))
        ids = sorted(common.seeded_sample(range(1, len(counts) + 1), args.resolution_frames, args.seed))
        out["mall_points"] = _mall_points(frames_dir, counts, ids, WIDTHS, args.seed)
        out["mall_n_frames"] = len(ids)
    except Exception as e:  # noqa: BLE001
        out["mall_status"] = f"SKIPPED: {e}"
    if "crowdhuman_points" not in out and "mall_points" not in out:
        out["status"] = "SKIPPED: neither CrowdHuman nor Mall available"
    return out


def minimum_width(points, key, target, higher_is_better, rel=0.9):
    """Smallest width such that IT AND EVERY LARGER WIDTH meet both the
    absolute target and >= rel x best (<= best / rel for errors).
    Returns (width_or_None, best, reason)."""
    vals = [(p["width"] or float("inf"), p[key]["value"]) for p in points if p[key]["value"] is not None]
    if not vals:
        return None, None, "no data"
    best = max(v for _, v in vals) if higher_is_better else min(v for _, v in vals)

    def ok(v):
        if higher_is_better:
            return v >= target and v >= rel * best
        return v <= target and v <= best / rel

    vals.sort(key=lambda t: -t[0])  # largest first
    chosen = None
    for w, v in vals:
        if ok(v):
            chosen = w
        else:
            break
    if chosen is None:
        return None, best, (f"target not met even at the largest size (best "
                            f"{'recall' if higher_is_better else 'relative MAE'} {best:.2f})")
    return (None if chosen == float("inf") else chosen), best, "ok"


__all__ = ["run", "minimum_width", "WIDTHS"]
