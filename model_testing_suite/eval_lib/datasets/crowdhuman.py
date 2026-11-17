"""CrowdHuman (person detection).

RESEARCH NOTE (verified live during this build, not assumed): the official
crowdhuman.org Google Drive links for annotation_val.odgt and
CrowdHuman_val.zip both return HTTP 404 — the files appear to have been
removed or made private since they were documented. Per the spec's required
fallback behaviour, we do not fail the whole dataset silently; instead we
use the equivalent, actively-hosted Mendeley mirror in YOLO .txt format
(same underlying CrowdHuman images/annotations, re-packaged):

  "CrowdHuman Dataset with YOLO Annotations" - Mendeley Data
  https://data.mendeley.com/datasets/tttr2h6pz4/1
  (single ~8.4GB zip, AWS S3-backed, Accept-Ranges: bytes confirmed live ->
   range-extracted, so only the handful of val images we need are pulled,
   not the whole archive)

YOLO label line: ``class cx cy w h`` (normalized 0-1), class 0 = person.
"""
from __future__ import annotations

import os
import zipfile

from .. import http_utils as hu

ZIP_URL = ("https://data.mendeley.com/public-files/datasets/tttr2h6pz4/"
           "files/a3ee1565-6626-41e8-bfc2-109b4367e241/file_downloaded")
ORIGINAL_SOURCE_NOTE = (
    "Official source https://www.crowdhuman.org/download.html links "
    "(annotation_val.odgt, CrowdHuman_val.zip on Google Drive) returned "
    "HTTP 404 when checked live -- using the Mendeley YOLO-format mirror "
    "of the same dataset instead: https://data.mendeley.com/datasets/tttr2h6pz4/1"
)


def _yolo_to_xyxy(line: str, img_w: int, img_h: int):
    parts = line.split()
    if len(parts) < 5:
        return None
    _, cx, cy, w, h = parts[:5]
    cx, cy, w, h = float(cx) * img_w, float(cy) * img_h, float(w) * img_w, float(h) * img_h
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def run(args) -> dict:
    import cv2
    from .. import metrics as m
    from .. import model_runner as mr

    root = os.path.join(args.data_root, "crowdhuman")
    img_dir = os.path.join(root, "images")
    lbl_dir = os.path.join(root, "labels")
    result = {"name": "CrowdHuman", "source_note": ORIGINAL_SOURCE_NOTE, "status": None}

    limit = args.crowdhuman_limit if not args.full else 100000
    have = (len([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
            if os.path.isdir(img_dir) else 0)
    if have < limit:
        try:
            rf = hu.RemoteZipFile(ZIP_URL, timeout=60)
            with zipfile.ZipFile(rf) as zf:
                val_images = sorted(n for n in zf.namelist()
                                     if n.endswith(".jpg") and "/val/images/" in n)[:limit]
                stems = {os.path.splitext(os.path.basename(n))[0] for n in val_images}
                val_labels = [n for n in zf.namelist()
                              if n.endswith(".txt") and "/val/labels/" in n
                              and os.path.splitext(os.path.basename(n))[0] in stems]
                os.makedirs(img_dir, exist_ok=True)
                os.makedirs(lbl_dir, exist_ok=True)
                for n in val_images:
                    data = zf.read(n)
                    with open(os.path.join(img_dir, os.path.basename(n)), "wb") as f:
                        f.write(data)
                for n in val_labels:
                    data = zf.read(n)
                    with open(os.path.join(lbl_dir, os.path.basename(n)), "wb") as f:
                        f.write(data)
            hu.log(f"CrowdHuman: range-extracted {len(val_images)} images + "
                   f"{len(val_labels)} labels from the 8.4GB mirror zip "
                   f"(full archive NOT downloaded)")
        except Exception as e:  # noqa: BLE001
            result["status"] = f"SKIPPED: range-extraction from Mendeley mirror failed ({e})"
            return result

    images = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))[:limit]
    if not images:
        result["status"] = "SKIPPED: no images available after download attempt"
        return result

    frames_pred, frames_gt = [], []
    latencies = []
    mr.warmup()
    for fname in images:
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
                box = _yolo_to_xyxy(line, w, h)
                if box:
                    gt_boxes.append(box)
        det_result, latency = mr.detect_persons(frame, persist=False)
        pred_boxes = [(x1, y1, x2, y2, c) for x1, y1, x2, y2, c in mr.boxes_xyxy_conf(det_result)]
        frames_pred.append(pred_boxes)
        frames_gt.append(gt_boxes)
        latencies.append(latency)

    if not frames_gt or sum(len(g) for g in frames_gt) == 0:
        result["status"] = "SKIPPED: no usable ground-truth boxes parsed"
        return result

    tp = fp = fn = 0
    for pred, gt in zip(frames_pred, frames_gt):
        t, f_, n_, _ = m.match_boxes(pred, gt, iou_thresh=0.5)
        tp += t; fp += f_; fn += n_
    precision, recall, f1 = m.precision_recall_f1(tp, fp, fn)
    ap50, pr_precisions, pr_recalls = m.average_precision_at_iou(
        frames_pred, frames_gt, iou_thresh=0.5)

    result.update({
        "status": "OK",
        "n_images": len(frames_gt),
        "n_gt_boxes": sum(len(g) for g in frames_gt),
        "n_pred_boxes": sum(len(p) for p in frames_pred),
        "precision": precision, "recall": recall, "f1": f1,
        "mAP50": ap50,
        "pr_curve_precisions": pr_precisions, "pr_curve_recalls": pr_recalls,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "conf_scores": [p[4] for preds in frames_pred for p in preds if p[4] is not None],
    })
    return result
