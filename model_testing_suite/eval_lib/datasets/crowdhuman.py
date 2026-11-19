"""CrowdHuman — person detection under occlusion, with OFFICIAL annotations.

Audit issue #2: the previous version used a Mendeley "YOLO-format" mirror whose
label files contain COCO class ids (car, umbrella, chair, tie, ...) and ~4.7
boxes/image, which is not CrowdHuman's human annotation (~22.6 persons/image
in the CrowdHuman paper). Its parser also ignored the class, counting the
non-person boxes as people.

This module now uses the official files, re-hosted unchanged on Hugging Face
(verified live 2026-11-18: HTTP 200, Accept-Ranges: bytes):
  annotation_val.odgt (23.3 MB, downloaded once)
  CrowdHuman_val.zip  (2.49 GB, only the sampled images are range-extracted)
None of the 80 Mendeley-mirror images appear in the official val split, so
the mirror is kept only as a clearly-labelled pilot (``run_mirror_pilot``)
with the class filter applied.

Ground truth (official CrowdHuman evaluation conventions):
- each tag == "person" box that is not marked extra.ignore == 1 is a GT person;
  its VISIBLE box ("vbox") is used, because a COCO-trained detector (the
  production yolov8n) boxes the visible part of a person
- tag == "mask" regions and ignore == 1 persons are IGNORE regions:
  predictions falling in them are neither TP nor FP
The models are stock COCO weights, so no CrowdHuman image was seen in training.
"""
from __future__ import annotations

import json
import os
import zipfile
from collections import Counter

from .. import common
from .. import http_utils as hu
from .. import metrics as m

HF_BASE = "https://huggingface.co/datasets/sshao0516/CrowdHuman/resolve/main"
ODGT_URL = f"{HF_BASE}/annotation_val.odgt"
VAL_ZIP_URL = f"{HF_BASE}/CrowdHuman_val.zip"
SOURCE_NOTE = ("Official CrowdHuman val annotations and images (annotation_val.odgt, "
               "CrowdHuman_val.zip), re-hosted unchanged at "
               "https://huggingface.co/datasets/sshao0516/CrowdHuman; the crowdhuman.org "
               "Google Drive links returned 404 when checked.")
MIN_VIS_FOR_SIZE = 0.5


# ---------------------------------------------------------------- parsing
def _xywh_to_xyxy(b):
    x, y, w, h = b
    return (float(x), float(y), float(x + w), float(y + h))


def parse_odgt_record(rec):
    """-> (gt_vboxes, ignore_boxes, meta) for one odgt line (dict)."""
    gt, ignore, meta = [], [], []
    for b in rec.get("gtboxes", []):
        extra = b.get("extra", {}) or {}
        if b.get("tag") == "person" and not extra.get("ignore", 0):
            v = _xywh_to_xyxy(b["vbox"])
            f = _xywh_to_xyxy(b["fbox"])
            v_area = max(0.0, v[2] - v[0]) * max(0.0, v[3] - v[1])
            f_area = max(1e-9, (f[2] - f[0]) * (f[3] - f[1]))
            gt.append(v)
            meta.append({"visibility": min(1.0, v_area / f_area), "fbox": f})
        else:
            ignore.append(_xywh_to_xyxy(b["fbox"]))
    return gt, ignore, meta


def parse_yolo_person_labels(lines, img_w, img_h):
    """YOLO label lines -> person boxes (class 0 ONLY) and a Counter of the
    other class ids that were dropped."""
    boxes, dropped = [], Counter()
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = parts[0]
        if cls != "0":
            dropped[cls] += 1
            continue
        cx, cy, w, h = (float(v) for v in parts[1:5])
        cx, cy, w, h = cx * img_w, cy * img_h, w * img_w, h * img_h
        boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes, dict(dropped)


def load_official(root, n_images, seed, log=hu.log):
    """Return [(image_path, gt, ignore, meta)] for a seeded sample of val
    images, downloading the odgt and range-extracting images as needed.
    Raises on network failure (caller reports SKIPPED)."""
    os.makedirs(root, exist_ok=True)
    odgt = os.path.join(root, "annotation_val.odgt")
    r = hu.simple_download(ODGT_URL, odgt, timeout=300)
    if not r.ok:
        raise RuntimeError(r.message)
    with open(odgt, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    recs.sort(key=lambda rr: rr["ID"])
    chosen = common.seeded_sample(recs, n_images, seed)
    img_dir = os.path.join(root, "official_val_images")
    os.makedirs(img_dir, exist_ok=True)
    missing = [rr["ID"] for rr in chosen
               if not os.path.isfile(os.path.join(img_dir, rr["ID"] + ".jpg"))]
    if missing:
        rf = hu.RemoteZipFile(VAL_ZIP_URL, timeout=120)
        with zipfile.ZipFile(rf) as zf:
            names = {os.path.splitext(os.path.basename(n))[0]: n for n in zf.namelist()
                     if n.endswith(".jpg")}
            for i, img_id in enumerate(missing, 1):
                data = zf.read(names[img_id])
                with open(os.path.join(img_dir, img_id + ".jpg"), "wb") as f:
                    f.write(data)
                if i % 25 == 0:
                    log(f"CrowdHuman: extracted {i}/{len(missing)} val images")
    out = []
    for rr in chosen:
        gt, ign, meta = parse_odgt_record(rr)
        out.append((os.path.join(img_dir, rr["ID"] + ".jpg"), gt, ign, meta))
    return {"n_val_records": len(recs), "items": out}


# ---------------------------------------------------------------- evaluation
def run(args) -> dict:
    import cv2
    from .. import model_runner as mr

    root = os.path.join(args.data_root, "crowdhuman")
    result = {"name": "CrowdHuman (official val annotations)", "source_note": SOURCE_NOTE,
              "status": None}
    try:
        loaded = load_official(root, args.crowdhuman_images, args.seed)
    except Exception as e:  # noqa: BLE001
        result["status"] = f"SKIPPED: could not obtain official CrowdHuman val data ({e})"
        return result
    items = loaded["items"]
    errors = common.ErrorExamples("crowdhuman", args.out_dir)

    prod_pred, ap_pred, base_pred, legacy_pred = [], [], [], []
    gts, igns, sizes, lat = [], [], [], []
    pred_counts, gt_counts = [], []
    for path, gt, ign, meta in items:
        frame = cv2.imread(path)
        if frame is None:
            continue
        h = frame.shape[0]
        res, ms = mr.detect_persons(frame)
        pred = mr.boxes_xyxy_conf(res)
        prod_pred.append(pred)
        lat.append(ms)
        ap_res, _ = mr.detect_persons(frame, conf=0.001)
        ap_pred.append(mr.boxes_xyxy_conf(ap_res))
        if args.baselines:
            b_res, _ = mr.detect_persons(frame, det_model_path=mr.model_path("yolov8s.pt"))
            base_pred.append(mr.boxes_xyxy_conf(b_res))
            l_res, _ = mr.detect_persons_legacy_tracked(frame, new_sequence=True)
            legacy_pred.append(mr.boxes_xyxy_conf(l_res))
        gts.append(gt)
        igns.append(ign)
        pred_counts.append(len(pred))
        gt_counts.append(len(gt))
        match = m.match_boxes(pred, gt, ignore_boxes=ign)
        hit = {gi for _, gi, _ in match["matched"]}
        for gi, (box, mt) in enumerate(zip(gt, meta)):
            sizes.append({"h_pct": (box[3] - box[1]) / h * 100, "hit": gi in hit,
                          "visibility": mt["visibility"]})
        errors.add(os.path.splitext(os.path.basename(path))[0], path, gt, pred,
                   match["fp"], match["fn"], ignore_boxes=ign)

    if not gts:
        result["status"] = "SKIPPED: no images could be read"
        return result

    det = common.evaluate_detection(prod_pred, gts, igns, seed=args.seed)
    det.pop("_per_frame")
    ap101, pr_p, pr_r = m.average_precision(ap_pred, gts, igns, points=101)
    ap11_legacy, _, _ = m.average_precision(prod_pred, gts, igns, points=11)
    result.update({
        "status": "OK",
        "status_quality": common.quality(det["n_gt"], args.headline_min_n),
        "n_val_records_available": loaded["n_val_records"],
        "n_images": len(gts),
        "sampling": f"seeded random sample (seed={args.seed}) of the official val split",
        "gt_definition": "visible boxes (vbox) of non-ignored persons; mask + ignore boxes = ignore regions",
        "mean_gt_persons_per_image": sum(gt_counts) / len(gt_counts),
        "detection_production": det,
        "ap50_coco101": {"value": ap101, "definition":
                         "COCO-style 101-point AP@IoU0.5, production model + nested-box "
                         "removal at conf=0.001 (standard AP protocol)"},
        "ap50_11pt_conf025_legacy_definition": {
            "value": ap11_legacy, "definition": "11-point AP on conf>=0.25 detections; "
            "only for comparison with the archived pre-audit run, NOT a standard mAP"},
        "count_errors_per_image": m.count_errors(pred_counts, gt_counts, seed=args.seed),
        "model_latency_ms": m.summarize_mean(lat, seed=args.seed),
        "pr_curve": {"precision": pr_p[::max(1, len(pr_p) // 400)],
                     "recall": pr_r[::max(1, len(pr_r) // 400)]},
        "conf_scores": [p[4] for preds in prod_pred for p in preds],
        "_size_instances": sizes,
    })
    if args.baselines and base_pred:
        b = common.evaluate_detection(base_pred, gts, igns, seed=args.seed)
        b.pop("_per_frame")
        lg = common.evaluate_detection(legacy_pred, gts, igns, seed=args.seed)
        lg.pop("_per_frame")
        result["baselines"] = {
            "yolov8s_detect_same_pipeline (NOT production)": b,
            "legacy_tracked_pipeline_pre_41ed380 (NOT production)": lg,
        }
    result["error_examples"] = errors.save()
    return result


def run_mirror_pilot(args) -> dict:
    """The 80 Mendeley-mirror images, evaluated with the class-0 filter.
    Kept only so the change from the archived run is explainable."""
    import cv2
    from .. import model_runner as mr

    root = os.path.join(args.data_root, "crowdhuman")
    img_dir, lbl_dir = os.path.join(root, "images"), os.path.join(root, "labels")
    result = {"name": "CrowdHuman images, third-party YOLO labels (provenance unverified)",
              "status": None, "status_quality": "pilot"}
    if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
        result["status"] = "SKIPPED: mirror images/labels not cached (not re-downloaded)"
        return result
    frames_pred, frames_gt = [], []
    dropped_total = Counter()
    for fname in sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg")):
        lbl = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
        frame = cv2.imread(os.path.join(img_dir, fname))
        if frame is None or not os.path.isfile(lbl):
            continue
        with open(lbl, encoding="utf-8") as f:
            gt, dropped = parse_yolo_person_labels(f.readlines(), frame.shape[1], frame.shape[0])
        dropped_total.update(dropped)
        res, _ = mr.detect_persons(frame)
        frames_pred.append(mr.boxes_xyxy_conf(res))
        frames_gt.append(gt)
    det = common.evaluate_detection(frames_pred, frames_gt, seed=args.seed)
    det.pop("_per_frame")
    result.update({
        "status": "PARTIAL: label provenance unverified (COCO class ids present, ~4.7 boxes/"
                  "image); none of these images are in the official val split. Pilot only.",
        "n_images": len(frames_gt),
        "non_person_label_boxes_dropped_by_class": dict(dropped_total),
        "detection_production": det,
    })
    return result
