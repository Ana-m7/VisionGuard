"""Mall Dataset (CUHK) — crowd counting MAE/RMSE.

Fixed indoor camera, shopping mall corridor — structurally the closest
public dataset to VisionGuard's own "Corridor"/"Entrance" camera framing.
Downloaded in full (87.6MB, small enough), but by default only the first
``--mall-frames`` frames are run through the model to keep a default run's
wall-clock time reasonable; ``--full`` evaluates all 2,000.

Research-only licence (CUHK): not for commercial use. Logged, not enforced
in code since this script never redistributes the data.
"""
from __future__ import annotations

import os
import zipfile

from .. import http_utils as hu

ZIP_URL = "https://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/mall_dataset.zip"


def run(args) -> dict:
    import cv2
    import scipy.io as sio
    from .. import metrics as m
    from .. import model_runner as mr

    root = os.path.join(args.data_root, "mall")
    zip_path = os.path.join(root, "mall_dataset.zip")
    extract_dir = os.path.join(root, "mall_dataset")
    result = {"name": "Mall Dataset", "status": None,
              "licence_note": "CUHK research-use-only licence"}

    r = hu.simple_download(ZIP_URL, zip_path, timeout=600)
    if not r.ok:
        result["status"] = f"SKIPPED: {r.message}"
        return result

    gt_mat = os.path.join(extract_dir, "mall_gt.mat")
    frames_dir = os.path.join(extract_dir, "frames")
    if not os.path.isfile(gt_mat) or not os.path.isdir(frames_dir) or not os.listdir(frames_dir):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root)

    try:
        gt = sio.loadmat(gt_mat)
        gt_counts_all = gt["count"].flatten().tolist()
    except Exception as e:  # noqa: BLE001
        result["status"] = f"SKIPPED: could not parse mall_gt.mat ({e})"
        return result

    n_frames = args.mall_frames if not args.full else len(gt_counts_all)
    n_frames = min(n_frames, len(gt_counts_all))

    mr.warmup()
    pred_counts = []
    for i in range(1, n_frames + 1):
        fpath = os.path.join(frames_dir, f"seq_{i:06d}.jpg")
        frame = cv2.imread(fpath)
        if frame is None:
            pred_counts.append(None)
            continue
        det_result, _ = mr.detect_persons(frame, persist=False)
        pred_counts.append(len(mr.boxes_xyxy_conf(det_result)))

    valid = [(p, g) for p, g in zip(pred_counts, gt_counts_all[:n_frames]) if p is not None]
    if not valid:
        result["status"] = "SKIPPED: no frames could be read"
        return result
    preds, gts = zip(*valid)
    mae, rmse = m.mae_rmse(list(preds), list(gts))

    result.update({
        "status": "OK",
        "n_frames": len(valid),
        "count_mae": mae, "count_rmse": rmse,
        "pred_counts": list(preds), "gt_counts": list(gts),
    })
    return result
