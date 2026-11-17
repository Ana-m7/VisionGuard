"""PETS2009 — zone/region ground-truth cross-check (best-effort).

VERIFIED LIVE during this build: the "People Counting Data for PETS2009"
archive (visal.cs.cityu.edu.hk/static/downloads/pets2009_feats.zip, 4.2MB)
contains 3 regions-of-interest (R0/R1/R2 .mat masks), pre-extracted holistic
features + ground-truth counts (``cnt`` field) per region, and per-frame
foreground SEGMENTATION MASKS for a few short clips — but it does **not**
bundle the raw RGB video frames themselves. Per the spec's instruction, we
do not fail the dataset outright: ROI/count metadata is downloaded and
reported, but the person-detector-vs-zone_densities() comparison that needs
real RGB frames is explicitly marked unavailable rather than faked.
"""
from __future__ import annotations

import os
import zipfile

from .. import http_utils as hu

FEATS_URL = "http://visal.cs.cityu.edu.hk/static/downloads/pets2009_feats.zip"
README_URL = "http://visal.cs.cityu.edu.hk/static/downloads/crowddoc/README-pets.pdf"


def run(args) -> dict:
    import scipy.io as sio

    root = os.path.join(args.data_root, "pets2009")
    zip_path = os.path.join(root, "pets2009_feats.zip")
    result = {"name": "PETS2009", "status": None}

    r = hu.simple_download(FEATS_URL, zip_path, timeout=180)
    if not r.ok:
        result["status"] = f"SKIPPED: {r.message}"
        return result
    hu.simple_download(README_URL, os.path.join(root, "README-pets.pdf"), timeout=60)

    roi_regions = {}
    gt_counts = {}
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for region in ("R0", "R1", "R2"):
                roi_name = f"gt/pets2009/pets_{region}_roi.mat"
                feat_name = f"features/pets2009_{region}_feats.mat"
                if roi_name in zf.namelist():
                    zf.extract(roi_name, root)
                    roi_regions[region] = "downloaded"
                if feat_name in zf.namelist():
                    zf.extract(feat_name, root)
                    try:
                        feats = sio.loadmat(os.path.join(root, feat_name))
                        cnt = feats.get("cnt")
                        gt_counts[region] = {
                            "has_ground_truth_counts": cnt is not None,
                            "shape": list(cnt.shape) if cnt is not None else None,
                        }
                    except Exception as e:  # noqa: BLE001
                        gt_counts[region] = {"error": str(e)}
    except Exception as e:  # noqa: BLE001
        result["status"] = f"PARTIAL: archive downloaded but could not be parsed ({e})"
        return result

    result.update({
        "status": ("PARTIAL: ROI + ground-truth count metadata downloaded for "
                   f"{len(roi_regions)}/3 regions, but this archive contains only "
                   "pre-extracted features/segmentation masks, NOT raw RGB video "
                   "frames -- a VisionGuard zone_densities() vs. PETS2009-region "
                   "comparison requires the original multi-camera video sequences, "
                   "which are hosted separately and were not reliably retrievable "
                   "in this session. Reporting dataset metadata only."),
        "regions_available": roi_regions,
        "ground_truth_counts_present": gt_counts,
    })
    return result
