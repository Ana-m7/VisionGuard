#!/usr/bin/env python
"""VisionGuard external-dataset accuracy AND minimum-CCTV-hardware-
requirement evaluation suite.

Run: python test_models.py [options]

This script does not modify any existing VisionGuard file. It only
creates/reads:
  external_datasets/      downloaded (idempotent -- re-runs skip cached files)
  model_eval_results/     charts/, summary.json, summary.csv

See eval_lib/ for the implementation. Each dataset module documents, in its
own docstring, exactly what was verified live (URLs, formats, fallbacks)
while this suite was built -- read those before trusting any one number.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_lib import http_utils as hu
from eval_lib import charts, report
from eval_lib.datasets import (
    crowdhuman, mot20, mall, pets2009, le2i, ur_fall, caucafall, uhctd,
    synthetic_tamper,
)
from eval_lib.sweeps import fps_sweep, resolution_sweep, person_size_sweep, min_spec


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="external_datasets")
    p.add_argument("--out-dir", default="model_eval_results")
    p.add_argument("--full", action="store_true",
                   help="Fetch/evaluate full datasets instead of the right-sized "
                        "default subsets (will download several GB and take much longer).")
    p.add_argument("--crowdhuman-limit", type=int, default=80)
    p.add_argument("--mot20-frames", type=int, default=80)
    p.add_argument("--mall-frames", type=int, default=150)
    p.add_argument("--le2i-clips-per-class", type=int, default=6)
    p.add_argument("--urfall-sequences", type=int, default=3,
                   help="number of UR Fall fall-sequences AND adl-sequences to "
                        "download (each sequence's RGB zip is 50-70MB with no "
                        "Range support -- keep this small by default).")
    p.add_argument("--caucafall-subjects", type=int, default=5,
                   help="(documented for future use -- Mendeley's file-listing API "
                        "currently has no working pagination, see eval_lib/datasets/"
                        "caucafall.py; this flag is accepted but has no effect yet.)")
    p.add_argument("--min-spec-threshold", type=float, default=0.9)
    p.add_argument("--skip", nargs="*", default=[],
                   help="dataset keys to skip, e.g. --skip mall pets2009")
    p.add_argument("--only", nargs="*", default=None,
                   help="run only these dataset keys")
    return p.parse_args()


DATASET_MODULES = {
    "crowdhuman": crowdhuman,
    "mot20": mot20,
    "mall": mall,
    "pets2009": pets2009,
    "le2i": le2i,
    "ur_fall": ur_fall,
    "caucafall": caucafall,
    "uhctd": uhctd,
    "synthetic_tamper": synthetic_tamper,
}


def run_dataset(key, module, args):
    hu.log(f"--- {key} ---")
    t0 = time.time()
    try:
        result = module.run(args)
    except Exception as e:  # noqa: BLE001 - one dataset's bug must not kill the run
        result = {"name": key, "status": f"FAILED: unhandled exception: {e}"}
        hu.log(f"{key}: unhandled exception -> {e}")
        traceback.print_exc()
    result["elapsed_seconds"] = round(time.time() - t0, 1)
    hu.log(f"{key}: {result.get('status')} ({result['elapsed_seconds']}s)")
    return result


def main():
    args = parse_args()
    keys = list(DATASET_MODULES.keys())
    if args.only:
        keys = [k for k in keys if k in args.only]
    keys = [k for k in keys if k not in args.skip]

    hu.log(f"VisionGuard evaluation suite starting. data_root={args.data_root} "
           f"out_dir={args.out_dir} full={args.full}")

    datasets_results = {}
    for key in keys:
        datasets_results[key] = run_dataset(key, DATASET_MODULES[key], args)

    # ---- Part 2: minimum CCTV hardware requirement sweeps ----
    mot20_dir = os.path.join(args.data_root, "mot20", "mot20", "MOT20-01")
    # mot20.py stores images under external_datasets/mot20/MOT20-01/img1
    mot20_dir = os.path.join(args.data_root, "mot20", "MOT20-01")
    mot20_gt_path = os.path.join(args.data_root, "mot20", "MOT20Labels", "train",
                                 "MOT20-01", "gt", "gt.txt")
    le2i_clips_dir = os.path.join(args.data_root, "le2i", "clips")
    le2i_manifest = os.path.join(args.data_root, "le2i", "manifest.txt")
    crowdhuman_root = os.path.join(args.data_root, "crowdhuman")

    hu.log("--- FPS sweep ---")
    fps_result = {}
    try:
        fps_result = fps_sweep.run(mot20_dir, le2i_clips_dir, le2i_manifest)
    except Exception as e:  # noqa: BLE001
        hu.log(f"FPS sweep failed: {e}")
        traceback.print_exc()

    mall_root = os.path.join(args.data_root, "mall")

    hu.log("--- Resolution sweep ---")
    resolution_result = {}
    try:
        resolution_result = resolution_sweep.run(mot20_dir, mot20_gt_path, mall_root)
    except Exception as e:  # noqa: BLE001
        hu.log(f"Resolution sweep failed: {e}")
        traceback.print_exc()

    hu.log("--- Person-size sweep ---")
    person_size_result = {}
    try:
        person_size_result = person_size_sweep.run(mot20_dir, mot20_gt_path, crowdhuman_root)
    except Exception as e:  # noqa: BLE001
        hu.log(f"Person-size sweep failed: {e}")
        traceback.print_exc()

    min_spec_result = min_spec.derive(fps_result, resolution_result, person_size_result,
                                      threshold=args.min_spec_threshold)

    # ---- Charts ----
    chart_dir = os.path.join(args.out_dir, "charts")
    chart_paths = []
    ch = datasets_results.get("crowdhuman", {})
    if ch.get("status") == "OK":
        chart_paths.append(charts.confidence_histogram(
            ch.get("conf_scores", []), chart_dir, "crowdhuman_confidence_hist.png",
            "CrowdHuman: detection confidence distribution"))
        chart_paths.append(charts.pr_curve(
            ch.get("pr_curve_precisions"), ch.get("pr_curve_recalls"), ch.get("mAP50", 0),
            chart_dir, "crowdhuman_pr_curve.png", "CrowdHuman: Precision-Recall curve"))
    mo = datasets_results.get("mot20", {})
    if mo.get("status") == "OK":
        chart_paths.append(charts.count_comparison_line(
            mo.get("pred_counts", []), mo.get("gt_counts", []), chart_dir,
            "mot20_count_over_time.png", "MOT20-01: predicted vs ground-truth person count"))
        chart_paths.append(charts.prf1_bar(
            mo.get("precision", 0), mo.get("recall", 0), mo.get("f1", 0), chart_dir,
            "mot20_detection_prf1.png", "MOT20-01: detection Precision/Recall/F1"))
        chart_paths.append(charts.tracking_bar(mo.get("tracking", {}), chart_dir))
    ma = datasets_results.get("mall", {})
    if ma.get("status") == "OK":
        chart_paths.append(charts.count_comparison_line(
            ma.get("pred_counts", []), ma.get("gt_counts", []), chart_dir,
            "mall_count_over_time.png", "Mall Dataset: predicted vs ground-truth count"))
        chart_paths.append(charts.mae_rmse_bar(
            ma.get("count_mae", 0), ma.get("count_rmse", 0), chart_dir,
            "mall_mae_rmse.png", "Mall Dataset: counting MAE/RMSE"))
    le = datasets_results.get("le2i", {})
    if le.get("status") == "OK":
        chart_paths.append(charts.prf1_bar(
            le.get("precision", 0), le.get("recall", 0), le.get("f1", 0), chart_dir,
            "le2i_fall_prf1.png", "Le2i: event-level fall-detection Precision/Recall/F1"))
    uf = datasets_results.get("ur_fall", {})
    if uf.get("status") == "OK":
        chart_paths.append(charts.prf1_bar(
            uf.get("precision", 0), uf.get("recall", 0), uf.get("f1", 0), chart_dir,
            "ur_fall_prf1.png", "UR Fall: event-level fall-detection Precision/Recall/F1"))
    latency_series = {}
    if le.get("status") == "OK" and le.get("latency_samples_frames"):
        latency_series["Le2i (clip-start proxy, not true onset)"] = le["latency_samples_frames"]
    if uf.get("status") == "OK" and uf.get("latency_samples_frames"):
        latency_series["UR Fall (true onset, label-derived)"] = uf["latency_samples_frames"]
    if latency_series:
        chart_paths.append(charts.latency_histogram(latency_series, chart_dir))
    ca = datasets_results.get("caucafall", {})
    if ca.get("status", "").startswith("PARTIAL") or ca.get("status") == "OK":
        if "detection_precision" in ca:
            chart_paths.append(charts.prf1_bar(
                ca.get("detection_precision", 0), ca.get("detection_recall", 0),
                ca.get("detection_f1", 0), chart_dir,
                "caucafall_detection_prf1.png", "CAUCAFall: person-detection Precision/Recall/F1"))
    st = datasets_results.get("synthetic_tamper", {})
    if st.get("status") == "OK":
        chart_paths.append(charts.confusion_matrix_chart(
            st.get("confusion_counts", {}), ["normal", "covered", "frozen"], chart_dir,
            "synthetic_tamper_confusion.png", "Synthetic tamper test: confusion matrix"))

    chart_paths.append(charts.fps_sweep_chart(fps_result, chart_dir))
    chart_paths.append(charts.resolution_sweep_chart(resolution_result, chart_dir))
    chart_paths.append(charts.person_size_sweep_chart(person_size_result, chart_dir))
    chart_paths = [p for p in chart_paths if p]

    all_results = {
        "datasets": datasets_results,
        "sweeps": {
            "fps": fps_result,
            "resolution": resolution_result,
            "person_size": person_size_result,
            "min_spec": min_spec_result,
        },
        "charts_generated": chart_paths,
        "args": vars(args),
    }
    json_path, csv_path = report.write_summary(all_results, args.out_dir)

    report.print_min_spec(min_spec_result)
    report.print_overall_summary(all_results)
    print(f"Charts written to: {chart_dir} ({len(chart_paths)} files)")
    print(f"Full results: {json_path}")
    print(f"Flat CSV:     {csv_path}")


if __name__ == "__main__":
    main()
