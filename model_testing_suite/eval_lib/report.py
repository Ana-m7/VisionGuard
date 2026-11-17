"""summary.json / summary.csv writers + the console report, in the order
the spec requires: (1) MINIMUM CCTV REQUIREMENTS, (2) OVERALL ACCURACY
SUMMARY."""
from __future__ import annotations

import csv
import json
import os


def write_summary(all_results: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    csv_path = os.path.join(out_dir, "summary.csv")
    rows = []
    for dataset_key, r in all_results.get("datasets", {}).items():
        row = {"dataset": dataset_key, "status": r.get("status")}
        for k, v in r.items():
            if k in ("status", "per_clip", "cases_summary", "pred_counts",
                     "gt_counts", "crowd_alert_log", "conf_scores", "latency",
                     "per_class_metrics", "confusion_counts"):
                continue
            if isinstance(v, (int, float, str, bool)) or v is None:
                row[k] = v
        rows.append(row)
    fieldnames = sorted({k for row in rows for k in row})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return json_path, csv_path


def print_min_spec(min_spec_result: dict):
    print("\n" + "=" * 70)
    print("MINIMUM CCTV REQUIREMENTS")
    print("=" * 70)
    for line in min_spec_result.get("lines", []):
        print(f"  - {line}")
    print()


def print_overall_summary(all_results: dict):
    print("=" * 70)
    print("OVERALL ACCURACY SUMMARY")
    print("=" * 70)
    rows = [
        ("Person detection", "CrowdHuman (Mendeley YOLO mirror), MOT20-01, CAUCAFall",
         _fmt_detection(all_results)),
        ("Tracking", "MOT20-01",
         _fmt_tracking(all_results)),
        ("Crowd / zone-density", "MOT20-01, Mall Dataset, PETS2009 (metadata only)",
         _fmt_crowd(all_results)),
        ("Fall detection", "Le2i, UR Fall (CAUCAFall: detection-only, see status)",
         _fmt_fall(all_results)),
        ("Camera tamper", "Synthetic test (UHCTD gated -- see status)",
         _fmt_tamper(all_results)),
    ]
    width1, width2 = 22, 46
    for name, datasets, metric in rows:
        print(f"{name:<{width1}} | {datasets:<{width2}} | {metric}")
    print()


def _get(all_results, key):
    return all_results.get("datasets", {}).get(key, {})


def _fmt_detection(r):
    ch = _get(r, "crowdhuman")
    parts = []
    if ch.get("status") == "OK":
        parts.append(f"CrowdHuman mAP50={ch.get('mAP50', 0):.2f} F1={ch.get('f1', 0):.2f}")
    mo = _get(r, "mot20")
    if mo.get("status") == "OK":
        parts.append(f"MOT20-01 F1={mo.get('f1', 0):.2f}")
    return "; ".join(parts) if parts else "no successful dataset"


def _fmt_tracking(r):
    mo = _get(r, "mot20")
    t = mo.get("tracking", {})
    if t and "id_switches" in t:
        return f"ID switches={t['id_switches']}, fragmentations={t['fragmentations']} ({t.get('implementation')})"
    return "unavailable"


def _fmt_crowd(r):
    mo = _get(r, "mot20")
    ma = _get(r, "mall")
    parts = []
    if mo.get("status") == "OK":
        parts.append(f"MOT20-01 count MAE={mo.get('count_mae', 0):.1f}, "
                     f"hysteresis agreement={mo.get('crowd_hysteresis_agreement_with_gt_threshold', 0):.2f}")
    if ma.get("status") == "OK":
        parts.append(f"Mall MAE={ma.get('count_mae', 0):.1f}")
    return "; ".join(parts) if parts else "no successful dataset"


def _fmt_fall(r):
    parts = []
    le = _get(r, "le2i")
    if le.get("status") == "OK":
        parts.append(f"Le2i P={le.get('precision', 0):.2f} R={le.get('recall', 0):.2f} "
                     f"F1={le.get('f1', 0):.2f} FPR(ADL)={le.get('false_positive_rate_on_ADL_clips')} "
                     f"avg_latency={le.get('avg_latency_frames')}frames")
    uf = _get(r, "ur_fall")
    if uf.get("status") == "OK":
        parts.append(f"UR-Fall P={uf.get('precision', 0):.2f} R={uf.get('recall', 0):.2f} "
                     f"F1={uf.get('f1', 0):.2f} FPR(ADL)={uf.get('false_positive_rate_on_ADL_sequences')} "
                     f"avg_latency={uf.get('avg_latency_frames')}frames")
    return "; ".join(parts) if parts else "unavailable"


def _fmt_tamper(r):
    st = _get(r, "synthetic_tamper")
    if st.get("status") == "OK":
        pc = st.get("per_class_metrics", {})
        avg_f1 = sum(c["f1"] for c in pc.values()) / len(pc) if pc else 0
        return f"synthetic avg F1={avg_f1:.2f} (per-class: {pc})"
    return "unavailable"
