"""summary.json / summary.csv writers and the console report.

summary.csv is LONG format: one row per headline metric with value, 95% CI,
n and quality ("headline" / "pilot" / ...), so no number can be quoted
without its uncertainty and sample size.
"""
from __future__ import annotations

import csv
import json
import os


def strip_private(obj):
    """Drop keys starting with '_' (large intermediate data) recursively."""
    if isinstance(obj, dict):
        return {k: strip_private(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [strip_private(v) for v in obj]
    return obj


def _row(rows, dataset, metric, obj, quality=None, note=""):
    if obj is None:
        return
    if isinstance(obj, dict):
        rows.append({"dataset": dataset, "metric": metric, "value": obj.get("value"),
                     "ci_low": obj.get("ci_low"), "ci_high": obj.get("ci_high"),
                     "n": obj.get("n"), "quality": quality, "note": note})
    else:
        rows.append({"dataset": dataset, "metric": metric, "value": obj, "ci_low": None,
                     "ci_high": None, "n": None, "quality": quality, "note": note})


def headline_rows(all_results):
    ds = all_results.get("datasets", {})
    sw = all_results.get("sweeps", {})
    rows = []
    for key, r in ds.items():
        status = str(r.get("status"))
        if not (status.startswith("OK") or status.startswith("PARTIAL")):
            rows.append({"dataset": key, "metric": "status", "value": status, "quality": "not run"})
            continue
        q = r.get("status_quality")
        if key in ("crowdhuman", "crowdhuman_mirror_pilot", "mot20", "caucafall"):
            d = r.get("detection_production") or {}
            for k in ("precision", "recall"):
                _row(rows, key, f"detection_{k}", d.get(k), q, "IoU 0.5, production pipeline")
            _row(rows, key, "detection_f1", d.get("f1"), q)
        if key == "crowdhuman":
            _row(rows, key, "ap50_coco101", r["ap50_coco101"]["value"], q, "conf>=0.001, 101-pt")
            ce = r["count_errors_per_image"]
            _row(rows, key, "count_mae_per_image", {**ce["mae"], "n": ce["n"]}, q)
            _row(rows, key, "count_relative_mae", {**ce["relative_mae"], "n": ce["n"]}, q)
        if key == "mot20":
            ce = r["count_errors"]
            _row(rows, key, "count_mae", {**ce["mae"], "n": ce["n"]}, q)
            _row(rows, key, "count_relative_mae", {**ce["relative_mae"], "n": ce["n"]}, q)
            t = r["tracking_pose_model"]
            for k in ("mota", "idf1", "motp_mean_iou", "idsw", "frag"):
                _row(rows, key, f"tracking_{k}", t.get(k), q, "pose-model tracker")
            cc = t.get("crosscheck_motmetrics")
            if isinstance(cc, dict):
                for k in ("mota", "idf1", "idsw", "frag"):
                    _row(rows, key, f"tracking_{k}_motmetrics", cc.get(k), q,
                         f"reference cross-check ({cc.get('package')})")
        if key == "mall":
            ce = r["count_errors"]
            for k in ("mae", "rmse", "bias", "relative_mae"):
                _row(rows, key, f"count_{k}", {**ce[k], "n": ce["n"]}, q, "block bootstrap")
            z = r["zone_density"]
            _row(rows, key, "zone_per_cell_mae", z["per_cell_mae"], q)
            _row(rows, key, "red_cell_precision", z["red_cell_flag"]["precision"], q)
            _row(rows, key, "red_cell_recall", z["red_cell_flag"]["recall"], q)
            a = r["alert_threshold_crossing_crop"]["production_logic"]
            for k in ("frame_agreement", "episode_recall", "episode_precision"):
                _row(rows, key, f"crowd_alert_{k}", a[k], q, r["alert_threshold_crossing_crop"]["crop_name"])
            syn = r["alert_logic_synthetic"]
            _row(rows, key, "crowd_alert_synthetic_checks_passed",
                 f"{sum(c['pass'] for c in syn.values())}/{len(syn)}", "functional check")
        if key == "le2i":
            for mode, res in r["results"].items():
                h = res["headline_fall_vs_stand"]
                for k in ("recall_fall", "precision", "false_alarm_rate_stand"):
                    _row(rows, key, f"{mode}:{k}", h[k], h["status_quality"], res["rate"])
                for cls, p in res["per_class_alert_rate"].items():
                    _row(rows, key, f"{mode}:alert_rate {cls}", p, h["status_quality"], res["rate"])
        if key == "ur_fall":
            for label, res in r["by_rate"].items():
                qq = res["status_quality"]
                _row(rows, key, f"{label}:recall", res["recall"], qq)
                _row(rows, key, f"{label}:precision_event_level", res["precision_event_level"], qq)
                _row(rows, key, f"{label}:adl_sequences_with_false_alarm",
                     res["adl_sequences_with_false_alarm"], qq)
                _row(rows, key, f"{label}:false_alarms_per_hour", res["false_alarms_per_hour_of_adl_footage"], qq)
                _row(rows, key, f"{label}:latency_s_from_fall_onset", res["latency_s_from_fall_onset"], qq)
                split = ((r.get("adl_false_alarms_by_lying_label") or {}).get("by_rate") or {}).get(label)
                for g, p in (split or {}).items():
                    _row(rows, key, f"{label}:false_alarm_rate {g}", p, qq,
                         "dataset label: person lying somewhere in the ADL clip" if g == "adl_with_lying"
                         else "no lying frame in the ADL clip")
        if key == "whatif_detection":
            wq = "what-if (NOT production)"
            for k, c in ((r.get("crowdhuman") or {}).get("configs") or {}).items():
                d = c["detection"]
                _row(rows, key, f"crowdhuman {k}: recall", d["recall"], wq, c["label"])
                _row(rows, key, f"crowdhuman {k}: precision", d["precision"], wq, c["label"])
                _row(rows, key, f"crowdhuman {k}: model_ms_per_frame", c["model_ms_per_frame"], wq, c["label"])
            for k, c in ((r.get("mall") or {}).get("configs") or {}).items():
                ce = c["count_errors"]
                _row(rows, key, f"mall {k}: count_mae", {**ce["mae"], "n": ce["n"]}, wq, c["label"])
                _row(rows, key, f"mall {k}: count_relative_mae", {**ce["relative_mae"], "n": ce["n"]},
                     wq, c["label"])
            for k, c in (((r.get("cycle_time") or {}).get("variants")) or {}).items():
                _row(rows, key, f"cycle {k}: seconds_per_cycle", c["seconds_per_cycle"], wq, c["label"])
        if key == "tamper_functional":
            _row(rows, key, "scenarios_matching_design",
                 f"{sum(s['matches_design'] for s in r['scenarios'])}/{len(r['scenarios'])}", q)
            for s in r["scenarios"]:
                _row(rows, key, f"scenario: {s['scenario']}", s["outcome"], q,
                     f"expected {s['expected_by_design']}, observed {s['first_detection'] or 'normal'}")
    ms = sw.get("min_spec") or {}
    for i, line in enumerate(ms.get("lines", [])):
        rows.append({"dataset": "min_spec", "metric": f"line_{i + 1}", "value": line, "quality": ""})
    return rows


def write_summary(all_results: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    clean = strip_private(all_results)
    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, default=str)
    csv_path = os.path.join(out_dir, "summary.csv")
    rows = headline_rows(clean)
    fields = ["dataset", "metric", "value", "ci_low", "ci_high", "n", "quality", "note"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return json_path, csv_path, rows


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def print_report(rows, min_spec):
    print("\n" + "=" * 78)
    print("MINIMUM CAMERA REQUIREMENTS (only where the data supports a conclusion)")
    print("=" * 78)
    for line in (min_spec or {}).get("lines", []):
        print(f"  - {line}")
    print("\n" + "=" * 78)
    print("HEADLINE METRICS  (value [95% CI] n=..., quality)")
    print("=" * 78)
    for r in rows:
        if r["dataset"] == "min_spec":
            continue
        ci = (f" [{_fmt(r['ci_low'])}-{_fmt(r['ci_high'])}]"
              if r.get("ci_low") is not None else "")
        n = f" n={r['n']}" if r.get("n") is not None else ""
        print(f"  {r['dataset']:<24} {r['metric']:<58} {_fmt(r['value'])}{ci}{n}  {r.get('quality') or ''}")
    print()
