"""UR Fall Detection Dataset (University of Rzeszow) — fall-event detection,
false alarms and LATENCY IN SECONDS, at every frame and at production cadence.

Source (verified live): http://fenix.ur.edu.pl/~mkepski/ds/data/
  fall-NN-cam0-rgb.zip (30 falls), adl-NN-cam0-rgb.zip (40 daily-activity
  sequences), 30 fps; per-frame labels in urfall-cam0-falls.csv /
  urfall-cam0-adls.csv: -1 = not lying, 0 = transitional (falling), 1 = lying.

Fixes after the audit (#8, #3, #5):
- TRUE fall onset = first frame of the transitional (label 0) run that leads
  into the first lying frame; latency is measured from there, in SECONDS, with
  no tolerance window and therefore no clamping. Latency from the first lying
  frame is reported too.
- An alert raised in a fall sequence BEFORE the fall starts is a false alarm,
  not ignored.
- The pose tracker is reset at the start of every sequence and every rate.
- Each sequence is replayed at several effective inference rates by TIME
  (eval_lib/cadence.py), including production's ~2.8 Hz, through the
  production fall alert logic (confirm_falls imported from detector.engine +
  FALL_CLEAR_SECONDS mirrored in production_logic). The inference-rate sweep
  reads its numbers from here, so the two can never disagree.
- Large downloads are gated: uncached sequences beyond --max-download-mb are
  skipped (and reported) unless --allow-large-downloads is given.
"""
from __future__ import annotations

import os
import zipfile
from collections import defaultdict

from .. import cadence
from .. import common
from .. import frames as fr
from .. import http_utils as hu
from .. import metrics as m
from .. import production_logic as pl

import config  # noqa: E402

BASE = "http://fenix.ur.edu.pl/~mkepski/ds/data"
FALLS_CSV_URL = f"{BASE}/urfall-cam0-falls.csv"
ADLS_CSV_URL = f"{BASE}/urfall-cam0-adls.csv"
NATIVE_FPS = 30.0


def load_labels(csv_path):
    by_seq = defaultdict(dict)
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 3:
                by_seq[p[0]][int(p[1])] = int(p[2])
    return by_seq


def fall_timing(labels: dict):
    """-> dict(onset, lying_start, lying_end) in frame numbers, or None."""
    lying = sorted(f for f, lab in labels.items() if lab == 1)
    if not lying:
        return None
    onset = lying[0]
    while labels.get(onset - 1) == 0:
        onset -= 1
    return {"onset": onset, "lying_start": lying[0], "lying_end": lying[-1]}


def plan_sequences(args):
    falls = [f"fall-{i:02d}" for i in range(1, args.urfall_falls + 1)]
    adls = [f"adl-{i:02d}" for i in range(1, args.urfall_adls + 1)]
    seqs = []
    for i in range(max(len(falls), len(adls))):  # interleave so a budget cut stays balanced
        if i < len(falls):
            seqs.append((falls[i], True))
        if i < len(adls):
            seqs.append((adls[i], False))
    return seqs


def ensure_sequences(root, seqs, args):
    ready, skipped = [], []
    budget = None if args.allow_large_downloads else args.max_download_mb * 1024 * 1024
    spent = 0
    for name, is_fall in seqs:
        zip_name = f"{name}-cam0-rgb.zip"
        zip_path = os.path.join(root, zip_name)
        frames_dir = os.path.join(root, "frames", f"{name}-cam0-rgb")
        if not fr.list_frames(frames_dir):
            if not os.path.isfile(zip_path):
                size = hu.remote_size(f"{BASE}/{zip_name}") or 0
                if budget is not None and spent + size > budget:
                    skipped.append({"sequence": name, "reason":
                                    f"download budget {args.max_download_mb} MB reached "
                                    "(use --allow-large-downloads)"})
                    continue
                r = hu.simple_download(f"{BASE}/{zip_name}", zip_path, timeout=600)
                if not r.ok:
                    skipped.append({"sequence": name, "reason": r.message})
                    continue
                spent += size
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(os.path.join(root, "frames"))
        ready.append((name, is_fall, frames_dir))
    return ready, skipped, spent


def eval_sequence(frames_dir, labels, is_fall, rate_hz, pose_model_path=None):
    import cv2
    from .. import model_runner as mr
    files = fr.list_frames(frames_dir)
    idx = cadence.sample_indices(len(files), NATIVE_FPS, rate_hz)
    machine = pl.FallAlertMachine()
    raises, on_frames = [], []
    raise_frames = []
    for k, i in enumerate(idx):
        fno = fr.frame_number(files[i])
        frame = cv2.imread(os.path.join(frames_dir, files[i]))
        if frame is None:
            continue
        res, _ = mr.pose_infer(frame, new_sequence=(k == 0), pose_model_path=pose_model_path)
        out = machine.step(mr.classify_persons(res), fno / NATIVE_FPS)
        if out["raised"]:
            raises.append(fno)
            raise_frames.append((os.path.join(frames_dir, files[i]),
                                 [p["box"] for p in out["confirmed"]]))
        if out["alert"]:
            on_frames.append(fno)
    row = {"n_frames": len(files), "n_cycles": len(idx), "alerts_raised_at_frames": raises}
    timing = fall_timing(labels) if is_fall else None
    if is_fall and timing:
        early = [f for f in raises if f < timing["onset"]]
        in_window = [f for f in on_frames if timing["onset"] <= f <= timing["lying_end"]]
        detected = bool(in_window)
        first = in_window[0] if detected else None
        row.update({
            "timing_frames": timing, "detected": detected,
            "early_false_alarms": len(early),
            "latency_s_from_fall_onset": (first - timing["onset"]) / NATIVE_FPS if detected else None,
            "latency_s_from_lying_start": (first - timing["lying_start"]) / NATIVE_FPS if detected else None,
            "alert_already_on_at_onset": detected and first == timing["onset"] and bool(early),
        })
    else:
        row.update({"false_alarms": len(raises), "duration_s": len(files) / NATIVE_FPS})
    row["_raise_frames"] = raise_frames
    return row, idx


def aggregate(rows, min_n):
    falls = [r for r in rows if r["is_fall"] and "detected" in r]
    adls = [r for r in rows if not r["is_fall"]]
    tp = sum(1 for r in falls if r["detected"])
    adl_fp_seqs = sum(1 for r in adls if r["false_alarms"] > 0)
    fp_events = sum(r["false_alarms"] for r in adls) + sum(r["early_false_alarms"] for r in falls)
    adl_hours = sum(r["duration_s"] for r in adls) / 3600
    p, rec, f1 = m.precision_recall_f1(tp, fp_events, len(falls) - tp)
    return {
        "n_fall_sequences": len(falls), "n_adl_sequences": len(adls),
        "recall": m.proportion(tp, len(falls)),
        "precision_event_level": m.proportion(tp, tp + fp_events),
        "f1": f1,
        "adl_sequences_with_false_alarm": m.proportion(adl_fp_seqs, len(adls)),
        "fall_sequences_with_early_false_alarm": m.proportion(
            sum(1 for r in falls if r["early_false_alarms"] > 0), len(falls)),
        "false_alarm_events_total": fp_events,
        "false_alarms_per_hour_of_adl_footage": (sum(r["false_alarms"] for r in adls) / adl_hours)
        if adl_hours else None,
        "latency_s_from_fall_onset": m.summarize_mean(
            [r["latency_s_from_fall_onset"] for r in falls if r["detected"]]),
        "latency_s_from_lying_start": m.summarize_mean(
            [r["latency_s_from_lying_start"] for r in falls if r["detected"]]),
        "status_quality": common.quality(min(len(falls), len(adls)), min_n),
    }


def adl_split_by_lying(by_rate, labels):
    """Split ADL false alarms by whether the DATASET's own per-frame labels
    mark the person as lying (label 1) somewhere in the sequence. Several UR
    Fall ADL clips show someone deliberately lying down (e.g. on a bed or the
    floor); production's fall rule fires on a lying pose, so an alert there is
    a false alarm by the dataset's definition but not a random one."""
    lying = {s for s, lab in labels.items() if s.startswith("adl-") and any(v == 1 for v in lab.values())}
    out = {"definition": "ADL sequence 'with lying' = at least one frame labelled 1 (lying) in "
                         "urfall-cam0-adls.csv",
           "by_rate": {}}
    for key, r in by_rate.items():
        adl = [p for p in r.get("per_sequence", []) if not p["is_fall"]]
        groups = {"adl_with_lying": [p for p in adl if p["sequence"] in lying],
                  "adl_without_lying": [p for p in adl if p["sequence"] not in lying]}
        out["by_rate"][key] = {
            g: {**m.proportion(sum(1 for p in rows if p["alerts_raised_at_frames"]), len(rows)),
                "sequences_alarmed": [p["sequence"] for p in rows if p["alerts_raised_at_frames"]]}
            for g, rows in groups.items()}
    return out


def measure_cycle_rate(frame_paths):
    """Wall-clock time of one production cycle (detect + pose) on this CPU."""
    import time
    import cv2
    from .. import model_runner as mr
    times = []
    prev_dir = None
    for p in frame_paths:
        frame = cv2.imread(p)
        if frame is None:
            continue
        new_seq = os.path.dirname(p) != prev_dir
        prev_dir = os.path.dirname(p)
        t0 = time.perf_counter()
        mr.detect_persons(frame)
        mr.pose_infer(frame, new_sequence=new_seq)
        times.append(time.perf_counter() - t0)
    if not times:
        return None
    times = times[1:] or times   # drop the first (cold) cycle
    mean = sum(times) / len(times)
    return {"n_cycles_timed": len(times), "mean_cycle_s": mean,
            "achievable_rate_hz": 1 / max(mean, config.DETECT_INTERVAL),
            "detect_interval_cap_hz": cadence.max_rate_hz()}


def run(args) -> dict:
    from .. import model_runner as mr
    root = os.path.join(args.data_root, "ur_fall")
    os.makedirs(root, exist_ok=True)
    result = {"name": "UR Fall Detection Dataset", "status": None,
              "licence_note": "CC BY-NC-SA 4.0 (non-commercial academic use)"}
    falls_csv, adls_csv = (os.path.join(root, "urfall-cam0-falls.csv"),
                           os.path.join(root, "urfall-cam0-adls.csv"))
    r1 = hu.simple_download(FALLS_CSV_URL, falls_csv, timeout=60)
    r2 = hu.simple_download(ADLS_CSV_URL, adls_csv, timeout=60)
    if not (r1.ok and r2.ok):
        result["status"] = f"SKIPPED: label CSVs unavailable ({r1.message}; {r2.message})"
        return result
    labels = {**load_labels(falls_csv), **load_labels(adls_csv)}
    ready, skipped, spent = ensure_sequences(root, plan_sequences(args), args)
    if not ready:
        result["status"] = "SKIPPED: no sequences available"
        result["sequences_skipped"] = skipped
        return result

    errors = common.ErrorExamples("ur_fall", args.out_dir)
    rates = list(dict.fromkeys(args.rates))
    by_rate, distinct_ok = {}, True
    for rate in rates:
        rows = []
        for name, is_fall, d in ready:
            row, _ = eval_sequence(d, labels.get(name, {}), is_fall, rate)
            raise_frames = row.pop("_raise_frames")
            rows.append({"sequence": name, "is_fall": is_fall, **row})
            if rate == args.inference_hz:
                early = row.get("early_false_alarms", 0)
                if (not is_fall or early) and raise_frames:
                    path, boxes = raise_frames[0]
                    errors.add(f"{name}_falsealarm", path, [], boxes, 1, 0, note="false alarm")
                if is_fall and not row.get("detected", True):
                    files = fr.list_frames(d)
                    t = row["timing_frames"]
                    mid = next((f for f in files if fr.frame_number(f) >= t["lying_start"]), files[-1])
                    errors.add(f"{name}_missed", os.path.join(d, mid), [], [], 0, 1, note="missed fall")
        by_rate[cadence.rate_label(rate)] = {"rate_hz": rate, **aggregate(rows, args.headline_min_n),
                                             "per_sequence": rows}
    # every rate must look at a different frame set (audit #5)
    for _, _, d in ready:
        n = len(fr.list_frames(d))
        sets = {tuple(cadence.sample_indices(n, NATIVE_FPS, r)) for r in rates}
        distinct_ok &= len(sets) == len(rates)

    sample_paths = [os.path.join(d, f) for _, _, d in ready[:2] for f in fr.list_frames(d)[:15]]
    prod_key = cadence.rate_label(args.inference_hz)
    result.update({
        "status": "OK" if not skipped else "PARTIAL: some sequences not downloaded (see sequences_skipped)",
        "status_quality": by_rate[prod_key]["status_quality"],
        "native_fps": NATIVE_FPS,
        "n_sequences_requested": len(plan_sequences(args)),
        "n_sequences_evaluated": len(ready),
        "sequences_skipped": skipped,
        "downloaded_this_run_mb": round(spent / 1024 / 1024, 1),
        "production_rate_key": prod_key,
        "rates_evaluated": [cadence.rate_label(r) for r in rates],
        "every_rate_uses_distinct_frames": distinct_ok,
        "by_rate": by_rate,
        "adl_false_alarms_by_lying_label": adl_split_by_lying(by_rate, labels),
        "measured_production_cycle": measure_cycle_rate(sample_paths),
    })
    if args.baselines:
        rows = []
        for name, is_fall, d in ready:
            row, _ = eval_sequence(d, labels.get(name, {}), is_fall, args.inference_hz,
                                   pose_model_path=mr.model_path("yolov8n-pose.pt"))
            row.pop("_raise_frames")
            rows.append({"sequence": name, "is_fall": is_fall, **row})
        result["baselines"] = {
            f"yolov8n-pose at {prod_key} (NOT production)": aggregate(rows, args.headline_min_n)}
    result["error_examples"] = errors.save()
    return result
