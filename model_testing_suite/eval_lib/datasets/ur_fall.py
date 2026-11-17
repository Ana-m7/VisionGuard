"""UR Fall Detection Dataset (University of Rzeszow) — event-level fall
detection accuracy, WITH a true ground-truth onset frame (unlike Le2i's
re-packaged clips, which have no separate pre-fall baseline). This is the
second, independent fall-accuracy source requested to back up Le2i after
CAUCAFall's Mendeley pagination made its fall/ADL labels unreachable.

VERIFIED LIVE during this build: fenix.ur.edu.pl serves each sequence's RGB
frames as its own zip (``fall-NN-cam0-rgb.zip``, ``adl-NN-cam0-rgb.zip``,
each 50-70MB, no Range support advertised) plus two small shared feature
CSVs (``urfall-cam0-falls.csv``, ``urfall-cam0-adls.csv``) with a genuine
PER-FRAME label: -1 = not lying, 1 = lying on the ground, 0 = transitional
(excluded from classification, per the dataset's own documentation). The
first/last frame with label==1 gives a real fall-onset/offset interval,
so -- unlike Le2i -- actual detection LATENCY (frames from true onset to
VisionGuard's confirmed alert) can be computed here.

Each sequence zip is tens of MB with no Range support, so by default only
``--urfall-sequences`` fall + the same number of ADL sequences are
downloaded (documented trade-off: this is a real, one-time, bandwidth-
bound cost -- idempotent on re-runs).
"""
from __future__ import annotations

import os
import zipfile
from collections import defaultdict

from .. import http_utils as hu

BASE = "http://fenix.ur.edu.pl/~mkepski/ds/data"
FALLS_CSV_URL = f"{BASE}/urfall-cam0-falls.csv"
ADLS_CSV_URL = f"{BASE}/urfall-cam0-adls.csv"


def _load_labels(csv_path):
    """sequence -> {frame_number: label}"""
    by_seq = defaultdict(dict)
    with open(csv_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            seq, frame, label = parts[0], int(parts[1]), int(parts[2])
            by_seq[seq][frame] = label
    return by_seq


def _fall_interval(frame_labels):
    lying_frames = [fr for fr, lbl in frame_labels.items() if lbl == 1]
    if not lying_frames:
        return None
    return (min(lying_frames), max(lying_frames))


def run(args) -> dict:
    import cv2
    from .. import metrics as m
    from .. import model_runner as mr

    root = os.path.join(args.data_root, "ur_fall")
    os.makedirs(root, exist_ok=True)
    result = {"name": "UR Fall Detection Dataset", "status": None,
              "licence_note": "CC BY-NC-SA 4.0 (non-commercial academic use)"}

    n_seq = args.urfall_sequences if not args.full else 30
    falls_csv = os.path.join(root, "urfall-cam0-falls.csv")
    adls_csv = os.path.join(root, "urfall-cam0-adls.csv")
    r1 = hu.simple_download(FALLS_CSV_URL, falls_csv, timeout=60)
    r2 = hu.simple_download(ADLS_CSV_URL, adls_csv, timeout=60)
    if not (r1.ok and r2.ok):
        result["status"] = f"SKIPPED: could not download feature CSVs ({r1.message}; {r2.message})"
        return result

    fall_labels = _load_labels(falls_csv)
    adl_labels = _load_labels(adls_csv)

    sequences = ([(f"fall-{i:02d}", True) for i in range(1, n_seq + 1)] +
                [(f"adl-{i:02d}", False) for i in range(1, n_seq + 1)])

    mr.warmup()
    per_sequence = []
    for seq_name, is_fall in sequences:
        zip_name = f"{seq_name}-cam0-rgb.zip"
        zip_path = os.path.join(root, zip_name)
        extract_dir = os.path.join(root, "frames", f"{seq_name}-cam0-rgb")
        dl = hu.simple_download(f"{BASE}/{zip_name}", zip_path, timeout=300)
        if not dl.ok:
            per_sequence.append({"sequence": seq_name, "is_fall": is_fall,
                                 "status": f"SKIPPED: {dl.message}"})
            continue
        if not os.path.isdir(extract_dir) or not os.listdir(extract_dir):
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(os.path.join(root, "frames"))

        frame_files = sorted(
            f for f in os.listdir(extract_dir) if f.endswith(".png"))
        labels = fall_labels.get(seq_name) if is_fall else adl_labels.get(seq_name, {})
        gt_interval = _fall_interval(labels) if is_fall else None

        streaks = {}
        confirmed_frames = []
        for fname in frame_files:
            frame_no = int(os.path.splitext(fname)[0].split("-")[-1])
            frame = cv2.imread(os.path.join(extract_dir, fname))
            if frame is None:
                continue
            pose_result, _ = mr.pose_infer(frame, persist=True)
            persons = mr.classify_persons(pose_result)
            if mr.confirm_falls(persons, streaks):
                confirmed_frames.append(frame_no)

        if gt_interval:
            precision, recall, f1, latencies = m.event_level_prf1(
                [gt_interval], confirmed_frames, tolerance_frames=5)
            latency = latencies[0] if latencies else None
        else:
            # ADL sequence: any confirmation at all is a false positive.
            precision, recall, f1 = (0.0, None, 0.0) if confirmed_frames else (1.0, None, 1.0)
            latency = None

        per_sequence.append({
            "sequence": seq_name, "is_fall": is_fall, "status": "OK",
            "n_frames": len(frame_files), "gt_interval": gt_interval,
            "confirmed_frames": confirmed_frames, "latency_frames": latency,
        })

    ok_rows = [r for r in per_sequence if r.get("status") == "OK"]
    pos = [r for r in ok_rows if r["is_fall"]]
    neg = [r for r in ok_rows if not r["is_fall"]]
    tp = sum(1 for r in pos if r["confirmed_frames"] and r["gt_interval"] and
             any(r["gt_interval"][0] - 5 <= f <= r["gt_interval"][1] + 5
                 for f in r["confirmed_frames"]))
    fn = len(pos) - tp
    fp = sum(1 for r in neg if r["confirmed_frames"])
    tn = len(neg) - fp
    precision, recall, f1 = m.precision_recall_f1(tp, fp, fn)
    fpr_on_adl = fp / len(neg) if neg else None
    latencies = [r["latency_frames"] for r in pos if r.get("latency_frames") is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else None
    skipped = [r for r in per_sequence if r.get("status") != "OK"]

    result.update({
        "status": "OK" if ok_rows else "SKIPPED: no sequences downloaded successfully",
        "n_sequences_requested": len(sequences),
        "n_sequences_ok": len(ok_rows), "n_sequences_skipped": len(skipped),
        "n_positive": len(pos), "n_negative": len(neg),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "false_positive_rate_on_ADL_sequences": fpr_on_adl,
        "avg_latency_frames": avg_latency,
        "latency_samples_frames": latencies,
        "per_sequence_skipped": [r for r in skipped],
        "per_sequence": per_sequence,
    })
    return result
