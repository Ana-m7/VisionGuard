"""Synthetic covered/frozen/normal camera-tamper test — ALWAYS runs
(guaranteed-available fallback/complement to UHCTD, whose real dataset is
gated behind a Dataset Agreement). Ground truth is known exactly by
construction, same technique as tests/test_health_tamper.py, exercised
through the real production ``TamperDetector`` class and its configured
confirm-frame thresholds from config.py.
"""
from __future__ import annotations

import numpy as np

from .. import model_runner as mr
from .. import metrics as m

H, W = 270, 480
_rng = np.random.default_rng(0)


def _dark():
    return np.full((H, W, 3), 5, np.uint8)


def _scene():
    return _rng.integers(40, 220, (H, W, 3), dtype=np.uint8)


def _bright_featureless():
    return _rng.integers(118, 126, (H, W, 3), dtype=np.uint8)


def _run_sequence(frames, label):
    config = mr.config
    det = mr.TamperDetector()
    results = []
    first_hit_frame = None
    for i, frame in enumerate(frames):
        r = det.check(frame)
        results.append(r)
        if r is not None and first_hit_frame is None:
            first_hit_frame = i
    final = results[-1]
    return {
        "true_label": label,
        "predicted_label": final if final is not None else "normal",
        "first_detected_at_frame": first_hit_frame,
        "per_frame": results,
    }


def run(args) -> dict:
    config = mr.config
    cases = []

    # 1. Normal / live-changing feed -- must NEVER be flagged.
    n = config.TAMPER_FROZEN_CONFIRM_FRAMES + 10
    cases.append(_run_sequence([_scene() for _ in range(n)], "normal"))

    # 2. Covered: lens blacked out.
    n = config.TAMPER_COVERED_CONFIRM_FRAMES + 5
    cases.append(_run_sequence([_dark() for _ in range(n)], "covered"))

    # 3. Covered: bright but featureless (hand over a webcam w/ auto-exposure).
    n = config.TAMPER_FROZEN_CONFIRM_FRAMES + 5
    cases.append(_run_sequence([_bright_featureless() for _ in range(n)], "covered"))

    # 4. Frozen: identical frame repeated (simulated replay/stuck feed).
    still = _scene()
    n = config.TAMPER_FROZEN_CONFIRM_FRAMES + 5
    cases.append(_run_sequence([still.copy() for _ in range(n)], "frozen"))

    # 5. A second, independent normal case and a second frozen case for a
    #    slightly less trivial confusion matrix (still fully synthetic/known).
    cases.append(_run_sequence([_scene() for _ in range(n)], "normal"))
    still2 = _scene()
    cases.append(_run_sequence([still2.copy() for _ in range(n)], "frozen"))

    y_true = [c["true_label"] for c in cases]
    y_pred = [c["predicted_label"] for c in cases]
    classes = ["normal", "covered", "frozen"]
    counts, per_class = m.confusion_counts(y_true, y_pred, classes)

    latencies = {
        "covered_confirm_frames_configured": config.TAMPER_COVERED_CONFIRM_FRAMES,
        "frozen_confirm_frames_configured": config.TAMPER_FROZEN_CONFIRM_FRAMES,
        "observed": [
            {"true_label": c["true_label"], "first_detected_at_frame": c["first_detected_at_frame"]}
            for c in cases
        ],
    }

    return {
        "name": "Synthetic Tamper Test",
        "status": "OK",
        "n_cases": len(cases),
        "confusion_counts": {f"{t}->{p}": n for (t, p), n in counts.items()},
        "per_class_metrics": per_class,
        "latency": latencies,
        "cases_summary": [{"true": c["true_label"], "pred": c["predicted_label"]} for c in cases],
    }
