"""Thin wrapper around the two production models, calling them exactly the
way detector/engine.py does (same .track() args, same confidence floor, same
classify_persons()/confirm_falls()/zone_densities() logic) so every metric in
this evaluation suite reflects the real VisionGuard pipeline, not a
re-implementation of it.

Nothing here starts a thread, opens the SQLite DB, or touches a camera —
only detector.engine's pure functions/classes and the two .pt model files
are used.
"""
from __future__ import annotations

import logging
import os
import sys
import time

# Silence ultralytics' BoT-SORT/GMC "falling back to identity" warning noise --
# it fires whenever consecutive tracked frames differ in size (expected here,
# since several sweeps deliberately resize frames) and is harmless: ByteTrack/
# BoT-SORT falls back to an identity motion-compensation transform and keeps
# working correctly. verbose=False on .track() does not suppress this because
# it's logged at WARNING level, not INFO.
logging.getLogger("ultralytics").setLevel(logging.ERROR)

def _find_project_root(start, max_levels=6):
    """Walk upward from `start` until a directory containing config.py is
    found -- robust to this package living at any depth under the real
    VisionGuard project root (currently model_testing_suite/eval_lib/, two
    levels deep, but this does not hardcode that number so a future move
    doesn't silently break these imports again)."""
    current = os.path.abspath(start)
    for _ in range(max_levels):
        if os.path.isfile(os.path.join(current, "config.py")):
            return current
        parent = os.path.dirname(current)
        if parent == current:  # reached filesystem root
            break
        current = parent
    raise RuntimeError(
        f"Could not find the VisionGuard project root (a directory containing "
        f"config.py) within {max_levels} levels above {start}. "
        f"eval_lib/model_runner.py may have been moved somewhere unexpected.")


_PROJECT_ROOT = _find_project_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402
from detector.engine import (  # noqa: E402
    classify_persons, confirm_falls, zone_densities, TamperDetector,
)

_det_model = None
_pose_model = None


def _lazy_models():
    global _det_model, _pose_model
    if _det_model is None or _pose_model is None:
        from ultralytics import YOLO
        _det_model = YOLO(config.DETECTION_MODEL_PATH)
        _pose_model = YOLO(config.POSE_MODEL_PATH)
    return _det_model, _pose_model


def warmup():
    """Load both models once up front so the first timed frame isn't
    penalised by lazy weight loading."""
    _lazy_models()


def detect_persons(frame, persist=True):
    """Exactly detector/engine.py's crowd-counting call. Returns
    (ultralytics Results, latency_ms)."""
    det_model, _ = _lazy_models()
    t0 = time.perf_counter()
    result = det_model.track(frame, classes=[0], persist=persist,
                             conf=config.DETECTION_CONF_MIN, verbose=False)[0]
    latency_ms = (time.perf_counter() - t0) * 1000
    return result, latency_ms


def pose_infer(frame, persist=True):
    """Exactly detector/engine.py's fall-detection call. Returns
    (ultralytics Results, latency_ms)."""
    _, pose_model = _lazy_models()
    t0 = time.perf_counter()
    result = pose_model.track(frame, classes=[0], persist=persist,
                              conf=config.DETECTION_CONF_MIN, verbose=False)[0]
    latency_ms = (time.perf_counter() - t0) * 1000
    return result, latency_ms


def boxes_xyxy_conf(result):
    """[(x1,y1,x2,y2,conf), ...] from a detection Results object, or []."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    out = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        conf = float(boxes.conf[i]) if boxes.conf is not None else None
        out.append((x1, y1, x2, y2, conf))
    return out


__all__ = [
    "warmup", "detect_persons", "pose_infer", "boxes_xyxy_conf",
    "classify_persons", "confirm_falls", "zone_densities", "TamperDetector",
    "config",
]
