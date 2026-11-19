"""The ONE place the suite touches the production models.

Every call mirrors ``detector/engine.py::inference_thread`` exactly, and every
piece of logic that production exposes as a function is IMPORTED from
``detector.engine`` (never copied):

  crowd counting  : det_model.predict(frame, classes=[0], conf=DETECTION_CONF_MIN)
                    -> engine.drop_nested_boxes(result)        (engine.py:742-747)
  fall detection  : pose_model.track(frame, classes=[0], persist=True,
                    conf=DETECTION_CONF_MIN)                    (engine.py:744)
                    -> engine.classify_persons -> engine.confirm_falls
  zones / tamper  : engine.zone_densities, engine.TamperDetector

``check_production_parity()`` re-reads the engine source and aborts the run if
any of those calls change, so the suite cannot silently drift from the system
it claims to measure (audit issues #1 and #4).

Tracker hygiene (audit issue #3): the pose tracker is reset at the start of
every independent clip / sequence / sweep point (``new_sequence=True``), which
mirrors production starting a fresh tracker per camera thread. Within a
sequence ``persist=True`` is used, exactly as production does. No ``tracker=``
argument is passed, because production passes none: the installed
ultralytics default is used and recorded in the results.
"""
from __future__ import annotations

import inspect
import logging
import os
import time

from . import paths  # noqa: F401  (puts the VisionGuard root on sys.path)

import config  # noqa: E402
from detector import engine  # noqa: E402
from detector.engine import (  # noqa: E402,F401  (re-exported for dataset modules)
    TamperDetector, classify_persons, confirm_falls, drop_nested_boxes,
    nested_duplicates, zone_densities,
)


class _DropGmcWarnings(logging.Filter):
    """Only the tracker's global-motion-compensation fallback warnings are
    dropped (they fire whenever consecutive frames differ in size, e.g. in the
    resolution sweep, and are harmless). Everything else still logs."""

    _NOISE = ("findTransformECC failed", "not enough matching points",
              "GMC failed, falling back to identity")

    def filter(self, record):
        return not any(s in record.getMessage() for s in self._NOISE)


logging.getLogger("ultralytics").addFilter(_DropGmcWarnings())


# ------------------------------------------------------------- parity guard
class ProductionParityError(RuntimeError):
    pass


PARITY_TOKENS = (
    "det_model.predict(",
    "drop_nested_boxes(",
    "pose_model.track(",
    "persist=True",
    "classes=[0]",
    "conf=config.DETECTION_CONF_MIN",
    "classify_persons(pose_result)",
    "confirm_falls(persons, fall_streaks)",
    "zone_densities(boxes, frame.shape)",
    "CROWD_CONFIRM_CYCLES",
    "CROWD_CLEAR_MARGIN",
    "FALL_CLEAR_SECONDS",
)
PARITY_CONFIG_ATTRS = (
    "DETECTION_MODEL_PATH", "POSE_MODEL_PATH", "DETECTION_CONF_MIN",
    "NESTED_BOX_CONTAINMENT", "CROWD_MIN_PERSONS", "CROWD_CLEAR_MARGIN",
    "CROWD_CONFIRM_CYCLES", "CROWD_THRESHOLD", "GRID_ROWS", "GRID_COLS",
    "FALL_CONFIRM_FRAMES", "FALL_CLEAR_SECONDS", "DETECT_INTERVAL",
    "TAMPER_COVERED_CONFIRM_FRAMES", "TAMPER_FROZEN_CONFIRM_FRAMES",
)


def check_production_parity() -> dict:
    """Abort (ProductionParityError) if detector/engine.py no longer makes the
    calls this suite mirrors. Returns a small report for summary.json."""
    src = inspect.getsource(engine.inference_thread)
    missing = [t for t in PARITY_TOKENS if t not in src]
    tracked_det = "det_model.track(" in src
    missing_cfg = [a for a in PARITY_CONFIG_ATTRS if not hasattr(config, a)]
    if missing or tracked_det or missing_cfg:
        raise ProductionParityError(
            "detector/engine.py no longer matches what model_testing_suite mirrors. "
            f"Missing statements: {missing}; det model tracked again: {tracked_det}; "
            f"missing config attributes: {missing_cfg}. Update "
            "eval_lib/model_runner.py and eval_lib/production_logic.py before "
            "trusting any result.")
    return {"status": "OK", "checked_tokens": list(PARITY_TOKENS),
            "checked_config_attrs": list(PARITY_CONFIG_ATTRS),
            "engine_file": os.path.relpath(inspect.getsourcefile(engine), paths.PROJECT_ROOT)}


# ------------------------------------------------------------- model cache
_models: dict[tuple, object] = {}


def get_model(path: str, role: str = "predict"):
    """One YOLO instance per (weights file, role). Roles are kept on SEPARATE
    instances because calling .track() on a YOLO object permanently attaches
    tracker callbacks to it, after which even .predict() on that object is
    filtered through a tracker. Production never tracks its detection model,
    so a tracked call must never share an instance with a predict call.
    Model files are opened read-only."""
    key = (path, role)
    if key not in _models:
        from ultralytics import YOLO
        _models[key] = YOLO(path)
    return _models[key]


def _assert_untracked(model):
    """Guard: a predict-role model must have no tracker attached."""
    pred = getattr(model, "predictor", None)
    if pred is not None and hasattr(pred, "trackers"):
        raise RuntimeError("tracker callbacks found on a predict-only model instance; "
                           "results would be tracker-filtered (see get_model docstring)")


def model_path(name: str) -> str:
    """Path of a weights file in the project's models/ folder (read-only)."""
    return os.path.join(paths.PROJECT_ROOT, "models", name)


def warmup():
    import numpy as np
    blank = np.zeros((480, 640, 3), np.uint8)
    get_model(config.DETECTION_MODEL_PATH, "predict").predict(blank, verbose=False)
    get_model(config.POSE_MODEL_PATH, "track").predict(blank, verbose=False)


# ------------------------------------------------------------- inference
def _size_kw(imgsz):
    """Production never passes imgsz (ultralytics default 640); only what-if
    configurations set it, so the default call stays byte-identical."""
    return {} if imgsz is None else {"imgsz": int(imgsz)}


def detect_persons(frame, det_model_path: str | None = None, conf: float | None = None,
                   imgsz: int | None = None, role: str = "predict"):
    """Production crowd-counting call (engine.py:742-747).

    Returns (result_after_nested_box_removal, model_latency_ms).
    ``det_model_path`` / ``conf`` / ``imgsz`` exist only for clearly-labelled
    baselines, what-if configurations and the conf=0.001 AP pass; production
    uses config values. ``role`` lets a what-if use the POSE weights as a
    counter on an untracked instance (never the tracked one).
    """
    model = get_model(det_model_path or config.DETECTION_MODEL_PATH, role)
    t0 = time.perf_counter()
    result = model.predict(frame, classes=[0],
                           conf=config.DETECTION_CONF_MIN if conf is None else conf,
                           verbose=False, **_size_kw(imgsz))[0]
    latency_ms = (time.perf_counter() - t0) * 1000
    _assert_untracked(model)
    return drop_nested_boxes(result), latency_ms


def detect_persons_legacy_tracked(frame, new_sequence: bool):
    """ABLATION ONLY: the pre-41ed380 crowd pipeline (det_model.track, no
    nested-box removal) that produced the archived results. Never used for
    headline numbers."""
    model = get_model(config.DETECTION_MODEL_PATH, "legacy_track")
    t0 = time.perf_counter()
    result = model.track(frame, classes=[0], persist=not new_sequence,
                         conf=config.DETECTION_CONF_MIN, verbose=False)[0]
    return result, (time.perf_counter() - t0) * 1000


def pose_infer(frame, new_sequence: bool, pose_model_path: str | None = None,
               imgsz: int | None = None, role: str = "track"):
    """Production fall-detection call (engine.py:744).

    new_sequence=True resets the tracker (fresh trackers are created because
    persist=False on that call); every later frame of the same sequence uses
    persist=True, exactly as production. Returns (result, model_latency_ms).
    """
    model = get_model(pose_model_path or config.POSE_MODEL_PATH, role)
    t0 = time.perf_counter()
    result = model.track(frame, classes=[0], persist=not new_sequence,
                         conf=config.DETECTION_CONF_MIN, verbose=False, **_size_kw(imgsz))[0]
    return result, (time.perf_counter() - t0) * 1000


def boxes_xyxy_conf(result):
    """[(x1, y1, x2, y2, conf), ...] from a Results object, or []."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.tolist()
    confs = boxes.conf.tolist() if boxes.conf is not None else [None] * len(xyxy)
    return [(b[0], b[1], b[2], b[3], c) for b, c in zip(xyxy, confs)]


def tracks_xyxy(result):
    """[(track_id, x1, y1, x2, y2), ...] for boxes that carry a tracker id."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0 or boxes.id is None:
        return []
    return [(int(i), *b) for i, b in zip(boxes.id.tolist(), boxes.xyxy.tolist())]


def tracker_info() -> dict:
    """Which tracker production (and therefore this suite) actually uses."""
    try:
        from ultralytics.cfg import DEFAULT_CFG
        from ultralytics.utils.checks import check_yaml
        name = DEFAULT_CFG.tracker
        with open(check_yaml(name), encoding="utf-8") as f:
            text = f.read()
        return {"tracker": name, "yaml": text,
                "note": "production calls pose_model.track() without a tracker= "
                        "argument, so the installed ultralytics default applies"}
    except Exception as e:  # noqa: BLE001
        return {"tracker": None, "error": str(e)}


__all__ = [
    "check_production_parity", "ProductionParityError", "get_model", "model_path",
    "warmup", "detect_persons", "detect_persons_legacy_tracked", "pose_infer",
    "boxes_xyxy_conf", "tracks_xyxy", "tracker_info",
    "classify_persons", "confirm_falls", "zone_densities", "drop_nested_boxes",
    "nested_duplicates", "TamperDetector", "config",
]
