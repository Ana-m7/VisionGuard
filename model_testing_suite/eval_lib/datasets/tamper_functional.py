"""Camera-tamper FUNCTIONAL CHARACTERISATION (replaces "synthetic_tamper").

This is not an accuracy measurement: there is no independent tamper dataset
(UHCTD is gated behind a signed agreement). Instead, the REAL production
``detector.engine.TamperDetector`` is driven, with the CURRENT config values
(TAMPER_COVERED_CONFIRM_FRAMES, TAMPER_FROZEN_CONFIRM_FRAMES = 1440, ...),
through a set of scenarios built on REAL footage (Mall dataset frames) and
documented manipulations. For each scenario the suite records:

  expected_by_design : what the detector's rules imply it SHOULD report
  observed           : what it actually reported, and on which frame
  tamper_present     : whether the scenario is genuinely tampering
  outcome            : tamper detected / tamper missed / false alarm / correctly quiet

So the table shows both whether the code does what its design says (a
functional check) and what that design means operationally: which tamper
types are caught, which are missed, and which legitimate scenes raise false
alarms. Audit issue #14.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .. import model_runner as mr

ASSUMED_CAMERA_FPS = 24  # config.py states the frozen window as "~60s at 24 fps"


def _mall_stream(frames_dir, n, start=1):
    for i in range(start, start + n):
        img = cv2.imread(os.path.join(frames_dir, f"seq_{(i - 1) % 2000 + 1:06d}.jpg"))
        if img is not None:
            yield img


def build_cases(frames_dir, seed):
    cfg = mr.config
    rng = np.random.default_rng(seed)
    frozen_n = cfg.TAMPER_FROZEN_CONFIRM_FRAMES + 20
    covered_n = cfg.TAMPER_COVERED_CONFIRM_FRAMES + 20
    base = cv2.imread(os.path.join(frames_dir, "seq_000001.jpg"))
    h, w = base.shape[:2]

    def black(n):
        for _ in range(n):
            yield np.full((h, w, 3), 3, np.uint8)

    def featureless(n):
        for _ in range(n):
            yield rng.integers(118, 126, (h, w, 3), dtype=np.uint8)

    def repeat(img, n):
        for _ in range(n):
            yield img

    def jpeg_replay(img, n):
        for k in range(n):
            q = 70 if k % 2 else 90
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
            yield cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def sensor_noise(img, n, sigma=2.0):
        for _ in range(n):
            noise = rng.normal(0, sigma, img.shape)
            yield np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def scaled(stream, factor):
        for img in stream:
            yield np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    def half_covered(stream):
        for img in stream:
            out = img.copy()
            out[:, : w // 2] = 0
            yield out

    def blurred(stream):
        for img in stream:
            yield cv2.GaussianBlur(img, (51, 51), 0)

    def redirected(n_before, n_after):
        yield from _mall_stream(frames_dir, n_before)
        mat = np.float32([[1, 0, w * 0.35], [0, 1, h * 0.25]])
        for img in _mall_stream(frames_dir, n_after, start=n_before + 1):
            yield cv2.warpAffine(img, mat, (w, h))

    def dimming(n_ramp, n_hold):
        for k, img in enumerate(_mall_stream(frames_dir, n_ramp + n_hold)):
            f = max(0.02, 1 - k / n_ramp)
            yield np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)

    def cover_uncover(n_cover, n_open):
        yield from black(n_cover)
        yield from _mall_stream(frames_dir, n_open)

    return [
        ("real moving footage (Mall)", "normal", False, frozen_n,
         lambda: _mall_stream(frames_dir, frozen_n)),
        ("lens fully covered (black)", "covered", True, covered_n, lambda: black(covered_n)),
        ("hand over lens, auto-exposure (bright, featureless)", "covered", True, covered_n,
         lambda: featureless(covered_n)),
        ("frozen feed: identical frame replayed", "frozen", True, frozen_n,
         lambda: repeat(base, frozen_n)),
        ("frozen feed replayed through JPEG re-encoding", "frozen", True, frozen_n,
         lambda: jpeg_replay(base, frozen_n)),
        ("legitimately static scene with sensor noise (sigma 2)", "frozen", False, frozen_n,
         lambda: sensor_noise(base, frozen_n)),
        ("legitimately dark scene (real footage at 8% brightness)", "covered", False, covered_n + 40,
         lambda: scaled(_mall_stream(frames_dir, covered_n + 40), 0.08)),
        ("half of lens covered (moving scene)", "normal", True, covered_n + 40,
         lambda: half_covered(_mall_stream(frames_dir, covered_n + 40))),
        ("defocused / smeared lens (Gaussian blur 51 px)", "covered if contrast < TAMPER_UNIFORM_STD else normal",
         True, covered_n + 40, lambda: blurred(_mall_stream(frames_dir, covered_n + 40))),
        ("camera re-pointed (view shifted 35% x, 25% y)", "normal", True, 90,
         lambda: redirected(30, 60)),
        ("gradual dimming to near black over 100 frames", "covered", True, 130,
         lambda: dimming(100, 30)),
        ("covered for 30 frames, then uncovered", "covered then cleared", True, 60,
         lambda: cover_uncover(30, 30)),
    ]


def run_case(stream_fn):
    det = mr.TamperDetector()
    first_type, first_frame, cleared_at, n = None, None, None, 0
    prev = None
    for i, frame in enumerate(stream_fn()):
        r = det.check(frame)
        n += 1
        if r is not None and first_type is None:
            first_type, first_frame = r, i
        if prev is not None and r is None and cleared_at is None:
            cleared_at = i
        prev = r
    return {"frames_fed": n, "first_detection": first_type, "first_detection_frame": first_frame,
            "cleared_at_frame": cleared_at, "final_state": prev or "normal"}


def _matches(expected, obs):
    detected = obs["first_detection"] or "normal"
    if expected == "covered then cleared":
        return detected == "covered" and obs["cleared_at_frame"] is not None
    if expected.startswith("covered if"):
        return True  # either outcome is consistent with the rule; reported, not judged
    return detected == expected


def run(args) -> dict:
    from . import mall
    cfg = mr.config
    result = {"name": "Camera tamper - functional characterisation", "status": None,
              "status_quality": "functional check (not an accuracy measurement)"}
    try:
        frames_dir, _, _ = mall.load(os.path.join(args.data_root, "mall"))
    except Exception as e:  # noqa: BLE001
        result["status"] = f"SKIPPED: real footage (Mall frames) unavailable ({e})"
        return result
    rows = []
    for name, expected, tamper_present, _n, fn in build_cases(frames_dir, args.seed):
        obs = run_case(fn)
        detected = obs["first_detection"] is not None
        outcome = ("tamper detected" if tamper_present and detected else
                   "tamper missed" if tamper_present else
                   "FALSE ALARM" if detected else "correctly quiet")
        ff = obs["first_detection_frame"]
        rows.append({"scenario": name, "tamper_present": tamper_present,
                     "expected_by_design": expected, **obs,
                     "seconds_to_detection_at_24fps": (ff + 1) / ASSUMED_CAMERA_FPS if ff is not None else None,
                     "matches_design": _matches(expected, obs), "outcome": outcome})
    result.update({
        "status": "OK",
        "config_used": {k: getattr(cfg, k) for k in dir(cfg) if k.startswith("TAMPER_")},
        "n_scenarios": len(rows),
        "all_match_design": all(r["matches_design"] for r in rows),
        "tamper_types_detected": [r["scenario"] for r in rows if r["outcome"] == "tamper detected"],
        "tamper_types_missed": [r["scenario"] for r in rows if r["outcome"] == "tamper missed"],
        "false_alarms_on_legitimate_scenes": [r["scenario"] for r in rows if r["outcome"] == "FALSE ALARM"],
        "scenarios": rows,
        "note": "Synthetic manipulations of real footage, not a tamper dataset. UHCTD (gated) "
                "would be needed for accuracy figures.",
    })
    return result
