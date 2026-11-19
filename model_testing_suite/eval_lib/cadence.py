"""Production-cadence frame sampling.

Production does not look at every camera frame. Capture and inference are
decoupled threads (detector/engine.py); the inference thread always takes the
NEWEST frame, at whatever rate detection + pose inference allows (config.py
documents ~2.8 cycles/s on the target i5-1135G7 CPU, with
``DETECT_INTERVAL`` as a floor). A recorded clip is therefore replayed here by
TIME: at an effective inference rate r, the model sees the frame on screen at
t = 0, 1/r, 2/r, ...

``rate_hz=None`` (or >= native fps) means every frame.
"""
from __future__ import annotations

import math

import config

DOCUMENTED_PRODUCTION_HZ = 2.8   # config.py comment: "about 2.8 detection cycles/sec"


def max_rate_hz() -> float:
    """Upper bound production can never exceed: one cycle per DETECT_INTERVAL."""
    return 1.0 / config.DETECT_INTERVAL if config.DETECT_INTERVAL > 0 else float("inf")


def sample_indices(n_frames: int, native_fps: float, rate_hz: float | None) -> list[int]:
    """0-based indices of the frames inference would process."""
    if n_frames <= 0:
        return []
    if rate_hz is None or rate_hz >= native_fps:
        return list(range(n_frames))
    step = native_fps / rate_hz
    out, k = [], 0
    while True:
        i = int(math.floor(k * step + 1e-9))
        if i >= n_frames:
            break
        if not out or out[-1] != i:
            out.append(i)
        k += 1
    return out


def rate_label(rate_hz: float | None) -> str:
    return "every frame" if rate_hz is None else f"{rate_hz:g} Hz"
