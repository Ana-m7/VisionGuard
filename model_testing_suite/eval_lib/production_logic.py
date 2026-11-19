"""Alert state machines that exist ONLY inline inside
``detector/engine.py::inference_thread`` and therefore cannot be imported.

They are mirrored here line-for-line, reading EVERY constant from ``config``
at call time (never a copied number). ``model_runner.check_production_parity``
aborts the evaluation if the engine source no longer contains the statements
these classes mirror, so the two cannot drift apart silently.

Mirrored source (detector/engine.py at commit 41ed380):
- CrowdAlertMachine : engine.py lines 701, 705, 758-763, 794 and 829
    crowd_streak = crowd_streak + 1 if person_count >= CROWD_MIN_PERSONS else 0
    if crowd_state: crowd_alert = person_count >= CROWD_MIN_PERSONS - CROWD_CLEAR_MARGIN
    else:           crowd_alert = crowd_streak >= CROWD_CONFIRM_CYCLES
    (crowd_state starts as None and is set to crowd_alert on every change)
- FallAlertMachine  : engine.py lines 702-704, 765-774, 831 and 865
    confirmed = confirm_falls(persons, fall_streaks)       # imported, not copied
    if confirmed: last_fall_seen = now
    fall_alert = bool(confirmed) or (fall_state and now - last_fall_seen < FALL_CLEAR_SECONDS)

One deliberate difference: production uses wall-clock ``time.time()``; here
the caller passes the frame's VIDEO time, so a recorded clip is replayed in
its own timeline regardless of how fast this machine runs inference.
"""
from __future__ import annotations

import config
from detector.engine import confirm_falls


class CrowdAlertMachine:
    """Per-camera crowd alert state (engine.py inference_thread)."""

    def __init__(self):
        self.state = None     # None = neutral until first detection
        self.streak = 0

    def step(self, person_count: int) -> dict:
        self.streak = self.streak + 1 if person_count >= config.CROWD_MIN_PERSONS else 0
        if self.state:
            alert = person_count >= (config.CROWD_MIN_PERSONS - config.CROWD_CLEAR_MARGIN)
        else:
            alert = self.streak >= config.CROWD_CONFIRM_CYCLES
        raised = bool(alert) and not self.state
        cleared = (not alert) and bool(self.state)
        if alert != self.state:
            self.state = alert
        return {"alert": bool(alert), "raised": raised, "cleared": cleared}


def run_crowd_machine(counts) -> list[bool]:
    m = CrowdAlertMachine()
    return [m.step(c)["alert"] for c in counts]


class FallAlertMachine:
    """Per-camera fall alert state (engine.py inference_thread)."""

    def __init__(self):
        self.streaks = {}        # tracker ID -> consecutive fallen detections
        self.state = False
        self.last_fall_seen = 0.0

    def reset_streaks(self):
        """What production does when a video file loops (engine.py:727)."""
        self.streaks = {}

    def step(self, persons, now: float) -> dict:
        confirmed = confirm_falls(persons, self.streaks)
        if confirmed:
            self.last_fall_seen = now
        alert = bool(confirmed) or (
            self.state and now - self.last_fall_seen < config.FALL_CLEAR_SECONDS)
        raised = alert and not self.state
        cleared = (not alert) and self.state
        self.state = alert
        return {"alert": alert, "raised": raised, "cleared": cleared,
                "confirmed": confirmed}


def red_cells(counts: dict) -> set:
    """Grid cells the dashboard highlights red (engine.py draw_grid:352)."""
    return {cell for cell, c in counts.items() if c >= config.CROWD_THRESHOLD}
