"""Logging configuration for VisionGuard.

Call setup_logging() once at process startup from backend/app.py's
lifespan handler. After that every module gets its own named logger:

    import logging
    logger = logging.getLogger("visionguard.engine")

All loggers under the visionguard namespace inherit the same two handlers:
  Console  — same terminal output as before, now with level and timestamp.
  File     — rotating file in LOG_DIR; rolls at LOG_MAX_BYTES and keeps
             LOG_BACKUP_COUNT backups so the disk never fills up.
"""
import logging
import logging.handlers
import os
import sys

import config


def setup_logging() -> logging.Logger:
    os.makedirs(config.LOG_DIR, exist_ok=True)

    logger = logging.getLogger("visionguard")
    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))

    if logger.handlers:
        return logger

    # Our handlers are the only output. Without this, any library that adds
    # a root handler (logging.basicConfig) would print every line twice.
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)-30s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    log_path = os.path.join(config.LOG_DIR, "visionguard.log")
    fh = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("Logging initialised — file: %s", log_path)
    return logger


# ── Alert lines ──────────────────────────────────────────────────
# Every step of an alert's life is one self-describing line:
#
#   ACTION    TYPE             CAMERA           #id — detail
#   ALERT     CROWD GATHERING  CAM-04 (Webcam) #57 — 2 persons in view, peak zone 2
#   ACK       CROWD GATHERING  CAM-04 (Webcam) #57 — by admin
#   CLEARED   CROWD GATHERING  CAM-04 (Webcam) #58 — 1 person in view
#
# The type names match the dashboard, so `grep "FALL DETECTED"` or
# `grep ESCALATED` pulls a whole incident trail, and #id is the alert's
# database row — the same id as its dashboard entry and its CSV export row.
ALERT_LABELS = {
    "gathering": "CROWD GATHERING",
    "fall": "FALL DETECTED",
    "tamper": "CAMERA TAMPERING",
    "offline": "CAMERA OFFLINE",
}


def camera_name(cam_id):
    label = config.CAMERA_LABELS.get(cam_id)
    return f"CAM-{cam_id:02d} ({label})" if label else f"CAM-{cam_id:02d}"


def log_alert(log, level, action, kind, cam_id, detail="", row_id=None):
    """Write one alert line. action: ALERT | CLEARED | ACK | ESCALATED."""
    log.log(level, "%-9s %-16s %s%s%s", action, ALERT_LABELS[kind],
            camera_name(cam_id), f" #{row_id}" if row_id else "",
            f" — {detail}" if detail else "")
