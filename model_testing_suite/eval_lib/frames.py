"""Frame-file listing in TEMPORAL order.

Le2i frames are named ``86.jpg ... 101.jpg``; a plain ``sorted()`` puts
``100.jpg`` before ``86.jpg`` and silently scrambles the clip (audit issue #6).
Every module that walks a frame folder must use :func:`list_frames`.
"""
from __future__ import annotations

import os
import re

_TRAILING_INT = re.compile(r"(\d+)(?!.*\d)")


def numeric_frame_key(fname: str):
    """Sort key: the LAST integer in the file stem (``adl-01-cam0-rgb-012.png``
    -> 12, ``86.jpg`` -> 86, ``seq_000007.jpg`` -> 7). Files without a number
    sort after numbered ones, by name."""
    stem = os.path.splitext(os.path.basename(fname))[0]
    m = _TRAILING_INT.search(stem)
    return (0, int(m.group(1)), stem) if m else (1, 0, stem)


def frame_number(fname: str) -> int:
    key = numeric_frame_key(fname)
    if key[0] != 0:
        raise ValueError(f"no frame number in {fname!r}")
    return key[1]


def list_frames(directory: str, exts=(".jpg", ".jpeg", ".png")) -> list[str]:
    """File names (not paths) of the image frames in ``directory``, ordered by
    frame number."""
    if not os.path.isdir(directory):
        return []
    files = [f for f in os.listdir(directory) if f.lower().endswith(tuple(exts))]
    return sorted(files, key=numeric_frame_key)
