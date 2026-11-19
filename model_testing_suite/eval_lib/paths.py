"""Filesystem anchors for the evaluation suite.

Every default path is resolved relative to THIS package, never to the current
working directory, so ``python model_testing_suite/test_models.py`` from the
repo root and ``python test_models.py`` from inside model_testing_suite/ read
and write exactly the same folders (audit issue #15).
"""
from __future__ import annotations

import os
import sys

EVAL_LIB_DIR = os.path.dirname(os.path.abspath(__file__))
SUITE_DIR = os.path.dirname(EVAL_LIB_DIR)
DEFAULT_DATA_ROOT = os.path.join(SUITE_DIR, "external_datasets")
DEFAULT_OUT_DIR = os.path.join(SUITE_DIR, "model_eval_results")
DEFAULT_QUICK_OUT_DIR = os.path.join(SUITE_DIR, "model_eval_results_quick")


def _find_project_root(start: str, max_levels: int = 6) -> str:
    """Walk upward until a directory containing config.py AND detector/ is
    found (the VisionGuard project root)."""
    current = os.path.abspath(start)
    for _ in range(max_levels):
        if (os.path.isfile(os.path.join(current, "config.py"))
                and os.path.isdir(os.path.join(current, "detector"))):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    raise RuntimeError(
        f"Could not find the VisionGuard project root (config.py + detector/) "
        f"within {max_levels} levels above {start}.")


PROJECT_ROOT = _find_project_root(SUITE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def resolve_under_suite(path: str) -> str:
    """Relative CLI paths are interpreted relative to the suite folder."""
    return path if os.path.isabs(path) else os.path.join(SUITE_DIR, path)
