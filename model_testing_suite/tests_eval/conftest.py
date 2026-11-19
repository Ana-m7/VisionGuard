import os
import sys

SUITE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SUITE_DIR not in sys.path:
    sys.path.insert(0, SUITE_DIR)

from eval_lib import paths  # noqa: E402,F401  (adds the VisionGuard root to sys.path)
