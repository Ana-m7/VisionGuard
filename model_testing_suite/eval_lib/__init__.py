"""External-dataset evaluation suite for VisionGuard.

Everything under this package is evaluation code. Nothing here is imported
by, or modifies, the live application (backend/, detector/, frontend/,
config.py). Production logic is IMPORTED from detector/engine.py and every
threshold is read from config.py, so the evaluation exercises the actual
production code rather than a copy of it.
"""
from . import paths  # noqa: F401  (puts the VisionGuard project root on sys.path first)
