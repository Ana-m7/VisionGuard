"""External-dataset accuracy + minimum-hardware-requirement evaluation suite
for VisionGuard.

Everything under this package is NEW code written for offline evaluation.
Nothing here is imported by, or modifies, the live application
(backend/*, detector/*, frontend/*, config.py). Several modules import
read-only pure functions from detector/engine.py and constants from
config.py so the evaluation exercises the *actual* production logic rather
than a re-implementation of it.
"""
