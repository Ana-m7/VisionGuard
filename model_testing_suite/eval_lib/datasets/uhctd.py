"""UHCTD (University of Houston Camera Tampering Dataset) — gated behind a
signed Dataset Agreement (confirmed live: pmantini.github.io/.../uhctd/
links to a "Dataset Agreement" step, not an instant download). We do not
attempt to bypass that; we report SKIPPED with manual instructions.

The camera-tamper functional characterisation (eval_lib/datasets/
tamper_functional.py) drives the real TamperDetector through scenarios built
on real footage; it is a functional check, not a substitute for this dataset.
"""
from __future__ import annotations

UHCTD_PAGE = "https://pmantini.github.io/profile/projects/uhctd/"
DEVKIT = "https://github.com/QuantitativeImagingLaboratory/ctd-devkit"


def run(args) -> dict:
    return {
        "name": "UHCTD",
        "status": (
            "SKIPPED: requires a signed Dataset Agreement (not an anonymous/"
            "instant download) -- verified live by visiting the project page. "
            f"Manual steps: open {UHCTD_PAGE}, follow the 'Dataset Agreement' "
            f"link, and once granted access use the devkit at {DEVKIT} to "
            "prepare the data. Until then, tamper results come only from the "
            "functional characterisation (tamper_functional), which is not an accuracy measure."
        ),
        "manual_steps_url": UHCTD_PAGE,
        "devkit_url": DEVKIT,
    }
