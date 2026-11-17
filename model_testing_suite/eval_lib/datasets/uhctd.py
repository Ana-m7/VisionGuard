"""UHCTD (University of Houston Camera Tampering Dataset) — gated behind a
signed Dataset Agreement (confirmed live: pmantini.github.io/.../uhctd/
links to a "Dataset Agreement" step, not an instant download). We do not
attempt to bypass that; we report SKIPPED with manual instructions.

The synthetic covered/frozen/normal tamper test (same technique as
tests/test_health_tamper.py, with real known ground truth by construction)
ALWAYS runs regardless, in ``synthetic_tamper.run()`` — see that module for
the guaranteed-available fallback this dataset's absence doesn't block.
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
            "prepare the data. The synthetic covered/frozen tamper test "
            "(see 'Synthetic Tamper Test' in this report) runs regardless "
            "and exercises the same TamperDetector classes/thresholds."
        ),
        "manual_steps_url": UHCTD_PAGE,
        "devkit_url": DEVKIT,
    }
