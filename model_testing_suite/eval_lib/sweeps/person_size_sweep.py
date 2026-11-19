"""Person-size sweep (audit #13): detection recall by person height as a
percentage of frame height.

- Instances come from the dataset runs themselves (official CrowdHuman vbox
  GT and MOT20-01 pedestrian GT, production detection at native resolution),
  so no extra inference is needed and nothing is double-counted.
- Heavily occluded people are excluded (visibility < 0.5: CrowdHuman
  vbox/fbox area ratio, MOT20 visibility field), so the sweep measures SIZE,
  not occlusion. The number dropped is reported.
- Datasets are reported SEPARATELY (pooling made size a proxy for dataset).
- A bucket with fewer than HEADLINE_MIN_N instances is "insufficient data",
  and the minimum size uses an ABSOLUTE recall target, not 90% of a low best.
"""
from __future__ import annotations

from .. import metrics as m

BUCKETS = [(0, 2), (2, 4), (4, 8), (8, 15), (15, 25), (25, 1000)]
MIN_VISIBILITY = 0.5


def label(lo, hi):
    return f">={lo}%" if hi >= 1000 else f"{lo}-{hi}%"


def bucketize(instances, min_n):
    kept = [x for x in instances if x["visibility"] >= MIN_VISIBILITY]
    pts = []
    for lo, hi in BUCKETS:
        hits = [x["hit"] for x in kept if lo <= x["h_pct"] < hi]
        prop = m.proportion(sum(hits), len(hits))
        pts.append({"bucket": label(lo, hi), "lo": lo, "hi": hi, "recall": prop,
                    "sufficient_data": len(hits) >= min_n})
    return pts, len(instances) - len(kept)


def minimum_bucket(points, target):
    """Smallest bucket such that it and every larger bucket with sufficient
    data reach the absolute recall target."""
    usable = [p for p in points if p["sufficient_data"]]
    if not usable:
        return None, "no bucket has enough instances"
    chosen = None
    for p in reversed(usable):
        if p["recall"]["value"] >= target:
            chosen = p
        else:
            break
    if chosen is None:
        return None, f"recall target {target} not reached even for the largest people"
    return chosen["bucket"], "ok"


def run(dataset_results: dict, min_n: int, target: float) -> dict:
    out = {"status": "OK", "min_visibility": MIN_VISIBILITY, "recall_target": target,
           "per_dataset": {}}
    for key in ("crowdhuman", "mot20"):
        inst = (dataset_results.get(key) or {}).get("_size_instances")
        if not inst:
            out["per_dataset"][key] = {"status": "SKIPPED: dataset result unavailable"}
            continue
        pts, dropped = bucketize(inst, min_n)
        mb, reason = minimum_bucket(pts, target)
        out["per_dataset"][key] = {"status": "OK", "n_instances": len(inst),
                                   "dropped_low_visibility": dropped, "points": pts,
                                   "minimum_height_bucket": mb, "reason": reason}
    if all(v.get("status") != "OK" for v in out["per_dataset"].values()):
        out["status"] = "SKIPPED: no instances available"
    return out
