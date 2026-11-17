"""Derive a plain-language 'minimum viable CCTV specification' from the
three sweeps: the smallest FPS / lowest resolution / smallest person-size
at which the relevant metric stays at or above ``threshold`` (default 0.9,
i.e. 90%) of the best value achieved anywhere in that sweep.
"""
from __future__ import annotations


def _min_crossing(points, value_key, x_key, threshold, higher_is_better=True,
                   descending_x=False):
    """points sorted by x ascending (or pass descending_x=True if the list is
    already ordered from worst to best, e.g. resolution high->low)."""
    valid = [p for p in points if p.get(value_key) is not None]
    if not valid:
        return None, None
    best = max(p[value_key] for p in valid) if higher_is_better else min(p[value_key] for p in valid)
    cutoff = best * threshold if higher_is_better else best / threshold
    ordered = sorted(valid, key=lambda p: p[x_key], reverse=descending_x)
    for p in ordered:
        ok = (p[value_key] >= cutoff) if higher_is_better else (p[value_key] <= cutoff)
        if ok:
            return p[x_key], best
    return None, best


def derive(fps_result, resolution_result, person_size_result, threshold=0.9):
    out = {"threshold_used": threshold, "lines": []}

    # FPS -- crowd.
    crowd_pts = fps_result.get("crowd", [])
    if crowd_pts:
        min_fps_crowd, best_f1 = _min_crossing(crowd_pts, "f1", "target_fps", threshold)
        out["min_fps_crowd_detection"] = min_fps_crowd
        out["lines"].append(
            f"Minimum frame rate for reliable CROWD detection: ~{min_fps_crowd} fps "
            f"(F1 stays >= {threshold*100:.0f}% of the best observed F1={best_f1:.2f})"
            if min_fps_crowd else
            "Minimum frame rate for crowd detection: could not be determined "
            "(no sweep point met the threshold)")

    # FPS -- fall.
    fall_pts = fps_result.get("fall", [])
    if fall_pts:
        min_fps_fall, best_recall = _min_crossing(fall_pts, "recall", "target_fps", threshold)
        impossible = [p["target_fps"] for p in fall_pts
                     if p.get("clips_with_fewer_sampled_frames_than_FALL_CONFIRM_FRAMES", 0) > 0]
        out["min_fps_fall_detection"] = min_fps_fall
        out["lines"].append(
            f"Minimum frame rate for reliable FALL confirmation: ~{min_fps_fall} fps "
            f"(Recall stays >= {threshold*100:.0f}% of the best observed Recall={best_recall:.2f})"
            if min_fps_fall else
            "Minimum frame rate for fall confirmation: could not be determined")
        if impossible:
            out["lines"].append(
                f"Below {max(impossible)} fps, some fall clips no longer contain enough "
                f"sampled frames to reach FALL_CONFIRM_FRAMES consecutive detections under "
                f"the current config -- confirmation becomes physically impossible for "
                f"those events, independent of model accuracy.")

    # Resolution -- prefer Mall's count-accuracy-proxy signal: MOT20-01 is so
    # densely crowded that yolov8n is saturated by occlusion at EVERY
    # resolution (near-flat recall ~0.09-0.10 regardless of pixel size),
    # which gives a technically-computable but untrustworthy crossing point.
    # Mall (far less saturated, corridor-style scene) has an actual gradient.
    mall_pts = resolution_result.get("mall_points") or []
    mot20_pts = resolution_result.get("mot20_points") or resolution_result.get("points") or []
    chosen_source = None
    chosen = None
    best_val = None
    if mall_pts and any(p.get("accuracy_proxy") is not None for p in mall_pts):
        pts_for_area = [dict(p, area=p["width"] * p["height"]) for p in mall_pts]
        min_area, best_val = _min_crossing(pts_for_area, "accuracy_proxy", "area", threshold)
        chosen = next((p for p in pts_for_area if p["area"] == min_area), None)
        chosen_source = "Mall Dataset (count-accuracy proxy = 1 - MAE/mean_count)"
    if chosen is None and mot20_pts:
        pts_for_area = [dict(p, area=p["width"] * p["height"]) for p in mot20_pts]
        min_area, best_val = _min_crossing(pts_for_area, "recall", "area", threshold)
        chosen = next((p for p in pts_for_area if p["area"] == min_area), None)
        chosen_source = ("MOT20-01 (Recall -- NOTE: this scene is so densely "
                         "crowded the signal is nearly flat across all "
                         "resolutions; treat this number with low confidence)")
    out["min_resolution"] = chosen["label"] if chosen else None
    out["min_resolution_source"] = chosen_source
    out["lines"].append(
        f"Minimum resolution: {chosen['label']} ({chosen['width']}x{chosen['height']}) "
        f"-- derived from {chosen_source}, metric stays >= {threshold*100:.0f}% "
        f"of its best observed value ({best_val:.2f})"
        if chosen else
        "Minimum resolution: could not be determined (no resolution sweep data)")
    if mot20_pts:
        mot20_recalls = [p["recall"] for p in mot20_pts if p.get("recall") is not None]
        if mot20_recalls and (max(mot20_recalls) - min(mot20_recalls)) < 0.05:
            out["lines"].append(
                "Note: on MOT20-01 (extremely dense crowd scene), detection "
                f"recall barely varies across resolutions ({min(mot20_recalls):.2f}-"
                f"{max(mot20_recalls):.2f}) -- in that scene, crowd DENSITY/occlusion "
                "is the bottleneck, not camera resolution.")

    # Person-to-frame size.
    size_pts = person_size_result.get("points", [])
    if size_pts:
        order = {"0-2%": 0, "2-4%": 1, "4-8%": 2, "8-15%": 3, ">15%": 4}
        pts_sorted = sorted([p for p in size_pts if p.get("recall") is not None],
                            key=lambda p: order.get(p["bucket"], 99))
        if pts_sorted:
            best_recall_size = max(p["recall"] for p in pts_sorted)
            cutoff = best_recall_size * threshold
            chosen = next((p for p in pts_sorted if p["recall"] >= cutoff), None)
            out["min_person_height_pct"] = chosen["bucket"] if chosen else None
            out["lines"].append(
                f"Minimum person height in frame: {chosen['bucket']} of frame height "
                f"(~{chosen['n_instances']} instances sampled at this size; Recall "
                f">= {threshold*100:.0f}% of best observed Recall={best_recall_size:.2f})"
                if chosen else
                "Minimum person-to-frame size: could not be determined")

    return out
