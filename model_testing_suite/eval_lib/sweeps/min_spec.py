"""Plain-language minimum camera specification, derived ONLY where the data
supports it (audit #5, #12, #13). Every line states its basis (n, CI) or says
"insufficient data" instead of producing a number from noise.
"""
from __future__ import annotations

from .resolution_sweep import minimum_width


def _fmt_prop(p):
    if not p or p.get("value") is None:
        return "n/a"
    return f"{p['value']:.2f} [{p['ci_low']:.2f}-{p['ci_high']:.2f}], n={p['n']}"


def derive(rate_result, resolution_result, size_result, args) -> dict:
    out = {"status": "OK", "relative_threshold": args.min_spec_threshold, "lines": []}
    L = out["lines"]

    # ---- inference rate / camera fps
    pts = (rate_result or {}).get("points") or []
    if pts:
        L.append("Camera frame rate: production analyses about one frame per inference cycle "
                 f"(measured here: {_cycle(rate_result)}), so a camera only needs a frame rate at "
                 "or above that rate; extra camera fps is never seen by the model.")
        enough = all(p["status_quality"] == "headline" for p in pts)
        prod = next((p for p in pts if p["rate"] == rate_result.get("production_rate")), None)
        if prod:
            L.append(f"Fall detection at production cadence ({prod['rate']}): recall "
                     f"{_fmt_prop(prod['recall'])}; ADL sequences with a false alarm "
                     f"{_fmt_prop(prod['adl_sequences_with_false_alarm'])}.")
        if enough:
            ordered = sorted([p for p in pts if p["rate_hz"]], key=lambda p: -p["rate_hz"])
            best = max(p["recall"]["value"] for p in pts)
            chosen = None
            for p in ordered:
                if p["recall"]["value"] >= args.min_spec_threshold * best:
                    chosen = p
                else:
                    break
            out["min_inference_rate_hz"] = chosen["rate_hz"] if chosen else None
            L.append(f"Minimum inference rate keeping fall recall >= "
                     f"{args.min_spec_threshold:.0%} of best ({best:.2f}): "
                     f"{chosen['rate'] if chosen else 'not met by any sub-native rate'}.")
        else:
            out["min_inference_rate_hz"] = None
            L.append("Minimum inference rate: INSUFFICIENT DATA (fewer than "
                     f"{args.headline_min_n} fall and ADL sequences); see the sweep table only.")

    # ---- resolution
    res = resolution_result or {}
    if res.get("crowdhuman_points"):
        w, best, why = minimum_width(res["crowdhuman_points"], "recall",
                                     args.resolution_recall_target, True, args.min_spec_threshold)
        out["min_width_crowdhuman"] = w
        L.append(f"Resolution (CrowdHuman person recall, target >= {args.resolution_recall_target}): "
                 + (f"minimum frame width ~{w}px" if why == "ok" and w else
                    "native resolution only" if why == "ok" else why) + ".")
    if res.get("mall_points"):
        w, best, why = minimum_width(res["mall_points"], "relative_mae",
                                     args.resolution_rel_mae_target, False, args.min_spec_threshold)
        out["min_width_mall"] = w
        L.append(f"Resolution (Mall counting, target relative MAE <= {args.resolution_rel_mae_target}): "
                 + (f"minimum frame width ~{w}px" if why == "ok" else why) + ".")

    # ---- person size
    for key, d in ((size_result or {}).get("per_dataset") or {}).items():
        if d.get("status") != "OK":
            continue
        out[f"min_person_height_{key}"] = d["minimum_height_bucket"]
        L.append(f"Person height ({key}, recall target >= {size_result['recall_target']}, "
                 f"visibility >= {size_result['min_visibility']}): "
                 + (f"people must be at least {d['minimum_height_bucket'].lstrip('>=').split('-')[0].rstrip('%')}% "
                    "of frame height"
                    if d["minimum_height_bucket"] else d["reason"]) + ".")
    return out


def _cycle(rate_result):
    c = (rate_result or {}).get("measured_production_cycle")
    if not c:
        return "not measured"
    return f"{c['mean_cycle_s']:.2f} s per cycle = {c['achievable_rate_hz']:.1f} Hz on this machine"
