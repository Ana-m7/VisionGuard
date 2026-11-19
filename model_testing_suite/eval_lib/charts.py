"""Chart generation (matplotlib, Agg). Every chart is drawn ONLY from the
values already written to summary.json (the ``all_results`` dict), so a
chart can never show a number the summary does not contain.

Conventions (audit #21): one y-axis per chart (no twin axes), zero-based
value axes, 95% CI error bars wherever the metric has one, x ordered by the
quantity it represents, and buckets/points with too little data are marked.
Palette: validated categorical slots (blue, orange, aqua, yellow), direct
value labels on bars (two slots sit under 3:1 contrast on the surface).
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "lines.linewidth": 2,
})


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _err(p):
    """(value, lower_err, upper_err) from a {'value','ci_low','ci_high'} dict."""
    if not p or p.get("value") is None:
        return None, 0, 0
    v = p["value"]
    lo = p.get("ci_low") if p.get("ci_low") is not None else v
    hi = p.get("ci_high") if p.get("ci_high") is not None else v
    return v, max(0, v - lo), max(0, hi - v)


def _bars(ax, groups, series_names, getter, ylim=(0, 1.05), fmt="{:.2f}"):
    import numpy as np
    n = len(series_names)
    width = 0.8 / n
    x = np.arange(len(groups))
    for si, s in enumerate(series_names):
        vals, lo, hi = [], [], []
        for g in groups:
            v, el, eh = _err(getter(g, s))
            vals.append(v if v is not None else 0)
            lo.append(el)
            hi.append(eh)
        xs = x + (si - (n - 1) / 2) * width
        ax.bar(xs, vals, width * 0.92, color=SERIES[si % len(SERIES)], label=s, zorder=2)
        ax.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=2, zorder=3)
        for xi, v, h in zip(xs, vals, hi):
            ax.text(xi, v + h + 0.01 * (ylim[1] or 1), fmt.format(v), ha="center", va="bottom",
                    fontsize=7, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    if ylim:
        ax.set_ylim(*ylim)
    if n > 1:
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1),
                  ncol=n)


# ------------------------------------------------------------------ charts
def detection_prf1(results, out_dir):
    rows = []
    for key, label in (("crowdhuman", "CrowdHuman (official val)"), ("mot20", "MOT20-01"),
                       ("caucafall", "CAUCAFall (pilot)")):
        d = (results.get(key) or {}).get("detection_production")
        if d:
            rows.append((label, d))
    if not rows:
        return None
    data = dict(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    _bars(ax, list(data), ["Precision", "Recall", "F1"],
          lambda g, s: data[g][s.lower()])
    ax.set_title("Person detection, production pipeline (IoU 0.5, 95% CI)")
    ax.set_ylabel("Score")
    return _save(fig, out_dir, "detection_prf1.png")


def counting_errors(results, out_dir):
    rows = []
    for key, label, field in (("mall", "Mall\n({n} frames)", "count_errors"),
                              ("mot20", "MOT20-01\n({n} frames)", "count_errors"),
                              ("crowdhuman", "CrowdHuman\n({n} images)", "count_errors_per_image")):
        ce = (results.get(key) or {}).get(field)
        if ce and ce.get("n"):
            rows.append((label.format(n=ce["n"]), ce))
    if not rows:
        return None
    data = dict(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ymax = max(_err(c["mae"])[0] + _err(c["mae"])[2] for c in data.values()) * 1.2
    _bars(axes[0], list(data), ["MAE"], lambda g, s: data[g]["mae"], ylim=(0, ymax), fmt="{:.1f}")
    axes[0].set_title("Count MAE (people per frame, 95% CI)")
    _bars(axes[1], list(data), ["Relative MAE"], lambda g, s: data[g]["relative_mae"],
          ylim=(0, 1.05))
    axes[1].set_title("Relative MAE (MAE / mean true count)")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    return _save(fig, out_dir, "counting_errors.png")


def count_over_time(pred, gt, title, out_dir, name):
    if not pred:
        return None
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = range(1, len(pred) + 1)
    ax.plot(x, gt, color=SERIES[0], label="Ground truth")
    ax.plot(x, pred, color=SERIES[1], label="Production (yolov8n + nested-box removal)")
    ax.set_ylim(0, max(max(gt), max(pred)) * 1.1)
    ax.set_xlabel("Frame")
    ax.set_ylabel("People in frame")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, out_dir, name)


def mall_alert_timeline(results, out_dir):
    a = (results.get("mall") or {}).get("alert_threshold_crossing_crop")
    if not a:
        return None
    import config
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    x = range(1, len(a["gt_counts"]) + 1)
    axes[0].plot(x, a["gt_counts"], color=SERIES[0], label="Ground-truth count", linewidth=1.5)
    axes[0].plot(x, a["pred_counts"], color=SERIES[1], label="Predicted count", linewidth=1.5)
    axes[0].axhline(config.CROWD_MIN_PERSONS, color=INK2, linestyle="--", linewidth=1)
    axes[0].text(len(a["gt_counts"]), config.CROWD_MIN_PERSONS, " CROWD_MIN_PERSONS",
                 va="bottom", ha="right", fontsize=7, color=INK2)
    axes[0].set_ylim(0, max(max(a["gt_counts"]), max(a["pred_counts"])) * 1.15 + 1)
    axes[0].set_ylabel("People in crop")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_title(f"Mall {a['crop_name']}: crowd alert state, production logic on GT vs predictions")
    axes[1].fill_between(x, 0.55, [0.55 + 0.4 * v for v in a["gt_alerts"]], step="mid",
                         color=SERIES[0], linewidth=0)
    axes[1].fill_between(x, 0.05, [0.05 + 0.4 * v for v in a["pred_alerts"]], step="mid",
                         color=SERIES[1], linewidth=0)
    axes[1].set_yticks([0.25, 0.75])
    axes[1].set_yticklabels(["alert (pred)", "alert (GT)"])
    axes[1].set_ylim(0, 1)
    axes[1].grid(False)
    axes[1].set_xlabel("Frame (~ one detection cycle)")
    return _save(fig, out_dir, "mall_alert_timeline.png")


def tracking(results, out_dir):
    t = (results.get("mot20") or {}).get("tracking_pose_model")
    if not t or t.get("mota") is None:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 4))
    names = ["MOTA", "IDF1", "MOTP (mean IoU)"]
    vals = [t["mota"], t["idf1"] or 0, t["motp_mean_iou"] or 0]
    lo = min(0, min(vals)) * 1.1
    ax.bar(names, vals, color=SERIES[0], width=0.5, zorder=2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_ylim(lo, 1.05)
    ax.set_title(f"MOT20-01 tracking (pose model tracker)\nIDSW={t['idsw']}  Frag={t['frag']}  "
                 f"GT ids={t['n_gt_ids']}")
    return _save(fig, out_dir, "mot20_tracking.png")


def le2i_per_class(results, out_dir):
    le = results.get("le2i") or {}
    res = le.get("results")
    if not res:
        return None
    modes = [("every_frame", "Every frame"), ("production_cadence", "Production cadence")]
    classes = list(res["every_frame"]["per_class_alert_rate"].keys())
    short = [c.split(" (")[0] for c in classes]
    fig, ax = plt.subplots(figsize=(8, 4))
    _bars(ax, short, [m[1] for m in modes],
          lambda g, s: res[dict((b, a) for a, b in modes)[s]]["per_class_alert_rate"][classes[short.index(g)]])
    ax.set_ylabel("Fraction of clips with a fall alert")
    ax.set_title("Le2i: fall-alert rate per clip class (95% Wilson CI)\n"
                 "Fall should fire; Stand and Blank should not; Lie = posture only")
    return _save(fig, out_dir, "le2i_per_class.png")


def rate_sweep(sweep, out_dir):
    pts = (sweep or {}).get("points")
    if not pts:
        return None
    pts = sorted(pts, key=lambda p: p["rate_hz"] or 1e9)
    labels = [p["rate"] for p in pts]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in ((axes[0], "recall", "Fall recall"),
                           (axes[1], "adl_sequences_with_false_alarm",
                            "Daily-activity sequences with a false alarm")):
        v = [_err(p[key]) for p in pts]
        ax.errorbar(range(len(pts)), [a[0] or 0 for a in v], yerr=[[a[1] for a in v], [a[2] for a in v]],
                    color=SERIES[0] if key == "recall" else SERIES[1], marker="o", markersize=6,
                    capsize=3)
        for i, a in enumerate(v):
            ax.text(i, (a[0] or 0) + a[2] + 0.02, f"{a[0]:.2f}" if a[0] is not None else "n/a",
                    ha="center", fontsize=7)
        ax.set_xticks(range(len(pts)))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Effective inference rate")
        ax.set_title(f"{title} (95% CI)")
    n = pts[0]
    fig.suptitle(f"UR Fall, replayed at each inference rate (n = {n['n_fall_sequences']} falls, "
                 f"{n['n_adl_sequences']} ADL)", fontsize=10)
    return _save(fig, out_dir, "inference_rate_sweep.png")


def fall_latency(results, out_dir):
    uf = (results.get("ur_fall") or {}).get("by_rate")
    if not uf:
        return None
    items = sorted(uf.items(), key=lambda kv: kv[1]["rate_hz"] or 1e9)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (label, r) in enumerate(items):
        lat = [s["latency_s_from_fall_onset"] for s in r["per_sequence"]
               if s.get("latency_s_from_fall_onset") is not None]
        ax.scatter([i] * len(lat), lat, color=SERIES[0], s=24, zorder=3, alpha=0.8)
        summ = r["latency_s_from_fall_onset"]
        if summ.get("value") is not None:
            ax.hlines(summ["median"], i - 0.25, i + 0.25, color=SERIES[1], linewidth=2)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([k for k, _ in items])
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel("Effective inference rate")
    ax.set_ylabel("Seconds from fall onset to alert")
    ax.set_title("UR Fall: alert latency from true fall onset (dots = sequences, bar = median)")
    return _save(fig, out_dir, "fall_latency.png")


def resolution(sweep, out_dir):
    ch, ma = (sweep or {}).get("crowdhuman_points"), (sweep or {}).get("mall_points")
    if not ch and not ma:
        return None
    panels = [p for p in ((ch, "recall", "CrowdHuman person recall (higher is better)"),
                          (ma, "relative_mae", "Mall relative count error (lower is better)")) if p[0]]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4))
    axes = axes if len(panels) > 1 else [axes]
    for ax, (pts, key, title) in zip(axes, panels):
        pts = sorted(pts, key=lambda p: -(p["width"] or 1e9))
        v = [_err(p[key]) for p in pts]
        ax.errorbar(range(len(pts)), [a[0] for a in v], yerr=[[a[1] for a in v], [a[2] for a in v]],
                    color=SERIES[0], marker="o", markersize=6, capsize=3)
        ax.set_xticks(range(len(pts)))
        ax.set_xticklabels([p["label"] for p in pts], rotation=20, ha="right")
        ax.set_ylim(0, max(1.05, max(a[0] + a[2] for a in v) * 1.1))
        ax.set_title(title)
        ax.set_xlabel("Frame width fed to the model (aspect ratio kept)")
    fig.suptitle("Resolution sweep (model letterboxes to 640 px: wider frames add no detail)",
                 fontsize=10)
    return _save(fig, out_dir, "resolution_sweep.png")


def person_size(sweep, out_dir):
    per = (sweep or {}).get("per_dataset") or {}
    ok = {k: v for k, v in per.items() if v.get("status") == "OK"}
    if not ok:
        return None
    buckets = [p["bucket"] for p in next(iter(ok.values()))["points"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    import numpy as np
    width = 0.8 / len(ok)
    x = np.arange(len(buckets))
    for si, (k, d) in enumerate(ok.items()):
        xs = x + (si - (len(ok) - 1) / 2) * width
        for xi, p in zip(xs, d["points"]):
            v, lo, hi = _err(p["recall"])
            n = p["recall"]["n"]
            if v is None:
                ax.text(xi, 0.02, "n=0", ha="center", fontsize=7, color=INK2, rotation=90)
                continue
            ax.bar(xi, v, width * 0.92, color=SERIES[si], hatch=None if p["sufficient_data"] else "//",
                   edgecolor=SURFACE, zorder=2, label=k if xi == xs[0] else None)
            ax.errorbar(xi, v, yerr=[[lo], [hi]], ecolor=INK2, capsize=2, elinewidth=1, zorder=3)
            ax.text(xi, v + hi + 0.02, f"n={n}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Person height as % of frame height (visibility >= 0.5)")
    ax.set_ylabel("Detection recall")
    ax.set_title("Person-size sweep, per dataset (hatched = fewer than the minimum n)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[i]) for i in range(len(ok))]
    ax.legend(handles, list(ok), frameon=False, fontsize=8)
    return _save(fig, out_dir, "person_size_sweep.png")


def tamper_table(results, out_dir):
    t = (results.get("tamper_functional") or {}).get("scenarios")
    if not t:
        return None
    rows = [[r["scenario"], "yes" if r["tamper_present"] else "no", r["expected_by_design"],
             r["first_detection"] or "normal",
             "" if r["first_detection_frame"] is None else str(r["first_detection_frame"] + 1),
             r["outcome"]] for r in t]
    fig, ax = plt.subplots(figsize=(13, 0.28 * len(rows) + 0.5))
    ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=["Scenario", "Tamper?", "Expected by design", "Observed",
                                             "Frames to alert", "Outcome"],
                   loc="upper center", cellLoc="left", colLoc="left")
    tab.auto_set_font_size(False)
    tab.set_fontsize(7.5)
    tab.auto_set_column_width(list(range(6)))
    tab.scale(1, 1.3)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_text_props(weight="bold")
    ax.set_title("Tamper detector functional characterisation (real TamperDetector, current config)")
    return _save(fig, out_dir, "tamper_characterisation.png")


def pr_curve(results, out_dir):
    ch = results.get("crowdhuman") or {}
    pr = ch.get("pr_curve")
    if not pr or not pr.get("precision"):
        return None
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(pr["recall"], pr["precision"], color=SERIES[0])
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"CrowdHuman val, conf >= 0.001\nAP50 (COCO 101-pt) = {ch['ap50_coco101']['value']:.3f}")
    return _save(fig, out_dir, "crowdhuman_pr_curve.png")


def make_all(all_results, chart_dir):
    ds = all_results.get("datasets", {})
    sw = all_results.get("sweeps", {})
    out = [
        detection_prf1(ds, chart_dir), counting_errors(ds, chart_dir), pr_curve(ds, chart_dir),
        count_over_time((ds.get("mall") or {}).get("pred_counts"), (ds.get("mall") or {}).get("gt_counts"),
                        "Mall: predicted vs true people per frame", chart_dir, "mall_count_over_time.png"),
        count_over_time((ds.get("mot20") or {}).get("pred_counts"), (ds.get("mot20") or {}).get("gt_counts"),
                        "MOT20-01: predicted vs true people per frame", chart_dir, "mot20_count_over_time.png"),
        mall_alert_timeline(ds, chart_dir), tracking(ds, chart_dir), le2i_per_class(ds, chart_dir),
        rate_sweep(sw.get("inference_rate"), chart_dir), fall_latency(ds, chart_dir),
        resolution(sw.get("resolution"), chart_dir), person_size(sw.get("person_size"), chart_dir),
        tamper_table(ds, chart_dir), whatif(ds, chart_dir),
    ]
    return [p for p in out if p]


def whatif(results, out_dir):
    """Small multiples (shared config order, no twin axes): CrowdHuman recall,
    Mall relative count error, and model time per frame. Production is the
    orange bar; every other bar is a NOT-production what-if."""
    w = results.get("whatif_detection") or {}
    ch = ((w.get("crowdhuman") or {}).get("configs")) or {}
    ml = ((w.get("mall") or {}).get("configs")) or {}
    keys = list(ch) or list(ml)
    if not keys:
        return None
    labels = [(ch.get(k) or ml.get(k))["label"].replace("production: ", "PRODUCTION\n") for k in keys]
    colors = [SERIES[1] if k == "production" else SERIES[0] for k in keys]
    panels = [
        ("CrowdHuman recall\n(higher is better)",
         [((ch.get(k) or {}).get("detection") or {}).get("recall") for k in keys], (0, 1.0), "{:.2f}"),
        ("Mall relative count error\n(lower is better)",
         [((ml.get(k) or {}).get("count_errors") or {}).get("relative_mae") for k in keys], (0, 1.0), "{:.2f}"),
        ("Model time per frame, ms\n(CrowdHuman, this CPU; lower is faster)",
         [(ch.get(k) or {}).get("model_ms_per_frame") for k in keys], None, "{:.0f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.45 * len(keys) + 1.4), sharey=True)
    ys = list(range(len(keys)))[::-1]
    for ax, (title, vals, lim, fmt) in zip(axes, panels):
        v = [_err(p) for p in vals]
        xs = [a[0] or 0 for a in v]
        ax.barh(ys, xs, 0.7, color=colors, zorder=2)
        ax.errorbar(xs, ys, xerr=[[a[1] for a in v], [a[2] for a in v]], fmt="none",
                    ecolor=INK2, elinewidth=1, capsize=2, zorder=3)
        top = lim[1] if lim else max(x + a[2] for x, a in zip(xs, v)) * 1.25 or 1
        for y, x, a in zip(ys, xs, v):
            if a[0] is not None:
                ax.text(x + a[2] + top * 0.015, y, fmt.format(x), va="center", fontsize=7, color=INK)
        ax.set_xlim(0, top)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(labels, fontsize=8)
    fig.suptitle("What-if settings vs production (orange) - same images; blue bars are NOT production",
                 fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir, "whatif_detection.png")
