"""All matplotlib chart generation (Agg backend -- headless, no plt.show()).
Each function takes a results dict and a destination directory, and writes
one PNG, returning its path (or None if the inputs don't support a chart).
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def confidence_histogram(scores, out_dir, name, title):
    if not scores:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores, bins=20, color="#3366cc", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Detection confidence")
    ax.set_ylabel("Count")
    return _save(fig, out_dir, name)


def count_comparison_line(pred_counts, gt_counts, out_dir, name, title):
    if not pred_counts:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    x = list(range(1, len(pred_counts) + 1))
    ax.plot(x, gt_counts, label="Ground truth", color="#2ca02c", linewidth=2)
    ax.plot(x, pred_counts, label="VisionGuard (yolov8n)", color="#d62728",
            linewidth=1.5, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Person count")
    ax.legend()
    return _save(fig, out_dir, name)


def prf1_bar(precision, recall, f1, out_dir, name, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    vals = [precision, recall, f1]
    labels = ["Precision", "Recall", "F1"]
    bars = ax.bar(labels, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
    return _save(fig, out_dir, name)


def mae_rmse_bar(mae, rmse, out_dir, name, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    bars = ax.bar(["MAE", "RMSE"], [mae, rmse], color=["#9467bd", "#8c564b"])
    ax.set_title(title)
    for b, v in zip(bars, [mae, rmse]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom")
    return _save(fig, out_dir, name)


def latency_histogram(series: dict, out_dir, name="fall_detection_latency.png",
                       title="Fall-detection latency (frames from onset to confirmed alert)"):
    """series: {label: [latency_values_in_frames, ...]}. Different sources
    may use different latency definitions (e.g. UR Fall's true onset vs.
    Le2i's clip-start proxy) -- plotted as separate overlaid histograms so
    they are never visually conflated into one number."""
    series = {k: v for k, v in series.items() if v}
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for (label, values), color in zip(series.items(), colors):
        ax.hist(values, bins=min(10, max(3, len(values))), alpha=0.6,
                label=f"{label} (n={len(values)})", color=color, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1, linestyle=":")
    ax.set_xlabel("Latency (frames) -- negative = confirmed before the "
                 "labelled/baseline onset")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    return _save(fig, out_dir, name)


def tracking_bar(tracking_result, out_dir, name="mot20_tracking.png",
                 title="MOT20-01: tracking quality"):
    if not tracking_result or "id_switches" not in tracking_result:
        return None
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["ID switches", "Fragmentations"]
    vals = [tracking_result["id_switches"], tracking_result["fragmentations"]]
    bars = ax.bar(labels, vals, color=["#ff7f0e", "#9467bd"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom")
    ax.set_title(f"{title}\n({tracking_result.get('implementation', '')})")
    ax.set_ylabel("Count")
    return _save(fig, out_dir, name)


def pr_curve(precisions, recalls, ap, out_dir, name, title):
    if not precisions or not recalls:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(recalls, precisions, color="#1f77b4", linewidth=2)
    ax.fill_between(recalls, precisions, alpha=0.15, color="#1f77b4")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{title}\nAP@0.5 = {ap:.3f}")
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, name)


def confusion_matrix_chart(confusion_counts, classes, out_dir, name, title):
    import numpy as np
    mat = np.zeros((len(classes), len(classes)))
    for i, t in enumerate(classes):
        for j, p in enumerate(classes):
            mat[i, j] = confusion_counts.get(f"{t}->{p}", 0)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black")
    fig.colorbar(im)
    return _save(fig, out_dir, name)


def fps_sweep_chart(fps_result, out_dir, name="fps_sweep.png"):
    crowd = fps_result.get("crowd", [])
    fall = fps_result.get("fall", [])
    if not crowd and not fall:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    if crowd:
        ax.plot([p["target_fps"] for p in crowd], [p["f1"] for p in crowd],
                marker="o", label="Crowd detection F1 (MOT20-01)", color="#1f77b4")
    if fall:
        ax.plot([p["target_fps"] for p in fall], [p["recall"] for p in fall],
                marker="s", label="Fall confirmation Recall (Le2i)", color="#d62728")
    ax.set_xlabel("Simulated camera FPS")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Minimum-FPS sweep")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, name)


def resolution_sweep_chart(resolution_result, out_dir, name="resolution_sweep.png"):
    mot20_pts = resolution_result.get("mot20_points") or resolution_result.get("points") or []
    mall_pts = resolution_result.get("mall_points") or []
    if not mot20_pts and not mall_pts:
        return None
    n_panels = sum([bool(mot20_pts), bool(mall_pts)])
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    axes = [axes] if n_panels == 1 else list(axes)
    idx = 0
    if mot20_pts:
        ax1 = axes[idx]; idx += 1
        labels = [p["label"] for p in mot20_pts]
        x = list(range(len(mot20_pts)))
        ax1.plot(x, [p["recall"] for p in mot20_pts], marker="o", color="#1f77b4", label="Recall")
        ax1.set_ylabel("Recall", color="#1f77b4")
        ax1.set_ylim(0, 1.05)
        ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=20, ha="right")
        ax1b = ax1.twinx()
        ax1b.plot(x, [p["count_mae"] for p in mot20_pts], marker="s", color="#d62728", label="Count MAE")
        ax1b.set_ylabel("Count MAE", color="#d62728")
        ax1.set_title("MOT20-01 (densely crowded -- expect a near-flat line)")
    if mall_pts:
        ax2 = axes[idx]
        labels = [p["label"] for p in mall_pts]
        x = list(range(len(mall_pts)))
        ax2.plot(x, [p["accuracy_proxy"] for p in mall_pts], marker="o", color="#2ca02c",
                label="Accuracy proxy (1 - MAE/mean_count)")
        ax2.set_ylabel("Accuracy proxy", color="#2ca02c")
        ax2.set_ylim(0, 1.05)
        ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=20, ha="right")
        ax2b = ax2.twinx()
        ax2b.plot(x, [p["count_mae"] for p in mall_pts], marker="s", color="#d62728", label="Count MAE")
        ax2b.set_ylabel("Count MAE", color="#d62728")
        ax2.set_title("Mall Dataset (less saturated -- primary source for min-resolution)")
    fig.suptitle("Minimum-resolution sweep")
    return _save(fig, out_dir, name)


def person_size_sweep_chart(person_size_result, out_dir, name="person_size_sweep.png"):
    points = person_size_result.get("points", [])
    if not points:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = [p["bucket"] for p in points]
    vals = [p["recall"] if p["recall"] is not None else 0 for p in points]
    ns = [p["n_instances"] for p in points]
    bars = ax.bar(labels, vals, color="#2ca02c")
    for b, v, n in zip(bars, vals, ns):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"n={n}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Person height as % of frame height")
    ax.set_ylabel("Detection recall")
    ax.set_title("Minimum person-to-frame-size sweep (MOT20-01 + CrowdHuman)")
    return _save(fig, out_dir, name)
