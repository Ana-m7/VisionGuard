"""Metric implementations used across the evaluation suite.

Dependency-light (numpy + scipy only). Every function takes plain Python
lists/tuples so it can be unit-tested by hand (see tests_eval/).

Box convention everywhere: (x1, y1, x2, y2[, conf]).

Contents
- iou / ioa, greedy single-threshold matching (with optional ignore regions)
- precision / recall / F1
- Wilson score interval for proportions; (block) bootstrap CIs
- count MAE / RMSE / bias with bootstrap CIs
- COCO-style 101-point AP50 (and the legacy 11-point variant, for comparison
  with the archived pre-audit run only)
- CLEAR-MOT (MOTA, MOTP, IDSW, Frag) and IDF1
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

Z_95 = 1.959963984540054


# ---------------------------------------------------------------- geometry
def iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def ioa(box, region) -> float:
    """Intersection over the area of ``box`` (how much of box lies in region)."""
    ax1, ay1, ax2, ay2 = box[:4]
    bx1, by1, bx2, by2 = region[:4]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    return (iw * ih) / area if area > 0 else 0.0


def _hits_ignore(pred, ignore_boxes, iou_thresh, ioa_thresh) -> bool:
    for r in ignore_boxes or ():
        if iou(pred, r) >= iou_thresh or (ioa_thresh is not None and ioa(pred, r) >= ioa_thresh):
            return True
    return False


# ---------------------------------------------------------------- matching
def match_boxes(pred_boxes, gt_boxes, iou_thresh: float = 0.5, ignore_boxes=None,
                ioa_thresh: float | None = 0.5):
    """Greedy highest-IoU-first one-to-one matching at a single IoU threshold.

    Predictions left unmatched that overlap an ignore region (IoU >= iou_thresh,
    or intersection-over-prediction-area >= ioa_thresh) are neither TP nor FP
    (CrowdHuman "mask"/ignore regions, MOTChallenge distractor classes).

    Returns dict(tp, fp, fn, matched=[(pi, gi, iou)], ignored=[pi], fp_idx, fn_idx).
    """
    pairs = []
    for pi, p in enumerate(pred_boxes):
        for gi, g in enumerate(gt_boxes):
            v = iou(p, g)
            if v >= iou_thresh:
                pairs.append((v, pi, gi))
    pairs.sort(key=lambda t: -t[0])
    used_p, used_g, matched = set(), set(), []
    for v, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matched.append((pi, gi, v))
    ignored, fp_idx = [], []
    for pi, p in enumerate(pred_boxes):
        if pi in used_p:
            continue
        if ignore_boxes and _hits_ignore(p, ignore_boxes, iou_thresh, ioa_thresh):
            ignored.append(pi)
        else:
            fp_idx.append(pi)
    fn_idx = [gi for gi in range(len(gt_boxes)) if gi not in used_g]
    return {"tp": len(matched), "fp": len(fp_idx), "fn": len(fn_idx),
            "matched": matched, "ignored": ignored, "fp_idx": fp_idx, "fn_idx": fn_idx}


def precision_recall_f1(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------- intervals
def wilson_ci(k: int, n: int, z: float = Z_95):
    """Wilson score interval for k successes out of n. (None, None) if n == 0."""
    if n <= 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def proportion(k: int, n: int) -> dict:
    lo, hi = wilson_ci(k, n)
    return {"value": (k / n) if n else None, "k": int(k), "n": int(n),
            "ci_low": lo, "ci_high": hi, "ci_method": "wilson95"}


def bootstrap_ci(n_items: int, stat_fn: Callable[[np.ndarray], float], n_boot: int = 2000,
                 seed: int = 0, block: int = 1, alpha: float = 0.05):
    """Percentile bootstrap CI of ``stat_fn(indices)``.

    ``block`` > 1 gives a moving-block bootstrap, which keeps runs of
    consecutive video frames together; ordinary i.i.d. resampling of
    neighbouring (highly correlated) frames would give a falsely narrow CI.
    """
    if n_items <= 0:
        return None, None
    rng = np.random.default_rng(seed)
    block = max(1, min(block, n_items))
    n_blocks = math.ceil(n_items / block)
    starts_max = n_items - block + 1
    stats = []
    for _ in range(n_boot):
        starts = rng.integers(0, starts_max, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n_items]
        stats.append(stat_fn(idx))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def count_errors(pred_counts: Sequence[float], gt_counts: Sequence[float], seed: int = 0,
                 block: int = 1) -> dict:
    """MAE / RMSE / mean signed error (bias) / relative MAE, each with a
    bootstrap 95% CI."""
    assert len(pred_counts) == len(gt_counts)
    p = np.asarray(pred_counts, float)
    g = np.asarray(gt_counts, float)
    n = len(p)
    if n == 0:
        return {"n": 0}
    e = p - g
    mean_gt = float(g.mean())

    def _mae(i):
        return float(np.abs(e[i]).mean())

    def _rmse(i):
        return float(np.sqrt((e[i] ** 2).mean()))

    def _bias(i):
        return float(e[i].mean())

    def _rel(i):
        mg = g[i].mean()
        return float(np.abs(e[i]).mean() / mg) if mg else float("nan")

    out = {"n": n, "mean_gt_count": mean_gt, "mean_pred_count": float(p.mean()),
           "bootstrap_block": block}
    for name, fn in (("mae", _mae), ("rmse", _rmse), ("bias", _bias), ("relative_mae", _rel)):
        lo, hi = bootstrap_ci(n, fn, seed=seed, block=block)
        out[name] = {"value": fn(np.arange(n)), "ci_low": lo, "ci_high": hi,
                     "ci_method": f"bootstrap95(block={block})"}
    return out


def mae_rmse(pred_counts, gt_counts):
    assert len(pred_counts) == len(gt_counts)
    if not pred_counts:
        return 0.0, 0.0
    errs = [p - g for p, g in zip(pred_counts, gt_counts)]
    return (sum(abs(x) for x in errs) / len(errs),
            math.sqrt(sum(x * x for x in errs) / len(errs)))


def summarize_mean(values: Sequence[float], seed: int = 0) -> dict:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"n": 0, "value": None}
    arr = np.asarray(vals)
    lo, hi = bootstrap_ci(len(arr), lambda i: float(arr[i].mean()), seed=seed)
    return {"n": len(arr), "value": float(arr.mean()), "median": float(np.median(arr)),
            "min": float(arr.min()), "max": float(arr.max()),
            "ci_low": lo, "ci_high": hi, "ci_method": "bootstrap95"}


# ---------------------------------------------------------------- AP
def _ranked_tp_fp(frames_pred, frames_gt, frames_ignore, iou_thresh):
    flat = []
    for fi, preds in enumerate(frames_pred):
        for p in preds:
            conf = p[4] if len(p) > 4 and p[4] is not None else 1.0
            flat.append((conf, fi, p[:4]))
    flat.sort(key=lambda t: -t[0])
    matched_gt = [set() for _ in frames_gt]
    flags = []  # 1 = TP, 0 = FP (ignored predictions are dropped)
    confs = []
    for conf, fi, pbox in flat:
        best_iou, best_gi = 0.0, -1
        for gi, gbox in enumerate(frames_gt[fi]):
            if gi in matched_gt[fi]:
                continue
            v = iou(pbox, gbox)
            if v > best_iou:
                best_iou, best_gi = v, gi
        if best_gi >= 0 and best_iou >= iou_thresh:
            matched_gt[fi].add(best_gi)
            flags.append(1)
            confs.append(conf)
        elif frames_ignore and _hits_ignore(pbox, frames_ignore[fi], iou_thresh, 0.5):
            continue
        else:
            flags.append(0)
            confs.append(conf)
    return flags, confs


def average_precision(frames_pred, frames_gt, frames_ignore=None, iou_thresh: float = 0.5,
                      points: int = 101):
    """Single-class AP at one IoU threshold.

    points=101 -> COCO-style (precision envelope sampled at recall 0,0.01,..,1).
    points=11  -> PASCAL-VOC-2007 11-point (kept ONLY to compare with the
    archived pre-audit run, which used it on conf>=0.25 detections).
    Returns (ap, precisions, recalls).
    """
    assert len(frames_pred) == len(frames_gt)
    n_gt = sum(len(g) for g in frames_gt)
    if n_gt == 0:
        return 0.0, [], []
    flags, _ = _ranked_tp_fp(frames_pred, frames_gt, frames_ignore, iou_thresh)
    tp = np.cumsum(flags) if flags else np.zeros(0)
    fp = np.cumsum([1 - f for f in flags]) if flags else np.zeros(0)
    if len(tp) == 0:
        return 0.0, [], []
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-12)
    env = prec.copy()
    for i in range(len(env) - 2, -1, -1):
        env[i] = max(env[i], env[i + 1])
    levels = np.linspace(0, 1, points)
    samples = []
    for r in levels:
        idx = np.searchsorted(rec, r, side="left")
        samples.append(env[idx] if idx < len(env) else 0.0)
    return float(np.mean(samples)), prec.tolist(), rec.tolist()


# ---------------------------------------------------------------- tracking
def filter_ignored_preds(pred_frame, gt_boxes, ignore_boxes, iou_thresh: float = 0.5):
    """MOTChallenge-style distractor handling: drop a prediction whose best
    overlap is with a distractor/ignore box (IoU >= thresh and higher than its
    IoU with every real GT box)."""
    if not ignore_boxes:
        return list(pred_frame)
    kept = []
    for p in pred_frame:
        box = p[1:5] if len(p) >= 5 else p
        best_ign = max((iou(box, r) for r in ignore_boxes), default=0.0)
        best_gt = max((iou(box, g) for g in gt_boxes), default=0.0)
        if best_ign >= iou_thresh and best_ign > best_gt:
            continue
        kept.append(p)
    return kept


def clear_mot(frames_pred_tracks, frames_gt_tracks, iou_thresh: float = 0.5) -> dict:
    """CLEAR-MOT (Bernardin & Stiefelhagen 2008), following py-motmetrics.

    frames_*: list over frames of [(id, x1, y1, x2, y2), ...].
    Per frame: keep last frame's correspondences that still overlap
    (IoU >= thresh), Hungarian-assign the rest on (1 - IoU), count an ID switch
    when a GT object is matched to a different track id than the one it was
    last matched to. Frag = transitions tracked -> missed between an object's
    first and last tracked frame.
    """
    from scipy.optimize import linear_sum_assignment

    assert len(frames_pred_tracks) == len(frames_gt_tracks)
    last_match = {}           # gt id -> pred id it was last matched to
    status = {}               # gt id -> list of tracked flags per frame present
    n_gt = tp = fp = fn = idsw = 0
    iou_sum = 0.0
    for preds, gts in zip(frames_pred_tracks, frames_gt_tracks):
        n_gt += len(gts)
        pred_by_id = {p[0]: p[1:5] for p in preds}
        gt_by_id = {g[0]: g[1:5] for g in gts}
        matches = {}
        used_p = set()
        for gid, pid in last_match.items():  # continuity first
            if gid in gt_by_id and pid in pred_by_id and pid not in used_p:
                v = iou(gt_by_id[gid], pred_by_id[pid])
                if v >= iou_thresh:
                    matches[gid] = (pid, v)
                    used_p.add(pid)
        rem_g = [g for g in gt_by_id if g not in matches]
        rem_p = [p for p in pred_by_id if p not in used_p]
        if rem_g and rem_p:
            cost = np.ones((len(rem_g), len(rem_p)))
            for i, g in enumerate(rem_g):
                for j, p in enumerate(rem_p):
                    v = iou(gt_by_id[g], pred_by_id[p])
                    if v >= iou_thresh:
                        cost[i, j] = 1.0 - v
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] < 1.0:
                    g, p = rem_g[i], rem_p[j]
                    if g in last_match and last_match[g] != p:
                        idsw += 1
                    matches[g] = (p, 1.0 - cost[i, j])
        for g, (p, v) in matches.items():
            last_match[g] = p
            iou_sum += v
        tp += len(matches)
        fp += len(pred_by_id) - len(matches)
        fn += len(gt_by_id) - len(matches)
        for g in gt_by_id:
            status.setdefault(g, []).append(g in matches)
    frag = 0
    for flags in status.values():
        tracked = [i for i, f in enumerate(flags) if f]
        if not tracked:
            continue
        seg = flags[tracked[0]:tracked[-1] + 1]
        frag += sum(1 for a, b in zip(seg, seg[1:]) if a and not b)
    mota = 1.0 - (fn + fp + idsw) / n_gt if n_gt else None
    motp = iou_sum / tp if tp else None
    return {"mota": mota, "motp_mean_iou": motp, "idsw": idsw, "frag": frag,
            "tp": tp, "fp": fp, "fn": fn, "n_gt": n_gt,
            "n_gt_ids": len(status)}


def idf1(frames_pred_tracks, frames_gt_tracks, iou_thresh: float = 0.5) -> dict:
    """IDF1 (Ristani et al. 2016): one-to-one GT-id <-> track-id assignment
    maximising the number of co-occurring frames with IoU >= thresh."""
    from scipy.optimize import linear_sum_assignment

    gt_ids, pr_ids = {}, {}
    co = {}
    n_gt_dets = n_pr_dets = 0
    for preds, gts in zip(frames_pred_tracks, frames_gt_tracks):
        n_gt_dets += len(gts)
        n_pr_dets += len(preds)
        for g in gts:
            gt_ids.setdefault(g[0], len(gt_ids))
            for p in preds:
                pr_ids.setdefault(p[0], len(pr_ids))
                if iou(g[1:5], p[1:5]) >= iou_thresh:
                    key = (gt_ids[g[0]], pr_ids[p[0]])
                    co[key] = co.get(key, 0) + 1
        for p in preds:
            pr_ids.setdefault(p[0], len(pr_ids))
    idtp = 0
    if gt_ids and pr_ids and co:
        m = np.zeros((len(gt_ids), len(pr_ids)))
        for (i, j), c in co.items():
            m[i, j] = c
        rows, cols = linear_sum_assignment(-m)
        idtp = int(m[rows, cols].sum())
    idfp = n_pr_dets - idtp
    idfn = n_gt_dets - idtp
    denom = 2 * idtp + idfp + idfn
    return {"idf1": (2 * idtp / denom) if denom else None,
            "idp": idtp / (idtp + idfp) if (idtp + idfp) else None,
            "idr": idtp / (idtp + idfn) if (idtp + idfn) else None,
            "idtp": idtp, "idfp": idfp, "idfn": idfn}


# ---------------------------------------------------------------- cross-check
def crosscheck_motmetrics(frames_pred_tracks, frames_gt_tracks, iou_thresh: float = 0.5):
    """Same tracking scores computed by the reference py-motmetrics package,
    to cross-check clear_mot()/idf1(). Returns None if motmetrics is not
    installed. The IoU distance matrix is built here (with iou()) because
    motmetrics 1.4.0's own iou_matrix() calls np.asfarray, removed in
    numpy 2; the accumulator and metric computation are motmetrics' own."""
    try:
        import motmetrics as mm
    except ImportError:
        return None
    acc = mm.MOTAccumulator(auto_id=True)
    for preds, gts in zip(frames_pred_tracks, frames_gt_tracks):
        d = np.full((len(gts), len(preds)), np.nan)
        for i, g in enumerate(gts):
            for j, p in enumerate(preds):
                ov = iou(g[1:5], p[1:5])
                if ov >= iou_thresh:
                    d[i, j] = 1.0 - ov
        acc.update([g[0] for g in gts], [p[0] for p in preds], d)
    names = ["mota", "motp", "idf1", "num_switches", "num_fragmentations",
             "num_false_positives", "num_misses", "num_objects"]
    s = mm.metrics.create().compute(acc, metrics=names, name="x")
    row = {k: float(s[k].iloc[0]) for k in names}
    return {"package": f"motmetrics {mm.__version__}",
            "mota": row["mota"], "motp_mean_iou": 1.0 - row["motp"], "idf1": row["idf1"],
            "idsw": int(row["num_switches"]), "frag": int(row["num_fragmentations"]),
            "fp": int(row["num_false_positives"]), "fn": int(row["num_misses"]),
            "n_gt": int(row["num_objects"])}
