"""Shared helpers: detection scoring with CIs, result-quality labels, seeded
sampling and FP/FN example dumps (error analysis)."""
from __future__ import annotations

import heapq
import os

import numpy as np

from . import metrics as m

QUALITY_NOTE = ("'headline' = at least HEADLINE_MIN_N positive events/instances "
                "behind the main metric; 'pilot' = fewer, indicative only")


def quality(n_positive: int, min_n: int) -> str:
    return "headline" if n_positive >= min_n else "pilot"


def seeded_sample(items, k, seed):
    """Deterministic sample of k items (all items if k >= len), in input order."""
    items = list(items)
    if k is None or k >= len(items):
        return items
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(items), size=k, replace=False).tolist())
    return [items[i] for i in idx]


def evaluate_detection(frames_pred, frames_gt, frames_ignore=None, iou_thresh=0.5,
                       seed=0, block=1) -> dict:
    """Pooled P/R/F1 at one IoU threshold. Precision and recall get Wilson CIs
    (pooled counts); F1 gets a (block-)bootstrap CI over frames/images."""
    per = []
    for i, (p, g) in enumerate(zip(frames_pred, frames_gt)):
        ign = frames_ignore[i] if frames_ignore else None
        per.append(m.match_boxes(p, g, iou_thresh, ignore_boxes=ign))
    tp = sum(r["tp"] for r in per)
    fp = sum(r["fp"] for r in per)
    fn = sum(r["fn"] for r in per)
    precision, recall, f1 = m.precision_recall_f1(tp, fp, fn)
    tps = np.array([r["tp"] for r in per], float)
    fps_ = np.array([r["fp"] for r in per], float)
    fns = np.array([r["fn"] for r in per], float)

    def _f1(idx):
        return m.precision_recall_f1(tps[idx].sum(), fps_[idx].sum(), fns[idx].sum())[2]

    lo, hi = m.bootstrap_ci(len(per), _f1, seed=seed, block=block)
    return {
        "iou_thresh": iou_thresh, "n_frames": len(per), "tp": tp, "fp": fp, "fn": fn,
        "n_gt": tp + fn, "n_pred_scored": tp + fp,
        "n_pred_ignored": sum(len(r["ignored"]) for r in per),
        "precision": m.proportion(tp, tp + fp),
        "recall": m.proportion(tp, tp + fn),
        "f1": {"value": f1, "ci_low": lo, "ci_high": hi, "ci_method": f"bootstrap95(block={block})"},
        "_per_frame": per,
    }


class ErrorExamples:
    """Keeps the K frames with the most false positives and the K with the
    most false negatives, then draws them (GT green, predictions red,
    ignore regions grey) into out_dir/error_examples/<dataset>/."""

    def __init__(self, dataset: str, out_dir: str, k: int = 10):
        self.dataset, self.k = dataset, k
        self.dir = os.path.join(out_dir, "error_examples", dataset)
        self._fp, self._fn = [], []
        self._n = 0

    def add(self, frame_id, image_path, gt_boxes, pred_boxes, n_fp, n_fn, ignore_boxes=None,
            gt_points=None, note=""):
        self._n += 1
        item = (frame_id, image_path, list(gt_boxes or []), list(pred_boxes or []),
                list(ignore_boxes or []), list(gt_points or []), note)
        for heap, score in ((self._fp, n_fp), (self._fn, n_fn)):
            if score <= 0:
                continue
            entry = (score, self._n, item)
            if len(heap) < self.k:
                heapq.heappush(heap, entry)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, entry)

    def save(self, crop=None) -> dict:
        import cv2
        written = {"false_positives": [], "false_negatives": []}
        for kind, heap in (("false_positives", self._fp), ("false_negatives", self._fn)):
            for score, _, (fid, path, gt, pred, ign, pts, note) in sorted(heap, reverse=True):
                img = cv2.imread(path)
                if img is None:
                    continue
                if crop:
                    x1, y1, x2, y2 = crop
                    img = img[y1:y2, x1:x2].copy()
                for b in ign:
                    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (150, 150, 150), 1)
                for b in gt:
                    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 0), 2)
                for (x, y) in pts:
                    cv2.circle(img, (int(x), int(y)), 3, (0, 200, 0), -1)
                for b in pred:
                    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 230), 1)
                label = f"{kind.split('_')[1][:-1]} x{score} {note}".strip()
                cv2.putText(img, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                os.makedirs(os.path.join(self.dir, kind), exist_ok=True)
                safe = str(fid).replace("/", "_").replace("\\", "_").replace(",", "_")
                out = os.path.join(self.dir, kind, f"{safe}.jpg")
                cv2.imwrite(out, img)
                written[kind].append(os.path.relpath(out, os.path.dirname(os.path.dirname(self.dir))))
        return written
