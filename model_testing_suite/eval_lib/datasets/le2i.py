"""Le2i Fall Detection Dataset ("Le2i-raw" re-packaging) — per-class
fall / posture / false-alarm behaviour.

The re-packaging (Google Drive file 16sv5CLT3pBI0kcLvAYJqT_zscGK-GF91, 309 MB,
linked from github.com/YifeiYang210/Fall_Detection_dataset) holds 16-frame
RGB clips (320x240) in train/ and val/, each in one of five classes. It has no
fall start/end annotations, so ground truth is per CLIP.

Fixes after the audit (#6, #7):
- frames are processed in NUMERIC order (eval_lib/frames.py); the previous
  string sort scrambled 3 of 30 clips
- every class is reported SEPARATELY:
    Fall     -> fall-event recall (the headline positive class)
    Stand    -> false-alarm rate on real daily activity (the headline negative)
    Lie      -> posture detection rate (the person is already lying: a posture
                classifier, not a fall-event detector, is being measured)
    Likefall -> fire rate on fall-like movements (reported, not scored)
    Blank    -> empty-room sanity check (no person; NOT counted in the ADL rate)
- clips are a SEEDED random sample from BOTH splits, stratified round-robin
  by scene (coffee room / home / ...). The models are stock COCO weights, so
  neither split was seen in training; using both only increases n
- the pose tracker is reset at the start of every clip
- the old "latency" (frames from clip start) is dropped: the clips have no
  pre-fall baseline, so it measured nothing
- each clip is run on every frame AND at production cadence (frames sampled
  by time at --inference-hz, see eval_lib/cadence.py). Native rate assumed
  25 fps (the original Le2i recordings); the re-packaging does not state it.
  A 16-frame clip lasts 0.64 s, so at ~2.8 Hz production sees only 2 frames:
  that row shows how short events behave under production sampling.
"""
from __future__ import annotations

import os
import zipfile
from collections import defaultdict

from .. import cadence
from .. import common
from .. import frames as fr
from .. import http_utils as hu
from .. import metrics as m
from .. import production_logic as pl

GDRIVE_FILE_ID = "16sv5CLT3pBI0kcLvAYJqT_zscGK-GF91"
CLASSES = ("Fall", "Stand", "Lie", "Likefall", "Blank")
NATIVE_FPS_ASSUMED = 25.0


def scene_of(clip: str) -> str:
    s = clip.lower().split("_v")[0]
    return "coffee" if s.startswith("coffee") else s


def _index_path(root):
    return os.path.join(root, "index_v2.tsv")


def build_or_load_index(root):
    """All (split, cls, clip) in the archive; cached so later runs work offline."""
    path = _index_path(root)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return [tuple(line.rstrip("\n").split("\t")) for line in f if line.strip()], None
    sess, url = hu.GoogleDriveDownloader.resolve_direct_url(GDRIVE_FILE_ID, timeout=30)
    if not url:
        raise RuntimeError(
            "Google Drive could not resolve a direct URL for Le2i-raw. Manual step: download "
            f"https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view and extract to {root}")
    rf = hu.RemoteZipFile(url, session=sess, timeout=120)
    zf = zipfile.ZipFile(rf)
    clips = set()
    for n in zf.namelist():
        parts = n.split("/")
        if len(parts) == 4 and parts[1] in CLASSES and parts[-1].lower().endswith(".jpg"):
            clips.add((parts[0], parts[1], parts[2]))
    index = sorted(clips)
    os.makedirs(root, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in index:
            f.write("\t".join(row) + "\n")
    return index, zf


def stratified_sample(index, per_class, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    chosen = []
    for cls in CLASSES:
        by_scene = defaultdict(list)
        for split, c, clip in index:
            if c == cls:
                by_scene[scene_of(clip)].append((split, c, clip))
        queues = []
        for scene in sorted(by_scene):
            items = sorted(by_scene[scene])
            order = rng.permutation(len(items)).tolist()
            queues.append([items[i] for i in order])
        picked = []
        while len(picked) < per_class and any(queues):
            for q in queues:
                if q and len(picked) < per_class:
                    picked.append(q.pop(0))
        chosen.extend(picked)
    return chosen


def ensure_clips(root, chosen, zf=None):
    clips_dir = os.path.join(root, "clips_v2")
    missing = [c for c in chosen if not fr.list_frames(os.path.join(clips_dir, *c))]
    if missing:
        if zf is None:
            sess, url = hu.GoogleDriveDownloader.resolve_direct_url(GDRIVE_FILE_ID, timeout=30)
            if not url:
                raise RuntimeError("Google Drive direct URL could not be resolved")
            zf = zipfile.ZipFile(hu.RemoteZipFile(url, session=sess, timeout=120))
        names = zf.namelist()
        for k, (split, cls, clip) in enumerate(missing, 1):
            prefix = f"{split}/{cls}/{clip}/"
            dest = os.path.join(clips_dir, split, cls, clip)
            os.makedirs(dest, exist_ok=True)
            for n in names:
                if n.startswith(prefix) and n.lower().endswith(".jpg"):
                    with open(os.path.join(dest, os.path.basename(n)), "wb") as f:
                        f.write(zf.read(n))
            if k % 10 == 0:
                hu.log(f"Le2i: extracted {k}/{len(missing)} clips")
    return clips_dir


def run_clip(clip_dir, rate_hz, pose_model_path=None):
    """Replay one clip through the production fall path. Returns dict."""
    import cv2
    from .. import model_runner as mr
    files = fr.list_frames(clip_dir)
    idx = cadence.sample_indices(len(files), NATIVE_FPS_ASSUMED, rate_hz)
    machine = pl.FallAlertMachine()
    first_raise, raise_frame = None, None
    for k, i in enumerate(idx):
        frame = cv2.imread(os.path.join(clip_dir, files[i]))
        if frame is None:
            continue
        res, _ = mr.pose_infer(frame, new_sequence=(k == 0), pose_model_path=pose_model_path)
        out = machine.step(mr.classify_persons(res), i / NATIVE_FPS_ASSUMED)
        if out["raised"] and first_raise is None:
            first_raise = i
            raise_frame = (os.path.join(clip_dir, files[i]),
                           [p["box"] for p in out["confirmed"]])
    return {"n_frames": len(files), "n_cycles": len(idx), "alert": first_raise is not None,
            "first_alert_frame": first_raise, "_raise_frame": raise_frame,
            "_mid_frame": os.path.join(clip_dir, files[len(files) // 2]) if files else None}


def summarize(rows, min_n):
    by = defaultdict(list)
    for r in rows:
        by[r["class"]].append(r)
    fired = {c: sum(1 for r in by[c] if r["alert"]) for c in CLASSES}
    n = {c: len(by[c]) for c in CLASSES}
    tp, fn, fp = fired["Fall"], n["Fall"] - fired["Fall"], fired["Stand"]
    p, r, f1 = m.precision_recall_f1(tp, fp, fn)
    return {
        "headline_fall_vs_stand": {
            "tp": tp, "fn": fn, "fp": fp, "tn": n["Stand"] - fp,
            "recall_fall": m.proportion(tp, n["Fall"]),
            "precision": m.proportion(tp, tp + fp),
            "f1": f1,
            "false_alarm_rate_stand": m.proportion(fp, n["Stand"]),
            "status_quality": common.quality(min(n["Fall"], n["Stand"]), min_n),
        },
        "per_class_alert_rate": {
            "Fall (fall events; should fire)": m.proportion(fired["Fall"], n["Fall"]),
            "Lie (already lying; posture detection)": m.proportion(fired["Lie"], n["Lie"]),
            "Stand (daily activity; should NOT fire)": m.proportion(fired["Stand"], n["Stand"]),
            "Likefall (fall-like movement; reported only)": m.proportion(fired["Likefall"], n["Likefall"]),
            "Blank (empty room sanity check; should NOT fire)": m.proportion(fired["Blank"], n["Blank"]),
        },
    }


def run(args) -> dict:
    from .. import model_runner as mr
    root = os.path.join(args.data_root, "le2i")
    result = {"name": "Le2i (Le2i-raw 16-frame clips)", "status": None}
    try:
        index, zf = build_or_load_index(root)
        chosen = stratified_sample(index, args.le2i_clips_per_class, args.seed)
        clips_dir = ensure_clips(root, chosen, zf)
    except Exception as e:  # noqa: BLE001
        result["status"] = f"SKIPPED: Le2i clips unavailable ({e})"
        return result

    errors = common.ErrorExamples("le2i", args.out_dir)
    modes = {"every_frame": None, "production_cadence": args.inference_hz}
    per_mode = {}
    for mode, rate in modes.items():
        rows = []
        for split, cls, clip in chosen:
            d = os.path.join(clips_dir, split, cls, clip)
            r = run_clip(d, rate)
            rows.append({"split": split, "class": cls, "clip": clip, "scene": scene_of(clip),
                         **{k: v for k, v in r.items() if not k.startswith("_")}})
            if mode == "every_frame":
                if cls in ("Stand", "Blank") and r["alert"] and r["_raise_frame"]:
                    path, boxes = r["_raise_frame"]
                    errors.add(f"{cls}_{clip}", path, [], boxes, 1, 0, note="false alarm")
                elif cls == "Fall" and not r["alert"] and r["_mid_frame"]:
                    errors.add(f"{cls}_{clip}", r["_mid_frame"], [], [], 0, 1, note="missed fall")
        per_mode[mode] = {"rate": cadence.rate_label(rate), **summarize(rows, args.headline_min_n),
                          "per_clip": rows}

    counts = defaultdict(int)
    scenes = defaultdict(int)
    for split, cls, clip in chosen:
        counts[cls] += 1
        scenes[scene_of(clip)] += 1
    result.update({
        "status": "OK",
        "status_quality": per_mode["every_frame"]["headline_fall_vs_stand"]["status_quality"],
        "n_clips": len(chosen), "clips_per_class": dict(counts), "clips_per_scene": dict(scenes),
        "n_clips_available_in_archive": len(index),
        "sampling": f"seeded (seed={args.seed}) random, stratified round-robin by scene, both splits",
        "native_fps_assumed": NATIVE_FPS_ASSUMED,
        "results": per_mode,
    })
    if args.baselines:
        rows = []
        for split, cls, clip in chosen:
            r = run_clip(os.path.join(clips_dir, split, cls, clip), None,
                         pose_model_path=mr.model_path("yolov8n-pose.pt"))
            rows.append({"class": cls, "alert": r["alert"]})
        result["baselines"] = {"yolov8n-pose every frame (NOT production)":
                               summarize(rows, args.headline_min_n)}
    result["error_examples"] = errors.save()
    return result
