"""Le2i Fall Detection Dataset — event-level fall-detection accuracy.

RESEARCH NOTE (verified live during this build): the GitHub repo listed as
a "mirror" (github.com/YifeiYang210/Fall_Detection_dataset) holds no video
files itself -- it only documents a Google Drive link labelled "Le2i-raw"
(309MB). That link resolves to a zip of real RGB frames (confirmed by
visually inspecting a sample image -- a genuine coffee-room scene, not a
binarized mask) pre-split into train/val, each with 5 class folders
(Blank, Stand = ADL negatives; Fall, Lie = positives; Likefall = an
ambiguous "unstable centre of gravity" class, reported separately rather
than forced into either bucket), each class folder containing many 16-frame
clip subfolders. There are no separate start/end-frame annotation files in
this re-packaging (that exists only in the *original* Le2i Home/Coffee-room
avi release, which the GitHub "mirror" does not actually host) -- so ground
truth here is per-CLIP (does this 16-frame clip show a fall or not), not
per-frame-interval. The evaluation below is adapted accordingly: each clip
is one event; VisionGuard's confirm_falls() must fire at least once inside
a positive clip (TP) and must never fire inside a negative clip (FP check).
"""
from __future__ import annotations

import os
import zipfile
from collections import defaultdict

from .. import http_utils as hu

GDRIVE_FILE_ID = "16sv5CLT3pBI0kcLvAYJqT_zscGK-GF91"  # "Le2i-raw", 309MB, confirmed live
POSITIVE_CLASSES = ("Fall", "Lie")
NEGATIVE_CLASSES = ("Blank", "Stand")
AMBIGUOUS_CLASSES = ("Likefall",)


def run(args) -> dict:
    import cv2
    from .. import metrics as m
    from .. import model_runner as mr
    import config as vg_config

    root = os.path.join(args.data_root, "le2i")
    clips_dir = os.path.join(root, "clips")
    result = {"name": "Le2i", "status": None}

    split = "val"
    clips_per_class = args.le2i_clips_per_class if not args.full else 10 ** 9

    manifest_path = os.path.join(root, "manifest.txt")
    have_per_class = defaultdict(int)
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            for line in f:
                cls, _clip, _n = line.strip().split("\t")
                have_per_class[cls] += 1
    need_more = (not have_per_class) or any(
        v < clips_per_class for v in have_per_class.values()) or len(have_per_class) < 5
    if need_more:
        sess, direct_url = hu.GoogleDriveDownloader.resolve_direct_url(GDRIVE_FILE_ID, timeout=30)
        if not direct_url:
            result["status"] = (
                "SKIPPED: Google Drive could not resolve a direct URL for the Le2i-raw "
                f"file (id={GDRIVE_FILE_ID}). Manual step: open "
                f"https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view, download it, "
                f"and extract to {root}")
            return result
        try:
            rf = hu.RemoteZipFile(direct_url, session=sess, timeout=60)
            with zipfile.ZipFile(rf) as zf:
                names = zf.namelist()
                clip_set = defaultdict(set)
                for n in names:
                    parts = n.split("/")
                    if len(parts) == 4 and parts[0] == split and parts[-1].endswith(".jpg"):
                        clip_set[parts[1]].add(parts[2])

                os.makedirs(clips_dir, exist_ok=True)
                manifest = []
                for cls, clip_names in clip_set.items():
                    chosen = sorted(clip_names)[:clips_per_class]
                    for clip in chosen:
                        frame_names = sorted(
                            n for n in names
                            if n.startswith(f"{split}/{cls}/{clip}/") and n.endswith(".jpg"))
                        dest_clip_dir = os.path.join(clips_dir, cls, clip)
                        os.makedirs(dest_clip_dir, exist_ok=True)
                        for fn in frame_names:
                            data = zf.read(fn)
                            with open(os.path.join(dest_clip_dir, os.path.basename(fn)), "wb") as f:
                                f.write(data)
                        manifest.append((cls, clip, len(frame_names)))
            with open(manifest_path, "w") as f:
                for cls, clip, n in manifest:
                    f.write(f"{cls}\t{clip}\t{n}\n")
            hu.log(f"Le2i: range-extracted {len(manifest)} clips "
                   f"(up to {clips_per_class}/class) from the 309MB mirror zip")
        except Exception as e:  # noqa: BLE001
            result["status"] = f"SKIPPED: range-extraction of Le2i clips failed ({e})"
            return result

    manifest = []
    with open(manifest_path) as f:
        for line in f:
            cls, clip, n = line.strip().split("\t")
            manifest.append((cls, clip, int(n)))
    if not manifest:
        result["status"] = "SKIPPED: no clips available"
        return result

    mr.warmup()
    per_clip_results = []
    for cls, clip, n_frames_expected in manifest:
        clip_dir = os.path.join(clips_dir, cls, clip)
        frame_files = sorted(f for f in os.listdir(clip_dir) if f.endswith(".jpg"))
        streaks = {}
        confirmed_any = False
        first_confirmed_at = None
        for i, fname in enumerate(frame_files):
            frame = cv2.imread(os.path.join(clip_dir, fname))
            if frame is None:
                continue
            pose_result, _ = mr.pose_infer(frame, persist=True)
            persons = mr.classify_persons(pose_result)
            confirmed = mr.confirm_falls(persons, streaks)
            if confirmed and not confirmed_any:
                confirmed_any = True
                first_confirmed_at = i
        per_clip_results.append({
            "class": cls, "clip": clip, "n_frames": len(frame_files),
            "confirmed_fall": confirmed_any, "first_confirmed_at_frame": first_confirmed_at,
        })

    pos = [r for r in per_clip_results if r["class"] in POSITIVE_CLASSES]
    neg = [r for r in per_clip_results if r["class"] in NEGATIVE_CLASSES]
    amb = [r for r in per_clip_results if r["class"] in AMBIGUOUS_CLASSES]

    tp = sum(1 for r in pos if r["confirmed_fall"])
    fn = sum(1 for r in pos if not r["confirmed_fall"])
    fp = sum(1 for r in neg if r["confirmed_fall"])
    tn = sum(1 for r in neg if not r["confirmed_fall"])
    precision, recall, f1 = m.precision_recall_f1(tp, fp, fn)
    fpr_on_adl = fp / len(neg) if neg else None
    ambiguous_fire_rate = (sum(1 for r in amb if r["confirmed_fall"]) / len(amb)) if amb else None

    # Latency caveat: these clips are pre-trimmed around the activity (no
    # separate pre-fall baseline exists in this re-packaged dataset, unlike
    # UR Fall's untrimmed sequences with a real per-frame label), so
    # "latency" here is frames-from-CLIP-START-to-confirmation, a coarser
    # proxy than UR Fall's true onset-to-confirmation latency -- reported
    # and charted separately from UR Fall's for that reason.
    latencies = [r["first_confirmed_at_frame"] for r in pos
                if r["confirmed_fall"] and r["first_confirmed_at_frame"] is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else None

    result.update({
        "status": "OK",
        "n_clips_total": len(per_clip_results),
        "n_positive_clips": len(pos), "n_negative_clips": len(neg), "n_ambiguous_clips": len(amb),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "false_positive_rate_on_ADL_clips": fpr_on_adl,
        "ambiguous_Likefall_fire_rate": ambiguous_fire_rate,
        "avg_latency_frames": avg_latency,
        "latency_samples_frames": latencies,
        "latency_caveat": ("frames-from-clip-start-to-confirmation, NOT true "
                          "fall-onset latency (this re-packaged dataset has no "
                          "pre-fall baseline frames) -- see UR Fall for true "
                          "onset-to-confirmation latency"),
        "per_clip": per_clip_results,
    })
    return result
