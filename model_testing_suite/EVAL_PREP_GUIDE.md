# VisionGuard Model Testing Suite — Explanation & Evaluation Prep Guide

This guide explains, in plain words, everything that was done to
`model_testing_suite/`: what was wrong, what was changed in each file, what the
accuracy numbers mean, and how to answer likely questions in an evaluation.
Every number here is copied from `model_eval_results/summary.csv` (full run,
18 Nov 2026, commit `41ed380`).

---

## 1. The one-minute summary

**VisionGuard** is a CCTV monitoring system. It has four main features:

- it counts people;
- it shows crowded zones on a 3×4 grid;
- it raises crowd alerts and fall alerts;
- it detects camera tampering (covered or frozen camera).

**The model testing suite** checks how accurate those features are, using public
datasets in which humans hand-labelled the right answers.

**Before:** an audit found the old suite was **NOT RELIABLE**, with 22 problems:

- it tested an old version of the pipeline;
- the "CrowdHuman" labels were fake;
- tracker memory leaked between tests;
- frames were played out of order;
- some "minimum camera" claims were really just noise;
- no number had an error range.

**After:**
- The suite now tests **exactly the code that runs live**: it imports the real functions and refuses to run if they change.
- It uses the official datasets.
- Running it twice in any order gives **identical results**.
- Every number has a **95% confidence interval** and a sample size.
- It was fully re-run on all the data.

**Main finding:**
- When VisionGuard marks someone as a person it is almost always right (92–95%).
- In crowded scenes, though, it only finds about **40% of the people**, so crowd counts come out 40–60% too low.
- Fall detection was tested on **all 70 UR Fall videos**. At the configured live speed (2.8 analyses per second) it catches **10 of 30 falls (33%)**; at 10 analyses per second it catches **24 of 30 (80%)**. So fall detection depends strongly on how fast the computer is.
- 12 of 40 normal-activity videos raised a fall alarm, but **10 of those 12 are videos where the person lies down on purpose**. In videos with no lying, only 2 of 24 raised an alarm.
- A "what-if" test measured settings that would improve counting: a bigger input image (recall 0.42 → 0.58) or a lower confidence threshold (0.42 → 0.49, at no speed cost). These are measurements only; production was not changed.

---

## 2. Glossary (every technical term, in easy words)

### Detection and accuracy
| Term | Easy meaning |
|---|---|
| **Detection** | The AI draws a box around something it thinks is a person. |
| **Ground truth (GT)** | The correct answer, hand-labelled by humans in the dataset. |
| **IoU (Intersection over Union)** | How much a predicted box overlaps the true box, from 0 (no overlap) to 1 (perfect). We count a detection as correct if IoU ≥ 0.5, i.e. it overlaps at least half. |
| **True positive (TP)** | Correctly found a real person. |
| **False positive (FP)** | Flagged something that isn't there: a false alarm. |
| **False negative (FN)** | Missed a real person. |
| **Precision** | Of everything VisionGuard flagged, how much was right. TP ÷ (TP + FP). Low precision means many false alarms. |
| **Recall** | Of everything really there, how much VisionGuard found. TP ÷ (TP + FN). Low recall means many misses. |
| **F1 score** | One number that balances precision and recall (their harmonic mean). 1.0 is perfect. |
| **AP50 (Average Precision at IoU 0.5)** | Precision averaged over all confidence thresholds: one number for the whole precision–recall trade-off. "COCO 101-point" is the standard way of computing it, so it can be compared with published papers. |
| **Confidence score** | How sure the AI is about a box (0–1). VisionGuard keeps boxes with confidence ≥ 0.25 (`DETECTION_CONF_MIN`). |
| **Ignore regions / distractors** | Areas the dataset authors marked "don't score" (e.g. reflections, mannequins, unclear crowds). Detections there count neither as right nor wrong. |
| **Visibility** | How much of a person can be seen (1.0 = fully visible). Used to exclude heavily hidden people from the size test. |

### Counting
| Term | Easy meaning |
|---|---|
| **MAE (Mean Absolute Error)** | On average, how many people the count was off by. MAE 13.4 means typically 13 people off. |
| **RMSE (Root Mean Square Error)** | Like MAE but punishes big mistakes more. |
| **Bias** | The average signed error. Negative means VisionGuard **under-counts**. |
| **Relative MAE** | MAE divided by the true average count. 0.43 means 43% off. |

### Statistics
| Term | Easy meaning |
|---|---|
| **n (sample size)** | How many cases a number is based on. |
| **95% confidence interval (CI)**, written [low–high] | The range the true value very probably lies in. Narrow means reliable; wide means not enough data. |
| **Wilson interval** | The standard, safe way to compute a CI for a proportion (like "4 of 8"). It works even with small n, or when the result is 0% or 100%. |
| **Bootstrap** | Re-sample the data thousands of times and see how much the number moves. Used for MAE. |
| **Block bootstrap** | The same, but resampling chunks of consecutive video frames, because neighbouring frames are nearly identical and would otherwise make the CI look falsely narrow. |
| **Headline vs pilot** | Headline = based on ≥ 30 cases, safe to quote with its CI. Pilot = fewer, indicative only: quote raw counts ("4 of 8"), not percentages. |
| **Seed** | A fixed starting number for random choices, so the same images are picked every time and results can be reproduced. |
| **Deterministic** | Same input gives exactly the same output, every run, in any order. |

### Tracking
| Term | Easy meaning |
|---|---|
| **Tracking** | Following the same person across frames and giving them an ID number. |
| **Track ID** | That ID number. VisionGuard uses it to require that the *same person* looks fallen for 2 cycles in a row. |
| **ID switch (IDSW)** | The tracker swapped two people's IDs. |
| **Fragmentation (Frag)** | One person's track broke into pieces. |
| **MOTA** | Overall tracking accuracy: combines misses, false boxes and ID switches. 1.0 is perfect; it can go negative. |
| **MOTP** | How tightly the tracked boxes fit (average IoU). |
| **IDF1** | How consistently each person keeps the same ID (1.0 is perfect). |

### System and video
| Term | Easy meaning |
|---|---|
| **YOLOv8n / YOLOv8s** | The AI detection models (n = nano, small and fast; s = small, bigger and more accurate). VisionGuard uses **yolov8n** for counting. |
| **Pose model (yolov8s-pose)** | Detects people **and** their body keypoints (head, shoulders, hips, ankles). Used for fall detection. |
| **Keypoints** | Body joint positions. The torso angle from them decides "fallen". |
| **COCO weights** | The models were trained by Ultralytics on the COCO dataset, not on any of our test datasets, so there is **no data leakage**. The AI has never seen the test images. |
| **fps (frames per second)** | How many images per second a camera produces. |
| **Inference** | One run of the AI on one frame. |
| **Inference cycle / cadence / Hz** | Live, VisionGuard does not analyse every frame. It analyses the newest frame each time it finishes the previous one. The number of analyses per second is the cadence, in Hz. `config.py` documents about 2.8 Hz. |
| **Letterboxing** | YOLO resizes every image to 640 px (with padding). This is why sizes above 640 px make no difference. |
| **Hysteresis** | Different thresholds to raise and to clear an alert, so it doesn't flicker on and off. Crowd raises at ≥ 4 people and clears below 2. |
| **Confirm cycles** | The condition must hold several cycles in a row before alerting: 3 for crowd, 2 for fall. |
| **ADL (Activities of Daily Living)** | Normal-activity videos (walking, sitting, lying on a bed). Used to measure false alarms. |
| **Fall onset** | The moment the fall *starts* (not when the person is already on the floor). Latency is measured from here. |
| **Latency** | The time from the fall starting to the alert firing. |
| **Tamper** | Someone covering, freezing, blurring or moving the camera. |

### Testing practice
| Term | Easy meaning |
|---|---|
| **Baseline** | A comparison model (e.g. yolov8s, or the old pipeline) that shows whether our result is good or bad relative to something. |
| **Parity check** | Before each run, the suite reads `detector/engine.py` and checks that the exact calls it mirrors are still there. If production changed, the suite refuses to run, so the tests can't silently go stale. |
| **Provenance** | A record saved with the results: git commit, settings, package versions and machine. It proves exactly what produced the numbers. |
| **Range extraction** | Downloading only the needed files from inside a huge online zip (e.g. 300 images from a 2.5 GB file) instead of the whole thing. |

---

## 3. What was wrong (the audit, in 7 stories)

1. **The wrong pipeline was tested.** Production (commit 41ed380) counts people with plain detection (`predict`) and then removes duplicate boxes. The old suite still used **tracking**, which hides people the tracker hasn't "confirmed" yet. So every counting number described an old system.
2. **The "CrowdHuman" labels were fake.** The files came from an unofficial mirror. They had about 4.7 boxes per image (real CrowdHuman averages about 23), and contained cars, umbrellas, chairs and ties, all counted as "persons". The labels looked machine-generated, so the test was the AI checked against another AI.
3. **Tracker memory leaked.** The tracker was never reset between videos, so results depended on what ran before. Identical frames gave different scores.
4. **Le2i problems.** Frames named `86.jpg … 101.jpg` were sorted as text ("100" before "86"), so 3 clips were played out of order. Half the "normal activity" clips were an **empty room**, which made the false-alarm rate look half as big as it was.
5. **The "minimum camera" claims were noise.**
   - The fps test compared 20 vs 25 fps, but both used the same frames.
   - Production only analyses about 3 frames per second anyway.
   - The resolution test squashed images out of shape.
   - The person-size claim rested on only 3 people.
6. **Out of date and unreproducible.**
   - The tamper test used the old 15-frame frozen setting (production now uses 1440).
   - Paths depended on the folder you ran from.
   - No versions were recorded.
   - No error ranges.
7. **Found during the re-run: tracker contamination.** In Ultralytics, calling `.track()` on a model **permanently** attaches the tracker to that model object. The comparison test (old tracked pipeline) used the *same* model object as the production test, so production numbers were secretly tracked too. Proof: the same images gave recall 0.96 alone vs 0.64 after another test. Fixed by giving each role its own model object, plus a guard that raises an error if a tracker is ever found on the production model.

---

## 4. What was changed — file by file

The rule throughout: **only files inside `model_testing_suite/` were changed.** Production code (`config.py`, `detector/`, `backend/` and so on) was not touched. The suite **imports** the real functions instead of copying them.

### Main script
| File | What it does / what changed |
|---|---|
| `test_models.py` | The entry point. Rewritten. Runs, in order: parity check → fixed random seed → record provenance → warm up models → each dataset → sweeps → charts → `summary.csv`/`summary.json`. Options: `--quick`, `--only` (runs in the given order), `--skip-sweeps`, `--redraw-charts`, `--seed`, `--max-download-mb` (500 MB default), `--allow-large-downloads`, `--no-baselines`. A failure in one dataset is recorded, not hidden. |

### Core library (`eval_lib/`)
| File | Status | What it does |
|---|---|---|
| `paths.py` | new | Finds the suite and project folders from the file location, so the suite works from any working directory. Puts the project on the Python path so `import config` works. |
| `__init__.py` | changed | Loads `paths` first. |
| `frames.py` | new | Sorts frame files by **number**, not text (fixes the Le2i scramble). |
| `cadence.py` | new | Picks frames **by time** to simulate the live analysis rate (e.g. 2.8 per second). Guarantees different rates use different frames. |
| `model_runner.py` | rewritten | Runs the AI exactly like production: `predict` + duplicate-box removal for counting, and `pose_model.track` for falls, with the tracker reset at the start of each video. Separate model objects per role (the contamination fix). Contains the **parity check**. Silences harmless tracker warnings. |
| `production_logic.py` | new | Line-for-line copies of the two alert rules that only exist inside the engine's main loop: the crowd alert state machine (confirm 3 cycles, hysteresis) and the fall alert hold (`FALL_CLEAR_SECONDS`). All thresholds are read from `config.py`. The parity check protects these copies. |
| `metrics.py` | rewritten | IoU, box matching (with ignore regions), precision/recall/F1, Wilson CIs, bootstrap CIs, count errors, COCO-style AP50, and MOTA/MOTP/IDF1 tracking metrics (own implementation, unit-tested). `crosscheck_motmetrics()` recomputes the tracking scores with the standard `motmetrics` package when it is installed; on MOT20 both give identical results. |
| `common.py` | new | Shared helpers: headline/pilot labelling, seeded sampling, detection scoring with CIs, and saving **error-example images** (green = truth, red = prediction, grey = ignore). |
| `provenance.py` | new | Records git commit, uncommitted files, config snapshot (secrets hidden), package versions vs the pinned ones, and machine details. |
| `report.py` | rewritten | Writes `summary.json` and the long-format `summary.csv` (value, CI, n, quality), and prints the console report. |
| `charts.py` | rewritten | All charts, using a colour-blind-safe palette, zero-based axes, error bars and sample sizes, with no misleading zoomed axes. |
| `http_utils.py` | changed | Added a remote file-size check (for the download budget) and **automatic retries** when a download connection drops. |

### Datasets (`eval_lib/datasets/`)
| File | Status | What it does now |
|---|---|---|
| `crowdhuman.py` | rewritten | Uses the **official** CrowdHuman validation annotations, a seeded random sample of 300 images, visible-body boxes and ignore regions. Gives precision/recall/F1, AP50, counting error and PR curve, plus comparisons (yolov8s; old tracked pipeline). The old mirror is kept only as a "pilot" with the person-class filter. |
| `mot20.py` | rewritten | All 429 frames of MOT20-01. Covers detection, counting and **tracking** (MOTA/IDF1) of the pose model, the one production actually tracks. Distractor classes are ignored. |
| `mall.py` | rewritten | All 2000 frames: counting, **zone density** on the real 3×4 grid (`zone_densities()`), and crowd alert accuracy on a view that actually crosses the threshold, plus 4 known-answer alert checks and comparison models. |
| `le2i.py` | rewritten | 20 clips per class, seeded and spread across scenes. Correct frame order. Results reported **per class** (Fall, Lie, Stand, Likefall, Blank), with empty-room clips excluded from the false-alarm rate. Run both at every frame and at production rate. |
| `ur_fall.py` | rewritten | Full recordings with labels marking the **fall onset**. Tests the alert at several analysis rates (1, 2, 2.8, 5 and 10 Hz, and every frame). Early alarms count as false alarms. Latency in **seconds**. Respects the 500 MB download budget (`--allow-large-downloads` fetches all 70). Measures the real cycle time on this machine. Also splits normal-activity false alarms by whether the dataset says the person lies down (`adl_false_alarms_by_lying_label`). |
| `whatif_detection.py` | **new** | NOT production. Measures what other settings *would* do on the same images: input size 960/1280 px, confidence 0.15, yolov8s at 960 px, and counting with the pose model's boxes. Reports recall/precision, Mall counting error and time per image, plus the time of one full inference cycle for faster variants. Each setting uses its own model copy so production is never affected. |
| `caucafall.py` | changed | Person-class filter, shared scoring, handles offline use. Pilot only. |
| `uhctd.py` | changed | Note pointing to the new tamper test (UHCTD needs a signed agreement). |
| `tamper_functional.py` | **new** (replaces `synthetic_tamper.py`) | 12 scenarios on **real Mall footage** through the real `TamperDetector` with current settings. Each is labelled with whether the result matches what the rules are designed to do. |
| `pets2009.py` | unchanged | Metadata only (no usable video). |

### Sweeps (`eval_lib/sweeps/`) — "what camera do you need?"
| File | Status | What it does now |
|---|---|---|
| `inference_rate_sweep.py` | **new** (replaces `fps_sweep.py`) | Fall detection at different analysis rates, using UR Fall results. |
| `resolution_sweep.py` | rewritten | Keeps the image shape. Widths 640/480/416/320/256/160 px. A minimum only counts if that size **and every larger size** pass. |
| `person_size_sweep.py` | rewritten | Recall by person height (as % of frame), visible people only, per dataset, with sample sizes. |
| `min_spec.py` | rewritten | Writes a plain-language camera requirement **only where the data supports it**; otherwise it says "insufficient data". |

### Tests, docs and outputs
| Item | What it is |
|---|---|
| `tests_eval/` | **New**: 31 unit tests for the suite itself (maths, frame sorting, cadence, alert rules, label parsing, model separation, parity, motmetrics agreement, what-if isolation, the lying split, and that every module imports). All pass. |
| `requirements-eval.txt` | **New**: exact package versions used. |
| `README.txt` | Rewritten plain-language guide with results and limitations. |
| `ROOT_README_UPDATE.md` | **New**: replacement text for the root README's evaluation section, to paste by hand. |
| `EVAL_PREP_GUIDE.md` | This file. |
| `model_eval_results/` | **Current results**: `summary.csv`, `summary.json`, `charts/` (14, incl. `whatif_detection.png`), `error_examples/`. |
| `model_eval_results_archive/2026-11-18_pre_audit/` | Old results, kept for history. **Do not quote.** |
| `model_eval_results_quick/`, `model_eval_results_pinned_quick/` | Output of short smoke tests. Not for quoting, and not pushed (listed in `model_testing_suite/.gitignore`). |
| `.venv-pinned/` | A separate Python 3.12 environment with the project's pinned versions (ultralytics 8.4.51, torch 2.8.0), used once for the version check. Not pushed. |
| `.gitignore` | **New**: keeps the scratch output folders and `.venv-pinned/` out of git. |

---

## 5. How one run works (step by step)

1. **Parity check.** Confirm `detector/engine.py` still makes the calls the suite mirrors. If not, stop (exit code 2).
2. **Seed everything** (random, numpy, torch) so the same samples are chosen.
3. **Record provenance** (commit, config, versions, machine).
4. **Warm up** the models (the first inference is slow and would distort timing).
5. **Run each dataset.** For every video, reset the tracker, replay frames at the chosen rate, call the real production functions, and compare with ground truth.
6. **Sweeps** (rate, resolution, person size) → minimum camera spec.
7. **Charts**, then `summary.csv` + `summary.json`.

---

## 6. The results explained

Format: value [95% CI], n. "Headline" means safe to quote; "pilot" means indicative only.

### 6.1 Person detection and counting (production: yolov8n, no tracking)
| Dataset | Result | Quality | What it means |
|---|---|---|---|
| **CrowdHuman**, 300 official images | Precision **0.92** [0.91–0.93] (n=3093); Recall **0.42** [0.40–0.43] (n=6848); F1 0.57; AP50 0.64; count MAE 12.4 per image | headline | Boxes are almost always right, but under half the people in crowd photos are found. |
| **MOT20-01**, 429 frames, 36–55 people/frame | Precision **0.95**; Recall **0.38**; count MAE **27.4** [25.5–29.1] (relative 0.59) | headline | In a very dense station crowd, it finds about 4 in 10 people. |
| **Mall**, 2000 frames | Count MAE **13.4** [12.8–14.1]; bias −13.4; relative MAE **0.43** | headline | It always under-counts, by about 43%. |
| CAUCAFall, 50 frames, 1 person | Precision 1.00, recall 0.76 | pilot | Fine on a single clear person, but too little data. |

**Why precision is high but recall is low:** the model only fires when it is
fairly sure, so it rarely invents people. But in crowds people overlap and are
small or partly hidden, so many are never confidently detected. The person-size
test below shows this directly.

**Comparison models (same images, not production):**
| Model | CrowdHuman recall | Mall MAE (first 500 frames) |
|---|---|---|
| Production (yolov8n, no tracking) | 0.42 | 11.9 |
| yolov8s (bigger model) | 0.46 | 11.0 |
| Old tracked pipeline (before 41ed380) | 0.21 | 24.1 |

→ Removing tracking from counting (commit 41ed380) **halved the counting error**.
A bigger model helps only a little.

### 6.2 Zones and crowd alert (Mall)
- Per-cell count error on the 3×4 grid: **1.49 people per cell**.
- "Red" (crowded) cells: precision **0.81**, recall **0.42**. A red cell is usually truly crowded, but over half of the truly crowded cells are not shown red, because of the same under-counting.
- Crowd alert on a view that crosses the threshold:
  - **98%** of real crowd episodes raised an alert [0.90–1.00], n=51;
  - **86%** of alerts were real [0.69–0.95], n=29;
  - alert state matched the truth in **81%** of frames.
- Known-answer alert checks (single spike, hold, clear, flapping on the threshold): **4/4 pass**.

### 6.3 Fall detection (headline)
**UR Fall**, **all 70 recordings** (30 falls, 40 normal activities = "ADL"), replayed at the live rate of 2.8 analyses per second:
- falls detected **10 of 30 = 0.33** [0.19–0.51];
- ADL recordings with a false alarm **12 of 40 = 0.30** [0.18–0.45];
  - **10 of those 12** are among the 16 ADL videos where the dataset's own labels say the person is **lying down on purpose** (e.g. lying on the floor or a bed). The fall rule fires on a lying posture, so these are "lying, but not a fall";
  - ADL videos with **no lying at all**: **2 of 24 = 0.08** [0.02–0.26];
- no fall video raised an alarm *before* the fall started (0 of 30);
- alert **1.5 s** after the fall *starts* [1.3–1.7], n=10 (about 0.5 s after the person is lying).

The same 70 recordings at other analysis rates:

| Rate | 1 Hz | 2 Hz | 2.8 Hz (live) | 5 Hz | 10 Hz |
|---|---|---|---|---|---|
| Falls caught (of 30) | 1 | 8 | 10 | 15 | **24** (0.80 [0.63–0.90]) |
| ADL with false alarm (of 40) | 8 | 11 | 12 | 13 | 13 |
| … of which no-lying ADL (of 24) | 0 | 1 | 2 | 3 | 3 |
| Latency from fall start (s) | 2.2 (n=1) | 1.8 | 1.5 | 1.3 | 1.0 |

→ **Analysing faster catches far more falls, and sooner**, with only a small rise in false alarms.
Every frame (30 Hz) was not repeated on all 70 videos (it would have taken ~4 more hours); on the earlier 13-video pilot it caught 7 of 8.

**Why the pilot looked better (4/8, 0/5):** every video it shared with the full run gave *exactly* the same result. The pilot's 13 videos just happened to be easier than the other 57. This is why small samples are labelled "pilot" and not quoted as percentages.

Comparison (NOT production): the smaller yolov8n-pose model at 2.8 Hz catches 13/30 falls with the same 12/40 false alarms.

**Le2i** (20 clips per class, every frame analysed):

| Class | Alerts | What it means |
|---|---|---|
| Fall | 16/20 (0.80 [0.58–0.92]) | Falls are usually caught. |
| Stand (normal activity) | 0/20 | No false alarms. |
| Lie (already lying) | 13/20 | Fires on posture; reported separately. |
| Likefall (fall-like movement) | 0/20 | Not fooled by fall-like moves. |
| Blank (empty room) | 0/20 | Sanity check passed. |

**At the production rate, Le2i catches 0/20 falls.** This does not mean the live
system misses every fall. Each Le2i clip is only 16 frames (0.64 s), which gives
just 2 analysed frames, and the alert needs 2 confirmed cycles. The clips are
simply too short for this test. UR Fall (full recordings) gives the live-rate
answer.

### 6.4 Tamper (12 scenarios, functional check)
| Result | Scenarios |
|---|---|
| Correctly no alarm | Normal moving footage. |
| Detected | Lens fully covered; hand over lens (bright, featureless); frozen feed; frozen feed through JPEG re-encoding; gradual dimming to black; covered then uncovered. |
| Missed (by design of current rules) | Half the lens covered; heavy blur; camera re-pointed to a different view. |
| False alarm | A genuinely still scene with only sensor noise (seen as frozen); a very dark scene (seen as covered). |

All **12/12 behave as the rules are designed**. This shows exactly what the
tamper feature can and cannot catch. It is not an accuracy rate.

### 6.5 Tracking (pose tracker, MOT20)
MOTA **0.13**, IDF1 **0.22**, 2 ID switches, 40 fragmentations. These are low
mainly because most people in this extremely dense scene are never detected.
The tracker cannot follow someone it never sees.

**Cross-check:** the same tracks scored with the standard `py-motmetrics`
package (1.4.0) give **identical** MOTA, IDF1, ID switches and fragmentations,
so our own implementation is confirmed.

### 6.5b What-if settings (NOT production)
Same 300 CrowdHuman images and first 500 Mall frames. Time is per image on this CPU.

| Setting | CrowdHuman recall | Precision | Mall relative error | ms / image |
|---|---|---|---|---|
| **Production** (yolov8n, 640 px, conf 0.25) | **0.42** | 0.92 | 0.43 | 601 |
| yolov8n, 960 px | 0.51 | 0.90 | 0.34 | 1126 |
| yolov8n, 1280 px | **0.58** | 0.90 | 0.31 | 1842 |
| yolov8n, 640 px, conf 0.15 | 0.49 | 0.86 | 0.34 | **539** |
| yolov8s, 960 px | 0.53 | 0.92 | 0.31 | 2372 |
| pose model's boxes as counter | 0.36 | 0.93 | 0.65 | 1210 |

→ The **input size** matters most (small people get more pixels), but it costs time.
→ **Lowering the confidence threshold to 0.15** is the cheapest gain: better recall
and Mall error at no extra time, with precision dropping from 0.92 to 0.86.
→ Using the pose model's boxes for counting would be *worse*.
→ Nothing reaches the recall ≥ 0.8 target.

Inference cycle time (detect + pose, Mall frames): production **1.55 s**; pose
model only **1.12 s**; pose at 480 px **1.16 s**. Only the time was measured
for these variants, not their effect on fall accuracy.

### 6.5c Version check (pinned ultralytics 8.4.51)
The project pins ultralytics 8.4.51 but the results were made with 8.4.138. A
quick run in a separate environment with the pinned versions gave **identical
numbers for 73 of 78 metrics**. The 5 that differ are all MOT20 tracking,
because 8.4.51's default tracker is BoT-SORT (8.4.138 uses TrackTrack):

| Tracking (quick run) | 8.4.138 | 8.4.51 |
|---|---|---|
| MOTA | 0.121 | 0.151 |
| IDF1 | 0.218 | 0.187 |
| ID switches | 0 | 286 |
| Fragmentations | 3 | 34 |

Detection, counting, zones, alerts, falls and tamper do not depend on the version.

### 6.6 Minimum camera requirements
| Question | Answer from the data |
|---|---|
| Camera fps? | VisionGuard analyses one frame per cycle. On this laptop a cycle took 1.0–1.5 s in three runs (0.65–1.0 per second). Any camera at ≥ 5 fps already gives more frames than are analysed. |
| Analysis speed for falls? | To keep fall recall within 90% of the best rate tested, the computer must analyse about **10 frames per second** (80% of falls caught). At 2.8 per second it is 33%. This is a requirement on the **computer**, not the camera. |
| Resolution? | 640 → 320 px costs little (CrowdHuman recall 0.44 → 0.41; Mall relative MAE 0.43 → 0.46). At 160 px it drops clearly. No size reaches the target, so **no minimum is claimed**. |
| Person size? | People ≥ 15% of frame height are found 81% of the time (MOT20, n=5849); ≥ 25% gives 97% (n=204). Below 8% recall falls sharply. → **Mount cameras so people fill at least about 15–25% of the frame height.** |

### 6.7 Old vs new
| Test | Old (do not quote) | New |
|---|---|---|
| CrowdHuman | P 0.66, R 0.57 (80 fake-labelled images) | P 0.92, R 0.42 (300 official) |
| MOT20 | P 0.99, R 0.24, MAE 28 (80 frames) | P 0.95, R 0.38, MAE 27.4 (429 frames) |
| Mall | MAE 18.9 (150 frames, tracked) | MAE 13.4 (2000 frames) |
| Crowd alert | "agreement 1.0" (meaningless, no negative frames) | 98% of episodes caught, 86% of alerts real |
| Le2i | P 0.89, 8% false alarms (empty rooms counted) | 16/20 falls, 0/20 standing clips |
| UR Fall | F1 0.80 (4 recordings) | All 70 recordings: 10/30 falls at live rate (24/30 at 10 Hz), 12/40 false alarms (10 of them deliberate lying), 1.5 s |
| Tamper | 6/6 correct (old settings) | 12 real-footage scenarios, strengths and gaps shown |
| Camera spec | 15/20 fps, 320×240, >15% height | Only person size is supported (15–25%) |

**Why some new numbers look "worse":** the old ones were inflated by fake labels,
easy empty-room clips and too few cases. Recall on CrowdHuman dropped because the
real annotations include many more small and hidden people. The new numbers are
lower but **true**.

---

## 7. Honest limitations
- **UR Fall is one dataset** (one room, a few actors, one camera angle). All 70 recordings were used, so its rates are headline, but other rooms may differ. The every-frame point exists only for the 13-video pilot.
- **"False alarm" includes deliberate lying.** Whether an alert for someone lying on the floor is wanted is a product decision; both splits are reported.
- **Le2i clips are too short** for the live-rate test, and their original fps is assumed (25).
- **Live speed:** this laptop managed 0.65–1.0 analyses per second (it varied between runs), slower than the 2.8 in `config.py`. Since fall recall drops steeply at low rates, the real live fall recall on this laptop is likely **below** the 2.8 Hz figure. Results are given at several rates so either can be read.
- **Mall** has under 2 fps, so each frame is treated as one cycle. The crowd-alert test uses the bottom-right quarter of the frame as a stand-in for a narrower camera. Zones compare head points (truth) with box centres (VisionGuard).
- **Tamper** uses real footage with artificial changes; real tamper recordings (UHCTD) need a signed agreement.
- **CrowdHuman** failed mid-run on a network reset. It was re-run from cached images with the same seed (together with MOT20 and the sweeps) and merged. MOT20 came out identical both times. A second merged re-run added the full UR Fall run, the what-if test and the motmetrics cross-check; every other number stayed identical. Both are recorded in `summary.json → provenance.partial_reruns`.
- **Ultralytics** used is 8.4.138; the project pins 8.4.51. Only tracking scores differ between them (see 6.5c).
- **What-if settings** are measurements, not changes; their effect on fall detection was not tested.
- **One machine only** (Snapdragon X Elite, CPU).

---

## 8. Why the results can be trusted
1. **Tests the real code.** The suite imports `classify_persons`, `confirm_falls`, `drop_nested_boxes`, `zone_densities` and `TamperDetector` from `detector/engine.py`, and reads every threshold from `config.py`.
2. **Parity guard.** If production changes, the suite refuses to run.
3. **Deterministic.** Datasets run in a different order gave identical numbers, and MOT20 was identical in two separate full runs.
4. **No data leakage.** The models are stock COCO weights (checked in the model file metadata), never trained on these datasets.
5. **Official ground truth**, with ignore regions respected.
6. **Every number has n and a 95% CI**, and is labelled headline or pilot.
7. **Provenance.** Commit, config and versions are saved with the results.
8. **Cross-checked.** Tracking scores match the standard `motmetrics` package exactly; the pinned ultralytics version gives the same non-tracking numbers.
9. **Tests pass.** 31/31 suite tests and 39/39 project tests. `git status` shows changes only inside `model_testing_suite/`.

---

## 9. Likely evaluation questions — with answers (34)

**Q1. What does your testing suite do?**
It measures VisionGuard's people counting, zone density, crowd alert, fall
detection and tamper detection on public, hand-labelled datasets. It calls the
real production code and reports every number with a confidence interval.

**Q2. Why is recall only about 0.4?**
Crowded scenes have small, overlapping, partly hidden people. yolov8n is a tiny,
fast model; it only reports people it is fairly sure about, so it misses many.
The person-size test confirms it: big people are found 81–97% of the time,
small ones rarely.

**Q3. Precision is 0.92 but recall 0.42. What does that mean?**
When it says "person" it's right 92% of the time, so there are few false alarms.
But it only finds 42% of the people who are there, so it under-counts.

**Q4. Your old results looked better. Why did they change?**
The old ones were flawed:
- the CrowdHuman labels were machine-generated and contained cars and umbrellas;
- the Le2i false-alarm rate included empty-room clips;
- the tracker leaked between tests;
- there were very small samples.

The new numbers use official labels, more data and correct methods.

**Q5. How do you know the test matches the live system?**
It imports the real functions instead of copying them. The two rules that only
exist inside the engine loop are mirrored line by line, and a parity check
verifies the engine still makes those exact calls before each run.

**Q6. Why didn't you change `engine.py` to make testing easier?**
The rule was not to touch production code. Testing must not change the thing
being tested.

**Q7. What is a confidence interval? Why Wilson?**
It is the range the true value probably lies in. Wilson is the recommended
method for proportions, because it stays correct with small samples and with
results of 0% or 100%, where the simple formula breaks.

**Q8. Why do you call some results "pilot"?**
They are based on fewer than 30 cases, so the interval is too wide to quote a
percentage. Example: the first UR Fall pilot gave 4 of 8 falls (CI 22–78%); the
full 70-video run then gave 10 of 30 (33%), inside that range but far from "50%".

**Q9. Why does Le2i show 0/20 falls at production rate?**
Each Le2i clip is only 0.64 s long, so at 2.8 analyses per second only 2 frames
are analysed. The fall rule needs 2 confirmed cycles of the same person. The
clips are too short; UR Fall (full videos) gives 10/30 at the same rate.

**Q10. What is the fall alert latency?**
About 1.5 s from the moment the fall starts (UR Fall, n=10, CI 1.3–1.7 s), about 0.5 s after the person is lying. At 10 Hz it is 1.0 s.

**Q11. Is there data leakage?**
No. The models are standard Ultralytics COCO weights. None of the test datasets
were used in training.

**Q12. What was the biggest bug you found?**
Tracker contamination. Calling `.track()` on a YOLO model permanently attaches a
tracker to that object, so the "untracked" production test was secretly
tracked. It was fixed with separate model objects per role and a runtime guard
that errors if it ever happens again.

**Q13. Why did removing tracking from counting help?**
A tracker only shows people after it has "confirmed" them over several frames,
so new or briefly hidden people disappear from the count. Plain detection counts
everyone visible. Mall error dropped from 24.1 to 11.9 on the same frames.

**Q14. Why not just use yolov8s?**
At the same 640 px it improves recall only slightly (0.42 → 0.46). At 960 px it
reaches 0.53 but takes about 4× longer per image (2.4 s vs 0.6 s) on a CPU-only
laptop, where VisionGuard already runs below its target rate. The what-if test
shows a larger input size or a lower confidence threshold with yolov8n gives
similar gains more cheaply.

**Q15. What camera do you recommend?**
Mount it so a person fills at least about 15–25% of the frame height. Any common
resolution (≥ 320 px wide) and frame rate (≥ 5 fps) is enough, because
VisionGuard only analyses about 1–3 frames per second.

**Q16. Can the tamper detector be fooled?**
Yes, and the tests show where:
- half-covering the lens, blurring it, or turning the camera away is not detected;
- a very still or very dark scene can cause a false alarm.

These are known design limits of the current brightness/motion rules.

**Q17. How is the crowd alert stopped from flickering?**
It needs ≥ 4 people for 3 cycles in a row to raise, and it only clears when the
count drops below 2 (hysteresis). The tests confirm a single spike never alerts
and flapping on the threshold raises only once.

**Q18. How did you measure tracking?**
With CLEAR-MOT (MOTA, MOTP, ID switches, fragmentation) and IDF1 on MOT20,
tracking the pose model (the only one production tracks), with the tracker reset
at the start.

**Q19. Are your results reproducible?**
Yes. There is a fixed seed and the tracker is reset per video. Re-running in a
different order gives identical numbers. The commit, config and package versions
are saved in `summary.json`.

**Q20. What is AP50 and why 101-point?**
Average precision over all confidence levels at IoU 0.5. The 101-point COCO
method is the standard, so our 0.64 can be compared with published numbers. The
old suite used a non-standard version.

**Q21. Why test at several analysis rates?**
The live system analyses only 1–3 frames per second, not every camera frame.
Testing every frame would overstate performance. The sweep shows the trade-off:
at 1 Hz 1/30 falls are caught, at 2.8 Hz 10/30, at 10 Hz 24/30, while false
alarms rise only from 8/40 to 13/40.

**Q22. What is the ground truth for Mall?**
Every person's head is marked as a point in all 2000 frames. The count is the
number of points; zone truth uses which grid cell each head falls in.

**Q23. What would you improve next?**
- Make the system analyse faster (pose-only cycles were measured at 1.12 s vs 1.55 s), since fall recall depends on it.
- Lower the detection confidence to 0.15 or raise the input size (measured gains, see 6.5b).
- Decide whether "person lying on the floor" should alert, and tune the fall rule accordingly.
- Fine-tune or use a crowd-trained detector to raise recall.
- Pin the tracker in production.
- Test on the target hardware.
- Get real tamper footage (UHCTD).
- Apply the README update.

**Q24. Why are some datasets skipped?**
- UHCTD needs a signed agreement.
- PETS2009's downloadable archive has no usable video.
- CAUCAFall's file index is broken, so only 50 frames could be fetched.

All of these are recorded honestly as SKIPPED or PARTIAL.

**Q25. How long does a full run take?**
About 4–5 hours on CPU including the what-if test (skip it with `--skip whatif_detection`). The full 70-video UR Fall run alone took about 4 hours. The quick smoke run takes about 25 minutes.

**Q26. What is an error example?**
A saved image where VisionGuard was wrong. Green is the truth, red is what it
detected. It is useful for seeing *why* it fails (small or hidden people).

**Q27. What's the difference between the fall classifier and the fall alert?**
The classifier decides per frame whether a person looks fallen (torso angle
> 65°, head level, leg veto). The alert needs the same tracked person fallen for
2 cycles in a row, and stays on until nobody has been down for 2 s.

**Q28. Why ignore some areas in CrowdHuman/MOT20?**
The dataset authors marked unclear areas (crowds too dense to label,
reflections, static mannequins) as "ignore". Counting detections there as wrong
would unfairly lower precision.

**Q29. Fall detection is only 33% at live speed. Isn't that bad?**
It is honest. The number depends heavily on how often a frame is analysed: 1/30
falls at 1 Hz, 10/30 at 2.8 Hz, 24/30 at 10 Hz. A fall lasts about a second, so at
2.8 analyses per second there are only 2–3 chances to see it, and the alert needs 2
confirmed cycles. The fix is a faster analysis loop, not a different camera.

**Q30. Why are there false alarms on normal-activity videos?**
10 of the 12 are videos where the person lies down on purpose (the dataset's own
labels say "lying"). VisionGuard's rule fires on a person lying flat, so it cannot
tell "lay down" from "fell down". Only 2 of 24 videos without any lying raised an
alarm. Whether "someone lying on the floor" should alert is a product decision.

**Q31. What is the "what-if" test? Did you change the system?**
No. It runs other settings on the same images, using separate model copies, to
show what a change *would* do: bigger input image (recall 0.42 → 0.58, 3× slower),
lower confidence threshold (0.42 → 0.49, same speed). Production is untouched,
and a unit test checks that what-if settings never reach the production models.

**Q32. How do you know your tracking metrics are correct?**
They are unit-tested, and on MOT20 they match the standard `py-motmetrics` package
exactly (MOTA 0.132, IDF1 0.221, 2 ID switches, 40 fragmentations).

**Q33. The project pins ultralytics 8.4.51 but you used 8.4.138. Does it matter?**
It was checked: a quick run with the pinned versions gave identical numbers for 73
of 78 metrics. Only MOT20 tracking differs, because the default tracker changed
(BoT-SORT in 8.4.51, TrackTrack in 8.4.138). Nothing else depends on the version.

**Q34. The pilot said 4/8 falls (50%); the full run says 33%. Which is right?**
The full run. Every video in both runs gave exactly the same result; the full run
simply added 57 harder videos. 33% lies inside the pilot's 22–78% interval, which
is exactly why pilot results are not quoted as percentages.

---

## 10. Next steps
1. Speed up the inference cycle or reduce work per cycle; fall recall rises from 33% at 2.8 Hz to 80% at 10 Hz.
2. Apply `ROOT_README_UPDATE.md` to the main README and correct the "ByteTrack" wording (it's actually the Ultralytics default tracker, `tracktrack.yaml`).
3. Align ultralytics with the pinned version, or update the pin (only tracking differs between them).
4. Consider a crowd-specialised or fine-tuned detector to raise recall.
5. Measure the live speed on the actual deployment laptop.
6. Commit the suite changes.

---

## 11. Command cheat-sheet
```bash
# use the Python that has ultralytics installed (on this machine: py -3.13)
python model_testing_suite/test_models.py --quick          # smoke run (~25 min)
python model_testing_suite/test_models.py                  # full run (~4-5 h)
python model_testing_suite/test_models.py --only mall le2i # selected datasets
python model_testing_suite/test_models.py --redraw-charts  # charts from saved results
python model_testing_suite/test_models.py --only ur_fall --allow-large-downloads --rates 1,2,2.8,5,10  # all 70 UR Fall videos (~4.2 GB, ~4 h)
python -m pytest model_testing_suite/tests_eval            # suite unit tests (31)
python -m pytest tests                                     # project unit tests (39)
```
