===============================================================================
VISIONGUARD MODEL TESTING SUITE
===============================================================================

This folder measures how well VisionGuard's computer vision works, using
public datasets that other researchers labelled by hand, and works out what
camera set-up VisionGuard needs. It never changes VisionGuard itself: it
IMPORTS the real detection functions from detector/engine.py and reads every
threshold from config.py, so it tests exactly the code that runs live.

Technical terms are explained the first time they appear.

Results of the current run:   model_eval_results/
  summary.csv      one row per number: value, 95% confidence interval, sample
                   size (n) and quality label
  summary.json     everything, including exactly which code, settings and
                   software versions produced it ("provenance")
  charts/          the graphs
  error_examples/  frames where the system was wrong (green = truth,
                   red = what VisionGuard detected), for error analysis
Earlier, superseded results (do not quote): model_eval_results_archive/


===============================================================================
1. HOW TO READ THE NUMBERS
===============================================================================

PRECISION  of everything VisionGuard flagged, the fraction that was right.
           Low precision = many false alarms.
RECALL     of everything that really was there, the fraction VisionGuard
           caught. Low recall = many misses.
F1         one number balancing precision and recall (1.0 is perfect).
MAE        mean absolute error: on average, how many people the head-count
           was off by. RELATIVE MAE divides that by the true average count
           (0.40 = off by 40% of the crowd). BIAS is the signed average:
           negative means VisionGuard UNDER-counts.
95% CONFIDENCE INTERVAL (CI), written [low-high]
           the range the true value very probably lies in, given how much
           data was tested. A wide range means "not enough data to be sure".
n          how many cases a number is based on.
QUALITY    "headline" = based on at least 30 cases (safe to quote with its
           CI); "pilot" = fewer, indicative only - do not quote as a rate.
MOTA / IDF1  standard tracking scores: MOTA combines misses, false boxes and
           identity mix-ups; IDF1 measures how consistently each person keeps
           the same ID number. 1.0 is perfect for both.


===============================================================================
2. WHAT IS TESTED, AND HOW
===============================================================================

Every test runs the production pipeline exactly as detector/engine.py does:

  Counting people   yolov8n detection, confidence >= DETECTION_CONF_MIN,
                    then the engine's own duplicate-box removal. No tracking
                    (production stopped tracking for counting in commit
                    41ed380).
  Fall detection    yolov8s-pose with the tracker, then the engine's own
                    classify_persons() and confirm_falls(), plus the alert's
                    FALL_CLEAR_SECONDS hold.
  Crowd alert       CROWD_MIN_PERSONS for CROWD_CONFIRM_CYCLES cycles to
                    raise, CROWD_CLEAR_MARGIN hysteresis to clear.
  Zones             the engine's own zone_densities() on the 3x4 grid.
  Tamper            the engine's own TamperDetector, current settings.

The two alert rules that only exist inside the engine's main loop are copied
line-for-line into eval_lib/production_logic.py. Before every run the suite
re-reads detector/engine.py and REFUSES TO RUN if any of the mirrored code has
changed, so the tests can never silently drift away from the real system.

Datasets
  CrowdHuman (official val split)  person detection in crowds, with
        occlusion. Official hand annotations; areas the annotators marked
        "ignore" are not scored either way.
  MOT20-01   a very dense train-station scene (36-39 labelled people per
        frame, 429 frames): detection, counting and TRACKING.
  Mall       a shopping-mall camera, 2000 consecutive frames with every head
        marked: counting, zone density, and the crowd-alert logic.
  Le2i       short 16-frame clips in five classes (Fall, Lie, Stand,
        Likefall, Blank): fall detection and false alarms.
  UR Fall    full fall and daily-activity recordings at 30 fps with a
        per-frame label, including the moment the fall STARTS: fall
        detection, false alarms and how many seconds an alert takes.
  CAUCAFall  50 frames of one person (pilot only - its file index is broken).
  PETS2009, UHCTD   could not be used (no video / signed agreement needed).
  Tamper     12 scenarios built from real footage (see section 4).

Dataset sources
  The datasets are NOT stored in this repository (they are large and contain
  footage of real people). test_models.py downloads what it can into
  model_testing_suite/external_datasets/, which is git-ignored. Sources:
  CrowdHuman  https://huggingface.co/datasets/sshao0516/CrowdHuman
  MOT20       https://motchallenge.net/data/MOT20/
              (MOT20.zip and MOT20Labels.zip from https://motchallenge.net/data/)
  Mall        https://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/mall_dataset.zip
  Le2i        https://drive.google.com/file/d/16sv5CLT3pBI0kcLvAYJqT_zscGK-GF91/view
  UR Fall     http://fenix.ur.edu.pl/~mkepski/ds/uf.html
              (files under http://fenix.ur.edu.pl/~mkepski/ds/data/)
  CAUCAFall   https://data.mendeley.com/datasets/7w7fccy7ky/4
  PETS2009    http://visal.cs.cityu.edu.hk/static/downloads/pets2009_feats.zip
  UHCTD       https://pmantini.github.io/profile/projects/uhctd/
              (devkit: https://github.com/QuantitativeImagingLaboratory/ctd-devkit)
  Use of each dataset is subject to its own licence / terms.

None of these datasets were used to train VisionGuard's models: they are the
standard Ultralytics COCO weights (checked from the model files' metadata).

Production cadence
  Live, VisionGuard does not analyse every camera frame: it analyses the
  newest frame each time the previous analysis finishes (about 2.8 times per
  second on the target laptop CPU; measured on this machine in the results).
  Fall tests are therefore replayed BY TIME at that rate as well as on every
  frame. The "production cadence" numbers are the ones that describe the live
  system.


===============================================================================
3. WHAT CHANGED AFTER THE AUDIT (2026-11-18)
===============================================================================

The earlier results (now in model_eval_results_archive/2026-11-18_pre_audit/)
must not be quoted. An audit found that:

  - counting was tested with tracking, which production no longer uses;
  - the "CrowdHuman" labels were not CrowdHuman's annotations (they contained
    cars, umbrellas, chairs and ties, all counted as people);
  - the tracker's memory leaked between clips and tests, so the same frames
    gave different scores depending on what ran before;
  - 3 of the Le2i clips were analysed with their frames out of order;
  - half the Le2i "normal activity" clips were an empty room, which made the
    false-alarm rate look half as large as it was;
  - the frame-rate test compared identical frame sets (20 vs 25 fps) and
    could not apply to a system that analyses ~3 frames per second anyway;
  - the resolution test squashed the picture's shape, and its "minimum" was
    noise;
  - the tamper test still used the old 15-frame frozen-feed setting
    (production now uses 1440 frames);
  - no number had a confidence interval.

All of these are fixed (details in the module docstrings under eval_lib/).
One more problem was found and fixed while re-running: calling the tracker on
a model permanently attaches it to that model, which had silently re-applied
tracking to "untracked" counts. Production and comparison models now use
separate instances, and the suite checks this on every call.


===============================================================================
4. RESULTS
===============================================================================

Full run of 2026-11-18, commit 41ed380, ultralytics 8.4.138, CPU only
(Snapdragon X Elite laptop), plus two merged re-runs with the same commit,
settings and seed (listed in summary.json -> provenance.partial_reruns):
round 1 re-ran CrowdHuman after a network failure; round 2 (2026-11-19)
added the full 70-video UR Fall run, the what-if comparison and the
motmetrics cross-check. Format: value [95% CI], n = cases, quality.
Exact numbers for everything: model_eval_results/summary.csv.

PERSON DETECTION AND COUNTING (production pipeline: yolov8n, no tracking)
  CrowdHuman, 300 official val images (seeded random sample)        headline
    precision 0.92 [0.91-0.93]  n=3093 detections
    recall    0.42 [0.40-0.43]  n=6848 visible people
    F1 0.57 [0.53-0.61];  AP50 (COCO-style, 101-pt) 0.64
    count error per image: MAE 12.4 [10.2-14.8]  (relative MAE 0.54)
  MOT20-01, all 429 frames (36-55 labelled people per frame)          headline
    precision 0.95 [0.94-0.95]  n=7999;  recall 0.38 [0.38-0.39]  n=19870
    count MAE 27.4 [25.5-29.1]  (relative MAE 0.59)
  Mall, all 2000 frames                                              headline
    count MAE 13.4 [12.8-14.1], bias -13.4 (always UNDER-counts),
    relative MAE 0.43 [0.42-0.44]
  CAUCAFall, 50 frames, 1 person (pilot): precision 1.00, recall 0.76

  In plain words: when VisionGuard draws a box around someone it is almost
  always a real person (precision 0.92-0.95), but in crowded scenes it only
  finds about 40% of the people (recall ~0.4), so crowd counts come out
  roughly 40-60% too low. Small and partly hidden people are the ones missed
  (see "Person size" below).

  Comparison models (NOT production, same images):
    yolov8s instead of yolov8n:  CrowdHuman recall 0.46, F1 0.61;
                                 Mall MAE 11.0 vs 11.9 (first 500 frames)
    old tracked pipeline (before commit 41ed380): CrowdHuman recall 0.21,
                                 Mall MAE 24.1 vs 11.9 (first 500 frames)
  -> the switch away from tracking in 41ed380 roughly halved the counting
     error; a larger model would help only a little.

ZONES AND CROWD ALERT (Mall, 2000 frames)                            headline
  Per-cell count error on the 3x4 grid: MAE 1.49 people per cell.
  "Red" cells (>= CROWD_THRESHOLD people): precision 0.81 [0.80-0.83],
  recall 0.42 [0.41-0.43] - a red cell is usually right, but more than half
  of the truly crowded cells are not shown red (same under-counting).
  Crowd alert on a view that crosses the threshold (bottom-right quarter of
  the Mall frame, chosen from the ground truth only):
    frames where the alert state matches the truth  0.81 [0.79-0.82], n=2000
    true crowd episodes that raised an alert        0.98 [0.90-1.00], n=51
    alerts that matched a true episode              0.86 [0.69-0.95], n=29
  Alert-rule known-answer checks (spike, hold, clear, flapping): 4/4 passed.

FALL DETECTION                                                     headline
  UR Fall, ALL 70 recordings (30 falls, 40 daily activities = "ADL"),
  replayed at the production rate of 2.8 analyses per second:
    falls detected          10/30 = 0.33 [0.19-0.51]
    ADL recordings with a false alarm  12/40 = 0.30 [0.18-0.45]
      - of which 10 are among the 16 ADL recordings where the dataset's own
        labels show the person LYING DOWN on purpose (10/16); the fall rule
        fires on a lying posture, so these are "lying, but not a fall".
      - ADL recordings with no lying at all: 2/24 = 0.08 [0.02-0.26]
    no alarm before any fall started: 0/30 fall recordings
    time from the START of the fall to the alert  1.5 s [1.3-1.7], n=10
    (0.5 s after the person is first lying on the floor)
  The same recordings analysed faster (sweep, see chart):
    falls detected:  1 Hz 1/30 | 2 Hz 8/30 | 2.8 Hz 10/30 | 5 Hz 15/30 |
                     10 Hz 24/30 = 0.80 [0.63-0.90]
    ADL false alarm: 8/40 | 11/40 | 12/40 | 13/40 | 13/40
    alert time:      2.2 s (n=1) | 1.8 s | 1.5 s | 1.3 s | 1.0 s
    Every frame (30 Hz) was not repeated on all 70; on the earlier 13-video
    pilot it found 7/8 falls (kept in summary.json for reference only).
  In plain words: fall detection depends strongly on how often a frame is
  analysed. At the configured 2.8 Hz it catches about 1 fall in 3; at 10 Hz
  about 4 in 5. Most false alarms are people deliberately lying down.
  The earlier 13-video pilot (4/8 falls, 0/5 false alarms) was too small and,
  as it turned out, unrepresentative: every recording it shared with the full
  run gave exactly the same result, the other 57 recordings were harder.
  Le2i, 20 clips per class, every frame analysed:
    Fall clips alerted 16/20 = 0.80 [0.58-0.92]
    Stand (daily activity) clips alerted 0/20 = 0.00 [0.00-0.16]
    Lie (already lying) 13/20; Likefall 0/20; Blank (empty room) 0/20
  Le2i at the production rate detects 0/20 falls: each clip is only 16
  frames (0.64 s), which gives just 2 analysed frames - too short for the
  2-cycle confirmation rule. This says the clips are too short, not that the
  live system misses every fall; use UR Fall for the live-rate result.
  Comparison (NOT production): yolov8n-pose on UR Fall at 2.8 Hz detects
  13/30 falls, with the same 12/40 ADL false alarms.

TAMPER (12 scenarios on real Mall footage, real TamperDetector,
        current settings; a functional check, not an accuracy rate)
  Normal moving footage: correctly no alarm.
  Detected: lens fully blacked out, hand over lens (auto-exposed bright and
            featureless), frozen feed,
            replayed JPEG frame, gradual dimming to black, cover then uncover.
  Not detected (by design of the current rules): half the lens covered,
            heavy blur, camera re-pointed to a different view.
  False tamper alarms: a genuinely still scene with only sensor noise
            (reported as frozen), a very dark scene (reported as covered).
  All 12 outcomes match what the current rules are designed to do.

TRACKING (pose-model tracker, MOT20-01, 429 frames)                  headline
  MOTA 0.13, IDF1 0.22, 2 identity switches, 40 fragmentations. Low because
  most people in this very dense scene are not detected at all.
  Cross-checked with the standard py-motmetrics package (1.4.0): identical
  MOTA, IDF1, identity switches and fragmentations.

WHAT-IF SETTINGS (NOT production - what a settings change WOULD do)
  Same 300 CrowdHuman images and the first 500 Mall frames; model time per
  image on this CPU. Production = yolov8n, 640 px input, confidence 0.25.
                               CrowdHuman  CrowdHuman  Mall rel.  ms per
                               recall      precision   MAE        image
    production                 0.42        0.92        0.43        601
    yolov8n, 960 px            0.51        0.90        0.34       1126
    yolov8n, 1280 px           0.58        0.90        0.31       1842
    yolov8n, conf 0.15         0.49        0.86        0.34        539
    yolov8s, 960 px            0.53        0.92        0.31       2372
    pose model's boxes         0.36        0.93        0.65       1210
  (all recall CIs are about +/-0.01; Mall MAE CIs about +/-0.03; see csv)
  Reading it: a larger input image is the change that helps counting most
  (recall 0.42 -> 0.58 at 1280 px) but triples the model time. Lowering the
  confidence threshold to 0.15 is a free improvement (0.42 -> 0.49, Mall
  error 0.43 -> 0.34) at a small cost in precision (0.92 -> 0.86). Counting
  with the pose model's boxes instead would be worse. None reaches the
  recall >= 0.8 target.
  Inference cycle (detect + pose, Mall frames, this CPU): production 1.55 s;
  pose model only 1.12 s; pose at 480 px 1.16 s. Only timing was measured
  for these variants, not their fall accuracy.

MINIMUM CAMERA REQUIREMENTS (only where the data supports a statement)
  Frame rate: production analyses one frame per inference cycle. On this
    laptop a cycle took 1.0-1.5 s (three runs), i.e. about 0.65-1.0
    analyses per second, below the 2.8 Hz in config.py. Any camera at 5 fps
    or more already supplies more frames than are analysed.
  Analysis rate for falls: keeping fall recall within 90% of the best rate
    tested needs 10 analyses per second (80% recall). At 2.8 Hz recall is
    0.33. This is a requirement on the computer, not on the camera.
  Resolution: reducing the frame width from 640 to 320 px costs little
    (CrowdHuman recall 0.44 -> 0.41; Mall relative MAE 0.43 -> 0.46); at 160
    px it drops clearly (0.35; 0.56). No resolution reaches the target
    (recall >= 0.8 / relative MAE <= 0.2), so no minimum is claimed.
  Person size (visible people only): on MOT20, people at least 15% of the
    frame height are found 81% of the time (n=5849) and those over 25% 97%
    (n=204); below 8% recall falls sharply. CrowdHuman shows the same trend
    but only reaches 80% for people over 25% of frame height (n=1852).
    -> Mount cameras so a person fills at least about 15-25% of the
       frame height.


===============================================================================
5. HOW TO RUN IT
===============================================================================

Use the Python that has VisionGuard's packages installed (on the development
machine: Python 3.13, `py -3.13`). Versions used: requirements-eval.txt.
Works from any folder:

    python model_testing_suite/test_models.py --quick   (smoke test, ~25 min)
    python model_testing_suite/test_models.py           (full run, ~4-5 hours incl. what-if)

Useful options
    --only crowdhuman mall        run only these datasets, in this order
    --skip-sweeps                 skip the camera-requirement sweeps
    --redraw-charts               redraw charts from an existing summary.json
    --max-download-mb 500         budget for new UR Fall downloads (default)
    --allow-large-downloads       fetch all 70 UR Fall recordings (~4.2 GB)
    --rates 1,2,2.8,5,10          UR Fall analysis rates (default also
                                  includes every frame)
    --skip whatif_detection       skip the what-if comparison (~1.5 h)
    --no-baselines                skip the comparison models
    --seed 0                      random seed for every sample drawn

Unit tests for the suite itself (no models run):
    python -m pytest model_testing_suite/tests_eval

Re-running with the same seed gives identical numbers (verified by running
the datasets in a different order and without the comparison models).


===============================================================================
6. LIMITATIONS
===============================================================================

- UR Fall now uses all 70 recordings (30 falls, 40 ADL), so its rates are
  headline. It is still ONE dataset: one room, a few actors, one camera
  angle. The default run (500 MB download budget) only uses 13 recordings;
  to reproduce the full result (~4.2 GB of zips, ~4 hours on this laptop):
      python model_testing_suite/test_models.py --only ur_fall --allow-large-downloads --rates 1,2,2.8,5,10
  The every-frame (30 Hz) point was not repeated on all 70 to save ~4 hours;
  only the 13-video pilot value exists for it.
- UR Fall counts any alert on a daily-activity recording as a false alarm,
  including recordings where the person lies down on purpose. Whether an
  alert for "someone lying on the floor" is wanted is a product decision;
  both splits are reported in summary.csv.
- Le2i clips are only 16 frames long and their original frame rate is not
  documented (25 fps assumed). They are useful for "does it fire on a fall
  vs on normal activity" with every frame analysed, but cannot test the
  live analysis rate.
- The live analysis rate on this laptop (1.0-1.4 s per cycle, measured
  twice) is slower than the 2.8 Hz in config.py, and it varied between runs.
  Fall results are reported at 2.8 Hz and at the other rates so either can
  be read off.
- Mall has fewer than 2 frames per second, so the crowd-alert test treats
  each Mall frame as one analysis cycle. The threshold-crossing test uses
  the bottom-right quarter of the frame as a stand-in for a camera with a
  narrower view; zones are scored with head points (truth) against box
  centres (VisionGuard).
- Tamper scenarios are real footage with artificial manipulation (covering,
  freezing, blurring, shifting). Real covered/moved camera recordings
  (UHCTD) need a signed agreement and were not used.
- CAUCAFall (one person, 50 frames) and the CrowdHuman mirror set are pilot
  only; PETS2009 provides no usable video in the downloadable archive.
- CrowdHuman (official) failed mid-run on a network reset; it was re-run
  from cached images with the same seed together with MOT20 and the camera
  sweeps, and merged. MOT20 gave identical numbers both times. Recorded in
  summary.json -> provenance.partial_rerun.
- Tracking scores come from the suite's own MOTA/IDF1 code and are
  cross-checked against py-motmetrics when it is installed (identical on
  MOT20). motmetrics is optional; without it the cross-check is skipped.
- Results were produced with ultralytics 8.4.138; VisionGuard's
  requirements.txt pins 8.4.51. A --quick run in a separate environment with
  the pinned versions (ultralytics 8.4.51, torch 2.8.0, numpy 2.0.2, Python
  3.12; .venv-pinned/, not pushed) gave identical numbers for 73 of 78 quick
  metrics. The 5 that differ are all MOT20 tracking, because 8.4.51's default
  tracker is BoT-SORT instead of TrackTrack: MOTA 0.15 vs 0.12, IDF1 0.19 vs
  0.22, identity switches 286 vs 0. Detection, counting, zones, alerts, fall
  and tamper results do not depend on the version tested.
- The what-if settings were measured on the same images as production but
  are NOT what the system runs; their fall-detection effect was not tested.
- All timing was measured on one machine (Snapdragon X Elite, CPU only).
