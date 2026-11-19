# Suggested update for the root README.md

The model testing suite was only allowed to change files inside
`model_testing_suite/`, so the root `README.md` still quotes the old,
superseded numbers. Below is replacement text to apply by hand.

## 1. Replace the whole "## Model Evaluation" section (README.md lines 420-442)

~~~markdown
## Model Evaluation

`model_testing_suite/` measures the production pipeline (it imports the real
functions from `detector/engine.py` and every threshold from `config.py`)
against public, hand-labelled datasets, without changing any VisionGuard code.
`model_testing_suite/README.txt` explains every test in plain language;
results, charts and error examples are in `model_testing_suite/model_eval_results/`.

```bash
python model_testing_suite/test_models.py           # full run, ~4-5 hours on CPU
python model_testing_suite/test_models.py --quick   # smoke run, ~25 minutes
```

Values are shown with a 95% confidence interval; "pilot" means too few cases
to quote as a rate.

| Dataset | What it tests | Result |
|---------|---------------|--------|
| CrowdHuman (300 official val images) | Person detection in crowds | Precision 0.92 [0.91-0.93], recall 0.42 [0.40-0.43], AP50 0.64 |
| MOT20-01 (429 frames) | Very dense crowd, counting, tracking | Precision 0.95, recall 0.38; count MAE 27.4 [25.5-29.1]; MOTA 0.13, IDF1 0.22 |
| Mall (2000 frames) | Crowd counting, zones, crowd alert | Count MAE 13.4 [12.8-14.1] (under-counts by ~43%); 98% of crowd episodes raised an alert (n=51) |
| UR Fall (all 70 recordings) | Fall detection at the live analysis rate (2.8 Hz) | 10/30 falls detected (0.33 [0.19-0.51]); 12/40 daily-activity recordings raised an alarm, 10 of them where the person lies down on purpose; alert 1.5 s after the fall starts. At 10 Hz: 24/30 falls (0.80) |
| Le2i (20 clips per class, pilot) | Fall vs. daily activity | 16/20 fall clips alerted, 0/20 standing clips |
| Tamper (12 scenarios) | Covered, frozen, blurred, moved camera | 12/12 behave as designed; half-cover, blur and re-pointing are not detected |
| CAUCAFall, PETS2009, UHCTD | — | Pilot or unavailable (dataset access limits, documented in the results) |

Main findings: detections are almost always correct, but in dense crowds only
about 40% of people are found, so counts run low. People should fill at least
about 15-25% of the frame height. Reducing resolution from 640 to 320 px
costs little. Raising the detector input size or lowering the confidence
threshold to 0.15 would find more people (measured, not yet applied). Fall
detection depends strongly on how often frames are analysed: about 1 fall in
3 at the configured 2.8 Hz, 4 in 5 at 10 Hz.
~~~

## 2. Tracker name (README.md lines 138 and 144)

Both lines say the pose model is tracked with **ByteTrack**. The code calls
`pose_model.track(persist=True)` without naming a tracker, so it uses the
installed ultralytics default. With the installed ultralytics 8.4.138 that
default is `tracktrack.yaml`; with the pinned 8.4.51 it is `botsort.yaml`
(BoT-SORT). Neither is ByteTrack. Either say "the ultralytics
default tracker", or pin `tracker="bytetrack.yaml"` in `detector/engine.py`
if ByteTrack is what is wanted (a code change outside the suite, so it is not
made here).
