===============================================================================
VISIONGUARD MODEL TESTING SUITE - WHAT THIS IS AND WHAT WAS DONE
===============================================================================

This folder contains a complete, independent testing system that checks how
accurate VisionGuard's actual computer-vision models are, using real datasets
built by other researchers - not just VisionGuard's own demo videos - and
also works out the minimum camera quality (frame rate, resolution, distance)
VisionGuard actually needs to work reliably.

Nothing in this folder changes how VisionGuard itself works. It only reads
VisionGuard's existing detection code to test it; it never edits it.

This document is written so that someone with no technical background can
read it and understand what was built, why, what was found, and what the
results mean. Any technical term is explained the first time it appears.


===============================================================================
1. WHAT WAS ASKED FOR
===============================================================================

VisionGuard is a surveillance system that watches camera footage and tries to
automatically detect three things: too many people gathering in one place
("crowd detection"), a person falling down ("fall detection"), and someone
tampering with the camera itself, such as covering the lens or freezing the
video feed ("camera tamper detection"). It also tracks how many people are in
different areas of a scene ("zone density").

The request was to actually test how accurate these detections are - with
real numbers and graphs, not guesses - using datasets that other researchers
have already labelled by hand (so there is a known "right answer" to compare
VisionGuard's output against). On top of that, there was a separate question
from a mentor: what is the minimum camera setup (how many pictures per
second the camera needs to capture, how sharp/detailed the picture needs to
be, and how close the camera needs to be to people) for VisionGuard to work
at all.


===============================================================================
2. WHY DATASETS FROM OUTSIDE VISIONGUARD WERE NEEDED
===============================================================================

VisionGuard ships with a few of its own demonstration videos. The problem is
that nobody ever sat down and manually counted exactly how many people are in
each frame of those videos, or marked the exact moment someone falls. Without
that known "right answer" (called GROUND TRUTH in this field), there is no
way to calculate real accuracy - you would just be comparing VisionGuard's
guess to... nothing.

Outside researchers, however, have already built and manually labelled
datasets specifically for testing things like "how many people are in this
picture" or "did a fall happen in this video". Using those lets us compare
VisionGuard's answer to a real, independently verified answer, and get an
honest accuracy score.


===============================================================================
3. THE 8 TESTS USED, AND WHY EACH ONE WAS CHOSEN
===============================================================================

1. CROWDHUMAN - A dataset of photos with every person's position marked by
   hand, specifically built to be hard (crowded scenes, people partly hidden
   behind each other). Used to test VisionGuard's basic "find every person in
   this picture" ability (PERSON DETECTION).

2. MOT20-01 - One scene from a benchmark called MOT20, filmed in a real,
   very crowded train station, with every person marked AND given a tracking
   number that follows them across frames. Used to test both person
   detection and VisionGuard's ability to keep track of the same person
   across multiple frames (TRACKING), plus how VisionGuard's crowd-alert
   rule performs on a genuinely crowded scene.

3. MALL DATASET - Footage from a fixed camera in a shopping mall corridor,
   with the number of people in every single frame counted by hand. Used to
   test how accurately VisionGuard counts people in a less extreme, more
   "every day" crowd scene (closer to how VisionGuard's own cameras are
   typically positioned) than MOT20-01's train-station crush.

4. PETS2009 - A well-known older research benchmark for crowd counting,
   originally filmed from multiple cameras with marked-out zones. Only the
   supporting measurement files could be obtained here, not the original
   video (explained in section 4) - so this one did not produce a usable
   comparison, but its region/zone definitions were still downloaded for
   reference.

5. LE2I - A dataset of short video clips showing people either falling down
   or doing normal activities (walking, sitting, picking something up),
   recorded in a coffee room and a home setting. Used to test VisionGuard's
   fall-detection logic against real fall and non-fall footage.

6. UR FALL DETECTION DATASET - A second, independent fall-detection dataset
   from the University of Rzeszow, Poland, with a genuine frame-by-frame
   label of exactly which video frames show a person "lying on the ground".
   This was added specifically to double-check Le2i's fall-detection result
   with a completely separate dataset, and because it has a real labelled
   starting point for each fall it also allowed measuring DETECTION LATENCY -
   how many video frames after the fall actually started does VisionGuard
   take to raise its alert.

7. CAUCAFALL - A Colombian research dataset of real people falling and doing
   normal activities, built specifically to have varied lighting and camera
   angles. The plan was to use it as a second check on fall detection
   (alongside Le2i and UR Fall), but its online file index turned out to be
   broken (explained in section 4), so it ended up only testing basic person
   detection, not fall detection specifically.

8. SYNTHETIC TAMPER TEST - Not a downloaded dataset at all. Because the
   proper camera-tampering dataset (UHCTD, see below) could not be used
   automatically, this test instead builds its own simple, controlled
   examples directly in code: a normal, moving video feed; a feed that goes
   dark (simulating a covered lens); and a feed that freezes on one frame
   (simulating a disconnected or hacked camera). Because these examples are
   built on purpose, the "right answer" is known exactly, which makes this
   one of the most reliable tests in the whole suite.

A ninth dataset, UHCTD (University of Houston Camera Tampering Dataset), was
the originally intended camera-tamper dataset but could not be used - see
section 4.


===============================================================================
4. WHAT WENT WRONG, AND HOW IT WAS HANDLED HONESTLY
===============================================================================

Not everything went smoothly, and rather than hide that or quietly make up
numbers, every problem was documented and handled so the final report always
tells the truth about what could and could not be tested.

- CROWDHUMAN'S OFFICIAL DOWNLOAD LINKS WERE DEAD. The dataset's own website
  pointed to Google Drive files that, when actually tried, returned a "file
  not found" error. A different, verified-working copy of the same dataset
  (re-packaged by someone else on a public research-data site called
  Mendeley) was used instead, and this substitution is written directly into
  the test code so anyone reading it knows exactly what happened.

- PETS2009'S DOWNLOAD DID NOT CONTAIN THE ACTUAL VIDEO. Its public download
  package only contains supporting measurement files (region definitions and
  pre-calculated statistics), not the original camera footage. Without the
  footage, VisionGuard cannot be run against it at all, so this dataset is
  marked as "PARTIAL" in the results rather than given a fake accuracy score.

- CAUCAFALL'S ONLINE FILE LISTING WAS BROKEN. The website that lists
  CAUCAFall's files is supposed to let you browse through all of them, but
  it kept returning only the very first small batch no matter what was
  asked for, with no way to see which files belonged to a "fall" versus a
  "normal activity". Because of this, only a small sample could be tested,
  and only for basic person detection - not fall detection, since there was
  no way to know which of those few files were falls. This is clearly
  labelled "PARTIAL" in the results, explaining exactly why.

- UHCTD (the camera-tampering dataset) REQUIRES SIGNED PERMISSION. Its
  creators require anyone wanting the data to sign a formal agreement first -
  it cannot simply be downloaded. Rather than skip camera-tamper testing
  entirely, the Synthetic Tamper Test (described above) was built to cover
  this gap with its own controlled, known-correct examples.

- SLOW INTERNET CONNECTION. The actual download speed available while
  building this was roughly 0.3-0.4 megabytes per second - quite slow by
  modern standards. Several of these datasets are many gigabytes in size if
  downloaded in full (one is 4.68 GB, another is 8.4 GB). Downloading them
  in full would have taken many hours. Instead, a technique was used to pull
  out only the small number of files actually needed from inside these large
  files without downloading the whole thing - similar to opening one page of
  a very large book without having to photocopy the entire book first. This
  is why each test only uses a modest sample (tens to low hundreds of
  images/frames) rather than the full dataset - it can be scaled up later by
  re-running with different settings if more certainty is wanted, at the
  cost of more download time.

In every one of these cases, the rule followed was: never invent a number.
If a real test could not be completed, the result says so plainly instead of
guessing.


===============================================================================
5. THE MINIMUM-CAMERA-REQUIREMENTS TESTS
===============================================================================

Separately from accuracy testing, three extra checks were run to answer the
mentor's question about minimum camera requirements:

FRAME RATE (FPS - "frames per second") is how many separate pictures a
camera captures every second. A higher number means smoother, more complete
video; a lower number means the camera might miss fast events because it
simply wasn't "looking" at that exact moment. This was tested by artificially
making the same real footage look like it came from a slower camera (by
skipping frames) and checking whether VisionGuard's crowd and fall detection
still worked.

RESOLUTION is how sharp/detailed each picture is - a higher resolution means
more visible detail per person, a lower resolution means people become
blurrier, smaller blobs of pixels that are harder for the software to
recognise as a person at all. This was tested by shrinking the same real
footage down to several lower resolutions and re-testing detection accuracy
at each size.

PERSON SIZE IN FRAME is about how close/zoomed-in the camera needs to be - if
a camera is mounted too far away or too wide-angle, each person only takes up
a handful of pixels, making them nearly invisible to the detection software
even at a decent resolution. This was tested by grouping real, hand-labelled
people in the test footage by how tall they appear as a percentage of the
whole picture's height, and checking detection accuracy for each size group.

WHAT WAS FOUND (exact numbers, calculated at a "90% of the best possible"
cut-off - meaning the point where performance has not yet dropped by more
than 10% from its best observed level):

  - Minimum frame rate for reliable CROWD detection: about 15 frames per
    second.
  - Minimum frame rate for reliable FALL detection: about 20 frames per
    second. Below this, a real fall (which only lasts a fraction of a
    second) may not be captured in enough separate frames for VisionGuard's
    "confirm it happened more than once before raising an alarm" safety rule
    to ever be satisfied - meaning a fall could be physically missed no
    matter how good the detection software is, purely because too few
    pictures were taken during it.
  - Minimum resolution: roughly 320x240 pixels (an old, low resolution by
    today's standards), based on the shopping-mall footage, which gave a
    clearer answer than the ultra-crowded train-station footage (in that
    scene, the camera was so packed with overlapping people that resolution
    stopped mattering - the real problem there was simply too many people
    blocking each other, not the resolution of the picture).
  - Minimum person height in frame: a person needs to take up more than 15%
    of the picture's height before VisionGuard reliably recognises them;
    below 8% of the picture's height, the detection software essentially
    never recognised a person at all in this testing.


===============================================================================
6. WHAT THE RESULTS ACTUALLY MEAN, FEATURE BY FEATURE
===============================================================================

A few short terms used below:
  PRECISION = out of everything VisionGuard flagged as "a person" (or "a
  fall"), what fraction were actually correct. Low precision means lots of
  false alarms.
  RECALL = out of everything that was actually a person (or an actual fall),
  what fraction did VisionGuard successfully catch. Low recall means it
  misses a lot of real cases.
  F1 SCORE = a single combined number that balances precision and recall
  (1.0 is perfect, 0.0 is worthless).
  FALSE POSITIVE = VisionGuard raised an alarm when nothing was actually
  wrong.
  MAE (Mean Absolute Error) = on average, how far off VisionGuard's people-
  count was from the real count, in number of people. Lower is better.

PERSON DETECTION: On CrowdHuman (80 test images, 374 real people marked by
hand), VisionGuard found about 57 out of every 100 real people (recall
0.57), and about 66 out of every 100 things it flagged as a person were
genuinely a person (precision 0.66). On the extremely crowded MOT20 train-
station footage (80 frames), precision was very high (0.99 - almost
everything it flagged WAS a real person) but recall was much lower (0.24) -
meaning in a true crowd crush it reliably avoids false alarms but misses
roughly three out of every four people, because people are heavily
overlapping and hard to tell apart. CAUCAFall's limited sample (50 frames)
showed stronger detection (recall 0.76, precision 1.00), but that dataset
only had one or two people per picture, which is a much easier case than a
real crowd. Honest takeaway: VisionGuard's person detection is solid for
individual or moderately-grouped people, but in a genuine dense crowd it
will significantly under-count.

TRACKING (following the same person across frames): on the MOT20 crowded
footage, VisionGuard never confused one tracked person's identity with
another's (0 "ID switches"), but did lose and re-acquire the same person 13
times across the test ("fragmentations") - expected in a dense, occluded
scene where people keep disappearing behind each other.

CROWD / ZONE DENSITY (counting how many people are in an area): on the
train-station footage, VisionGuard's count was off by about 28 people on
average per frame (which sounds large, but that scene had 60+ real people
per frame - the under-counting problem described above compounds here). On
the more realistic shopping-mall footage, the average error was about 19
people per frame. Separately, the pass/fail rule VisionGuard uses to decide
"is this officially a crowd" matched what the real headcount would have said
100% of the time in this test, meaning its ALERT LOGIC (the rule, not the
raw headcount) is sound even when the underlying headcount itself is
imperfect.

FALL DETECTION: On Le2i (30 test clips, half real falls/lying-down, half
normal activity), VisionGuard correctly caught two out of every three real
falls (recall 0.67), and when it did raise a fall alarm, it was right about
89% of the time (precision 0.89), with a low false-alarm rate on normal
activity clips (about 8%). On UR Fall (a second, smaller, independent
dataset - 4 test sequences), it caught every real fall in this sample
(recall 1.00) and, where a real onset frame was known, actually raised the
alarm a few frames before the person was officially marked as "fully lying
down" - which is a good sign (catching it early), though this is from a very
small sample (only 2 fall sequences and 2 normal-activity sequences) so it
should not be treated as a precise number yet. One of those 2 normal-
activity sequences did trigger a false alarm, which is the one area worth
re-checking with a larger sample before trusting that number.

CAMERA TAMPER DETECTION: on the built-in Synthetic Tamper Test (6 controlled
examples: normal footage, a covered/darkened lens, and a frozen feed), every
single case was identified correctly - perfect scores across the board. This
is the most reliable result in this whole report because the "right answer"
was built by hand with full certainty, but it has not yet been checked
against a real, independently-built tampering dataset (UHCTD), since that
one requires special permission that was not obtained in this session.


===============================================================================
7. WHERE EVERYTHING IS, INSIDE THIS FOLDER
===============================================================================

model_testing_suite/
  test_models.py          - the one file you actually run to do all of this
  README.txt              - this document
  eval_lib/                - all the supporting code the above file uses
    model_runner.py        - loads VisionGuard's real detection models and
                              calls them exactly the way VisionGuard's own
                              detector/engine.py does, so the test results
                              reflect VisionGuard's real behaviour
    metrics.py              - the maths behind precision/recall/F1/etc.
    charts.py                - draws all the graphs
    report.py                - writes the final summary files
    datasets/                - one file per dataset (download + test logic)
    sweeps/                  - the minimum-camera-requirements tests
  external_datasets/       - the downloaded test data itself (images, video
                              frames, label files) - this is what got
                              downloaded from the internet, kept here so
                              re-running the tests doesn't need to download
                              everything again
  model_eval_results/      - the actual output of the last test run:
    summary.json             - every single number from every test, in a
                                machine-readable format
    summary.csv               - the same numbers in a simple spreadsheet-
                                friendly table
    charts/                   - 15 graph images (.png files) you can open
                                and look at directly, covering accuracy,
                                the minimum-camera-requirement sweeps, and
                                more


===============================================================================
8. HOW TO RUN THIS YOURSELF
===============================================================================

Open a terminal, move into this folder (model_testing_suite), and run:

    python test_models.py

That single command re-downloads anything missing (skipping anything already
downloaded - see external_datasets/ above), re-tests every model, re-draws
every graph, and rewrites the summary files.

It accepts several options to make a run bigger/slower (more certain
results) or smaller/faster:

    --full                       use the complete version of every dataset
                                  instead of the small sample used here
                                  (warning: multiple gigabytes, hours on a
                                  slow connection)
    --crowdhuman-limit N         how many CrowdHuman images to test (default 80)
    --mot20-frames N             how many MOT20 video frames to test (default 80)
    --mall-frames N              how many Mall Dataset frames to test (default 150)
    --le2i-clips-per-class N     how many Le2i clips per category (default 6)
    --urfall-sequences N         how many UR Fall sequences, each (default 3)
    --skip NAME [NAME ...]       skip specific tests entirely
    --only NAME [NAME ...]       run only specific tests (fast way to re-check
                                  just one thing, e.g. --only synthetic_tamper)

Re-running is always safe - nothing gets re-downloaded if it's already
present, and the summary files simply get overwritten with the latest run's
numbers.


===============================================================================
9. HONEST LIMITATIONS OF THIS TESTING SUITE
===============================================================================

- Sample sizes are modest (tens to low hundreds of images/frames per test,
  and as few as 2-4 sequences for UR Fall) because of the slow internet
  connection available while building this. Numbers from the smallest
  samples (especially UR Fall's false-alarm rate) should be treated as an
  early indication, not a precise, final figure.

- PETS2009 could not be properly tested at all - only its supporting
  reference files were obtained, not the actual video.

- CAUCAFall could only be used for basic person detection, not fall
  detection specifically, because of its broken online file listing.

- UHCTD, the dedicated camera-tampering research dataset, could not be used
  because it requires signed permission from its creators. Camera-tamper
  testing here relies entirely on the built-in Synthetic Tamper Test, which,
  while reliable for what it tests, has not been checked against a real,
  independently-built dataset of actual tampering footage.

- The "tracking quality" numbers use a simplified, custom-built calculation
  rather than a more advanced, widely-used tracking-metrics library (because
  that library was not available in this environment) - the numbers are
  real and calculated the same way every time, just not identical to the
  most sophisticated version of this kind of measurement used in some
  published research.

- This testing suite checks VisionGuard's computer-vision accuracy and the
  minimum camera requirements. It does NOT test VisionGuard's other
  features, such as user accounts/login, the alert escalation ladder
  (deciding when to notify a guard vs. an administrator), Telegram/WhatsApp
  message sending, or the web dashboard itself - those are not things that
  can be tested against a photo/video dataset, since they are not computer-
  vision problems.

- No file that was part of the original VisionGuard project (its detection
  code, web server, configuration, etc.) was changed in any way while
  building or running this testing suite. This suite only reads from that
  code to test it.
