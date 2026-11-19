#!/usr/bin/env python
"""VisionGuard model evaluation suite: accuracy on public datasets, alert
logic, tamper characterisation and minimum camera requirements.

Run from anywhere (paths resolve relative to this file):
    python model_testing_suite/test_models.py --quick      # smoke run, minutes
    python model_testing_suite/test_models.py              # full run (asks nothing; see README.txt)

The suite never modifies VisionGuard itself. It imports the production
functions from detector/engine.py and every threshold from config.py, and it
aborts if detector/engine.py no longer makes the calls it mirrors.
Outputs: model_eval_results/ (or model_eval_results_quick/ with --quick):
summary.json, summary.csv, charts/, error_examples/.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_lib import paths  # noqa: E402  (must come first: puts the project root on sys.path)
from eval_lib import charts, provenance, report  # noqa: E402
from eval_lib import http_utils as hu  # noqa: E402
from eval_lib import model_runner as mr  # noqa: E402
from eval_lib import cadence  # noqa: E402
from eval_lib.datasets import (  # noqa: E402
    caucafall, crowdhuman, le2i, mall, mot20, pets2009, tamper_functional, uhctd, ur_fall,
    whatif_detection,
)
from eval_lib.sweeps import (  # noqa: E402
    inference_rate_sweep, min_spec, person_size_sweep, resolution_sweep,
)

DATASETS = {
    "crowdhuman": crowdhuman.run,
    "crowdhuman_mirror_pilot": crowdhuman.run_mirror_pilot,
    "mot20": mot20.run,
    "mall": mall.run,
    "le2i": le2i.run,
    "ur_fall": ur_fall.run,
    "caucafall": caucafall.run,
    "pets2009": pets2009.run,
    "uhctd": uhctd.run,
    "tamper_functional": tamper_functional.run,
    "whatif_detection": whatif_detection.run,   # NOT production: candidate setting changes
}

FULL = dict(crowdhuman_images=300, mot20_frames=429, mall_frames=2000, le2i_clips_per_class=20,
            urfall_falls=30, urfall_adls=40, resolution_images=100, resolution_frames=300,
            baseline_frames=500, rates="1,2,2.8,5,10,native")
QUICK = dict(crowdhuman_images=30, mot20_frames=60, mall_frames=200, le2i_clips_per_class=4,
             urfall_falls=2, urfall_adls=2, resolution_images=15, resolution_frames=40,
             baseline_frames=60, rates="1,2.8,native")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="small smoke run (minutes)")
    p.add_argument("--data-root", default=None, help="default: model_testing_suite/external_datasets")
    p.add_argument("--out-dir", default=None,
                   help="default: model_testing_suite/model_eval_results(_quick)")
    p.add_argument("--only", nargs="*", default=None, help=f"datasets to run: {list(DATASETS)}")
    p.add_argument("--skip", nargs="*", default=[])
    p.add_argument("--skip-sweeps", action="store_true")
    p.add_argument("--redraw-charts", action="store_true",
                   help="only redraw charts from <out-dir>/summary.json (no inference)")
    p.add_argument("--seed", type=int, default=0)
    for k in FULL:
        p.add_argument("--" + k.replace("_", "-"), type=(str if k == "rates" else int), default=None)
    p.add_argument("--inference-hz", type=float, default=cadence.DOCUMENTED_PRODUCTION_HZ,
                   help="effective production inference rate (config.py documents ~2.8 Hz)")
    p.add_argument("--max-download-mb", type=int, default=500,
                   help="budget for NEW large downloads (UR Fall sequences) without asking")
    p.add_argument("--allow-large-downloads", action="store_true")
    p.add_argument("--no-baselines", dest="baselines", action="store_false")
    p.add_argument("--headline-min-n", type=int, default=30)
    p.add_argument("--min-spec-threshold", type=float, default=0.9)
    p.add_argument("--size-recall-target", type=float, default=0.8)
    p.add_argument("--resolution-recall-target", type=float, default=0.8)
    p.add_argument("--resolution-rel-mae-target", type=float, default=0.2)
    a = p.parse_args(argv)
    defaults = QUICK if a.quick else FULL
    for k, v in defaults.items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    a.data_root = paths.resolve_under_suite(a.data_root) if a.data_root else paths.DEFAULT_DATA_ROOT
    a.out_dir = (paths.resolve_under_suite(a.out_dir) if a.out_dir else
                 (paths.DEFAULT_QUICK_OUT_DIR if a.quick else paths.DEFAULT_OUT_DIR))
    rates = []
    for tok in str(a.rates).split(","):
        tok = tok.strip().lower()
        rates.append(None if tok in ("native", "all", "every") else float(tok))
    if a.inference_hz not in rates:
        rates.append(a.inference_hz)
    a.rates = rates
    return a


def seed_everything(seed):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:  # noqa: BLE001
        pass


def guarded(name, fn, *a):
    t0 = time.time()
    hu.log(f"--- {name} ---")
    try:
        out = fn(*a)
    except Exception as e:  # noqa: BLE001 - one failure must not kill the run, but it is RECORDED
        out = {"status": f"FAILED: {type(e).__name__}: {e}",
               "traceback": traceback.format_exc()}
        traceback.print_exc()
    out["elapsed_seconds"] = round(time.time() - t0, 1)
    hu.log(f"{name}: {str(out.get('status'))[:100]} ({out['elapsed_seconds']}s)")
    return out


def main(argv=None):
    args = parse_args(argv)
    if args.redraw_charts:
        import json
        with open(os.path.join(args.out_dir, "summary.json"), encoding="utf-8") as f:
            saved = json.load(f)
        made = charts.make_all(saved, os.path.join(args.out_dir, "charts"))
        print(f"Redrew {len(made)} charts in {os.path.join(args.out_dir, 'charts')}")
        return 0
    try:
        parity = mr.check_production_parity()
    except mr.ProductionParityError as e:
        print(f"ABORTED: {e}", file=sys.stderr)
        return 2
    seed_everything(args.seed)
    args_dict = {k: (v if not isinstance(v, list) else list(v)) for k, v in vars(args).items()}
    prov = provenance.collect(args_dict, mr.tracker_info(), parity)
    hu.log(f"VisionGuard evaluation: commit {prov['git']['commit']}, "
           f"ultralytics {prov['versions']['ultralytics']}, tracker {prov['tracker'].get('tracker')}, "
           f"out_dir={args.out_dir}, quick={args.quick}")
    if prov["git"]["dirty_files_outside_suite"]:
        hu.log("WARNING: uncommitted changes outside model_testing_suite/: "
               f"{prov['git']['dirty_files_outside_suite']}")
    mr.warmup()

    order = [k for k in args.only if k in DATASETS] if args.only else list(DATASETS)
    keys = [k for k in order if k not in args.skip]   # --only runs in the order given
    datasets = {k: guarded(k, DATASETS[k], args) for k in keys}

    sweeps = {}
    if not args.skip_sweeps:
        sweeps["inference_rate"] = guarded("inference-rate sweep", inference_rate_sweep.run,
                                           datasets.get("ur_fall"))
        sweeps["resolution"] = guarded("resolution sweep", resolution_sweep.run, args)
        sweeps["person_size"] = guarded("person-size sweep", person_size_sweep.run, datasets,
                                        args.headline_min_n, args.size_recall_target)
        sweeps["min_spec"] = guarded("minimum spec", min_spec.derive, sweeps["inference_rate"],
                                     sweeps["resolution"], sweeps["person_size"], args)

    prov["finished_at"] = datetime.now().isoformat(timespec="seconds")
    all_results = {"provenance": prov, "datasets": datasets, "sweeps": sweeps,
                   "quality_definition": f"headline = n >= {args.headline_min_n} positive "
                                         "events/instances behind the main metric; pilot = fewer"}
    clean = report.strip_private(all_results)
    chart_dir = os.path.join(args.out_dir, "charts")
    try:
        chart_paths = charts.make_all(clean, chart_dir)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        chart_paths = []
    all_results["charts_generated"] = [os.path.relpath(p, args.out_dir) for p in chart_paths]
    json_path, csv_path, rows = report.write_summary(all_results, args.out_dir)
    report.print_report(rows, sweeps.get("min_spec"))
    print(f"Charts:  {chart_dir} ({len(chart_paths)} files)")
    print(f"Summary: {json_path}\nCSV:     {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
