"""Frame ordering, cadence sampling, production state machines, the
CrowdHuman class filter and the production-parity guard (no inference)."""
import config
import pytest

from eval_lib import cadence, frames, production_logic as pl
from eval_lib import model_runner as mr
from eval_lib.datasets import crowdhuman


# ---------------------------------------------------------------- frames
def test_numeric_sort_fixes_le2i_scramble():
    names = ["100.jpg", "101.jpg", "86.jpg", "99.jpg"]
    assert sorted(names, key=frames.numeric_frame_key) == ["86.jpg", "99.jpg", "100.jpg", "101.jpg"]


def test_numeric_sort_urfall_names():
    names = ["adl-01-cam0-rgb-010.png", "adl-01-cam0-rgb-002.png"]
    assert sorted(names, key=frames.numeric_frame_key)[0].endswith("002.png")
    assert frames.frame_number("adl-01-cam0-rgb-010.png") == 10


# ---------------------------------------------------------------- cadence
def test_cadence_distinct_rates_give_distinct_frame_sets():
    sets = {r: tuple(cadence.sample_indices(300, 30, r)) for r in (1, 2, 2.8, 5, 10, None)}
    assert len(set(sets.values())) == len(sets)


def test_cadence_is_time_based():
    assert cadence.sample_indices(31, 30, 10) == list(range(0, 31, 3))
    assert cadence.sample_indices(10, 30, None) == list(range(10))
    # 2.8 Hz on 25 fps: frames at floor(k * 8.93)
    assert cadence.sample_indices(30, 25, 2.8) == [0, 8, 17, 26]


# ---------------------------------------------------------------- crowd machine
def test_crowd_needs_confirm_cycles():
    n = config.CROWD_MIN_PERSONS
    alerts = pl.run_crowd_machine([n] * config.CROWD_CONFIRM_CYCLES)
    assert alerts == [False] * (config.CROWD_CONFIRM_CYCLES - 1) + [True]


def test_crowd_single_frame_spike_never_alerts():
    n = config.CROWD_MIN_PERSONS
    assert not any(pl.run_crowd_machine([0, n + 5, 0, 0, n + 5, 0]))


def test_crowd_hysteresis_holds_then_clears():
    n, margin = config.CROWD_MIN_PERSONS, config.CROWD_CLEAR_MARGIN
    seq = [n] * config.CROWD_CONFIRM_CYCLES + [n - margin, n - margin - 1]
    alerts = pl.run_crowd_machine(seq)
    assert alerts[config.CROWD_CONFIRM_CYCLES - 1] is True
    assert alerts[-2] is True            # still >= MIN - MARGIN: held
    assert alerts[-1] is False           # dropped below: cleared


def test_crowd_flapping_on_threshold_does_not_flap_alert():
    n = config.CROWD_MIN_PERSONS
    alerts = pl.run_crowd_machine([n] * config.CROWD_CONFIRM_CYCLES + [n - 1, n] * 10)
    raised = sum(1 for a, b in zip([False] + alerts, alerts) if b and not a)
    assert raised == 1


# ---------------------------------------------------------------- fall machine
def _person(key, fallen):
    return {"key": key, "box": (0, 0, 10, 10), "fallen": fallen, "angle": 80.0, "ratio": 1.0}


def test_fall_machine_confirms_and_holds_for_clear_seconds():
    m = pl.FallAlertMachine()
    t = 0.0
    out = []
    for i in range(config.FALL_CONFIRM_FRAMES):
        out.append(m.step([_person(1, True)], t))
        t += 0.3
    assert out[-1]["raised"] and out[-1]["alert"]
    # nobody seen down: alert holds until FALL_CLEAR_SECONDS have passed
    held = m.step([], t)
    assert held["alert"]
    gone = m.step([], t + config.FALL_CLEAR_SECONDS + 0.01)
    assert gone["cleared"] and not gone["alert"]


def test_fall_machine_single_frame_never_raises():
    m = pl.FallAlertMachine()
    assert not m.step([_person(1, True)], 0.0)["alert"] or config.FALL_CONFIRM_FRAMES <= 1


# ---------------------------------------------------------------- crowdhuman parsing
def test_yolo_label_class_filter_keeps_only_persons():
    lines = ["0 0.5 0.5 0.2 0.4", "2 0.5 0.5 0.2 0.4", "25 0.1 0.1 0.1 0.1", "0 0.2 0.2 0.1 0.1"]
    boxes, dropped = crowdhuman.parse_yolo_person_labels(lines, 100, 200)
    assert len(boxes) == 2
    assert dropped == {"2": 1, "25": 1}


def test_odgt_parsing_uses_vbox_and_ignore():
    rec = {"ID": "x", "gtboxes": [
        {"tag": "person", "fbox": [0, 0, 10, 40], "vbox": [0, 0, 10, 20], "extra": {}},
        {"tag": "person", "fbox": [50, 0, 10, 40], "vbox": [50, 0, 10, 20], "extra": {"ignore": 1}},
        {"tag": "mask", "fbox": [100, 0, 30, 30], "vbox": [100, 0, 30, 30], "extra": {}},
    ]}
    gt, ign, meta = crowdhuman.parse_odgt_record(rec)
    assert gt == [(0, 0, 10, 20)]
    assert len(ign) == 2
    assert meta[0]["visibility"] == pytest.approx(0.5)


# ---------------------------------------------------------------- model roles
def test_tracked_and_predict_roles_never_share_a_model_instance():
    """.track() permanently attaches tracker callbacks to a YOLO object, so the
    production (untracked) detection model must be a separate instance."""
    p = config.DETECTION_MODEL_PATH
    assert mr.get_model(p, "predict") is not mr.get_model(p, "legacy_track")
    assert mr.get_model(p, "predict") is mr.get_model(p, "predict")


# ---------------------------------------------------------------- parity
def test_production_parity_passes_on_current_engine():
    assert mr.check_production_parity()["status"] == "OK"


def test_production_call_passes_no_imgsz():
    """Production never passes imgsz; only what-if configurations do."""
    assert mr._size_kw(None) == {}
    assert mr._size_kw(960) == {"imgsz": 960}


def test_whatif_configs_never_reuse_production_instances():
    from eval_lib.datasets import whatif_detection as w
    for key, _, kw in w.CONFIGS:
        call = w._call_kwargs(key, kw, config, mr)
        if key == "production":
            assert "role" not in call and "imgsz" not in call and "conf" not in call
        else:
            assert call["role"].startswith("whatif_")


def test_every_suite_module_imports():
    """Catches syntax errors in modules the other tests never import (charts, main script)."""
    import importlib
    for name in ("eval_lib.charts", "eval_lib.report", "eval_lib.datasets.whatif_detection",
                 "test_models"):
        importlib.import_module(name)


def test_urfall_adl_split_by_lying_label():
    from eval_lib.datasets import ur_fall
    labels = {"adl-01": {1: -1, 2: -1}, "adl-02": {1: -1, 2: 0, 3: 1}, "fall-01": {1: 1}}
    seqs = [{"sequence": "adl-01", "is_fall": False, "alerts_raised_at_frames": [2]},
            {"sequence": "adl-02", "is_fall": False, "alerts_raised_at_frames": [3]},
            {"sequence": "fall-01", "is_fall": True, "alerts_raised_at_frames": [1]}]
    out = ur_fall.adl_split_by_lying({"2.8 Hz": {"per_sequence": seqs}}, labels)["by_rate"]["2.8 Hz"]
    assert out["adl_with_lying"]["k"] == 1 and out["adl_with_lying"]["n"] == 1
    assert out["adl_with_lying"]["sequences_alarmed"] == ["adl-02"]
    assert out["adl_without_lying"]["sequences_alarmed"] == ["adl-01"]
