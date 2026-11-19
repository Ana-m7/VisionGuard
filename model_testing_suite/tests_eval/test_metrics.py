"""Hand-computed checks for eval_lib/metrics.py (no model inference)."""
import math

import pytest

from eval_lib import metrics as m


def test_iou_known_values():
    assert m.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert m.iou((0, 0, 10, 10), (10, 10, 20, 20)) == 0.0
    # 5x10 overlap, union 100+100-50
    assert m.iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_ioa_is_fraction_of_first_box():
    assert m.ioa((0, 0, 10, 10), (0, 0, 100, 100)) == 1.0
    assert m.ioa((0, 0, 10, 10), (5, 0, 100, 100)) == pytest.approx(0.5)


def test_match_boxes_greedy_one_to_one():
    gt = [(0, 0, 10, 10), (20, 0, 30, 10)]
    pred = [(0, 0, 10, 10, .9), (1, 0, 11, 10, .8), (50, 50, 60, 60, .7)]
    r = m.match_boxes(pred, gt)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 2, 1)


def test_match_boxes_ignore_region_is_neither_tp_nor_fp():
    gt = [(0, 0, 10, 10)]
    pred = [(0, 0, 10, 10, .9), (100, 100, 110, 110, .8)]
    r = m.match_boxes(pred, gt, ignore_boxes=[(90, 90, 200, 200)])
    assert (r["tp"], r["fp"], r["fn"], r["ignored"]) == (1, 0, 0, [1])


def test_prf1_and_zero_division():
    assert m.precision_recall_f1(8, 2, 4) == pytest.approx((0.8, 8 / 12, 2 * .8 * (2 / 3) / (.8 + 2 / 3)))
    assert m.precision_recall_f1(0, 0, 0) == (0.0, 0.0, 0.0)


def test_wilson_matches_reference_values():
    # Reference: 8/10 -> (0.4902, 0.9433) (Wilson, z=1.96)
    lo, hi = m.wilson_ci(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)
    lo, hi = m.wilson_ci(0, 5)
    assert lo == 0.0 and hi == pytest.approx(0.4345, abs=1e-3)
    assert m.wilson_ci(0, 0) == (None, None)


def test_count_errors_point_values():
    r = m.count_errors([1, 2, 6], [2, 2, 2])
    assert r["mae"]["value"] == pytest.approx(5 / 3)
    assert r["rmse"]["value"] == pytest.approx(math.sqrt(17 / 3))
    assert r["bias"]["value"] == pytest.approx(1.0)
    assert r["mae"]["ci_low"] <= r["mae"]["value"] <= r["mae"]["ci_high"]


def test_bootstrap_is_seeded_and_deterministic():
    f = lambda i: float(len(set(i.tolist())))  # noqa: E731
    assert m.bootstrap_ci(20, f, seed=3) == m.bootstrap_ci(20, f, seed=3)


def test_ap101_perfect_and_half():
    gt = [[(0, 0, 10, 10)], [(0, 0, 10, 10)]]
    perfect = [[(0, 0, 10, 10, .9)], [(0, 0, 10, 10, .8)]]
    assert m.average_precision(perfect, gt)[0] == pytest.approx(1.0)
    half = [[(0, 0, 10, 10, .9)], []]
    # recall reaches only 0.5 with precision 1 -> 51 of 101 points at 1.0
    assert m.average_precision(half, gt)[0] == pytest.approx(51 / 101)


def test_clear_mot_counts_an_id_switch_and_fragment():
    gt = [[(1, 0, 0, 10, 10)], [(1, 0, 0, 10, 10)], [(1, 0, 0, 10, 10)], [(1, 0, 0, 10, 10)]]
    pred = [[(7, 0, 0, 10, 10)], [], [(8, 0, 0, 10, 10)], [(8, 0, 0, 10, 10)]]
    r = m.clear_mot(pred, gt)
    assert r["tp"] == 3 and r["fn"] == 1 and r["fp"] == 0
    assert r["idsw"] == 1          # 7 -> 8
    assert r["frag"] == 1          # tracked, missed, tracked
    assert r["mota"] == pytest.approx(1 - (1 + 0 + 1) / 4)


def test_idf1_hand_example():
    gt = [[(1, 0, 0, 10, 10)]] * 4
    pred = [[(7, 0, 0, 10, 10)], [(7, 0, 0, 10, 10)], [(8, 0, 0, 10, 10)], [(8, 0, 0, 10, 10)]]
    r = m.idf1(pred, gt)
    # best one-to-one: gt1<->7 (2 frames) -> IDTP 2, IDFP 2, IDFN 2
    assert (r["idtp"], r["idfp"], r["idfn"]) == (2, 2, 2)
    assert r["idf1"] == pytest.approx(0.5)


def test_filter_ignored_preds_drops_distractor_matches():
    preds = [(1, 0, 0, 10, 10), (2, 50, 50, 60, 60)]
    kept = m.filter_ignored_preds(preds, gt_boxes=[(0, 0, 10, 10)], ignore_boxes=[(50, 50, 60, 60)])
    assert [p[0] for p in kept] == [1]


def test_tracking_metrics_agree_with_motmetrics():
    """Our CLEAR-MOT / IDF1 must match the reference py-motmetrics package."""
    pytest.importorskip("motmetrics")
    gt = [[(1, 0, 0, 10, 10), (2, 50, 50, 60, 60)]] * 3 + [[(1, 0, 0, 10, 10)]] * 3
    pred = [[(7, 0, 0, 10, 10), (9, 51, 51, 60, 60)],
            [(7, 1, 0, 10, 10)],
            [(8, 0, 0, 10, 10), (9, 50, 50, 60, 60), (5, 200, 200, 210, 210)],
            [(8, 0, 0, 10, 10)], [], [(8, 0, 1, 10, 10)]]
    ours = {**m.clear_mot(pred, gt), **m.idf1(pred, gt)}
    ref = m.crosscheck_motmetrics(pred, gt)
    for k in ("mota", "idf1", "motp_mean_iou"):
        assert ours[k] == pytest.approx(ref[k], abs=1e-9), k
    for k in ("idsw", "frag", "fp", "fn", "n_gt"):
        assert ours[k] == ref[k], k
