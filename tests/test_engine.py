"""Unit tests for the detection logic in detector/engine.py.

Model inference is not run here — small stand-in objects mimic the
shapes ultralytics returns (boxes with .xyxy tensors, keypoints with
.xy/.conf), so these tests cover the pure logic: zone assignment,
torso-angle fall classification, the aspect-ratio fallback, and the
per-person confirmation streaks.
"""
import math

import config
from detector.engine import (
    KP_L_HIP, KP_L_SHOULDER, KP_R_HIP, KP_R_SHOULDER,
    classify_persons, confirm_falls, nested_duplicates, zone_densities,
)

N_KEYPOINTS = 17


# ── stand-ins for ultralytics result objects ─────────────────────
class FakeTensor:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class FakeBox:
    def __init__(self, xyxy):
        self.xyxy = [FakeTensor(xyxy)]


class FakeBoxes:
    def __init__(self, xyxy_list, ids=None, conf=None):
        self.xyxy = [FakeTensor(b) for b in xyxy_list]
        self.id = ids  # None (tracker unavailable) or list of ints
        self.conf = conf if conf is not None else [0.9] * len(xyxy_list)

    def __len__(self):
        return len(self.xyxy)


class FakeKeypoints:
    def __init__(self, xy_list, conf_list):
        self.xy = [FakeTensor(xy) for xy in xy_list]
        self.conf = [FakeTensor(c) for c in conf_list]


class FakeResult:
    def __init__(self, boxes, keypoints=None):
        self.boxes = boxes
        self.keypoints = keypoints


def make_keypoints(mid_shoulder, mid_hip, conf=0.9, nose=None, ankles=None):
    """17 keypoints, all zero except shoulders/hips placed around the
    given midpoints, plus optional nose and ankle positions."""
    xy = [[0.0, 0.0]] * N_KEYPOINTS
    confs = [0.0] * N_KEYPOINTS
    sx, sy = mid_shoulder
    hx, hy = mid_hip
    xy = list(xy)
    xy[KP_L_SHOULDER] = [sx - 5, sy]
    xy[KP_R_SHOULDER] = [sx + 5, sy]
    xy[KP_L_HIP] = [hx - 5, hy]
    xy[KP_R_HIP] = [hx + 5, hy]
    trusted = [KP_L_SHOULDER, KP_R_SHOULDER, KP_L_HIP, KP_R_HIP]
    if nose is not None:
        from detector.engine import KP_NOSE
        xy[KP_NOSE] = list(nose)
        trusted.append(KP_NOSE)
    if ankles is not None:
        from detector.engine import KP_L_ANKLE, KP_R_ANKLE
        ax, ay = ankles
        xy[KP_L_ANKLE] = [ax - 5, ay]
        xy[KP_R_ANKLE] = [ax + 5, ay]
        trusted.extend([KP_L_ANKLE, KP_R_ANKLE])
    for i in trusted:
        confs[i] = conf
    return xy, confs


# ── zone densities ───────────────────────────────────────────────
# frame 400x300 with the 3x4 grid → cells are 100x100
FRAME_SHAPE = (300, 400, 3)


def test_no_people_means_zero_density():
    counts, max_density = zone_densities([], FRAME_SHAPE)
    assert counts == {}
    assert max_density == 0


def test_person_lands_in_cell_containing_box_centre():
    box = FakeBox([10, 10, 90, 90])  # centre (50, 50) → row 0, col 0
    counts, max_density = zone_densities([box], FRAME_SHAPE)
    assert counts[(0, 0)] == 1
    assert max_density == 1


def test_three_people_in_one_cell_reach_alert_threshold():
    boxes = [FakeBox([120, 120, 160, 180]) for _ in range(3)]  # all in (1, 1)
    counts, max_density = zone_densities(boxes, FRAME_SHAPE)
    assert counts[(1, 1)] == 3
    assert max_density >= config.CROWD_THRESHOLD


def test_centre_beyond_frame_edge_clamps_to_last_cell():
    box = FakeBox([380, 280, 460, 340])  # centre (420, 310) — off frame
    counts, _ = zone_densities([box], FRAME_SHAPE)
    assert counts[(config.GRID_ROWS - 1, config.GRID_COLS - 1)] == 1


# ── fall classification ──────────────────────────────────────────
def test_standing_person_is_not_fallen():
    # vertical torso: shoulders directly above hips
    xy, confs = make_keypoints(mid_shoulder=(100, 50), mid_hip=(100, 120))
    result = FakeResult(FakeBoxes([[80, 20, 120, 200]]),
                        FakeKeypoints([xy], [confs]))
    (person,) = classify_persons(result)
    assert person["angle"] is not None and person["angle"] < 5
    assert not person["fallen"]


def test_horizontal_torso_is_fallen():
    # shoulders far to the side of hips → torso ~76 deg off vertical
    xy, confs = make_keypoints(mid_shoulder=(200, 95), mid_hip=(100, 120))
    result = FakeResult(FakeBoxes([[80, 80, 260, 140]]),
                        FakeKeypoints([xy], [confs]))
    (person,) = classify_persons(result)
    assert person["angle"] > config.FALL_TORSO_ANGLE_DEG
    assert person["fallen"]


def test_upright_torso_wins_over_wide_bounding_box():
    # wide bounding box but upright torso → NOT fallen
    xy, confs = make_keypoints(mid_shoulder=(100, 50), mid_hip=(100, 120))
    wide_box = [0, 60, 300, 160]  # ratio 3.0
    result = FakeResult(FakeBoxes([wide_box]), FakeKeypoints([xy], [confs]))
    (person,) = classify_persons(result)
    assert person["ratio"] > 1.3
    assert not person["fallen"]


def test_no_usable_keypoints_never_falls():
    # upper-body-only view (e.g. seated at a desk): hips occluded, box
    # wider than tall — must NOT be classified as a fall
    xy, confs = make_keypoints((100, 50), (100, 120), conf=0.0)  # below CONF_MIN
    wide_box = [0, 60, 300, 160]
    result = FakeResult(FakeBoxes([wide_box]),
                        FakeKeypoints([xy], [confs]))
    (person,) = classify_persons(result)
    assert person["angle"] is None
    assert not person["fallen"]


def test_head_at_hip_level_is_fallen_despite_small_torso_angle():
    # falling toward the camera: torso looks near-vertical in projection,
    # but the head never rises above the hips
    xy, confs = make_keypoints(mid_shoulder=(100, 110), mid_hip=(100, 120),
                               nose=(100, 118))
    result = FakeResult(FakeBoxes([[40, 60, 240, 160]]),
                        FakeKeypoints([xy], [confs]))
    (person,) = classify_persons(result)
    assert person["angle"] is not None and person["angle"] < 60
    assert person["fallen"]


def test_bending_over_is_not_a_fall():
    # picking something up: torso tilts past the angle threshold, but the
    # ankles stay far below the hips — upright legs veto the fall
    xy, confs = make_keypoints(mid_shoulder=(160, 100), mid_hip=(100, 120),
                               ankles=(100, 220))
    result = FakeResult(FakeBoxes([[60, 80, 200, 240]]),
                        FakeKeypoints([xy], [confs]))
    (person,) = classify_persons(result)
    assert person["angle"] > 65  # the torso alone says "fallen"
    assert not person["fallen"]  # but the legs say "standing"


def test_lying_with_ankles_at_hip_level_is_fallen():
    # on the ground: body horizontal, ankles at roughly hip height
    xy, confs = make_keypoints(mid_shoulder=(200, 95), mid_hip=(100, 120),
                               ankles=(30, 125))
    result = FakeResult(FakeBoxes([[10, 80, 260, 150]]),
                        FakeKeypoints([xy], [confs]))
    (person,) = classify_persons(result)
    assert person["fallen"]


def test_tracker_ids_key_people_when_available():
    xy, confs = make_keypoints((100, 50), (100, 120))
    result = FakeResult(FakeBoxes([[0, 0, 50, 100]], ids=[42]),
                        FakeKeypoints([xy], [confs]))
    (person,) = classify_persons(result)
    assert person["key"] == 42

    result_untracked = FakeResult(FakeBoxes([[0, 0, 50, 100]]),
                                  FakeKeypoints([xy], [confs]))
    (person,) = classify_persons(result_untracked)
    assert person["key"] == "det0"


# ── fall confirmation streaks ────────────────────────────────────
def fallen_person(key):
    return {"key": key, "box": (0, 0, 10, 10), "fallen": True,
            "angle": 80.0, "ratio": 2.0}


def upright_person(key):
    return {"key": key, "box": (0, 0, 10, 10), "fallen": False,
            "angle": 5.0, "ratio": 0.4}


def test_single_frame_fall_does_not_confirm():
    streaks = {}
    confirmed = confirm_falls([fallen_person(1)], streaks)
    assert confirmed == []
    assert streaks[1] == 1


def test_consecutive_falls_confirm_after_threshold():
    streaks = {}
    for _ in range(config.FALL_CONFIRM_FRAMES - 1):
        assert confirm_falls([fallen_person(1)], streaks) == []
    confirmed = confirm_falls([fallen_person(1)], streaks)
    assert [p["key"] for p in confirmed] == [1]


def test_standing_up_resets_the_streak():
    streaks = {}
    confirm_falls([fallen_person(1)], streaks)
    confirm_falls([upright_person(1)], streaks)  # recovered
    assert 1 not in streaks
    assert confirm_falls([fallen_person(1)], streaks) == []  # starts over


def test_leaving_the_frame_clears_the_streak():
    streaks = {}
    confirm_falls([fallen_person(1)], streaks)
    confirm_falls([fallen_person(2)], streaks)  # person 1 gone
    assert 1 not in streaks and streaks[2] == 1


# ── Nested duplicate boxes ──────────────────────────────────────
def test_head_box_inside_body_box_is_a_duplicate():
    # body box at 0.9, a head box fully inside it at 0.4 → head dropped
    assert nested_duplicates(
        [[100, 100, 200, 400], [130, 110, 170, 160]], [0.9, 0.4]) == {1}


def test_overlapping_people_are_both_kept():
    # two people side by side, overlapping by a third — neither is nested
    assert nested_duplicates(
        [[100, 100, 200, 400], [170, 100, 270, 400]], [0.9, 0.5]) == set()


def test_more_confident_inner_box_is_kept():
    # a nested box is only dropped in favour of a MORE confident container
    assert nested_duplicates(
        [[100, 100, 200, 400], [130, 110, 170, 160]], [0.4, 0.9]) == set()
