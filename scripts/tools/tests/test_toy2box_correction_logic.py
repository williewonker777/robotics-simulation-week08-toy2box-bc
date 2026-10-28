import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "toy2box_correction_logic.py"
SPEC = importlib.util.spec_from_file_location("toy2box_correction_logic", SCRIPT)
LOGIC = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LOGIC
SPEC.loader.exec_module(LOGIC)


def observation(
    step,
    *,
    closed=False,
    eef=(0.0, 0.0, 0.0),
    truck=(0.0, 0.0, 0.0),
    box=(0.0, 0.0, 0.0),
    settled_z=0.0,
):
    return LOGIC.RecoveryObservation(step, truck, eef, box, closed, settled_z)


def advance(detector, start, stop, **kwargs):
    result = None
    for step in range(start, stop + 1):
        result = detector.update(observation(step, **kwargs))
    return result


def test_grasp_failure_triggers_at_60_not_59_steps_after_near_close():
    detector = LOGIC.FailureDetector()
    assert detector.update(observation(1, closed=True, eef=(0.06, 0, 0))) is None
    assert advance(detector, 2, 60, closed=True, eef=(0.061, 0, 0)) is None
    event = detector.update(observation(61, closed=True, eef=(0.061, 0, 0)))
    assert event.kind == "grasp_failure"
    assert event.detected_step == 61
    assert event.details["near_close_step"] == 1


def test_tenth_acquisition_sample_on_deadline_prevents_grasp_failure():
    detector = LOGIC.FailureDetector()
    detector.update(observation(1, closed=True, eef=(0.06, 0, 0)))
    advance(detector, 2, 51, closed=True, eef=(0.061, 0, 0))
    assert advance(
        detector,
        52,
        60,
        closed=True,
        truck=(0, 0, 0.03),
        eef=(0.06, 0, 0.03),
    ) is None
    assert detector.acquisition_run == 9
    assert detector.update(
        observation(61, closed=True, truck=(0, 0, 0.03), eef=(0.06, 0, 0.03))
    ) is None
    assert detector.acquired is True
    assert detector.acquired_step == 61


def test_nine_acquisition_samples_then_break_still_fails_on_deadline():
    detector = LOGIC.FailureDetector()
    detector.update(observation(1, closed=True))
    advance(detector, 2, 51, closed=True, eef=(0.061, 0, 0))
    advance(detector, 52, 60, closed=True, truck=(0, 0, 0.03), eef=(0.06, 0, 0.03))
    event = detector.update(observation(61, closed=True, eef=(0.061, 0, 0)))
    assert event.kind == "grasp_failure"
    assert event.details["acquisition_consecutive_steps"] == 0


def test_close_onset_requires_open_to_closed_and_equality_six_cm_counts():
    detector = LOGIC.FailureDetector()
    detector.update(observation(1, closed=True, eef=(0.0600001, 0, 0)))
    detector.update(observation(2, closed=True, eef=(0.06, 0, 0)))
    assert detector.near_close_seen is False
    detector.update(observation(3, closed=False, eef=(0.06, 0, 0)))
    detector.update(observation(4, closed=True, eef=(0.06, 0, 0)))
    assert detector.near_close_seen is True
    assert detector.pending_near_close_step == 4


def test_open_gripper_does_not_count_as_close_or_block_step_450_trigger():
    detector = LOGIC.FailureDetector()
    advance(detector, 1, 449, closed=False, eef=(0, 0, 0))
    event = detector.update(observation(450, closed=False, eef=(0, 0, 0)))
    assert event.kind == "no_near_close_by_450"
    assert event.detected_step == 450


def test_near_close_at_step_450_has_priority_over_approach_stall():
    detector = LOGIC.FailureDetector()
    advance(detector, 1, 449, closed=False, eef=(1, 0, 0))
    assert detector.update(observation(450, closed=True, eef=(0.06, 0, 0))) is None
    assert detector.near_close_seen is True


def test_acquisition_requires_ten_consecutive_samples_after_close():
    detector = LOGIC.FailureDetector()
    detector.update(observation(1, closed=True))
    advance(detector, 2, 10, truck=(0, 0, 0.03), eef=(0.06, 0, 0.03))
    assert detector.acquired is False
    detector.update(observation(11, closed=False, truck=(0, 0, 0.03), eef=(0.06, 0, 0.03)))
    assert detector.acquired is True


def test_acquisition_boundaries_are_inclusive():
    detector = LOGIC.FailureDetector()
    detector.update(observation(1, closed=True))
    advance(detector, 2, 11, truck=(0, 0, 0.03), eef=(0.06, 0, 0.03))
    assert detector.acquired is True


def acquired_detector():
    detector = LOGIC.FailureDetector()
    detector.update(observation(1, closed=True))
    advance(detector, 2, 11, closed=True, truck=(0, 0, 0.03), eef=(0.05, 0, 0.03))
    assert detector.acquired
    return detector


def test_drop_thresholds_are_strict_not_inclusive():
    detector = acquired_detector()
    assert detector.update(
        observation(12, closed=True, truck=(0.12, 0, 0.03), eef=(0.181, 0, 0.03))
    ) is None

    detector = acquired_detector()
    assert detector.update(
        observation(12, closed=True, truck=(0.121, 0, 0.03), eef=(0.181, 0, 0.03))
    ) is None

    detector = acquired_detector()
    event = detector.update(
        observation(12, closed=True, truck=(0.121, 0, 0.03), eef=(0.181001, 0, 0.03))
    )
    assert event.kind == "drop"


def test_detector_rejects_gaps_and_updates_after_detection():
    detector = LOGIC.FailureDetector()
    with pytest.raises(ValueError, match="consecutive step 1"):
        detector.update(observation(2))
    detector.update(observation(1))
    with pytest.raises(ValueError, match="consecutive step 2"):
        detector.update(observation(3))

    detector = LOGIC.FailureDetector()
    advance(detector, 1, 450)
    with pytest.raises(RuntimeError, match="after takeover"):
        detector.update(observation(451))


@pytest.mark.parametrize("axis", ["roll", "pitch"])
def test_tilt_equal_25_allowed_and_above_25_skipped(axis):
    def quaternion(degrees):
        angle = math.radians(degrees) / 2
        return (
            (math.cos(angle), math.sin(angle), 0, 0)
            if axis == "roll"
            else (math.cos(angle), 0, math.sin(angle), 0)
        )

    kwargs = dict(
        truck_height=0.4,
        settled_truck_height=0.4,
        truck_box_xy_distance=0.13,
        grasped=False,
    )
    assert LOGIC.skip_reason(quaternion(25), (0, -0.2, 0), **kwargs) is None
    assert (
        LOGIC.skip_reason(quaternion(-25.0001), (0, -0.2, 0), **kwargs)
        == "truck_tilt_exceeds_25_deg"
    )


def skip_at(position=(0, -0.2, 99), *, height=0.4, settled=0.4, distance=0.13, grasped=False):
    return LOGIC.skip_reason(
        (1, 0, 0, 0),
        position,
        truck_height=height,
        settled_truck_height=settled,
        truck_box_xy_distance=distance,
        grasped=grasped,
    )


@pytest.mark.parametrize(
    "position",
    [(-0.20, -0.35, 0), (-0.20, -0.10, 0), (0.25, -0.35, 0), (0.25, -0.10, 0)],
)
def test_env_origin_xy_bounds_are_inclusive(position):
    assert skip_at(position) is None


@pytest.mark.parametrize(
    "position",
    [(-0.200001, -0.2, 0), (0.250001, -0.2, 0), (0, -0.350001, 0), (0, -0.099999, 0)],
)
def test_env_origin_xy_outside_fixed_bounds_is_skipped(position):
    assert skip_at(position) == "truck_xy_outside_bounds"


def test_height_boundaries_and_grasp_exception_are_exact():
    assert skip_at(height=0.37) is None
    assert skip_at(height=0.369999) == "truck_below_settled_height"
    assert skip_at(height=0.43) is None
    assert skip_at(height=0.430001) == "truck_above_settled_ungrasped"
    assert skip_at(height=0.430001, grasped=True) is None


def test_box_exclusion_radius_is_inclusive_at_12_cm():
    assert skip_at(distance=0.12) == "truck_within_box_exclusion_radius"
    assert skip_at(distance=0.120001) is None


def test_skip_input_validation_fails_closed_for_ambiguous_grasp_and_distance():
    with pytest.raises(ValueError, match="observed boolean"):
        skip_at(grasped=None)
    with pytest.raises(ValueError, match="nonnegative"):
        skip_at(distance=-0.01)


def test_slice_aligned_episode_preserves_takeover_alignment_and_initial_state():
    samples = {
        "actions": np.arange(35).reshape(5, 7),
        "obs": {"eef_pos": np.arange(15).reshape(5, 3)},
        "states": {"robot": {"joint_position": np.arange(20).reshape(5, 4)}},
    }
    initial = {"robot": {"joint_position": np.array([[101, 102, 103, 104]])}}
    result = LOGIC.slice_aligned_episode(samples, 2, initial)
    assert result["num_samples"] == 3
    np.testing.assert_array_equal(result["actions"][0], samples["actions"][2])
    np.testing.assert_array_equal(result["obs"]["eef_pos"][0], samples["obs"]["eef_pos"][2])
    np.testing.assert_array_equal(
        result["initial_state"]["robot"]["joint_position"],
        initial["robot"]["joint_position"],
    )
    samples["actions"][2, 0] = -1
    assert result["actions"][0, 0] != -1


def test_slice_rejects_misaligned_samples_and_invalid_initial_state():
    with pytest.raises(ValueError, match="misaligned"):
        LOGIC.slice_aligned_episode(
            {"actions": np.zeros((3, 7)), "obs": {"eef": np.zeros((2, 3))}},
            0,
            {"robot": np.zeros((1, 4))},
        )
    with pytest.raises(ValueError, match="leading dimension one"):
        LOGIC.slice_aligned_episode(
            {"actions": np.zeros((3, 7))},
            0,
            {"robot": np.zeros((2, 4))},
        )
