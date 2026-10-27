import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "toy2box_quality.py"
SPEC = importlib.util.spec_from_file_location("toy2box_quality", SCRIPT)
QUALITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALITY)


def trajectory(samples=160, dt=0.05):
    time = np.arange(samples) * dt
    actions = np.zeros((samples, 7))
    actions[:, 0] = 0.05
    positions = np.column_stack((0.02 * time, np.zeros(samples), np.ones(samples)))
    quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (samples, 1))
    joints = np.column_stack((0.1 * time, -0.05 * time))
    return actions, positions, quaternions, joints


def test_constant_velocity_is_smooth_and_json_serializable():
    metrics = QUALITY.trajectory_quality(*trajectory())

    assert metrics["eef_velocity_norm_p95"] == pytest.approx(0.02)
    assert metrics["eef_acceleration_norm_max"] < 1e-12
    assert metrics["eef_jerk_norm_max"] < 1e-10
    assert metrics["angular_velocity_norm_max"] == 0.0
    assert metrics["arm_joint_acceleration_max_component_max"] < 1e-11
    assert metrics["action_immediate_flip_count"] == 0
    assert metrics["rapid_eef_velocity_reversal_count"] == 0
    assert metrics["visible_rapid_eef_reversal_count"] == 0
    assert QUALITY.quality_rejections(metrics) == []
    json.dumps(metrics)


def test_smooth_sine_has_low_acceleration_and_no_rapid_reversals():
    actions, positions, quaternions, joints = trajectory(samples=400)
    time = np.arange(len(actions)) * 0.05
    positions[:, 0] = 0.01 * np.sin(0.5 * time)
    joints[:, 0] = 0.05 * np.sin(0.5 * time)
    metrics = QUALITY.trajectory_quality(actions, positions, quaternions, joints)

    assert metrics["eef_acceleration_norm_p95"] < 0.003
    assert metrics["eef_jerk_norm_p95"] < 0.002
    assert metrics["arm_joint_acceleration_max_component_p95"] < 0.02
    assert metrics["rapid_eef_velocity_reversal_count"] == 0
    assert metrics["rapid_arm_joint_velocity_reversal_count"] == 0
    assert metrics["visible_rapid_eef_reversal_count"] == 0
    assert metrics["visible_rapid_arm_joint_reversal_count"] == 0


def test_ten_hz_alternating_jitter_is_rejected():
    actions, positions, quaternions, joints = trajectory()
    alternating = np.where(np.arange(len(actions)) % 2 == 0, 1.0, -1.0)
    actions[:, 0] = 0.05 * alternating
    positions[:, 0] = 0.003 * alternating
    joints[:, 0] = 0.02 * alternating
    metrics = QUALITY.trajectory_quality(actions, positions, quaternions, joints)
    rejection_text = "\n".join(QUALITY.quality_rejections(metrics))

    assert metrics["action_direct_opposite_consecutive_count"] == len(actions) - 1
    assert metrics["action_nonzero_sign_reversal_counts"][0] == len(actions) - 1
    assert metrics["rapid_eef_velocity_reversal_count"] > 6
    assert metrics["rapid_arm_joint_velocity_reversal_count"] > 12
    assert metrics["visible_rapid_eef_reversal_count"] > 3
    assert metrics["visible_rapid_arm_joint_reversal_count"] > 6
    assert "action_immediate_flip_count" in rejection_text
    assert "eef_acceleration_norm_p95" in rejection_text
    assert "eef_jerk_norm_p95" in rejection_text


def test_quaternion_sign_is_invariant_and_rotation_rate_is_measured():
    args = list(trajectory())
    samples = len(args[0])
    angle = 0.2 * np.arange(samples) * 0.05
    quaternions = np.column_stack(
        (np.cos(angle / 2), np.zeros(samples), np.zeros(samples), np.sin(angle / 2))
    )
    args[2] = quaternions
    original = QUALITY.trajectory_quality(*args)
    quaternions[::2] *= -1
    sign_flipped = QUALITY.trajectory_quality(*args)

    assert original["angular_velocity_norm_p95"] == pytest.approx(0.2, rel=1e-10)
    assert original["angular_acceleration_norm_max"] < 1e-10
    assert sign_flipped["angular_velocity_norm_p95"] == pytest.approx(
        original["angular_velocity_norm_p95"], rel=1e-12
    )
    assert sign_flipped["angular_acceleration_norm_max"] == pytest.approx(
        original["angular_acceleration_norm_max"], abs=1e-12
    )


@pytest.mark.parametrize("index", range(4))
def test_nonfinite_input_is_rejected(index):
    args = list(trajectory())
    args[index][30, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        QUALITY.trajectory_quality(*args)


def test_shape_and_parameter_errors_are_rejected():
    args = list(trajectory())
    with pytest.raises(ValueError, match="same number of samples"):
        QUALITY.trajectory_quality(args[0][:-1], *args[1:])
    with pytest.raises(ValueError, match=r"shape \(T, 4\)"):
        QUALITY.trajectory_quality(args[0], args[1], args[2][:, :3], args[3])
    with pytest.raises(ValueError, match="dt must"):
        QUALITY.trajectory_quality(*args, dt=0)
    with pytest.raises(ValueError, match="skip must"):
        QUALITY.trajectory_quality(*args, skip=-1)
    with pytest.raises(ValueError, match=r"skip \+ 4"):
        QUALITY.trajectory_quality(*args, skip=len(args[0]) - 3)

    # Four post-skip states yield three velocities, two accelerations, and one
    # jerk sample, which is the smallest non-empty physical metric sequence.
    metrics = QUALITY.trajectory_quality(*args, skip=len(args[0]) - 4)
    assert metrics["samples_physical_metrics"] == 4
    assert metrics["eef_jerk_norm_max"] >= 0.0


def test_nonzero_action_sign_reversals_skip_zero_runs():
    actions, positions, quaternions, joints = trajectory()
    actions[:, 0] = 0.0
    actions[[30, 35, 40, 45], 0] = [0.05, -0.05, 0.05, -0.05]
    metrics = QUALITY.trajectory_quality(actions, positions, quaternions, joints)

    assert metrics["action_immediate_flip_count"] == 0
    assert metrics["action_nonzero_sign_reversal_counts"][0] == 3
    assert metrics["action_direct_opposite_consecutive_count"] == 0


def test_skip_removes_reset_transient_only_from_physical_metrics():
    actions, positions, quaternions, joints = trajectory()
    actions[:25, 0] = 0.05 * np.where(np.arange(25) % 2 == 0, 1, -1)
    # Third derivatives retain three samples of boundary context.  End reset
    # transients before that context; the action metric still uses all samples.
    positions[:22, 0] = 0.1 * np.where(np.arange(22) % 2 == 0, 1, -1)
    joints[:22, 0] = np.where(np.arange(22) % 2 == 0, 1, -1)
    skipped = QUALITY.trajectory_quality(actions, positions, quaternions, joints)
    unskipped = QUALITY.trajectory_quality(actions, positions, quaternions, joints, skip=0)

    assert skipped["eef_acceleration_norm_max"] < 1e-10
    assert skipped["arm_joint_acceleration_max_component_max"] < 1e-9
    assert unskipped["eef_acceleration_norm_max"] > 1.0
    assert unskipped["arm_joint_acceleration_max_component_max"] > 8.0
    assert skipped["action_immediate_flip_count"] == unskipped["action_immediate_flip_count"]


def test_first_post_settle_response_keeps_derivative_context():
    actions, positions, quaternions, joints = trajectory()
    actions[:] = 0.0
    positions[:] = [0.0, 0.0, 1.0]
    joints[:] = 0.0
    quaternions[:] = [1.0, 0.0, 0.0, 0.0]
    boundary = 25
    positions[boundary, 0] += 0.02
    joints[boundary, 0] += 0.5
    angle = 0.1
    quaternions[boundary] = [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]

    metrics = QUALITY.trajectory_quality(actions, positions, quaternions, joints, skip=boundary)

    assert metrics["eef_velocity_norm_max"] == pytest.approx(0.4)
    assert metrics["eef_acceleration_norm_max"] == pytest.approx(16.0)
    assert metrics["angular_velocity_norm_max"] == pytest.approx(2.0)
    assert metrics["angular_acceleration_norm_max"] == pytest.approx(80.0)
    assert metrics["arm_joint_velocity_max_component_max"] == pytest.approx(10.0)
    assert metrics["arm_joint_acceleration_max_component_max"] == pytest.approx(400.0)
    assert metrics["visible_rapid_eef_reversal_count"] == 1
    assert metrics["visible_rapid_arm_joint_reversal_count"] == 1
    assert QUALITY.quality_rejections(metrics)


def test_subthreshold_microjitter_is_not_a_visible_reversal():
    actions, positions, quaternions, joints = trajectory()
    alternating = np.where(np.arange(len(actions)) % 2 == 0, 1.0, -1.0)
    # Consecutive excursions are 0.8 mm and 0.004 rad: deliberately below the
    # visible-motion thresholds even though raw velocity reversals remain useful.
    positions[:, 0] = 0.0004 * alternating
    joints[:, 0] = 0.002 * alternating
    metrics = QUALITY.trajectory_quality(actions, positions, quaternions, joints)

    assert metrics["rapid_eef_velocity_reversal_count"] > 6
    assert metrics["rapid_arm_joint_velocity_reversal_count"] > 12
    assert metrics["visible_rapid_eef_reversal_count"] == 0
    assert metrics["visible_rapid_arm_joint_reversal_count"] == 0
    assert not any("visible_rapid" in reason for reason in QUALITY.quality_rejections(metrics))


def test_visible_two_millimetre_jitter_has_large_jerk_and_reversals():
    actions, positions, quaternions, joints = trajectory()
    alternating = np.where(np.arange(len(actions)) % 2 == 0, 1.0, -1.0)
    positions[:, 0] = 0.002 * alternating
    joints[:, 0] = 0.003 * alternating
    metrics = QUALITY.trajectory_quality(actions, positions, quaternions, joints)
    rejection_text = "\n".join(QUALITY.quality_rejections(metrics))

    assert metrics["eef_jerk_norm_p95"] > 95.0
    assert metrics["visible_rapid_eef_reversal_count"] > 3
    assert metrics["visible_rapid_arm_joint_reversal_count"] > 6
    assert "eef_jerk_norm_p95" in rejection_text
    assert "visible_rapid_eef_reversal_count" in rejection_text
    assert "visible_rapid_arm_joint_reversal_count" in rejection_text


def test_visible_reversal_threshold_is_inclusive():
    deltas = np.array([[0.001], [-0.001]])
    assert QUALITY._rapid_reversals(deltas, 0.001, 4, inclusive=True) == 1
    assert QUALITY._rapid_reversals(deltas, 0.001, 4) == 0


def test_isolated_two_millimetre_kick_is_rejected_by_max_not_p95():
    actions, positions, quaternions, joints = trajectory()
    actions[:] = 0.0
    positions[:] = [0.0, 0.0, 1.0]
    joints[:] = 0.0
    # One 2 mm per-axis EEF spike occupies under five percent of samples, so
    # percentile-only gating misses it while the vector-norm maximum catches it.
    positions[80] += 0.002
    metrics = QUALITY.trajectory_quality(actions, positions, quaternions, joints)
    rejections = QUALITY.quality_rejections(metrics)

    assert metrics["eef_jerk_norm_p95"] < QUALITY.DEFAULT_LIMITS["eef_jerk_norm_p95"]
    assert metrics["eef_jerk_norm_max"] > QUALITY.DEFAULT_LIMITS["eef_jerk_norm_max"]
    assert not any(reason.startswith("eef_jerk_norm_p95=") for reason in rejections)
    assert any(reason.startswith("eef_jerk_norm_max=") for reason in rejections)


def test_custom_limits_do_not_mutate_metrics_or_defaults():
    metrics = QUALITY.trajectory_quality(*trajectory())
    before = json.dumps(metrics, sort_keys=True)

    assert QUALITY.quality_rejections(metrics, {"eef_acceleration_norm_p95": 1e-10}) == []
    assert json.dumps(metrics, sort_keys=True) == before
    assert QUALITY.DEFAULT_LIMITS["eef_acceleration_norm_p95"] == 4.2
    with pytest.raises(ValueError, match="unknown quality limits"):
        QUALITY.quality_rejections(metrics, {"not_a_metric": 1.0})
