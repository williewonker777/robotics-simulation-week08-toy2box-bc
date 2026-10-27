# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the Toy2Box scripted demonstration policy.

These tests intentionally import only the script's pure-Python policy surface.  They
must never start Isaac Sim or require an Isaac Lab application context.
"""

from __future__ import annotations

import argparse
import builtins
import importlib.util
import itertools
import random
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "record_toy2box_scripted_demos.py"
POSE_ALPHABET = np.array([-0.05, 0.0, 0.05], dtype=np.float32)
GRIPPER_ALPHABET = np.array([-1.0, 1.0], dtype=np.float32)


def _values_are_in(actual, alphabet):
    actual = np.asarray(actual)
    return np.all(np.any(np.isclose(actual[..., None], alphabet, atol=1e-8), axis=-1))


@pytest.fixture(scope="module")
def collector_module():
    """Load the collector while making any eager simulator import a test failure."""
    forbidden_roots = {"gymnasium", "isaaclab", "isaaclab_tasks", "torch"}
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in forbidden_roots:
            raise AssertionError(f"simulator dependency imported at module load: {name}")
        return original_import(name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location("record_toy2box_scripted_demos_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        builtins.__import__ = guarded_import
        spec.loader.exec_module(module)
    finally:
        builtins.__import__ = original_import
    return module


def test_module_import_does_not_require_simulator(collector_module):
    assert collector_module.Toy2BoxPolicy is not None
    assert callable(collector_module.check_output_path)


@pytest.mark.parametrize("tolerance", [0.01, np.array([0.01, 0.02, 0.03])])
def test_discrete_command_is_bounded_to_keyboard_pose_alphabet(collector_module, tolerance):
    error = np.array([-100.0, 0.0, 100.0])
    command = collector_module.discrete_command(error, tolerance)

    assert _values_are_in(command, POSE_ALPHABET)
    assert np.max(np.abs(command)) <= 0.05


def test_discrete_command_obeys_dead_band_boundaries(collector_module):
    tolerance = np.array([0.01, 0.02, 0.03])
    error = np.array([0.01, -0.021, 0.031])

    np.testing.assert_array_equal(
        collector_module.discrete_command(error, tolerance),
        np.array([0.0, -0.05, 0.05]),
    )


def _axis_angle_quaternion(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    return np.r_[np.cos(angle / 2), axis * np.sin(angle / 2)]


def test_rotation_error_has_expected_axis_angle_and_shortest_sign(collector_module):
    target = _axis_angle_quaternion([0, 0, 1], np.pi / 2)
    identity = np.array([1.0, 0.0, 0.0, 0.0])

    error = collector_module.rotation_error(target, identity)
    np.testing.assert_allclose(error, [0.0, 0.0, np.pi / 2], atol=1e-7)
    np.testing.assert_allclose(
        collector_module.rotation_error(-target, identity), error, atol=1e-7
    )
    np.testing.assert_allclose(
        collector_module.rotation_error(target, -identity), error, atol=1e-7
    )


def test_rotation_error_is_zero_for_equivalent_quaternions(collector_module):
    orientation = _axis_angle_quaternion([1, -2, 3], 1.2)
    np.testing.assert_allclose(
        collector_module.rotation_error(orientation, orientation), np.zeros(3), atol=1e-7
    )
    np.testing.assert_allclose(
        collector_module.rotation_error(orientation, -orientation), np.zeros(3), atol=1e-7
    )


def test_output_path_accepts_new_file_and_requires_overwrite_for_existing_file(
    collector_module, tmp_path, monkeypatch
):
    protected = tmp_path / "protected.hdf5"
    protected.write_bytes(b"user demonstration")
    monkeypatch.setattr(collector_module, "PROTECTED_DATASET", protected)
    output = tmp_path / "new" / "codex.hdf5"

    assert collector_module.check_output_path(output, overwrite=False) == output.resolve()
    output.parent.mkdir()
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="Output already exists"):
        collector_module.check_output_path(output, overwrite=False)
    assert collector_module.check_output_path(output, overwrite=True) == output.resolve()


def test_output_path_always_rejects_protected_canonical_path(
    collector_module, tmp_path, monkeypatch
):
    protected = tmp_path / "toy2box_agibot.hdf5"
    protected.write_bytes(b"user demonstration")
    monkeypatch.setattr(collector_module, "PROTECTED_DATASET", protected)

    for overwrite in (False, True):
        with pytest.raises(ValueError, match="read-only"):
            collector_module.check_output_path(protected, overwrite=overwrite)


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_output_path_rejects_protected_file_aliases(
    collector_module, tmp_path, monkeypatch, alias_kind
):
    protected = tmp_path / "toy2box_agibot.hdf5"
    protected.write_bytes(b"user demonstration")
    alias = tmp_path / f"{alias_kind}.hdf5"
    if alias_kind == "symlink":
        alias.symlink_to(protected)
    else:
        alias.hardlink_to(protected)
    monkeypatch.setattr(collector_module, "PROTECTED_DATASET", protected)

    for overwrite in (False, True):
        with pytest.raises(ValueError, match="read-only"):
            collector_module.check_output_path(alias, overwrite=overwrite)


def _synthetic_state(collector_module, offset=(0.2, -0.2, 0.2)):
    truck = np.array([0.0, -0.2, 0.4], dtype=float)
    return {
        "eef": truck + np.asarray(offset, dtype=float),
        "quat": collector_module.GRASP_QUAT.copy(),
        "truck": truck,
        "truck_quat": _axis_angle_quaternion([0, 0, 1], 0.95),
        "box": np.array([0.3, 0.05, 0.5], dtype=float),
        "truck_vel": np.zeros(3, dtype=float),
        "grasp": False,
        "forces": [0.0, 0.0],
    }


def test_state_machine_uses_only_keyboard_action_alphabet_for_synthetic_states(collector_module):
    # Do not assert phase timing or tuned offsets: exercise each named policy mode with
    # varied finite feedback and lock only the externally required action contract.
    phases = ("settle", "hover", "descend", "close", "lift", "carry", "align", "lower", "release", "done")
    offsets = itertools.product((-0.2, 0.0, 0.2), repeat=3)

    for phase, offset in itertools.product(phases, offsets):
        policy = collector_module.Toy2BoxPolicy()
        policy.phase = phase
        policy.pickup = np.array([0.01, -0.18, 0.39], dtype=float)
        policy.lift_height = 0.8
        policy.hold_offset = np.array([0.01, 0.02, 0.25], dtype=float)
        action = policy.advance(_synthetic_state(collector_module, offset))

        assert action.shape == (7,)
        assert np.count_nonzero(action[3:6]) <= 1
        assert np.all(np.isfinite(action))
        assert _values_are_in(action[:6], POSE_ALPHABET), (phase, offset, action)
        assert _values_are_in(action[6], GRIPPER_ALPHABET), (phase, offset, action)



def test_reference_truck_yaw_preserves_reference_grasp_pose(collector_module):
    policy = collector_module.Toy2BoxPolicy()
    state = _synthetic_state(collector_module)
    for _ in range(25):
        policy.advance(state)
    np.testing.assert_allclose(policy.grasp_quat, collector_module.GRASP_QUAT, atol=1e-7)
    np.testing.assert_allclose(policy.grasp_offset, [0.020, 0.015, -0.010], atol=1e-7)


def test_rotation_commands_are_literal_keyboard_outputs(collector_module):
    from scipy.spatial.transform import Rotation

    for error in itertools.product((-0.5, 0.0, 0.5), repeat=3):
        command = collector_module.keyboard_rotation_command(np.array(error))
        assert np.count_nonzero(command) <= 1
        assert _values_are_in(command, POSE_ALPHABET)
        np.testing.assert_array_equal(
            Rotation.from_euler("XYZ", command).as_rotvec().astype(np.float32), command
        )


def _rotate_vector(quaternion, vector):
    """Rotate a vector by a unit wxyz quaternion without simulator dependencies."""
    w, x, y, z = np.asarray(quaternion, dtype=float)
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return rotation @ np.asarray(vector, dtype=float)


@pytest.mark.parametrize("truck_yaw", [-np.pi, -2.2, -0.1, 0.0, 0.95, 2.4, np.pi])
def test_top_down_settle_maps_local_z_down_and_uses_centered_grasp_offset(
    collector_module, truck_yaw
):
    policy = collector_module.Toy2BoxPolicy(top_down=True)
    state = _synthetic_state(collector_module)
    state["truck_quat"] = _axis_angle_quaternion([0, 0, 1], truck_yaw)

    for _ in range(25):
        policy.advance(state)

    assert policy.phase == "hover"
    np.testing.assert_allclose(policy.grasp_offset, [0.0, 0.0, 0.005], atol=1e-9)
    np.testing.assert_allclose(policy.pickup, state["truck"] + [0.0, 0.0, 0.005], atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(policy.grasp_quat), 1.0, atol=1e-9)
    np.testing.assert_allclose(
        _rotate_vector(policy.grasp_quat, [0.0, 0.0, 1.0]),
        [0.0, 0.0, -1.0],
        atol=1e-7,
    )


@pytest.mark.parametrize("phase", ["hover", "descend"])
@pytest.mark.parametrize(
    "current_quat",
    [
        np.array([1.0, 0.0, 0.0, 0.0]),
        _axis_angle_quaternion([1.0, 2.0, -3.0], 0.7),
    ],
)
def test_top_down_motion_uses_keyboard_alphabet_and_one_rotation_key(
    collector_module, phase, current_quat
):
    policy = collector_module.Toy2BoxPolicy(top_down=True)
    state = _synthetic_state(collector_module, offset=(0.2, -0.2, 0.2))
    state["truck_quat"] = _axis_angle_quaternion([0, 0, 1], 0.4)
    for _ in range(25):
        policy.advance(state)
    policy.phase = phase
    state["quat"] = current_quat

    action = policy.advance(state)

    assert action.shape == (7,)
    assert _values_are_in(action[:6], POSE_ALPHABET)
    assert _values_are_in(action[6], GRIPPER_ALPHABET)
    assert np.count_nonzero(action[3:6]) == 1


def test_top_down_release_nudge_is_short_and_keyboard_valued(collector_module):
    policy = collector_module.Toy2BoxPolicy(top_down=True)
    policy.phase = "release"
    state = _synthetic_state(collector_module)
    for step in range(1, 7):
        action = policy.advance(state)
        assert _values_are_in(action[:6], POSE_ALPHABET)
        assert action[6] == 1
        assert np.count_nonzero(action[3:6]) == 0
        if step <= 4:
            np.testing.assert_array_equal(action[:2], np.array([0.05, 0.05], dtype=np.float32))
        else:
            np.testing.assert_array_equal(action[:2], [0.0, 0.0])


def test_smooth_servo_caps_norm_and_slew_without_keyboard_quantization(collector_module):
    servo = collector_module.SmoothPoseServo()
    previous = np.zeros(6)
    for _ in range(100):
        command = servo.advance(np.ones(3), np.ones(3))
        assert np.linalg.norm(command[:3]) <= 0.025 + 1e-8
        assert np.linalg.norm(command[3:]) <= 0.030 + 1e-8
        assert np.linalg.norm(command[:3] - previous[:3]) <= 0.002 + 1e-8
        assert np.linalg.norm(command[3:] - previous[3:]) <= 0.003 + 1e-8
        previous = command
    assert not np.isin(command, POSE_ALPHABET).all()


def test_smooth_servo_reversals_pass_through_zero_dwell(collector_module):
    servo = collector_module.SmoothPoseServo()
    commands = []
    for sign in (1.0, -1.0, 1.0):
        for _ in range(40):
            commands.append(servo.advance(np.array([sign, 0, 0]), np.zeros(3)))
    commands = np.array(commands)
    assert not np.any(commands[1:] * commands[:-1] < 0)
    negative_start = np.flatnonzero(commands[:, 0] < 0)[0]
    assert np.all(commands[negative_start - 3:negative_start, 0] == 0)


def test_smooth_servo_hysteresis_avoids_small_error_chatter(collector_module):
    servo = collector_module.SmoothPoseServo()
    for _ in range(20):
        servo.advance(np.array([0.001, 0, 0]), np.zeros(3))
    for error in (0.003, -0.003, 0.002, -0.002):
        np.testing.assert_array_equal(servo.advance(np.array([error, 0, 0]), np.zeros(3)), np.zeros(6))
    assert servo.advance(np.array([0.006, 0, 0]), np.zeros(3))[0] > 0


def test_smooth_policy_is_explicitly_continuous_and_keeps_binary_gripper(collector_module):
    policy = collector_module.Toy2BoxPolicy(top_down=True, control_mode="smooth")
    state = _synthetic_state(collector_module)
    for _ in range(30):
        action = policy.advance(state)
    assert policy.phase == "hover"
    assert action[-1] == 1
    assert np.isfinite(action).all()
    assert np.max(np.abs(action[:6])) <= 0.05
    assert not np.isin(action[:6], POSE_ALPHABET).all()


def test_policy_rejects_unknown_control_mode(collector_module):
    with pytest.raises(ValueError, match="control mode"):
        collector_module.Toy2BoxPolicy(control_mode="undefined")


def test_smooth_alignment_targets_box_not_entry_pose_or_carry_loop(collector_module):
    policy = collector_module.Toy2BoxPolicy(top_down=True, control_mode="smooth")
    state = _synthetic_state(collector_module)
    policy.lift_height = 0.8
    policy.hold_offset = np.array([0.01, 0.01, 0.01])
    policy.transition("align", state)
    np.testing.assert_allclose(policy.hold_target[:2], state["box"][:2] + policy.hold_offset[:2])
    assert policy.hold_target[2] == max(policy.lift_height, state["box"][2] + policy.hold_offset[2] + 0.22)
    # A small outward drift cannot send the smooth planner back to carry repeatedly.
    policy.advance(state)
    assert policy.phase == "align"


class _FakeResetManager:
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.action_history = 7.0

    def reset(self, env_ids):
        self.env.calls.append((f"{self.name}.reset", list(env_ids)))
        if self.name == "action":
            self.action_history = 0.0

    def compute(self, update_history):
        self.env.calls.append(("observation.compute", update_history))
        return {"policy": {"action_history": self.env.action_manager.action_history}}


class _FakeRecorderManager(_FakeResetManager):
    def __init__(self, env):
        super().__init__(env, "recorder")
        self.episode = []
        self.initial_state = None

    def reset(self, env_ids):
        super().reset(env_ids)
        self.episode.clear()

    def record_post_reset(self, env_ids):
        self.env.calls.append(("recorder.record_post_reset", list(env_ids)))
        self.initial_state = self.env.raw_state.copy()


class _FakeWarmupEnv:
    def __init__(self, terminate_at=None):
        self.calls = []
        self.raw_state = np.zeros(2)
        self.obs_buf = {"policy": {"action_history": 7.0}}
        self.episode_length_buf = np.array([0])
        self.terminate_at = terminate_at
        self.observation_manager = _FakeResetManager(self, "observation")
        self.action_manager = _FakeResetManager(self, "action")
        self.recorder_manager = _FakeRecorderManager(self)

    def step(self, action):
        self.calls.append(("step", np.asarray(action).copy()))
        self.recorder_manager.episode.append(np.asarray(action).copy())
        self.raw_state += [0.001, -0.002]
        self.episode_length_buf[0] += 1
        done = self.terminate_at == self.episode_length_buf[0]
        observation = {"policy": {"action_history": np.asarray(action).copy()}}
        self.obs_buf = observation
        return observation, None, [done], [False], {}


def test_smooth_warmup_is_discarded_then_manager_state_is_reinitialized(collector_module):
    env = _FakeWarmupEnv()
    action = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    observation, steps, completed = collector_module.prepare_smooth_recording(env, action)

    assert completed and steps == 25
    step_calls = [entry for entry in env.calls if entry[0] == "step"]
    assert len(step_calls) == 25
    for _, actual in step_calls:
        np.testing.assert_array_equal(actual, action)
    assert env.calls[-5:] == [
        ("observation.reset", [0]),
        ("action.reset", [0]),
        ("recorder.reset", [0]),
        ("recorder.record_post_reset", [0]),
        ("observation.compute", True),
    ]
    assert env.recorder_manager.episode == []
    np.testing.assert_allclose(env.raw_state, [0.025, -0.050])
    assert env.recorder_manager.initial_state is not None
    np.testing.assert_array_equal(env.recorder_manager.initial_state, env.raw_state)
    assert env.episode_length_buf[0] == 0
    assert observation["policy"]["action_history"] == 0.0
    assert env.obs_buf is observation


def test_smooth_warmup_termination_discards_partial_episode(collector_module):
    env = _FakeWarmupEnv(terminate_at=3)
    action = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    _, steps, completed = collector_module.prepare_smooth_recording(env, action)

    assert not completed and steps == 3
    assert env.recorder_manager.episode == []
    assert ("recorder.reset", [0]) == env.calls[-1]
    assert not any(name in {"observation.reset", "action.reset", "recorder.record_post_reset"}
                   for name, _ in env.calls)


def test_direct_smooth_initialization_starts_hover_with_open_gripper(collector_module):
    policy = collector_module.Toy2BoxPolicy(top_down=True, control_mode="smooth")
    state = _synthetic_state(collector_module)

    policy.initialize_from_settled_state(state)
    action = policy.advance(state)

    assert policy.phase == "hover"
    assert policy.phase_history == ["hover"]
    assert policy.phase_steps == 1
    assert action[-1] == 1.0
    np.testing.assert_allclose(policy.pickup, state["truck"] + [0.0, 0.0, 0.005])


def test_smooth_quality_evaluation_uses_no_settle_skip(collector_module):
    samples = {
        "actions": [np.zeros(7) for _ in range(4)],
        "eef": [np.zeros(3) for _ in range(4)],
        "quat": [np.array([1.0, 0.0, 0.0, 0.0]) for _ in range(4)],
        "arm_joints": [np.zeros(7) for _ in range(4)],
    }
    received = {}

    def trajectory_quality(*arrays, skip):
        received["shapes"] = [array.shape for array in arrays]
        received["skip"] = skip
        return {"metric": 1.0}

    def quality_rejections(quality, limits):
        received["quality"] = quality
        received["limits"] = limits
        return []

    quality, rejections = collector_module.evaluate_smooth_quality(
        samples, trajectory_quality, quality_rejections, {"metric": 2.0}
    )

    assert collector_module.MIN_QUALITY_SAMPLES == 4
    assert received["skip"] == 0
    assert received["shapes"] == [(4, 7), (4, 3), (4, 4), (4, 7)]
    assert quality == {"metric": 1.0}
    assert rejections == []


def test_default_rng_skip_does_not_call_rng(collector_module):
    calls = []

    collector_module.consume_rng_skip_draws(0, lambda: calls.append("draw"))

    assert calls == []

    random.seed(42)
    expected_uniform = random.uniform(-1.0, 1.0)
    random.seed(42)
    collector_module.consume_rng_skip_draws(0)
    assert random.uniform(-1.0, 1.0) == expected_uniform


def test_positive_rng_skip_uses_exact_python_draw_count(collector_module):
    calls = []

    collector_module.consume_rng_skip_draws(17, lambda: calls.append("draw"))

    assert calls == ["draw"] * 17


def test_rng_skip_advances_seeded_uniform_placement_stream(collector_module):
    random.seed(42)
    unskipped = random.uniform(-1.0, 1.0)

    random.seed(42)
    expected_draw = random.random()
    expected_skipped = random.uniform(-1.0, 1.0)

    random.seed(42)
    collector_module.consume_rng_skip_draws(1)
    skipped = random.uniform(-1.0, 1.0)

    assert unskipped == 2.0 * expected_draw - 1.0
    assert skipped == expected_skipped
    assert skipped != unskipped


@pytest.mark.parametrize("value", ["-1", "1000001", "1.5", "many"])
def test_rng_skip_rejects_bad_values_before_collection(collector_module, value):
    with pytest.raises(argparse.ArgumentTypeError, match="rng_skip_draws"):
        collector_module.rng_skip_draws_arg(value)


@pytest.mark.parametrize("value", ["0", "42", "1000000"])
def test_rng_skip_accepts_inclusive_bounds(collector_module, value):
    assert collector_module.rng_skip_draws_arg(value) == int(value)
