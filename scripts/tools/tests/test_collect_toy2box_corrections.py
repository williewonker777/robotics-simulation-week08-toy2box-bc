"""Simulation-free focused tests for the Toy2Box correction collector."""

from __future__ import annotations

import builtins
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest


TOOLS = Path(__file__).resolve().parents[1]
SCRIPT = TOOLS / "collect_toy2box_corrections.py"


@pytest.fixture(scope="module")
def collector():
    forbidden = {"gymnasium", "isaaclab", "isaaclab_tasks", "robomimic", "torch"}
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in forbidden:
            raise AssertionError(f"simulator dependency imported at module load: {name}")
        return original_import(name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location("collect_toy2box_corrections_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(TOOLS))
    try:
        builtins.__import__ = guarded_import
        spec.loader.exec_module(module)
    finally:
        builtins.__import__ = original_import
        sys.path.remove(str(TOOLS))
    return module


def state(*, position=(0.5, -0.2, 0.05), quaternion=(1.0, 0.0, 0.0, 0.0)):
    return {
        "truck": np.asarray(position, dtype=float),
        "truck_quat": np.asarray(quaternion, dtype=float),
        "eef": np.array([0.5, -0.2, 0.2]),
        "quat": np.array([1.0, 0.0, 0.0, 0.0]),
        "box": np.array([0.8, 0.05, 0.05]),
        "arm_joints": np.zeros(7),
    }


def test_module_import_is_simulation_free(collector):
    assert collector.POLICY_HORIZON == 800
    assert collector.EXPERT_HORIZON == 600


def test_disable_fabric_cli_defaults_false_and_can_be_enabled(collector):
    parser = argparse.ArgumentParser()
    collector.add_collector_args(parser)
    required = [
        "--checkpoint", "model.pth",
        "--seed", "4361",
        "--num_corrections", "1",
        "--max_rollouts", "3",
        "--dataset_file", "out.hdf5",
        "--log_file", "out.jsonl",
    ]
    assert parser.parse_args(required).disable_fabric is False
    assert parser.parse_args([*required, "--disable_fabric"]).disable_fabric is True


def test_main_logs_exception_and_forces_nonzero_after_app_close(collector, tmp_path, monkeypatch):
    log_file = tmp_path / "collector.jsonl"
    log_file.touch()
    args = argparse.Namespace(log_file=log_file)
    close_observation = {}

    class KitApp:
        def post_quit(self, code):
            close_observation["post_quit"] = code

    class App:
        _app = KitApp()

        def close(self):
            close_observation["records"] = [json.loads(line) for line in log_file.read_text().splitlines()]

    class Launcher:
        def __init__(self, _args):
            self.app = App()

    monkeypatch.setattr(collector, "parse_args", lambda: (args, Launcher))

    def fail_collect(_args, _app):
        raise AttributeError("missing disable_fabric")

    monkeypatch.setattr(collector, "collect", fail_collect)
    monkeypatch.setattr(collector.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    with pytest.raises(SystemExit) as exit_info:
        collector.main()

    assert exit_info.value.code == 1
    assert close_observation["post_quit"] == 1
    [summary] = close_observation["records"]
    assert summary == {
        "error": "missing disable_fabric",
        "error_type": "AttributeError",
        "event": "summary",
        "phase": "collect",
        "record_type": "summary",
        "status": "exception",
    }


def test_main_logs_launcher_exception_without_attempting_close(collector, tmp_path, monkeypatch):
    log_file = tmp_path / "collector.jsonl"
    log_file.touch()
    args = argparse.Namespace(log_file=log_file)

    class BrokenLauncher:
        def __init__(self, _args):
            raise RuntimeError("launcher failed")

    monkeypatch.setattr(collector, "parse_args", lambda: (args, BrokenLauncher))
    with pytest.raises(RuntimeError, match="launcher failed"):
        collector.main()

    [summary] = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert summary["status"] == "exception"
    assert summary["phase"] == "app_launch"
    assert summary["error"] == "launcher failed"


def test_failure_close_forces_nonzero_even_when_close_raises(collector, monkeypatch):
    calls = []

    class App:
        def close(self):
            calls.append("close")
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(collector.sys.stdout, "flush", lambda: calls.append("stdout_flush"))
    monkeypatch.setattr(collector.sys.stderr, "flush", lambda: calls.append("stderr_flush"))
    monkeypatch.setattr(collector.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))

    with pytest.raises(SystemExit) as exit_info:
        collector.close_app_with_failure(App(), exit_code=7)

    assert exit_info.value.code == 7
    assert calls == ["close", "stdout_flush", "stderr_flush"]


def test_failure_close_continues_when_kit_exit_code_hook_raises(collector, monkeypatch):
    calls = []

    class KitApp:
        def post_quit(self, code):
            calls.append(("post_quit", code))
            raise RuntimeError("post_quit failed")

    class App:
        _app = KitApp()

        def close(self):
            calls.append("close")

    monkeypatch.setattr(collector.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))

    with pytest.raises(SystemExit) as exit_info:
        collector.close_app_with_failure(App(), exit_code=9)

    assert exit_info.value.code == 9
    assert calls == [("post_quit", 9), "close"]


def test_create_only_outputs_reject_existing_file(collector, tmp_path):
    output = tmp_path / "new.hdf5"
    assert collector.fresh_output(output, ".hdf5") == output.resolve()
    output.touch()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        collector.fresh_output(output, ".hdf5")


def test_smoke_mode_requires_exactly_three_rollouts(collector):
    collector.validate_collection_counts(1, 3, True)
    with pytest.raises(ValueError, match="requires --max_rollouts 3"):
        collector.validate_collection_counts(1, 4, True)
    with pytest.raises(ValueError, match="cannot exceed"):
        collector.validate_collection_counts(4, 3, False)


def test_output_reservation_is_create_only_and_distinct(collector, tmp_path):
    dataset = tmp_path / "corrections.hdf5"
    log = tmp_path / "corrections.jsonl"
    collector.reserve_outputs(dataset, log)
    assert dataset.exists() and log.exists()
    with pytest.raises(FileExistsError):
        collector.reserve_outputs(dataset, tmp_path / "second.jsonl")


def test_skip_tilt_boundary_and_fixed_env_workspace_are_fail_closed(collector):
    q25 = [math.cos(math.radians(25) / 2), math.sin(math.radians(25) / 2), 0, 0]
    q_over = [math.cos(math.radians(25.01) / 2), math.sin(math.radians(25.01) / 2), 0, 0]
    nominal = state(quaternion=q25)
    nominal["grasp"] = False
    assert collector.correction_skip_reason(nominal, [-0.15, -0.2, 0.05], 0.05) is None
    toppled = state(quaternion=q_over)
    toppled["grasp"] = False
    assert collector.correction_skip_reason(toppled, [-0.15, -0.2, 0.05], 0.05) == "truck_tilt_exceeds_25_deg"
    assert collector.correction_skip_reason(nominal, [0.251, -0.2, 0.05], 0.05) == "truck_xy_outside_bounds"


def test_skip_height_grasp_and_box_clearance_contract(collector):
    nominal = state()
    nominal["grasp"] = False
    assert collector.correction_skip_reason(nominal, [-0.15, -0.2, 0.05], 0.081) == "truck_below_settled_height"
    assert collector.correction_skip_reason(nominal, [-0.15, -0.2, 0.05], 0.019) == "truck_above_settled_ungrasped"
    nominal["grasp"] = True
    assert collector.correction_skip_reason(nominal, [-0.15, -0.2, 0.05], 0.019) is None
    nominal["box"] = nominal["truck"] + np.array([0.119, 0.0, 0.0])
    assert collector.correction_skip_reason(nominal, [-0.15, -0.2, 0.05], 0.05) == "truck_within_box_exclusion_radius"


def test_detector_observation_uses_policy_close_command(collector):
    observation = collector.detector_observation(
        9, state(), np.array([0, 0, 0, 0, 0, 0, -0.01]), settled_z=0.05
    )
    assert observation.step == 9
    assert observation.action_gripper_closed is True
    np.testing.assert_array_equal(observation.truck_pos_xyz, [0.5, -0.2, 0.05])
    with pytest.raises(ValueError, match="seven"):
        collector.detector_observation(1, state(), np.zeros(6), settled_z=0.05)


def test_settled_height_matches_preregistered_first_30_median(collector):
    states = [state(position=(0.5, -0.2, z)) for z in np.linspace(0.069, 0.050, 30)]
    assert collector.settled_height_from_prefix(states) == pytest.approx(
        np.median(np.linspace(0.069, 0.050, 30))
    )
    with pytest.raises(ValueError, match="exactly 30"):
        collector.settled_height_from_prefix(states[:-1])


def test_early_near_close_is_fail_closed_without_omitting_action(collector):
    near = state()
    near["eef"] = near["truck"] + np.array([0.059, 0.0, 0.0])
    assert collector.is_near_close_onset(False, np.r_[np.zeros(6), -1.0], near)
    assert not collector.is_near_close_onset(True, np.r_[np.zeros(6), -1.0], near)
    near["eef"] = near["truck"] + np.array([0.060001, 0.0, 0.0])
    assert not collector.is_near_close_onset(False, np.r_[np.zeros(6), -1.0], near)


def test_object_position_is_relative_to_env_origin(collector):
    class Data:
        root_pos_w = np.array([[1.2, 1.8, 3.1]])

    class Object:
        data = Data()

    class Scene(dict):
        env_origins = np.array([[1.0, 2.0, 3.0]])

    class Env:
        scene = Scene(toy_truck=Object())

    np.testing.assert_allclose(
        collector.object_position_in_env(Env(), "toy_truck"), [0.2, -0.2, 0.1]
    )


def test_takeover_resets_prefix_then_records_exact_initial_state(collector):
    class Recorder:
        def __init__(self):
            self.calls = []

        def reset(self, ids):
            self.calls.append(("reset", ids))

        def record_post_reset(self, ids):
            self.calls.append(("record_post_reset", ids))

    recorder = Recorder()
    collector.reset_recorder_at_takeover(recorder)
    assert recorder.calls == [("reset", [0]), ("record_post_reset", [0])]


def test_recovery_expert_first_action_opens_and_uses_existing_servo_limit(collector):
    expert = collector.RecoveryExpert(state())
    action = expert.advance(state())
    assert action.shape == (7,)
    assert action[-1] == 1.0
    assert 0 < action[2] <= 0.0020001
    assert expert.phase_history == ["recovery_open_and_rise"]


def test_airborne_drop_is_not_skipped_until_safe_rise_finishes(collector):
    airborne = state(position=(0.5, -0.2, 0.09))
    airborne["grasp"] = False
    assert (
        collector.correction_skip_reason(airborne, [-0.15, -0.2, 0.09], settled_z=0.05)
        == "truck_above_settled_ungrasped"
    )

    expert = collector.RecoveryExpert(airborne)
    assert collector.evaluate_skip_after_safe_rise(
        expert, airborne, [-0.15, -0.2, 0.09], settled_z=0.05
    ) == (False, None)
    for _ in range(7):
        safe = dict(airborne)
        safe["eef"] = expert.safe_target.copy()
        expert.advance(safe)
        assert not expert.safe_rise_ready

    expert.advance(safe)
    assert expert.safe_rise_ready
    assert expert.controller.phase == "recovery_open_and_rise"
    assert collector.evaluate_skip_after_safe_rise(
        expert, airborne, [-0.15, -0.2, 0.09], settled_z=0.05
    ) == (True, "truck_above_settled_ungrasped")


def test_skip_is_evaluated_after_safe_rise_before_expert_initialization(collector):
    takeover = state(position=(0.5, -0.2, 0.09))
    takeover["grasp"] = False
    expert = collector.RecoveryExpert(takeover)
    settled = dict(takeover)
    settled["truck"] = np.array([0.5, -0.2, 0.05])
    settled["eef"] = expert.safe_target.copy()

    for _ in range(8):
        expert.advance(settled)

    assert expert.safe_rise_ready
    assert collector.evaluate_skip_after_safe_rise(
        expert, settled, [-0.15, -0.2, 0.05], 0.05
    ) == (True, None)
    expert.finish_safe_rise(settled)
    assert not expert.safe_rise_ready
    assert expert.controller.phase != "recovery_open_and_rise"


def test_recovery_state_summary_preserves_detection_state(collector):
    detection = state(position=(0.5, -0.2, 0.09))
    detection["grasp"] = False
    summary = collector.recovery_state_summary(detection, [-0.15, -0.2, 0.09])
    assert summary["truck_position"] == [0.5, -0.2, 0.09]
    assert summary["truck_position_env"] == [-0.15, -0.2, 0.09]
    assert summary["grasped"] is False


def _make_valid_hdf(path: Path):
    with h5py.File(path, "w") as output:
        data = output.create_group("data")
        data.attrs["env_args"] = "{}"
        data.attrs["total"] = 3
        episode = data.create_group("demo_0")
        episode.attrs["num_samples"] = 3
        episode.attrs["success"] = True
        for key, width in {
            "actions": 7,
            "processed_actions": 9,
            "obs/actions": 7,
            "obs/box_orientations": 4,
            "obs/box_positions": 3,
            "obs/eef_pos": 3,
            "obs/eef_quat": 4,
            "obs/gripper_pos": 2,
            "obs/joint_pos": 34,
            "obs/joint_vel": 34,
            "obs/toy_truck_orientations": 4,
            "obs/toy_truck_positions": 3,
        }.items():
            episode.create_dataset(key, data=np.zeros((3, width), dtype=np.float32))
        widths = {
            "articulation/robot/joint_position": 34,
            "articulation/robot/joint_velocity": 34,
            "articulation/robot/root_pose": 7,
            "articulation/robot/root_velocity": 6,
            "rigid_object/box/root_pose": 7,
            "rigid_object/box/root_velocity": 6,
            "rigid_object/toy_truck/root_pose": 7,
            "rigid_object/toy_truck/root_velocity": 6,
        }
        for prefix, leading in (("initial_state", 1), ("states", 3)):
            for key, width in widths.items():
                episode.create_dataset(f"{prefix}/{key}", data=np.zeros((leading, width), dtype=np.float32))


def test_hdf_schema_matches_e0_recorder_contract(collector, tmp_path):
    path = tmp_path / "valid.hdf5"
    _make_valid_hdf(path)
    assert collector.validate_dataset_schema(path, expected_episodes=1) == {
        "episodes": 1,
        "total_samples": 3,
        "compatible": True,
    }


def test_hdf_schema_rejects_misaligned_or_missing_keys(collector, tmp_path):
    path = tmp_path / "invalid.hdf5"
    _make_valid_hdf(path)
    with h5py.File(path, "a") as output:
        del output["data/demo_0/obs/eef_pos"]
    with pytest.raises(ValueError, match="missing obs/eef_pos"):
        collector.validate_dataset_schema(path, expected_episodes=1)
