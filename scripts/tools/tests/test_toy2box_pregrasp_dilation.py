import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import types

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "generate_toy2box_settled.py"
SPEC = importlib.util.spec_from_file_location("generate_toy2box_settled_h5_test", SCRIPT)
SETTLED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SETTLED)


class Sequence:
    def __init__(self, values):
        self.sequence = list(values)

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index):
        return self.sequence[index]


class BoundedTrajectory:
    """Match WaypointTrajectory's bounded indexing and sequence storage."""

    def __init__(self, groups):
        self.waypoint_sequences = [Sequence(group) for group in groups]

    def __len__(self):
        return sum(len(sequence) for sequence in self.waypoint_sequences)

    def __getitem__(self, index):
        assert 0 <= index < len(self)
        offset = 0
        for sequence in self.waypoint_sequences:
            if index < offset + len(sequence):
                return sequence[index - offset]
            offset += len(sequence)
        raise AssertionError("unreachable")


class Log:
    def __init__(self):
        self.records = []

    def write(self, record):
        # Exercise JSON serializability of the audit contract.
        self.records.append(json.loads(json.dumps(record)))


def pose(x, yaw_deg):
    yaw = math.radians(yaw_deg)
    value = torch.eye(4, dtype=torch.float64)
    value[:2, :2] = torch.tensor(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=torch.float64,
    )
    value[0, 3] = x
    value[1, 3] = 0.25 * x
    value[2, 3] = 0.4
    return value


def yaw_deg(value):
    return math.degrees(math.atan2(float(value[1, 0]), float(value[0, 0])))


def waypoint(index, *, closed=False, noise=0.0, yaw=0.0):
    return types.SimpleNamespace(
        pose=pose(index * 0.01, yaw),
        gripper_action=torch.tensor([-1.0 if closed else 1.0], dtype=torch.float64),
        noise=torch.tensor(noise, dtype=torch.float64),
        marker=f"original-{index}",
    )


def config():
    return types.SimpleNamespace(
        subtask_configs={
            "agibot": [
                types.SimpleNamespace(object_ref="toy_truck"),
                types.SimpleNamespace(object_ref="box"),
            ]
        }
    )


def make_generator(pick, box, *, factor):
    source_pool = torch.arange(12, dtype=torch.float64).reshape(3, 4)

    class Base:
        def __init__(self, env):
            self.env = env
            self.env_cfg = config()
            self.source_pool = source_pool
            self.calls = []

        def generate_eef_subtask_trajectory(self, **kwargs):
            self.calls.append(kwargs)
            return pick if kwargs["subtask_ind"] == 0 else box

    logger = Log()
    coordinator = types.SimpleNamespace(logger=logger, attempts={7: 13}, torch=torch)
    generator = SETTLED.make_settled_generator(
        Base,
        coordinator,
        pregrasp_time_dilation=factor,
    )(object())
    return generator, logger, source_pool


def call(generator, subtask):
    return generator.generate_eef_subtask_trajectory(
        env_id=7,
        eef_name="agibot",
        subtask_ind=subtask,
        all_randomized_subtask_boundaries={},
        runtime_subtask_constraints_dict={},
        selected_src_demo_inds={"agibot": 2},
    )


def test_disabled_path_is_exact_identity_and_box_is_never_changed():
    pick = BoundedTrajectory([[waypoint(index) for index in range(12)]])
    box = BoundedTrajectory([[waypoint(20)]])
    generator, logger, source_pool = make_generator(pick, box, factor=1)
    pool_before = source_pool.clone()

    assert call(generator, 0) is pick
    assert call(generator, 1) is box
    assert logger.records == []
    assert torch.equal(source_pool, pool_before)


def test_factor_two_inserts_nine_shortest_rotation_midpoints_and_preserves_originals():
    yaws = [0.0] * 5 + [172, 174, 176, 178, -180, -178, -176, -174, -172, -170]
    originals = [waypoint(index, yaw=yaws[index]) for index in range(15)]
    originals += [waypoint(index, closed=True, yaw=-168 + index - 15) for index in range(15, 19)]
    original_snapshot = copy.deepcopy(originals)
    pick = BoundedTrajectory([originals[:7], originals[7:13], originals[13:]])
    box = BoundedTrajectory([[waypoint(30)]])
    generator, logger, source_pool = make_generator(pick, box, factor=2)
    pool_before = source_pool.clone()

    result = call(generator, 0)
    box_result = call(generator, 1)

    assert result is not pick
    assert len(result) == len(originals) + 9
    assert box_result is box
    assert torch.equal(source_pool, pool_before)
    assert len(pick) == len(originals)
    for index, expected in enumerate(original_snapshot):
        shifted = index + min(max(index - 5, 0), 9)
        actual = result[shifted]
        assert torch.equal(actual.pose, expected.pose)
        assert torch.equal(actual.gripper_action, expected.gripper_action)
        assert torch.equal(actual.noise, expected.noise)
        assert actual.marker == expected.marker
        assert torch.equal(pick[index].pose, expected.pose)

    midpoint_indices = [6 + 2 * index for index in range(9)]
    for midpoint_index in midpoint_indices:
        midpoint = result[midpoint_index]
        left, right = result[midpoint_index - 1], result[midpoint_index + 1]
        assert torch.allclose(midpoint.pose[:3, 3], 0.5 * (left.pose[:3, 3] + right.pose[:3, 3]))
        assert float(midpoint.gripper_action.item()) == 1.0
        assert float(midpoint.noise.item()) == 0.0
    # The midpoint between +178 and -180 degrees follows the two-degree path,
    # rather than rotating through zero.
    assert abs(abs(yaw_deg(result[12].pose)) - 179.0) < 1e-8

    assert len(logger.records) == 1
    audit = logger.records[0]
    assert audit["record_type"] == "pregrasp_time_dilation"
    assert audit["env_id"] == 7 and audit["attempt"] == 13
    assert audit["subtask"] == "agibot/0" and audit["factor"] == 2
    assert audit["original_waypoint_count"] == 19
    assert audit["new_waypoint_count"] == 28
    assert audit["inserted_waypoint_count"] == 9
    assert audit["first_close_index_before"] == 15
    assert audit["first_close_index_after"] == 24
    assert audit["midpoint_indices"] == midpoint_indices
    assert len(audit["window_pose_before"]) == 10
    assert len(audit["window_pose_after"]) == 19
    assert audit["linear_path_length_after_m"] == pytest.approx(
        audit["linear_path_length_before_m"], abs=1e-12
    )
    assert audit["minimum_expanded_translation_interval_m"] > 1e-6
    assert audit["application_count"] == 1 and audit["applied_exactly_once"] is True
    with pytest.raises(SETTLED.ConfigurationError, match="already applied"):
        SETTLED._dilate_pregrasp_trajectory(result, torch_module=torch)


@pytest.mark.parametrize(
    "trajectory, message",
    [
        (BoundedTrajectory([[waypoint(i, closed=i >= 9) for i in range(12)]]), "ten open"),
        (BoundedTrajectory([[waypoint(i) for i in range(12)]]), "negative gripper"),
        (
            BoundedTrajectory(
                [[waypoint(i, closed=i >= 10, noise=0.1 if i == 5 else 0.0) for i in range(13)]]
            ),
            "zero waypoint noise",
        ),
    ],
)
def test_boundary_contract_fails_closed(trajectory, message):
    with pytest.raises(SETTLED.ConfigurationError, match=message):
        SETTLED._dilate_pregrasp_trajectory(trajectory, torch_module=torch)


def test_antipodal_rotation_midpoint_fails_closed():
    first, second = pose(0.0, 0.0), pose(0.1, 180.0)
    with pytest.raises(SETTLED.ConfigurationError, match="antipodal"):
        SETTLED._se3_midpoint(first, second, torch)


def d3_args():
    return [
        "--settling_log", "log",
        "--box_robot_distance",
        "--enable_box_reselection",
        "--zero_action_noise",
        "--settle_arm_motion",
        "--settle_arm_joint_velocity", "0.15",
        "--settle_eef_velocity", "0.01",
        "--settle_max_steps", "20",
        "--grasp_offset_range", "50", "60",
        "--box_interpolation_steps", "30",
        "--per_env_rmp_reset",
        "--chunk_seed", "1",
        "--settle_reset_retries", "2",
    ]


def test_cli_accepts_only_registered_factor_and_exact_stock_d3_scope6():
    settings, remaining = SETTLED.parse_args([*d3_args(), "--pregrasp_time_dilation", "2"])
    assert settings.pregrasp_time_dilation == 2 and remaining == []
    for invalid in (
        ["--pregrasp_time_dilation", "3"],
        ["--pregrasp_time_dilation", "2", "--truck_pick_z_offset_m", "-0.005"],
        ["--pregrasp_time_dilation", "2", "--truck_yaw_fold", "--truck_yaw_fold_log", "fold.jsonl"],
    ):
        with pytest.raises(SystemExit):
            SETTLED.parse_args([*d3_args(), *invalid])
    with pytest.raises(SystemExit):
        SETTLED.parse_args(["--settling_log", "log", "--pregrasp_time_dilation", "2"])
    with pytest.raises(SETTLED.ConfigurationError, match="must be 1 or 2"):
        SETTLED.make_settled_generator(object, types.SimpleNamespace(), pregrasp_time_dilation=3)
