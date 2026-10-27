import importlib.util
import math
from pathlib import Path
import types

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "generate_toy2box_settled.py"
SPEC = importlib.util.spec_from_file_location("generate_toy2box_settled_yaw_fold", SCRIPT)
SETTLED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SETTLED)


def pose(yaw_degrees, x=0.0):
    yaw = math.radians(yaw_degrees)
    result = torch.eye(4, dtype=torch.float64)
    result[0, 0] = math.cos(yaw)
    result[0, 1] = -math.sin(yaw)
    result[1, 0] = math.sin(yaw)
    result[1, 1] = math.cos(yaw)
    result[0, 3] = x
    return result


class Records:
    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(record)


class Base:
    def __init__(self, env, pool):
        self.env = env
        self.env_cfg = types.SimpleNamespace(
            subtask_configs={
                "agibot": [
                    types.SimpleNamespace(object_ref="toy_truck"),
                    types.SimpleNamespace(object_ref="box"),
                ]
            }
        )
        self.src_demo_datagen_info_pool = pool
        self.selection_inputs = []

    def select_source_demo(self, **kwargs):
        self.selection_inputs.append(kwargs)
        return 0

    def generate_eef_subtask_trajectory(
        self,
        env_id,
        eef_name,
        subtask_ind,
        all_randomized_subtask_boundaries,
        runtime_subtask_constraints_dict,
        selected_src_demo_inds,
    ):
        object_ref = self.env_cfg.subtask_configs[eef_name][subtask_ind].object_ref
        object_pose = self.env.get_object_poses(env_ids=[env_id])[object_ref][0]
        return self.select_source_demo(
            eef_name=eef_name,
            eef_pose=torch.eye(4),
            object_pose=object_pose,
            src_demo_current_subtask_boundaries=all_randomized_subtask_boundaries[eef_name][
                :, subtask_ind
            ],
            subtask_object_name=object_ref,
            selection_strategy_name="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 3, "pos_weight": 1.0, "rot_weight": 1.0},
        )


def make_pool(truck_yaws=(120.0, 10.0, -30.0), box_yaws=(25.0, 25.0, 25.0)):
    info = types.SimpleNamespace(
        object_poses={
            "toy_truck": torch.stack([pose(yaw, x=index) for index, yaw in enumerate(truck_yaws)]),
            "box": torch.stack([pose(yaw, x=10 + index) for index, yaw in enumerate(box_yaws)]),
        }
    )
    return types.SimpleNamespace(
        datagen_infos=[info],
        subtask_boundaries={"agibot": [[(0, 2), (2, 3)]]},
    )


class Env:
    def __init__(self, truck_yaws):
        self.trucks = torch.stack([pose(yaw) for yaw in truck_yaws])
        self.boxes = torch.stack([pose(30 + index) for index in range(len(truck_yaws))])

    def get_object_poses(self, *, env_ids):
        return {"toy_truck": self.trucks[env_ids], "box": self.boxes[env_ids]}


@pytest.mark.parametrize(
    ("yaw", "expected_yaw", "expected_fold"),
    [(-90.0, 90.0, True), (90.0, 90.0, False), (180.0, 0.0, True)],
)
def test_fold_formula_and_half_open_boundaries(yaw, expected_yaw, expected_fold):
    original = pose(yaw, x=0.37)
    folded, did_fold, original_yaw, folded_yaw = SETTLED._fold_pose_to_truck_symmetry(original)
    assert math.degrees(original_yaw) == pytest.approx(180.0 if yaw == 180.0 else yaw)
    assert did_fold is expected_fold
    assert math.degrees(folded_yaw) == pytest.approx(expected_yaw, abs=1e-10)
    assert folded[:3, 3].tolist() == pytest.approx(original[:3, 3].tolist())
    assert torch.equal(original, pose(yaw, x=0.37))


def test_source_episode_uses_one_start_decision_for_all_steps_and_leaves_box_unchanged():
    pool = make_pool()
    truck_before = pool.datagen_infos[0].object_poses["toy_truck"].clone()
    box_before = pool.datagen_infos[0].object_poses["box"].clone()
    generator = types.SimpleNamespace(src_demo_datagen_info_pool=pool)

    SETTLED._fold_source_truck_poses(generator)

    expected = SETTLED._fold_pose_about_local_z(truck_before)
    assert torch.equal(pool.datagen_infos[0].object_poses["toy_truck"], expected)
    assert torch.equal(pool.datagen_infos[0].object_poses["box"], box_before)
    assert torch.equal(generator._truck_yaw_fold_source_original_poses[0], truck_before)
    assert generator._truck_yaw_fold_source_metadata[0]["folded"] is True


def test_option_off_preserves_source_and_target_objects_exactly():
    pool = make_pool()
    env = Env([120.0])
    truck_before = pool.datagen_infos[0].object_poses["toy_truck"].clone()
    coordinator = types.SimpleNamespace(attempts={0: 1}, logger=Records())
    cls = SETTLED.make_settled_generator(Base, coordinator)
    generator = cls(env, pool)
    boundaries = {"agibot": torch.tensor([[[0, 2], [2, 3]]])}

    result = generator.generate_eef_subtask_trajectory(0, "agibot", 0, boundaries, {}, {"agibot": None})

    assert result == 0
    assert torch.equal(pool.datagen_infos[0].object_poses["toy_truck"], truck_before)
    assert torch.equal(generator.selection_inputs[0]["object_pose"], env.trucks[0])
    assert not coordinator.logger.records


def test_selection_audit_reports_source_and_target_folds_and_restores_env_method():
    pool = make_pool()
    env = Env([120.0, 20.0])
    original_method = env.get_object_poses
    settling_log = Records()
    fold_log = Records()
    coordinator = types.SimpleNamespace(attempts={0: 4, 1: 7}, logger=settling_log)
    cls = SETTLED.make_settled_generator(
        Base,
        coordinator,
        enable_source_audit=True,
        enable_truck_yaw_fold=True,
        truck_yaw_fold_logger=fold_log,
    )
    generator = cls(env, pool)
    boundaries = {"agibot": torch.tensor([[[0, 2], [2, 3]]])}
    original_boxes = env.boxes.clone()

    generator.generate_eef_subtask_trajectory(0, "agibot", 0, boundaries, {}, {"agibot": None})
    generator.generate_eef_subtask_trajectory(1, "agibot", 0, boundaries, {}, {"agibot": None})

    assert env.get_object_poses.__func__ is original_method.__func__
    assert torch.equal(env.boxes, original_boxes)
    assert len(fold_log.records) == 2
    first, second = fold_log.records
    assert first == pytest.approx(
        {
            **{key: value for key, value in first.items() if not isinstance(value, float)},
            "target_original_yaw_deg": 120.0,
            "target_folded_yaw_deg": -60.0,
            "source_original_yaw_deg": 120.0,
            "source_folded_yaw_deg": -60.0,
            "abs_delta_yaw_deg": 0.0,
        }
    )
    assert first["env_id"] == 0 and first["attempt"] == 4
    assert first["subtask"] == "agibot/0" and first["selected_source_demo_index"] == 0
    assert first["target_folded"] is True and first["source_folded"] is True
    assert generator.selection_inputs[0]["selection_strategy_name"] == "nearest_neighbor_object"
    assert generator.selection_inputs[0]["selection_strategy_kwargs"] == {
        "nn_k": 3,
        "pos_weight": 1.0,
        "rot_weight": 1.0,
    }
    assert second["env_id"] == 1 and second["attempt"] == 7
    assert second["target_folded"] is False and second["source_folded"] is True
    assert second["abs_delta_yaw_deg"] == pytest.approx(80.0)
    assert torch.allclose(generator.selection_inputs[0]["object_pose"], pose(-60.0), atol=1e-12)
    assert torch.allclose(generator.selection_inputs[1]["object_pose"], pose(20.0), atol=1e-12)


def test_target_fold_proxy_restores_after_parent_error():
    class RaisingBase(Base):
        def generate_eef_subtask_trajectory(self, *args, **kwargs):
            self.env.get_object_poses(env_ids=[kwargs["env_id"]])
            raise RuntimeError("parent failed")

    pool = make_pool()
    env = Env([140.0])
    original_method = env.get_object_poses
    coordinator = types.SimpleNamespace(attempts={0: 1}, logger=Records())
    cls = SETTLED.make_settled_generator(
        RaisingBase,
        coordinator,
        enable_truck_yaw_fold=True,
        truck_yaw_fold_logger=Records(),
    )
    generator = cls(env, pool)
    boundaries = {"agibot": torch.tensor([[[0, 2], [2, 3]]])}

    with pytest.raises(RuntimeError, match="parent failed"):
        generator.generate_eef_subtask_trajectory(
            env_id=0,
            eef_name="agibot",
            subtask_ind=0,
            all_randomized_subtask_boundaries=boundaries,
            runtime_subtask_constraints_dict={},
            selected_src_demo_inds={"agibot": None},
        )

    assert env.get_object_poses.__func__ is original_method.__func__
    assert not hasattr(generator, "_source_selection_context")


def test_fold_cli_requires_a_fresh_dedicated_log(tmp_path):
    with pytest.raises(SystemExit):
        SETTLED.parse_args(["--settling_log", "settle.jsonl", "--truck_yaw_fold"])
    settings, remaining = SETTLED.parse_args(
        [
            "--settling_log",
            "settle.jsonl",
            "--truck_yaw_fold",
            "--truck_yaw_fold_log",
            str(tmp_path / "fold.jsonl"),
            "--headless",
        ]
    )
    assert settings.truck_yaw_fold is True
    assert settings.truck_yaw_fold_log == str(tmp_path / "fold.jsonl")
    assert remaining == ["--headless"]

    source = tmp_path / "source.hdf5"
    source.write_bytes(b"source")
    fold_log = tmp_path / "already-there.jsonl"
    fold_log.write_text("occupied\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already-there.jsonl"):
        SETTLED.validate_invocation(
            [
                "--headless",
                "--task",
                SETTLED.EXPECTED_TASK,
                "--input_file",
                str(source),
                "--output_file",
                str(tmp_path / "new.hdf5"),
            ],
            str(tmp_path / "settle.jsonl"),
            str(fold_log),
        )
