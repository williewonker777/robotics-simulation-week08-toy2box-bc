import asyncio
import ast
import builtins
import copy
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import types

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "generate_toy2box_settled.py"
SPEC = importlib.util.spec_from_file_location("generate_toy2box_settled", SCRIPT)
SETTLED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SETTLED)


class Recorder:
    def __init__(self):
        self.calls = []
        self.episode = []

    def reset(self, env_ids):
        self.calls.append(("reset", list(env_ids)))
        self.episode = []

    def record_post_reset(self, env_ids):
        self.calls.append(("post_reset", list(env_ids)))
        self.episode.append("post_reset")


class FakeEnv:
    def __init__(self, num_envs=1):
        truck_data = types.SimpleNamespace(
            root_pos_w=torch.tensor([[0.2, -0.1, 0.50]] * num_envs, dtype=torch.float64),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_envs, dtype=torch.float64),
            root_lin_vel_w=torch.zeros((num_envs, 3), dtype=torch.float64),
            root_ang_vel_w=torch.zeros((num_envs, 3), dtype=torch.float64),
        )
        box_data = types.SimpleNamespace(
            root_pos_w=torch.tensor([[0.7, 0.1, 0.40]] * num_envs, dtype=torch.float64),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_envs, dtype=torch.float64),
        )
        self.scene = {
            "toy_truck": types.SimpleNamespace(data=truck_data),
            "box": types.SimpleNamespace(data=box_data),
        }
        self.scene = Scene(self.scene, num_envs)
        self.recorder_manager = Recorder()
        self.episode_length_buf = torch.full((num_envs,), 37, dtype=torch.int64)
        self.reset_count = 0
        self.observation_compute_count = 0
        self.action_reset_count = 0


class ArmFakeEnv(FakeEnv):
    def __init__(self, num_envs=1):
        super().__init__(num_envs)
        robot_data = types.SimpleNamespace(
            joint_vel=torch.zeros((num_envs, 20), dtype=torch.float64),
            joint_pos=torch.zeros((num_envs, 20), dtype=torch.float64),
        )

        class Robot:
            data = robot_data

            @staticmethod
            def find_joints(pattern):
                assert pattern == "right_arm_joint[1-7]"
                ids = [5, 7, 9, 11, 13, 15, 17]
                return ids, [f"right_arm_joint{index}" for index in range(1, 8)]

        self.scene["robot"] = Robot()
        self.step_dt = 0.05
        self.eef_poses = torch.eye(4, dtype=torch.float64).repeat(num_envs, 1, 1)

    def get_robot_eef_pose(self, *, env_ids, eef_name):
        assert eef_name == "agibot"
        return self.eef_poses[env_ids]


class Scene(dict):
    def __init__(self, values, num_envs):
        super().__init__(values)
        self.env_origins = torch.zeros((num_envs, 3), dtype=torch.float64)


class FakeBase:
    def __init__(self, env):
        self.env = env
        self.received = []

    async def generate(self, **kwargs):
        self.received.append(kwargs.copy())
        env_id = kwargs["env_id"]
        self.env.recorder_manager.reset([env_id])
        await kwargs["env_reset_queue"].put(env_id)
        await kwargs["env_reset_queue"].join()
        return {"success": True, "marker": 19}


class FailingBase:
    def __init__(self, env):
        self.env = env

    async def generate(self, **kwargs):
        raise RuntimeError("failed before reset")


def logger(tmp_path):
    stream = (tmp_path / "settling.jsonl").open("x", encoding="utf-8")
    return SETTLED.JsonlLog(stream), stream


async def drive_reset(queue, env, env_id=0):
    assert await queue.get() == env_id
    env.reset_count += 1
    env.recorder_manager.record_post_reset([env_id])
    env.recorder_manager.episode.append("reset_prefix")
    queue.task_done()


async def drive_actions(queue, env, speeds, env_id=0):
    actions = []
    data = env.scene["toy_truck"].data
    for index, (linear, angular) in enumerate(speeds, 1):
        received_env, action = await queue.get()
        assert received_env == env_id
        actions.append(action.clone())
        data.root_pos_w[env_id, 2] = 0.50 - index * 0.006
        data.root_lin_vel_w[env_id] = torch.tensor([linear, 0.0, 0.0])
        data.root_ang_vel_w[env_id] = torch.tensor([0.0, 0.0, angular])
        env.recorder_manager.episode.append(f"hold-{index}")
        queue.task_done()
    return actions


async def drive_arm_actions(queue, env, samples, env_id=0):
    actions = []
    for linear, angular, joint_velocity, eef_x in samples:
        received_env, action = await queue.get()
        assert received_env == env_id
        actions.append(action.clone())
        env.scene["toy_truck"].data.root_lin_vel_w[env_id] = torch.tensor([linear, 0.0, 0.0])
        env.scene["toy_truck"].data.root_ang_vel_w[env_id] = torch.tensor([0.0, 0.0, angular])
        env.scene["robot"].data.joint_vel[env_id, [5, 7, 9, 11, 13, 15, 17]] = joint_velocity
        env.scene["robot"].data.joint_pos[env_id, [5, 7, 9, 11, 13, 15, 17]] += (
            joint_velocity * env.step_dt
        )
        env.eef_poses[env_id, 0, 3] = eef_x
        env.recorder_manager.episode.append("arm-hold")
        queue.task_done()
    return actions


def test_settles_after_gravity_step_and_excludes_hold_prefix(tmp_path):
    async def scenario():
        env = FakeEnv()
        reset_queue = asyncio.Queue()
        action_queue = asyncio.Queue()
        log, stream = logger(tmp_path)
        coordinator = SETTLED.SettlingCoordinator(log, torch, 0.001, 0.02, 2, 10)
        cls = SETTLED.make_settled_generator(FakeBase, coordinator)
        generator = cls(env)
        speeds = [(0.4905, 0.0), (0.0008, 0.010), (0.0007, 0.009)]
        reset_task = asyncio.create_task(drive_reset(reset_queue, env))
        action_task = asyncio.create_task(drive_actions(action_queue, env, speeds))
        result = await generator.generate(
            env_id=0,
            success_term="same",
            env_reset_queue=reset_queue,
            env_action_queue=action_queue,
            pause_subtask=True,
            export_demo=False,
            motion_planner="planner",
        )
        actions = await action_task
        await reset_task
        coordinator.restore()
        stream.close()
        return env, generator, result, actions

    env, generator, result, actions = asyncio.run(scenario())
    assert result == {"success": True, "marker": 19}
    assert len(actions) == 3
    assert all(torch.equal(action[:6], torch.zeros(6)) for action in actions)
    assert all(action[6].item() == 1.0 and action.device.type == "cpu" for action in actions)
    assert env.recorder_manager.episode == ["post_reset"]
    assert env.episode_length_buf.tolist() == [0]
    assert env.reset_count == 1
    assert env.observation_compute_count == 0
    assert env.action_reset_count == 0
    received = generator.received[0]
    assert received["success_term"] == "same"
    assert received["pause_subtask"] is True
    assert received["export_demo"] is False
    assert received["motion_planner"] == "planner"
    records = [json.loads(line) for line in (tmp_path / "settling.jsonl").read_text().splitlines()]
    attempt = records[0]
    assert attempt["record_type"] == "settling_attempt"
    assert attempt["steps"] == 3
    assert attempt["stable"] is True and attempt["capped"] is False
    assert attempt["hold_recorded"] is False
    assert attempt["trace"][0]["linear_speed"] == pytest.approx(0.4905)
    assert records[1]["record_type"] == "generation_result"


def test_other_environment_keeps_supplying_actions_while_one_settles(tmp_path):
    async def scenario():
        env = FakeEnv(num_envs=2)
        action_queue = asyncio.Queue()
        log, stream = logger(tmp_path)
        coordinator = SETTLED.SettlingCoordinator(log, torch, 0.001, 0.02, 2, 10)
        data = env.scene["toy_truck"].data
        other_actions = []

        async def companion():
            for _ in range(3):
                await action_queue.put((1, torch.full((7,), 9.0)))
                other_actions.append(1)
                await action_queue.join()

        async def env_loop_driver():
            speeds = [0.49, 0.0008, 0.0007]
            for speed in speeds:
                batch = [await action_queue.get(), await action_queue.get()]
                assert {item[0] for item in batch} == {0, 1}
                data.root_lin_vel_w[0] = torch.tensor([speed, 0.0, 0.0])
                for _ in batch:
                    action_queue.task_done()

        companion_task = asyncio.create_task(companion())
        driver_task = asyncio.create_task(env_loop_driver())
        await coordinator.settle(env, 0, action_queue)
        await companion_task
        await driver_task
        coordinator.restore()
        stream.close()
        return other_actions

    assert asyncio.run(scenario()) == [1, 1, 1]


def test_arm_settling_waits_for_two_combined_quiet_steps_and_discards_holds(tmp_path):
    async def scenario():
        env = ArmFakeEnv()
        queue = asyncio.Queue()
        log, stream = logger(tmp_path)
        coordinator = SETTLED.SettlingCoordinator(
            log, torch, 0.001, 0.02, 2, 20,
            settle_arm_motion=True,
            arm_joint_velocity_threshold=0.05,
            eef_velocity_threshold=0.01,
        )
        samples = [
            (0.0001, 0.001, 0.10, 0.0050),
            (0.0001, 0.001, 0.04, 0.0060),
            (0.0001, 0.001, 0.04, 0.0064),
            (0.0001, 0.001, 0.03, 0.0067),
        ]
        driver = asyncio.create_task(drive_arm_actions(queue, env, samples))
        await coordinator.settle(env, 0, queue)
        actions = await driver
        coordinator.restore()
        stream.close()
        return env, actions

    env, actions = asyncio.run(scenario())
    assert len(actions) == 4
    assert env.recorder_manager.episode == ["post_reset"]
    assert env.recorder_manager.calls == [("reset", [0]), ("post_reset", [0])]
    assert env.episode_length_buf.tolist() == [0]
    assert env.reset_count == 0
    assert env.observation_compute_count == 0
    assert env.action_reset_count == 0
    record = json.loads((tmp_path / "settling.jsonl").read_text())
    assert record["steps"] == 4 and record["stable"] is True
    assert record["arm_joint_ids"] == [5, 7, 9, 11, 13, 15, 17]
    assert record["step_dt"] == pytest.approx(0.05)
    assert record["trace"][0]["arm_joint_velocity_max"] == pytest.approx(0.10)
    assert record["trace"][0]["arm_joint_velocities"] == pytest.approx([0.10] * 7)
    assert record["trace"][0]["arm_joint_positions"] == pytest.approx([0.005] * 7)
    assert record["trace"][0]["arm_joint_position_delta_speed_max"] == pytest.approx(0.10)
    assert record["trace"][0]["eef_linear_speed"] == pytest.approx(0.10)
    assert record["trace"][-2]["eef_linear_speed"] < 0.01
    assert record["trace"][-1]["eef_linear_speed"] < 0.01


def test_other_environment_keeps_supplying_actions_during_arm_settling(tmp_path):
    async def scenario():
        env = ArmFakeEnv(num_envs=2)
        queue = asyncio.Queue()
        log, stream = logger(tmp_path)
        coordinator = SETTLED.SettlingCoordinator(
            log, torch, 0.001, 0.02, 2, 20, settle_arm_motion=True
        )
        samples = [
            (0.0001, 0.001, 0.10, 0.0050),
            (0.0001, 0.001, 0.04, 0.0060),
            (0.0001, 0.001, 0.04, 0.0064),
            (0.0001, 0.001, 0.03, 0.0067),
        ]
        other_actions = []

        async def companion():
            for _ in samples:
                await queue.put((1, torch.full((7,), 9.0)))
                other_actions.append(1)
                await queue.join()

        async def driver():
            for linear, angular, joint_velocity, eef_x in samples:
                batch = [await queue.get(), await queue.get()]
                assert {item[0] for item in batch} == {0, 1}
                env.scene["toy_truck"].data.root_lin_vel_w[0] = torch.tensor([linear, 0.0, 0.0])
                env.scene["toy_truck"].data.root_ang_vel_w[0] = torch.tensor([0.0, 0.0, angular])
                env.scene["robot"].data.joint_vel[0, [5, 7, 9, 11, 13, 15, 17]] = joint_velocity
                env.scene["robot"].data.joint_pos[0, [5, 7, 9, 11, 13, 15, 17]] += (
                    joint_velocity * env.step_dt
                )
                env.eef_poses[0, 0, 3] = eef_x
                for _ in batch:
                    queue.task_done()

        companion_task = asyncio.create_task(companion())
        driver_task = asyncio.create_task(driver())
        await coordinator.settle(env, 0, queue)
        await companion_task
        await driver_task
        coordinator.restore()
        stream.close()
        return other_actions

    assert asyncio.run(scenario()) == [1, 1, 1, 1]


@pytest.mark.parametrize("failure", ["invalid_ids", "nan_eef", "nan_joint_position", "arm_cap"])
def test_arm_settling_failures_trip_root_queue_guard(tmp_path, failure):
    async def scenario():
        env = ArmFakeEnv()
        queue = asyncio.Queue()
        log, stream = logger(tmp_path)
        max_steps = 2 if failure == "arm_cap" else 20
        coordinator = SETTLED.SettlingCoordinator(
            log, torch, 0.001, 0.02, 2, max_steps, settle_arm_motion=True
        )
        driver = None
        if failure == "invalid_ids":
            env.scene["robot"].find_joints = lambda pattern: ([1, 2, 3], ["bad"] * 3)
        elif failure == "nan_eef":
            env.eef_poses[0, 0, 3] = float("nan")
        elif failure == "nan_joint_position":
            env.scene["robot"].data.joint_pos[0, 5] = float("nan")
        else:
            samples = [(0.0001, 0.001, 0.2, 0.005), (0.0001, 0.001, 0.2, 0.010)]
            driver = asyncio.create_task(drive_arm_actions(queue, env, samples))
        with pytest.raises(SETTLED.SettlingError):
            await asyncio.wait_for(coordinator.settle(env, 0, queue), timeout=1)
        if driver is not None:
            await driver
        with pytest.raises(SETTLED.SettlingError):
            queue.qsize()
        coordinator.restore()
        stream.close()

    asyncio.run(scenario())


def test_cap_sets_fatal_and_root_qsize_raises_without_hang(tmp_path):
    async def scenario():
        env = FakeEnv()
        queue = asyncio.Queue()
        log, stream = logger(tmp_path)
        coordinator = SETTLED.SettlingCoordinator(log, torch, 0.001, 0.02, 2, 2)
        driver = asyncio.create_task(drive_actions(queue, env, [(0.1, 0.1), (0.1, 0.1)]))
        with pytest.raises(SETTLED.SettlingError, match="cap reached"):
            await asyncio.wait_for(coordinator.settle(env, 0, queue), timeout=1)
        await driver
        with pytest.raises(SETTLED.SettlingError, match="cap reached"):
            queue.qsize()
        coordinator.restore()
        assert queue.qsize() == 0
        stream.close()

    asyncio.run(scenario())


def test_cap_retries_only_target_env_twice_and_final_cycle_cleans_recorder(tmp_path):
    async def scenario():
        env = FakeEnv(num_envs=2)
        env.recorder_manager.pre_reset_exports = []
        reset_queue = asyncio.Queue()
        action_queue = asyncio.Queue()
        log, stream = logger(tmp_path)
        coordinator = SETTLED.SettlingCoordinator(
            log, torch, 0.001, 0.02, 2, 2,
            settle_reset_retries=2, chunk_seed=4, output_label="chunk.hdf5",
        )
        for env_id in range(1, 100):
            coordinator.begin_attempt(env_id)
        generator = SETTLED.make_settled_generator(FakeBase, coordinator)(env)
        reset_ids = []

        async def reset_driver():
            for cycle in range(3):
                env_id = await reset_queue.get()
                reset_ids.append(env_id)
                env.reset_count += 1
                env.scene["toy_truck"].data.root_pos_w[env_id, 0] = 0.20 + cycle * 0.01
                env.scene["box"].data.root_pos_w[env_id, 0] = 0.70 + cycle * 0.01
                # Match the stock env.reset ordering: record/export pre-reset,
                # then RecorderManager.reset, then record_post_reset.
                env.recorder_manager.pre_reset_exports.append(
                    (env_id, list(env.recorder_manager.episode))
                )
                env.recorder_manager.reset([env_id])
                env.recorder_manager.record_post_reset([env_id])
                env.recorder_manager.episode.append(f"reset-prefix-{cycle}")
                reset_queue.task_done()

        async def action_driver():
            speeds = [0.1, 0.1, 0.1, 0.1, 0.0008, 0.0007]
            for index, speed in enumerate(speeds):
                env_id, _ = await action_queue.get()
                assert env_id == 0
                env.scene["toy_truck"].data.root_pos_w[env_id, 2] -= 0.001
                env.scene["toy_truck"].data.root_lin_vel_w[env_id] = torch.tensor(
                    [speed, 0.0, 0.0]
                )
                env.scene["toy_truck"].data.root_ang_vel_w[env_id].zero_()
                env.recorder_manager.episode.append(f"hold-{index}")
                action_queue.task_done()

        reset_task = asyncio.create_task(reset_driver())
        action_task = asyncio.create_task(action_driver())
        result = await generator.generate(
            env_id=0,
            success_term="same",
            env_reset_queue=reset_queue,
            env_action_queue=action_queue,
        )
        await reset_task
        await action_task
        coordinator.restore()
        stream.close()
        return env, result, reset_ids, (
            coordinator.logger.logical_initial_settling_starts,
            coordinator.logger.settling_retry_operations,
            coordinator.logger.discarded_layouts,
        )

    env, result, reset_ids, counters = asyncio.run(scenario())
    assert result["success"] is True
    assert reset_ids == [0, 0, 0]
    assert env.reset_count == 3
    assert env.recorder_manager.episode == ["post_reset"]
    assert env.recorder_manager.pre_reset_exports == [(0, []), (0, []), (0, [])]
    assert all(env_id == 0 for env_id, _ in env.recorder_manager.pre_reset_exports)
    assert env.recorder_manager.calls.count(("reset", [0])) == 7
    assert not any(call == ("reset", [1]) for call in env.recorder_manager.calls)
    assert env.episode_length_buf.tolist() == [0, 37]
    rows = [json.loads(line) for line in (tmp_path / "settling.jsonl").read_text().splitlines()]
    attempts = [row for row in rows if row["record_type"] == "settling_attempt"]
    starts = [row for row in rows if row["record_type"] == "settling_attempt_start"]
    retries = [row for row in rows if row["record_type"] == "settling_retry"]
    results = [row for row in rows if row["record_type"] == "generation_result"]
    assert [row["record_type"] for row in rows if row["record_type"] != "settling_attempt_start"] == [
        "settling_attempt", "settling_retry", "settling_attempt",
        "settling_retry", "settling_attempt", "generation_result",
    ]
    assert len(starts) == 100
    assert [row["logical_initial_settling_starts"] for row in starts] == list(range(1, 101))
    assert starts[-1]["env_id"] == 0 and starts[-1]["attempt"] == 1
    assert starts[-1]["chunk_seed"] == 4 and starts[-1]["output_label"] == "chunk.hdf5"
    assert [(row["attempt"], row["retry_index"], row["capped"]) for row in attempts] == [
        (1, 0, True), (1, 1, True), (1, 2, False)
    ]
    assert [row["retry_count"] for row in retries] == [1, 2]
    assert [row["retry_ratio"] for row in retries] == pytest.approx([0.01, 0.02])
    assert all(row["env_id"] == 0 and row["attempt"] == 1 for row in retries)
    assert retries[0]["chunk_seed"] == 4 and retries[0]["output_label"] == "chunk.hdf5"
    assert retries[0]["original_layout"]["toy_truck"]["position_world"][0] == pytest.approx(0.20)
    assert retries[1]["original_layout"]["toy_truck"]["position_world"][0] == pytest.approx(0.21)
    assert retries[0]["original_layout"]["box"]["position_world"][0] == pytest.approx(0.70)
    assert retries[1]["original_layout"]["box"]["position_world"][0] == pytest.approx(0.71)
    assert retries[1]["original_layout"] != retries[0]["original_layout"]
    assert retries[0]["discarded_layout"]["box"]["position_env"][0] == pytest.approx(0.70)
    assert "frame_semantics" in retries[0]["discarded_layout"]
    assert attempts[0]["initial_layout"] == retries[0]["original_layout"]
    assert attempts[1]["initial_layout"] == retries[1]["original_layout"]
    assert attempts[2]["initial_layout"]["toy_truck"]["position_world"][0] == pytest.approx(0.22)
    assert len(results) == 1 and results[0]["attempt"] == 1
    assert counters == (100, 2, 2)


def test_scope6_attempt_start_is_logged_before_reset_wait_and_retry_does_not_add_start(tmp_path):
    # Seed 49 additional logical starts so one retry remains within the 2% cap.
    # They are real begin events but intentionally have no completed settle cycle.
    async def run():
        env = FakeEnv()
        reset_queue = asyncio.Queue()
        action_queue = asyncio.Queue()
        log, stream = logger(tmp_path)
        coordinator = SETTLED.SettlingCoordinator(
            log, torch, 0.001, 0.02, 2, 2,
            settle_reset_retries=2, chunk_seed=9, output_label="pending.hdf5",
        )
        for env_id in range(1, 50):
            coordinator.begin_attempt(env_id)
        proxy = SETTLED.ResetQueueProxy(reset_queue, action_queue, coordinator, env, 0)
        await proxy.put(0)
        coordinator.authorize_retry(
            0, 1, 1,
            {"frame_semantics": {}, "toy_truck": {}, "box": {}},
            {"frame_semantics": {}, "toy_truck": {}, "box": {}},
        )
        stream.close()
        return reset_queue.qsize(), coordinator.logger.logical_initial_settling_starts

    qsize, logical_starts = asyncio.run(run())
    records = [json.loads(line) for line in (tmp_path / "settling.jsonl").read_text().splitlines()]
    starts = [row for row in records if row["record_type"] == "settling_attempt_start"]
    assert qsize == 1 and logical_starts == 50
    assert len(starts) == 50 and starts[-1]["logical_initial_settling_starts"] == 50
    assert starts[-1]["env_id"] == 0 and starts[-1]["attempt"] == 1
    assert sum(row["record_type"] == "settling_retry" for row in records) == 1
    assert not any(row["record_type"] == "settling_attempt" for row in records)


def test_third_cap_after_two_retries_is_fatal_without_phantom_attempt(tmp_path):
    async def scenario():
        env = FakeEnv()
        reset_queue = asyncio.Queue()
        action_queue = asyncio.Queue()
        log, stream = logger(tmp_path)
        coordinator = SETTLED.SettlingCoordinator(
            log, torch, 0.001, 0.02, 2, 2, settle_reset_retries=2
        )
        for env_id in range(1, 100):
            coordinator.begin_attempt(env_id)
        generator = SETTLED.make_settled_generator(FakeBase, coordinator)(env)

        async def resets():
            for _ in range(3):
                assert await reset_queue.get() == 0
                env.reset_count += 1
                reset_queue.task_done()

        reset_task = asyncio.create_task(resets())
        action_task = asyncio.create_task(
            drive_actions(action_queue, env, [(0.1, 0.1)] * 6)
        )
        with pytest.raises(SETTLED.SettlingError, match="retries exhausted"):
            await generator.generate(
                env_id=0, success_term="same", env_reset_queue=reset_queue,
                env_action_queue=action_queue,
            )
        await reset_task
        await action_task
        with pytest.raises(SETTLED.SettlingError, match="retries exhausted"):
            action_queue.qsize()
        coordinator.restore()
        stream.close()
        return env

    env = asyncio.run(scenario())
    rows = [json.loads(line) for line in (tmp_path / "settling.jsonl").read_text().splitlines()]
    attempts = [row for row in rows if row["record_type"] == "settling_attempt"]
    retries = [row for row in rows if row["record_type"] == "settling_retry"]
    assert [row["retry_index"] for row in attempts] == [0, 1, 2]
    assert {row["attempt"] for row in attempts} == {1}
    assert len(retries) == 2 and env.reset_count == 3
    assert not any(row["record_type"] == "generation_result" for row in rows)


def test_retry_ratio_is_checked_before_reset_and_noncap_errors_never_retry(tmp_path):
    log, stream = logger(tmp_path)
    coordinator = SETTLED.SettlingCoordinator(
        log, torch, 0.001, 0.02, 2, 2, settle_reset_retries=2
    )
    for env_id in range(100):
        coordinator.begin_attempt(env_id)
    layout = {"frame_semantics": {}, "toy_truck": {}, "box": {}}
    coordinator.authorize_retry(0, 1, 1, layout, layout)
    coordinator.authorize_retry(1, 1, 1, layout, layout)
    with pytest.raises(SETTLED.SettlingError, match="exceed 2%"):
        coordinator.authorize_retry(2, 1, 1, layout, layout)
    assert coordinator.logger.settling_retry_operations == 2
    assert coordinator.logger.discarded_layouts == 2
    stream.close()

    async def malformed_scenario():
        env = FakeEnv()
        env.scene["box"].data.root_pos_w[0, 0] = float("nan")
        reset_queue = asyncio.Queue()
        action_queue = asyncio.Queue()
        log2, stream2 = logger(tmp_path / "malformed")
        coordinator2 = SETTLED.SettlingCoordinator(
            log2, torch, 0.001, 0.02, 2, 2, settle_reset_retries=2
        )
        generator = SETTLED.make_settled_generator(FakeBase, coordinator2)(env)
        reset_task = asyncio.create_task(drive_reset(reset_queue, env))
        with pytest.raises(SETTLED.SettlingError, match="malformed box pose"):
            await generator.generate(
                env_id=0, success_term="same", env_reset_queue=reset_queue,
                env_action_queue=action_queue,
            )
        await reset_task
        assert env.reset_count == 1
        assert coordinator2.logger.settling_retry_operations == 0
        with pytest.raises(SETTLED.SettlingError, match="malformed box pose"):
            action_queue.qsize()
        coordinator2.restore()
        stream2.close()

    (tmp_path / "malformed").mkdir()
    asyncio.run(malformed_scenario())


def test_base_failure_before_reset_is_visible_to_root_qsize(tmp_path):
    async def scenario():
        env = FakeEnv()
        reset_queue = asyncio.Queue()
        action_queue = asyncio.Queue()
        log, stream = logger(tmp_path)
        coordinator = SETTLED.SettlingCoordinator(log, torch, 0.001, 0.02, 2, 10)
        generator = SETTLED.make_settled_generator(FailingBase, coordinator)(env)
        with pytest.raises(SETTLED.SettlingError, match="failed before reset"):
            await generator.generate(
                env_id=0,
                success_term="same",
                env_reset_queue=reset_queue,
                env_action_queue=action_queue,
            )
        with pytest.raises(SETTLED.SettlingError, match="failed before reset"):
            action_queue.qsize()
        coordinator.restore()
        stream.close()

    asyncio.run(scenario())


def original_args(input_path, output_path):
    return [
        "--task",
        SETTLED.EXPECTED_TASK,
        "--headless",
        "--num_envs",
        "100",
        "--input_file",
        str(input_path),
        "--output_file",
        str(output_path),
    ]


def wrapper_settings(**overrides):
    values = {
        "settle_linear_velocity": 0.001,
        "settle_angular_velocity": 0.02,
        "settle_stable_steps": 2,
        "settle_max_steps": 10,
        "settle_arm_motion": False,
        "settle_arm_joint_velocity": 0.05,
        "settle_eef_velocity": 0.01,
        "box_robot_distance": False,
        "box_rotation_weight": 2.0,
        "box_nn_k": 1,
        "box_interpolation_steps": None,
        "enable_box_reselection": False,
        "zero_action_noise": False,
        "chunk_seed": None,
        "grasp_offset_range": None,
        "per_env_rmp_reset": False,
        "settle_reset_retries": 0,
        "truck_pick_z_offset_m": 0.0,
        "pregrasp_time_dilation": 1,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def toy2box_config():
    return types.SimpleNamespace(
        seed=17,
        datagen_config=types.SimpleNamespace(
            seed=1,
            generation_select_src_per_subtask=True,
            generation_select_src_per_arm=False,
            generation_interpolate_from_last_target_pose=True,
        ),
        subtask_configs={
            "agibot": [
                types.SimpleNamespace(
                    object_ref="toy_truck",
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={},
                    num_interpolation_steps=15,
                    action_noise=0.01,
                    subtask_term_signal="grasp",
                    subtask_term_offset_range=(2, 10),
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=False,
                    marker="untouched",
                ),
                types.SimpleNamespace(
                    object_ref="box",
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={},
                    num_interpolation_steps=15,
                    action_noise=0.01,
                    subtask_term_signal=None,
                    subtask_term_offset_range=(0, 0),
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=False,
                    marker="also-untouched",
                ),
            ]
        },
    )


def test_box_override_is_disabled_by_default_and_logs_unchanged_config(tmp_path):
    config = toy2box_config()
    before = copy.deepcopy(vars(config.subtask_configs["agibot"][1]))
    log, stream = logger(tmp_path)
    SETTLED.apply_box_selection_override(config, wrapper_settings(), log)
    stream.close()
    assert vars(config.subtask_configs["agibot"][1]) == before
    record = json.loads((tmp_path / "settling.jsonl").read_text())
    assert record["record_type"] == "runtime_config_override"
    assert record["subtask"] == "agibot/1"
    assert record["override_enabled"] is False
    assert record["before"] == record["after"]
    assert record["datagen_seed_original"] == record["datagen_seed_effective"] == 1
    assert record["reselection_enabled"] is False
    assert record["zero_action_noise_enabled"] is False
    assert record["generation_select_src_per_arm_original"] is False
    assert record["generation_select_src_per_arm_effective"] is False
    assert record["eef_names"] == ["agibot"]
    assert record["subtask1_before"] == record["subtask1_after"]
    assert record["subtask2_before"] == record["subtask2_after"]


def test_box_override_changes_only_second_subtask_and_preserves_success_term(tmp_path):
    config = toy2box_config()
    first_before = copy.deepcopy(vars(config.subtask_configs["agibot"][0]))
    second_before = copy.deepcopy(vars(config.subtask_configs["agibot"][1]))
    success_term = object()
    calls = []

    def original_setup(*args, **kwargs):
        calls.append((args, kwargs))
        return config, success_term

    log, stream = logger(tmp_path)
    wrapped = SETTLED.make_setup_env_config_wrapper(
        original_setup,
        wrapper_settings(
            box_robot_distance=True,
            enable_box_reselection=True,
        ),
        log,
    )
    returned_config, returned_success = wrapped("task", device="cuda")
    stream.close()

    assert calls == [(('task',), {"device": "cuda"})]
    assert returned_config is config
    assert returned_success is success_term
    assert vars(config.subtask_configs["agibot"][0]) == first_before
    box = config.subtask_configs["agibot"][1]
    assert box.selection_strategy == "nearest_neighbor_robot_distance"
    assert box.selection_strategy_kwargs == {"nn_k": 1, "pos_weight": 1.0, "rot_weight": 2.0}
    assert box.num_interpolation_steps == 15
    assert box.action_noise == second_before["action_noise"]
    assert box.marker == "also-untouched"
    assert config.datagen_config.seed == 1
    assert config.datagen_config.generation_select_src_per_arm is True
    assert config.datagen_config.generation_select_src_per_subtask is True
    record = json.loads((tmp_path / "settling.jsonl").read_text())
    assert record["before"]["selection_strategy"] == "nearest_neighbor_object"
    assert record["after"]["selection_strategy"] == "nearest_neighbor_robot_distance"
    assert record["subtask1_before"] == record["subtask1_after"]


def test_setup_wrapper_without_chunk_seed_preserves_both_seed_fields(tmp_path):
    config = toy2box_config()
    success_term = object()
    log, stream = logger(tmp_path)
    wrapped = SETTLED.make_setup_env_config_wrapper(
        lambda: (config, success_term), wrapper_settings(), log
    )
    returned_config, returned_success = wrapped()
    stream.close()

    assert returned_config is config
    assert returned_success is success_term
    assert config.seed == 17
    assert config.datagen_config.seed == 1
    records = [json.loads(line) for line in (tmp_path / "settling.jsonl").read_text().splitlines()]
    assert not any(record["record_type"] == "chunk_seed_override" for record in records)
    assert records[0]["datagen_seed_original"] == records[0]["datagen_seed_effective"] == 1


def test_per_env_rmp_install_is_opt_in_and_changes_only_action_class(tmp_path):
    class BaseAction:
        pass

    config = toy2box_config()
    config.actions = types.SimpleNamespace(
        arm_action=types.SimpleNamespace(class_type=BaseAction, scale=0.5, marker="same")
    )
    original_behavior = SETTLED._action_config_behavior(config.actions.arm_action)
    derived = type("DerivedAction", (BaseAction,), {})
    log, stream = logger(tmp_path)
    wrapped = SETTLED.make_setup_env_config_wrapper(
        lambda: (config, "success"),
        wrapper_settings(per_env_rmp_reset=True),
        log,
        derived,
    )
    returned, success = wrapped()
    stream.close()

    assert returned is config and success == "success"
    assert config.actions.arm_action.class_type is derived
    assert SETTLED._action_config_behavior(config.actions.arm_action) == original_behavior
    records = [json.loads(line) for line in (tmp_path / "settling.jsonl").read_text().splitlines()]
    install = records[-1]
    assert install == {
        "record_type": "per_env_rmp_install",
        "action_term": "arm_action",
        "base_class": SETTLED._qualified_name(BaseAction),
        "effective_class": SETTLED._qualified_name(derived),
        "config_behavior_unchanged": True,
    }

    untouched = toy2box_config()
    untouched.actions = types.SimpleNamespace(
        arm_action=types.SimpleNamespace(class_type=BaseAction, scale=0.5)
    )
    log2 = SETTLED.JsonlLog((tmp_path / "default.jsonl").open("x", encoding="utf-8"))
    default_wrapped = SETTLED.make_setup_env_config_wrapper(
        lambda: (untouched, object()), wrapper_settings(), log2, derived
    )
    default_wrapped()
    log2.stream.close()
    assert untouched.actions.arm_action.class_type is BaseAction
    default_records = [json.loads(line) for line in (tmp_path / "default.jsonl").read_text().splitlines()]
    assert not any(row["record_type"] == "per_env_rmp_install" for row in default_records)


def _actual_reset_idx():
    source = (
        Path(__file__).parents[3]
        / "source"
        / "isaaclab"
        / "isaaclab"
        / "controllers"
        / "rmp_flow.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    controller = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RmpFlowController"
    )
    method = copy.deepcopy(next(
        node for node in controller.body
        if isinstance(node, ast.FunctionDef) and node.name == "reset_idx"
    ))
    method.name = "actual_reset_idx"
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["actual_reset_idx"]


def _rmp_reset_fixture(tmp_path, prim_order=(0, 1, 2)):
    actual_reset_idx = _actual_reset_idx()

    class MotionPolicy:
        def __init__(self):
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

    class Policy:
        def __init__(self):
            self.motion_policy = MotionPolicy()

    class Controller:
        def __init__(self):
            self._device = "cpu"
            self.initialize_calls = 0

        def initialize(self, expression):
            self.initialize_calls += 1
            self.num_robots = 3
            self._prim_paths = [f"/World/envs/env_{index}/Robot" for index in prim_order]
            self.articulation_policies = [Policy() for _ in range(3)]
            self._command = torch.arange(21, dtype=torch.float64).reshape(3, 7)
            self.dof_pos_target = torch.arange(18, dtype=torch.float64).reshape(3, 6)
            self.dof_vel_target = torch.arange(18, dtype=torch.float64).reshape(3, 6) + 100

        reset_idx = actual_reset_idx

    class BaseAction:
        def __init__(self):
            self.num_envs = 3
            self.device = "cpu"
            self.cfg = types.SimpleNamespace(articulation_prim_expr="/World/envs/env_.*/Robot")
            self._rmpflow_controller = Controller()
            self._raw_actions = torch.arange(18, dtype=torch.float64).reshape(3, 6) + 200

        def reset(self, env_ids=None):
            self._raw_actions[env_ids] = 0.0
            self._rmpflow_controller.initialize(self.cfg.articulation_prim_expr)

    log, stream = logger(tmp_path)
    action_cls = SETTLED.make_per_env_rmpflow_action(BaseAction, log, torch)
    return action_cls(), stream


def test_per_env_rmp_reset_uses_actual_reset_idx_and_preserves_other_rows(tmp_path):
    action, stream = _rmp_reset_fixture(tmp_path)
    action.reset(None)
    controller = action._rmpflow_controller
    assert controller.initialize_calls == 1
    identities = tuple(id(value) for value in (
        controller._command, controller.dof_pos_target, controller.dof_vel_target
    ))
    action._raw_actions[:] = torch.arange(18, dtype=torch.float64).reshape(3, 6) + 200
    snapshots = {
        "raw": action._raw_actions.clone(),
        "command": controller._command.clone(),
        "pos": controller.dof_pos_target.clone(),
        "vel": controller.dof_vel_target.clone(),
    }

    action.reset(torch.tensor([1], dtype=torch.int64))
    assert controller.initialize_calls == 1
    assert [policy.motion_policy.reset_calls for policy in controller.articulation_policies] == [0, 1, 0]
    assert torch.equal(action._raw_actions[1], torch.zeros(6, dtype=torch.float64))
    assert torch.equal(action._raw_actions[[0, 2]], snapshots["raw"][[0, 2]])
    assert torch.equal(controller._command, snapshots["command"])
    assert torch.equal(controller.dof_pos_target, snapshots["pos"])
    assert torch.equal(controller.dof_vel_target, snapshots["vel"])
    assert identities == tuple(id(value) for value in (
        controller._command, controller.dof_pos_target, controller.dof_vel_target
    ))

    action.reset([2, 2])
    action.reset(0)
    action.reset(slice(None))
    assert controller.initialize_calls == 1
    assert [policy.motion_policy.reset_calls for policy in controller.articulation_policies] == [2, 2, 2]
    stream.close()
    records = [json.loads(line) for line in (tmp_path / "settling.jsonl").read_text().splitlines()]
    assert [row["record_type"] for row in records] == [
        "per_env_rmp_initialization",
        "per_env_rmp_reset",
        "per_env_rmp_reset",
        "per_env_rmp_reset",
        "per_env_rmp_reset",
        "per_env_rmp_reset",
    ]
    assert records[2]["env_ids"] == [1]
    assert records[2]["non_target_rows_unchanged"] == {
        "raw_actions": True,
        "command": True,
        "position_target": True,
        "velocity_target": True,
    }


@pytest.mark.parametrize("bad_ids", [[], slice(1, 1), [-1], [3], [1.0], [True], torch.tensor([1.0])])
def test_per_env_rmp_reset_rejects_invalid_or_empty_ids(tmp_path, bad_ids):
    action, stream = _rmp_reset_fixture(tmp_path)
    action.reset(None)
    with pytest.raises(SETTLED.ConfigurationError):
        action.reset(bad_ids)
    stream.close()


def test_per_env_rmp_reset_requires_initial_full_reset_and_ordered_prim_mapping(tmp_path):
    action, stream = _rmp_reset_fixture(tmp_path)
    with pytest.raises(SETTLED.ConfigurationError, match="every environment"):
        action.reset([0])
    stream.close()

    misordered = tmp_path / "misordered"
    misordered.mkdir()
    action, stream = _rmp_reset_fixture(misordered, prim_order=(1, 0, 2))
    with pytest.raises(SETTLED.ConfigurationError, match="prim-path order"):
        action.reset(None)
    stream.close()


@pytest.mark.parametrize("chunk_seed", [1, 2])
def test_chunk_seed_sets_env_and_datagen_before_runtime_override(tmp_path, chunk_seed):
    config = toy2box_config()
    success_term = object()
    calls = []

    def original_setup(*args, **kwargs):
        calls.append((args, kwargs))
        return config, success_term

    log, stream = logger(tmp_path)
    wrapped = SETTLED.make_setup_env_config_wrapper(
        original_setup,
        wrapper_settings(
            box_robot_distance=True,
            enable_box_reselection=True,
            zero_action_noise=True,
            chunk_seed=chunk_seed,
        ),
        log,
    )
    returned_config, returned_success = wrapped("task", device="cuda")
    stream.close()

    assert calls == [(("task",), {"device": "cuda"})]
    assert returned_config is config
    assert returned_success is success_term
    assert config.seed == chunk_seed
    assert config.datagen_config.seed == chunk_seed
    records = [json.loads(line) for line in (tmp_path / "settling.jsonl").read_text().splitlines()]
    assert records[0] == {
        "record_type": "chunk_seed_override",
        "env_seed_original": 17,
        "env_seed_effective": chunk_seed,
        "datagen_seed_original": 1,
        "datagen_seed_effective": chunk_seed,
    }
    runtime = records[1]
    assert runtime["record_type"] == "runtime_config_override"
    assert runtime["datagen_seed_original"] == chunk_seed
    assert runtime["datagen_seed_effective"] == chunk_seed


@pytest.mark.parametrize("missing", ["env", "datagen"])
def test_chunk_seed_missing_seed_field_fails_before_mutation(tmp_path, missing):
    config = toy2box_config()
    if missing == "env":
        delattr(config, "seed")
    else:
        delattr(config.datagen_config, "seed")
    env_seed_before = getattr(config, "seed", None)
    datagen_seed_before = getattr(config.datagen_config, "seed", None)
    log, stream = logger(tmp_path)
    wrapped = SETTLED.make_setup_env_config_wrapper(
        lambda: (config, object()),
        wrapper_settings(
            box_robot_distance=True,
            enable_box_reselection=True,
            zero_action_noise=True,
            chunk_seed=2,
        ),
        log,
    )
    with pytest.raises(SETTLED.ConfigurationError, match="env and datagen seed"):
        wrapped()
    stream.close()
    assert getattr(config, "seed", None) == env_seed_before
    assert getattr(config.datagen_config, "seed", None) == datagen_seed_before
    assert (tmp_path / "settling.jsonl").read_text() == ""


def test_zero_action_noise_changes_both_subtasks_and_box_interpolation_only(tmp_path):
    config = toy2box_config()
    first_before = copy.deepcopy(vars(config.subtask_configs["agibot"][0]))
    second_before = copy.deepcopy(vars(config.subtask_configs["agibot"][1]))
    log, stream = logger(tmp_path)
    SETTLED.apply_box_selection_override(
        config,
        wrapper_settings(
            box_robot_distance=True,
            enable_box_reselection=True,
            zero_action_noise=True,
            box_interpolation_steps=30,
            grasp_offset_range=[50, 60],
        ),
        log,
    )
    stream.close()

    grasp, box = config.subtask_configs["agibot"]
    assert grasp.action_noise == 0.0
    assert box.action_noise == 0.0
    assert grasp.num_interpolation_steps == first_before["num_interpolation_steps"] == 15
    assert grasp.subtask_term_offset_range == (50, 60)
    assert box.num_interpolation_steps == 30
    assert box.subtask_term_offset_range == second_before["subtask_term_offset_range"] == (0, 0)
    assert grasp.object_ref == first_before["object_ref"]
    assert grasp.selection_strategy == first_before["selection_strategy"]
    assert box.object_ref == second_before["object_ref"]
    assert box.selection_strategy == "nearest_neighbor_robot_distance"
    assert config.datagen_config.seed == 1
    assert config.datagen_config.generation_select_src_per_arm is True
    assert config.datagen_config.generation_select_src_per_subtask is True

    record = json.loads((tmp_path / "settling.jsonl").read_text())
    assert record["zero_action_noise_enabled"] is True
    assert record["grasp_offset_override_enabled"] is True
    assert record["requested_grasp_offset_range"] == [50, 60]
    assert record["subtask1_before"]["action_noise"] == pytest.approx(0.01)
    assert record["subtask1_after"]["action_noise"] == 0.0
    assert record["subtask1_after"]["num_interpolation_steps"] == 15
    assert record["subtask1_before"]["subtask_term_offset_range"] == [2, 10]
    assert record["subtask1_after"]["subtask_term_offset_range"] == [50, 60]
    assert record["subtask2_before"]["action_noise"] == pytest.approx(0.01)
    assert record["subtask2_after"]["action_noise"] == 0.0
    assert record["subtask2_before"]["num_interpolation_steps"] == 15
    assert record["subtask2_after"]["num_interpolation_steps"] == 30
    assert record["subtask2_after"]["subtask_term_offset_range"] == [0, 0]
    for key, value in record["subtask1_before"].items():
        if key not in {"action_noise", "subtask_term_offset_range"}:
            assert record["subtask1_after"][key] == value
    for key, value in record["subtask2_before"].items():
        if key not in {"action_noise", "selection_strategy", "selection_strategy_kwargs",
                       "num_interpolation_steps"}:
            assert record["subtask2_after"][key] == value


def test_zero_action_noise_fails_closed_when_noise_field_is_missing(tmp_path):
    config = toy2box_config()
    delattr(config.subtask_configs["agibot"][1], "action_noise")
    log, stream = logger(tmp_path)
    with pytest.raises(SETTLED.ConfigurationError, match="action_noise"):
        SETTLED.apply_box_selection_override(
            config,
            wrapper_settings(
                box_robot_distance=True,
                enable_box_reselection=True,
                zero_action_noise=True,
            ),
            log,
        )
    stream.close()


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda cfg: setattr(cfg, "subtask_configs", {}), "exactly one"),
        (lambda cfg: setattr(cfg.subtask_configs["agibot"][0], "object_ref", "box"), "subtask 1"),
        (lambda cfg: setattr(cfg.subtask_configs["agibot"][1], "object_ref", "wrong"), "subtask 2"),
        (lambda cfg: delattr(cfg.datagen_config, "seed"), "seed"),
        (
            lambda cfg: cfg.subtask_configs.update({"other_arm": cfg.subtask_configs["agibot"]}),
            "exactly one",
        ),
        (
            lambda cfg: setattr(cfg.datagen_config, "generation_select_src_per_subtask", False),
            "per-subtask",
        ),
    ],
)
def test_box_override_fails_closed_on_unexpected_task_config(tmp_path, mutate, message):
    config = toy2box_config()
    mutate(config)
    log, stream = logger(tmp_path)
    with pytest.raises(SETTLED.ConfigurationError, match=message):
        SETTLED.apply_box_selection_override(config, wrapper_settings(box_robot_distance=True), log)
    stream.close()


def test_source_selection_audit_preserves_parent_arguments_and_result(tmp_path):
    class SelectionBase:
        def __init__(self, env):
            self.env = env
            self.calls = []
            self.trajectory_calls = []

        def select_source_demo(self, **kwargs):
            self.calls.append(kwargs)
            return 7

        def generate_eef_subtask_trajectory(self, **kwargs):
            self.trajectory_calls.append(kwargs)
            return self.select_source_demo(**self.selection_arguments)

    log, stream = logger(tmp_path)
    coordinator = SETTLED.SettlingCoordinator(log, torch, 0.001, 0.02, 2, 10)
    generator = SETTLED.make_settled_generator(SelectionBase, coordinator, enable_source_audit=True)(object())
    arguments = {
        "eef_name": "agibot",
        "eef_pose": object(),
        "object_pose": object(),
        "src_demo_current_subtask_boundaries": object(),
        "subtask_object_name": "box",
        "selection_strategy_name": "nearest_neighbor_robot_distance",
        "selection_strategy_kwargs": {"nn_k": 1, "pos_weight": 1.0, "rot_weight": 2.0},
    }
    generator.selection_arguments = arguments
    coordinator.attempts[3] = 9
    trajectory_arguments = {
        "env_id": 3,
        "eef_name": "agibot",
        "subtask_ind": 1,
        "all_randomized_subtask_boundaries": {"preserved": True},
        "runtime_subtask_constraints_dict": {"also": "preserved"},
        "selected_src_demo_inds": {"agibot": 5},
    }
    assert generator.generate_eef_subtask_trajectory(**trajectory_arguments) == 7
    stream.close()
    assert generator.calls == [arguments]
    assert generator.trajectory_calls == [trajectory_arguments]
    assert not hasattr(generator, "_source_selection_context")
    record = json.loads((tmp_path / "settling.jsonl").read_text())
    assert record == {
        "attempt": 9,
        "eef_name": "agibot",
        "env_id": 3,
        "object_ref": "box",
        "record_type": "source_selection",
        "selected_source_demo_index": 7,
        "selection_strategy": "nearest_neighbor_robot_distance",
        "selection_strategy_kwargs": {"nn_k": 1, "pos_weight": 1.0, "rot_weight": 2.0},
        "subtask_index": 1,
    }


def test_truck_pick_z_offset_off_preserves_trajectory_and_pose_identity(tmp_path):
    source_pose = torch.eye(4, dtype=torch.float64)
    trajectory = [types.SimpleNamespace(pose=source_pose)]

    class TrajectoryBase:
        def __init__(self, env):
            self.env = env
            self.calls = []

        def generate_eef_subtask_trajectory(self, **kwargs):
            self.calls.append(kwargs)
            return trajectory

    log, stream = logger(tmp_path)
    coordinator = SETTLED.SettlingCoordinator(log, torch, 0.001, 0.02, 2, 10)
    generator = SETTLED.make_settled_generator(TrajectoryBase, coordinator)(object())
    arguments = {
        "env_id": 2,
        "eef_name": "agibot",
        "subtask_ind": 0,
        "all_randomized_subtask_boundaries": {"same": True},
        "runtime_subtask_constraints_dict": {"same": "constraints"},
        "selected_src_demo_inds": {"agibot": 4},
    }

    result = generator.generate_eef_subtask_trajectory(**arguments)
    stream.close()

    assert result is trajectory
    assert result[0].pose is source_pose
    assert generator.calls == [arguments]
    assert (tmp_path / "settling.jsonl").read_text() == ""


def test_truck_pick_z_offset_clones_pick_poses_and_leaves_box_and_source_unchanged(tmp_path):
    source_pool_poses = torch.eye(4, dtype=torch.float64).repeat(3, 1, 1)
    source_pool_poses[:, 0, 3] = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    source_pool_poses[:, 1, 3] = torch.tensor([-0.2, -0.1, 0.0], dtype=torch.float64)
    source_pool_poses[:, 2, 3] = torch.tensor([0.40, 0.41, 0.42], dtype=torch.float64)
    source_before = source_pool_poses.clone()
    class BoundedTrajectory:
        # Match the actual Mimic WaypointTrajectory indexing contract.
        def __init__(self, waypoints):
            self.waypoints = waypoints

        def __len__(self):
            return len(self.waypoints)

        def __getitem__(self, index):
            assert 0 <= index < len(self.waypoints)
            return self.waypoints[index]

    pick_trajectory = BoundedTrajectory([
        types.SimpleNamespace(pose=source_pool_poses[index], gripper_action=-1, noise=0.0)
        for index in range(3)
    ])
    box_pose = torch.eye(4, dtype=torch.float64)
    box_pose[2, 3] = 0.9
    box_trajectory = [types.SimpleNamespace(pose=box_pose)]

    class TrajectoryBase:
        def __init__(self, env):
            self.env = env
            self.env_cfg = toy2box_config()
            self.calls = []

        def generate_eef_subtask_trajectory(self, **kwargs):
            self.calls.append(kwargs)
            return pick_trajectory if kwargs["subtask_ind"] == 0 else box_trajectory

    log, stream = logger(tmp_path)
    coordinator = SETTLED.SettlingCoordinator(log, torch, 0.001, 0.02, 2, 10)
    coordinator.attempts[6] = 12
    generator = SETTLED.make_settled_generator(
        TrajectoryBase,
        coordinator,
        truck_pick_z_offset_m=-0.005,
    )(object())
    common = {
        "env_id": 6,
        "eef_name": "agibot",
        "all_randomized_subtask_boundaries": {},
        "runtime_subtask_constraints_dict": {},
        "selected_src_demo_inds": {"agibot": 1},
    }

    pick_result = generator.generate_eef_subtask_trajectory(subtask_ind=0, **common)
    box_result = generator.generate_eef_subtask_trajectory(subtask_ind=1, **common)
    stream.close()

    assert pick_result is pick_trajectory
    assert box_result is box_trajectory
    assert box_result[0].pose is box_pose
    assert torch.equal(box_result[0].pose, box_pose)
    assert torch.equal(source_pool_poses, source_before)
    for index in range(len(pick_result)):
        waypoint = pick_result[index]
        assert waypoint.gripper_action == -1 and waypoint.noise == 0.0
        assert waypoint.pose is not source_pool_poses[index]
        expected = source_before[index].clone()
        expected[2, 3] -= 0.005
        assert torch.equal(waypoint.pose, expected)
        assert torch.equal(waypoint.pose[:2], source_before[index, :2])
        assert torch.equal(waypoint.pose[3], source_before[index, 3])

    records = [json.loads(line) for line in (tmp_path / "settling.jsonl").read_text().splitlines()]
    assert records == [
        {
            "application_count": 1,
            "applied_exactly_once": True,
            "attempt": 12,
            "eef_name": "agibot",
            "env_id": 6,
            "first_z_after_m": pytest.approx(0.395),
            "first_z_before_m": pytest.approx(0.4),
            "last_z_after_m": pytest.approx(0.415),
            "last_z_before_m": pytest.approx(0.42),
            "object_ref": "toy_truck",
            "record_type": "truck_pick_z_offset",
            "shift_m": -0.005,
            "subtask": "agibot/0",
            "subtask_index": 0,
            "waypoint_count": 3,
        }
    ]


def test_source_selection_audit_logging_error_fails_closed():
    class FailingLogger:
        def write(self, record):
            raise OSError("audit log failed")

    class SelectionBase:
        def __init__(self, env):
            self.env = env

        def select_source_demo(self, **kwargs):
            return 2

    coordinator = types.SimpleNamespace(logger=FailingLogger(), attempts={0: 1})
    generator = SETTLED.make_settled_generator(SelectionBase, coordinator, enable_source_audit=True)(object())
    generator._source_selection_context = (0, "agibot", 1)
    with pytest.raises(OSError, match="audit log failed"):
        generator.select_source_demo(
            eef_name="agibot",
            eef_pose=object(),
            object_pose=object(),
            src_demo_current_subtask_boundaries=object(),
            subtask_object_name="box",
            selection_strategy_name="nearest_neighbor_robot_distance",
            selection_strategy_kwargs={"nn_k": 1},
        )


def test_box_transition_logs_actual_and_bridge_jumps_and_returns_parent_identity(tmp_path):
    def pose(x=0.0, y=0.0, yaw_degrees=0.0):
        angle = torch.tensor(yaw_degrees * 3.141592653589793 / 180.0)
        result = torch.eye(4)
        result[0, 0] = torch.cos(angle)
        result[0, 1] = -torch.sin(angle)
        result[1, 0] = torch.sin(angle)
        result[1, 1] = torch.cos(angle)
        result[0, 3] = x
        result[1, 3] = y
        return result

    actual_pose = pose()
    target_pose = pose(3.0, 4.0, 90.0)
    bridge_pose = pose(3.0, 0.0, 90.0)
    expected_result = [object()]

    class TransitionEnv:
        def get_robot_eef_pose(self, *, env_ids, eef_name):
            assert env_ids == [2] and eef_name == "agibot"
            return torch.stack([actual_pose])

    class TransitionBase:
        def __init__(self, env):
            self.env = env
            self.env_cfg = toy2box_config()
            self.calls = []

        def merge_eef_subtask_trajectory(self, **kwargs):
            self.calls.append(kwargs)
            return expected_result

    log, stream = logger(tmp_path)
    coordinator = SETTLED.SettlingCoordinator(log, torch, 0.001, 0.02, 2, 10)
    coordinator.attempts[2] = 4
    generator = SETTLED.make_settled_generator(TransitionBase, coordinator, enable_source_audit=True)(
        TransitionEnv()
    )
    previous = [types.SimpleNamespace(pose=bridge_pose)]
    subtask = [types.SimpleNamespace(pose=target_pose)]
    result = generator.merge_eef_subtask_trajectory(
        env_id=2,
        eef_name="agibot",
        subtask_index=1,
        prev_executed_traj=previous,
        subtask_trajectory=subtask,
    )
    stream.close()

    assert result is expected_result
    assert generator.calls == [
        {
            "env_id": 2,
            "eef_name": "agibot",
            "subtask_index": 1,
            "prev_executed_traj": previous,
            "subtask_trajectory": subtask,
        }
    ]
    record = json.loads((tmp_path / "settling.jsonl").read_text())
    assert record["record_type"] == "box_transition"
    assert record["env_id"] == 2 and record["attempt"] == 4
    assert record["position_jump_m"] == pytest.approx(5.0)
    assert record["rotation_jump_deg"] == pytest.approx(90.0)
    assert record["bridge_position_jump_m"] == pytest.approx(4.0)
    assert record["bridge_rotation_jump_deg"] == pytest.approx(0.0)
    assert record["num_interpolation_steps"] == 15


def test_transition_audit_fails_closed_for_nonfinite_pose(tmp_path):
    invalid = torch.eye(4)
    invalid[0, 0] = float("nan")
    with pytest.raises(SETTLED.ConfigurationError, match="non-finite"):
        SETTLED._pose_jump(torch, invalid, torch.eye(4))


def test_actual_core_generation_gate_reselects_only_when_per_arm_enabled():
    source = (
        Path(__file__).parents[3]
        / "source"
        / "isaaclab_mimic"
        / "isaaclab_mimic"
        / "datagen"
        / "data_generator.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    data_generator = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DataGenerator")
    method = next(
        node
        for node in data_generator.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate_eef_subtask_trajectory"
    )
    extracted = copy.deepcopy(method)
    extracted.name = "actual_generate_eef_subtask_trajectory"
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), extracted],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, str(source), "exec"), namespace)
    actual_method = namespace["actual_generate_eef_subtask_trajectory"]

    class StopAfterSelection(Exception):
        pass

    class StopPool:
        def __getitem__(self, index):
            raise StopAfterSelection(index)

    class CoreEnv:
        def get_object_poses(self, *, env_ids):
            return {"box": torch.eye(4).unsqueeze(0)}

        def get_robot_eef_pose(self, *, env_ids, eef_name):
            return torch.eye(4).unsqueeze(0)

    class CoreSelf:
        def __init__(self, per_arm):
            subtask = types.SimpleNamespace(
                object_ref="box",
                selection_strategy="nearest_neighbor_robot_distance",
                selection_strategy_kwargs={"nn_k": 1, "pos_weight": 1.0, "rot_weight": 2.0},
            )
            self.env_cfg = types.SimpleNamespace(
                subtask_configs={"agibot": [subtask, subtask]},
                datagen_config=types.SimpleNamespace(
                    generation_select_src_per_subtask=True,
                    generation_select_src_per_arm=per_arm,
                    generation_transform_first_robot_pose=False,
                ),
            )
            self.env = CoreEnv()
            self.src_demo_datagen_info_pool = types.SimpleNamespace(datagen_infos=StopPool())
            self.spy_calls = []

        def select_source_demo(self, **kwargs):
            self.spy_calls.append(kwargs)
            return 7

    boundaries = {"agibot": torch.tensor([[[0, 1], [1, 2]]] * 8)}
    for per_arm, expected_index, expected_calls in [(False, 5, 0), (True, 7, 1)]:
        instance = CoreSelf(per_arm)
        selected = {"agibot": 5}
        with pytest.raises(StopAfterSelection, match=str(expected_index)):
            actual_method(
                instance,
                env_id=0,
                eef_name="agibot",
                subtask_ind=1,
                all_randomized_subtask_boundaries=boundaries,
                runtime_subtask_constraints_dict={},
                selected_src_demo_inds=selected,
            )
        assert selected["agibot"] == expected_index
        assert len(instance.spy_calls) == expected_calls
        if expected_calls:
            call = instance.spy_calls[0]
            assert call["selection_strategy_name"] == "nearest_neighbor_robot_distance"
            assert call["selection_strategy_kwargs"] == {"nn_k": 1, "pos_weight": 1.0, "rot_weight": 2.0}


def test_validation_requires_expected_explicit_headless_invocation(tmp_path):
    source = tmp_path / "source.hdf5"
    source.touch()
    output = tmp_path / "new.hdf5"
    log = tmp_path / "settle.jsonl"
    args = original_args(source, output)
    paths = SETTLED.validate_invocation(args, str(log))
    assert paths["failed_output"] == tmp_path / "new_failed.hdf5"

    for remove, message in [
        ("--headless", "headless"),
        ("--task", "task"),
        ("--input_file", "input_file"),
        ("--output_file", "output_file"),
    ]:
        changed = list(args)
        index = changed.index(remove)
        del changed[index : index + (1 if remove == "--headless" else 2)]
        with pytest.raises(ValueError, match=message):
            SETTLED.validate_invocation(changed, str(log))

    wrong_suffix = tmp_path / "source.bin"
    wrong_suffix.touch()
    with pytest.raises(ValueError, match="lowercase .hdf5"):
        SETTLED.validate_invocation(original_args(wrong_suffix, output), str(log))


def test_validation_rejects_uppercase_suffix_before_stock_recorder_can_overwrite_lowercase(tmp_path):
    source = tmp_path / "source.hdf5"
    source.touch()
    lowercase_target = tmp_path / "new.hdf5"
    lowercase_target.write_text("must remain untouched")
    uppercase_cli = tmp_path / "new.HDF5"
    with pytest.raises(ValueError, match="lowercase .hdf5"):
        SETTLED.validate_invocation(original_args(source, uppercase_cli), str(tmp_path / "settle.jsonl"))
    assert lowercase_target.read_text() == "must remain untouched"


@pytest.mark.parametrize("kind", ["success", "failed", "log"])
def test_validation_refuses_each_existing_output(tmp_path, kind):
    source = tmp_path / "source.hdf5"
    source.touch()
    output = tmp_path / "new.hdf5"
    failed = tmp_path / "new_failed.hdf5"
    log = tmp_path / "settle.jsonl"
    {"success": output, "failed": failed, "log": log}[kind].touch()
    with pytest.raises(FileExistsError):
        SETTLED.validate_invocation(original_args(source, output), str(log))


@pytest.mark.parametrize("kind", ["success", "failed", "log"])
def test_validation_refuses_each_dangling_output_symlink(tmp_path, kind):
    source = tmp_path / "source.hdf5"
    source.touch()
    output = tmp_path / "new.hdf5"
    failed = tmp_path / "new_failed.hdf5"
    log = tmp_path / "settle.jsonl"
    selected = {"success": output, "failed": failed, "log": log}[kind]
    selected.symlink_to(tmp_path / "missing-target")
    with pytest.raises(FileExistsError):
        SETTLED.validate_invocation(original_args(source, output), str(log))


def test_validation_refuses_duplicates_symlinks_and_all_protected_paths(tmp_path, monkeypatch):
    source = tmp_path / "source.hdf5"
    source.touch()
    output = tmp_path / "new.hdf5"
    protected = tuple(tmp_path / f"protected-{index}.hdf5" for index in range(7))
    monkeypatch.setattr(SETTLED, "_protected_output_paths", lambda: protected)

    with pytest.raises(ValueError, match="distinct"):
        SETTLED.validate_invocation(original_args(source, output), str(output))
    with pytest.raises(ValueError, match="alias the input"):
        SETTLED.validate_invocation(original_args(source, source), str(tmp_path / "log"))
    for path in protected:
        with pytest.raises(ValueError, match="protected"):
            SETTLED.validate_invocation(original_args(source, path), str(tmp_path / "log"))

    target = protected[0]
    alias = tmp_path / "alias.hdf5"
    alias.symlink_to(target)
    with pytest.raises((ValueError, FileExistsError)):
        SETTLED.validate_invocation(original_args(source, alias), str(tmp_path / "log"))


def test_wrapper_passes_original_argv_flushes_summary_then_closes_once(tmp_path, monkeypatch):
    source = tmp_path / "source.hdf5"
    source.touch()
    output = tmp_path / "new.hdf5"
    log = tmp_path / "settle.jsonl"
    args = original_args(source, output)
    paths = SETTLED.validate_invocation(args, str(log))
    settings = wrapper_settings()
    seen = {}
    app = types.SimpleNamespace(close=lambda: seen.setdefault("close_count", 0))

    class OriginalDataGenerator:
        pass

    generation = types.ModuleType("isaaclab_mimic.datagen.generation")
    generation.DataGenerator = OriginalDataGenerator
    generation.num_success = 3
    generation.num_failures = 1
    generation.num_attempts = 4
    package = types.ModuleType("isaaclab_mimic")
    datagen = types.ModuleType("isaaclab_mimic.datagen")
    package.datagen = datagen
    datagen.generation = generation
    monkeypatch.setitem(sys.modules, "isaaclab_mimic", package)
    monkeypatch.setitem(sys.modules, "isaaclab_mimic.datagen", datagen)
    monkeypatch.setitem(sys.modules, "isaaclab_mimic.datagen.generation", generation)
    original_argv = sys.argv

    def fake_run_path(path, run_name):
        seen["argv"] = list(sys.argv)
        seen["run_name"] = run_name
        script_globals = {
            "generation": generation,
            "OriginalDataGenerator": OriginalDataGenerator,
            "seen": seen,
            "config": toy2box_config(),
            "success_term": object(),
        }
        exec(
            "def setup_env_config(*args, **kwargs):\n"
            "    return config, success_term\n"
            "def main():\n"
            "    configured, returned_success = setup_env_config()\n"
            "    assert configured is config\n"
            "    assert returned_success is success_term\n"
            "    assert generation.DataGenerator is not OriginalDataGenerator\n"
            "    seen['main_count'] = seen.get('main_count', 0) + 1\n",
            script_globals,
        )
        return {"simulation_app": app, "main": script_globals["main"]}

    def close():
        records = [json.loads(line) for line in log.read_text().splitlines()]
        assert records[0]["zero_action_noise_enabled"] is False
        assert records[0]["chunk_seed"] is None
        assert records[0]["settle_arm_motion"] is False
        assert records[0]["settle_arm_joint_velocity"] == pytest.approx(0.05)
        assert records[0]["settle_eef_velocity"] == pytest.approx(0.01)
        assert records[0]["per_env_rmp_reset_enabled"] is False
        assert records[0]["settle_reset_retries"] == 0
        assert records[0]["truck_pick_z_offset_m"] == 0.0
        assert records[0]["settle_retry_ratio_limit"] == pytest.approx(0.02)
        assert records[0]["output_label"] == "new.hdf5"
        assert records[0]["grasp_offset_override_enabled"] is False
        assert records[0]["grasp_offset_range"] is None
        assert records[-1]["record_type"] == "settling_summary"
        assert records[-1]["original_num_success"] == 3
        assert records[-1]["logical_initial_settling_starts"] == 0
        assert records[-1]["settling_retry_operations"] == 0
        assert records[-1]["discarded_layouts"] == 0
        assert records[-1]["settling_retry_ratio"] == 0.0
        assert records[-1]["settle_reset_retries"] == 0
        seen["close_count"] = seen.get("close_count", 0) + 1

    app.close = close
    SETTLED.execute(args, paths, settings, run_path=fake_run_path)

    assert seen["argv"] == [str(SETTLED.GENERATOR_SCRIPT), *args]
    assert seen["run_name"] == "__toy2box_settle_generate__"
    assert seen["main_count"] == 1
    assert seen["close_count"] == 1
    assert generation.DataGenerator is OriginalDataGenerator
    assert sys.argv is original_argv


@pytest.mark.parametrize("raise_in_main", [False, True])
def test_real_runpy_patches_main_globals_and_restores_setup_on_error(tmp_path, monkeypatch, raise_in_main):
    source = tmp_path / "source.hdf5"
    source.touch()
    output = tmp_path / "new.hdf5"
    log = tmp_path / "settle.jsonl"
    args = original_args(source, output)
    paths = SETTLED.validate_invocation(args, str(log))
    settings = wrapper_settings(box_robot_distance=True)

    class OriginalDataGenerator:
        pass

    generation = types.ModuleType("isaaclab_mimic.datagen.generation")
    generation.DataGenerator = OriginalDataGenerator
    generation.num_success = 0
    generation.num_failures = 0
    generation.num_attempts = 0
    package = types.ModuleType("isaaclab_mimic")
    datagen = types.ModuleType("isaaclab_mimic.datagen")
    package.datagen = datagen
    datagen.generation = generation
    monkeypatch.setitem(sys.modules, "isaaclab_mimic", package)
    monkeypatch.setitem(sys.modules, "isaaclab_mimic.datagen", datagen)
    monkeypatch.setitem(sys.modules, "isaaclab_mimic.datagen.generation", generation)

    app = types.SimpleNamespace(close_count=0)

    def close():
        app.close_count += 1

    app.close = close
    monkeypatch.setattr(builtins, "_toy2box_test_app", app, raising=False)
    monkeypatch.setattr(builtins, "_toy2box_test_cfg", toy2box_config(), raising=False)
    monkeypatch.setattr(builtins, "_toy2box_test_success", object(), raising=False)
    monkeypatch.setattr(builtins, "_toy2box_test_raise", raise_in_main, raising=False)
    fixture = tmp_path / "stock_fixture.py"
    fixture.write_text(
        "import builtins\n"
        "simulation_app = builtins._toy2box_test_app\n"
        "def setup_env_config(*args, **kwargs):\n"
        "    return builtins._toy2box_test_cfg, builtins._toy2box_test_success\n"
        "builtins._toy2box_test_original_setup = setup_env_config\n"
        "def main():\n"
        "    builtins._toy2box_test_main = main\n"
        "    cfg, success = setup_env_config('preserved', device='cuda')\n"
        "    builtins._toy2box_test_observed = (cfg, success)\n"
        "    if builtins._toy2box_test_raise:\n"
        "        raise RuntimeError('fixture failure')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(SETTLED, "GENERATOR_SCRIPT", fixture)

    if raise_in_main:
        with pytest.raises(RuntimeError, match="fixture failure"):
            SETTLED.execute(args, paths, settings, run_path=runpy.run_path)
    else:
        SETTLED.execute(args, paths, settings, run_path=runpy.run_path)

    main_function = builtins._toy2box_test_main
    assert main_function.__globals__["setup_env_config"] is builtins._toy2box_test_original_setup
    assert builtins._toy2box_test_observed == (
        builtins._toy2box_test_cfg,
        builtins._toy2box_test_success,
    )
    box = builtins._toy2box_test_cfg.subtask_configs["agibot"][1]
    assert box.selection_strategy == "nearest_neighbor_robot_distance"
    assert box.selection_strategy_kwargs == {"nn_k": 1, "pos_weight": 1.0, "rot_weight": 2.0}
    assert generation.DataGenerator is OriginalDataGenerator
    assert app.close_count == 1
    records = [json.loads(line) for line in log.read_text().splitlines()]
    config_records = [item for item in records if item["record_type"] == "runtime_config_override"]
    assert len(config_records) == 1
    assert records[-1]["status"] == ("error" if raise_in_main else "complete")


def test_parse_args_keeps_stock_arguments_order_and_validates_positive():
    settings, remaining = SETTLED.parse_args(
        ["--task", "x", "--settling_log", "log", "--num_envs", "100", "--headless"]
    )
    assert settings.settle_max_steps == 10
    assert settings.settle_arm_motion is False
    assert settings.settle_arm_joint_velocity == pytest.approx(0.05)
    assert settings.settle_eef_velocity == pytest.approx(0.01)
    assert settings.box_robot_distance is False
    assert settings.box_rotation_weight == 2.0
    assert settings.box_nn_k == 1
    assert settings.box_interpolation_steps is None
    assert settings.enable_box_reselection is False
    assert settings.zero_action_noise is False
    assert settings.chunk_seed is None
    assert settings.grasp_offset_range is None
    assert settings.per_env_rmp_reset is False
    assert settings.settle_reset_retries == 0
    assert settings.truck_pick_z_offset_m == 0.0
    assert settings.pregrasp_time_dilation == 1
    assert remaining == ["--task", "x", "--num_envs", "100", "--headless"]
    with pytest.raises(SystemExit):
        SETTLED.parse_args(["--settling_log", "log", "--settle_max_steps", "0"])
    with pytest.raises(SystemExit):
        SETTLED.parse_args(
            ["--settling_log", "log", "--settle_stable_steps", "11", "--settle_max_steps", "10"]
        )
    for option, value in [
        ("--box_rotation_weight", "0"),
        ("--box_rotation_weight", "nan"),
        ("--box_nn_k", "0"),
        ("--box_interpolation_steps", "0"),
    ]:
        with pytest.raises(SystemExit):
            SETTLED.parse_args(["--settling_log", "log", "--box_robot_distance", option, value])
    for option, value in [
        ("--settle_arm_joint_velocity", "0"),
        ("--settle_arm_joint_velocity", "nan"),
        ("--settle_eef_velocity", "0"),
    ]:
        with pytest.raises(SystemExit):
            SETTLED.parse_args(["--settling_log", "log", "--settle_arm_motion", option, value])
    for option in ("--settle_arm_joint_velocity", "--settle_eef_velocity"):
        with pytest.raises(SystemExit):
            SETTLED.parse_args(["--settling_log", "log", option, "0.1"])
    for option, value in [
        ("--box_rotation_weight", "3"),
        ("--box_nn_k", "2"),
        ("--box_interpolation_steps", "20"),
    ]:
        with pytest.raises(SystemExit, match="2"):
            SETTLED.parse_args(["--settling_log", "log", option, value])
    with pytest.raises(SystemExit):
        SETTLED.parse_args(["--settling_log", "log", "--enable_box_reselection"])
    with pytest.raises(SystemExit):
        SETTLED.parse_args(["--settling_log", "log", "--zero_action_noise"])
    with pytest.raises(SystemExit):
        SETTLED.parse_args(
            ["--settling_log", "log", "--box_robot_distance", "--zero_action_noise"]
        )
    with pytest.raises(SystemExit):
        SETTLED.parse_args(
            [
                "--settling_log", "log", "--box_robot_distance", "--enable_box_reselection",
                "--zero_action_noise", "--chunk_seed", "0",
            ]
        )
    for prerequisites in (
        [],
        ["--box_robot_distance"],
        ["--box_robot_distance", "--enable_box_reselection"],
    ):
        with pytest.raises(SystemExit):
            SETTLED.parse_args(["--settling_log", "log", *prerequisites, "--chunk_seed", "1"])
    for values in (("-1", "2"), ("5", "4")):
        with pytest.raises(SystemExit):
            SETTLED.parse_args([
                "--settling_log", "log", "--box_robot_distance", "--enable_box_reselection",
                "--zero_action_noise", "--grasp_offset_range", *values,
            ])
    with pytest.raises(SystemExit):
        SETTLED.parse_args(["--settling_log", "log", "--grasp_offset_range", "50", "60"])
    with pytest.raises(SystemExit):
        SETTLED.parse_args(["--settling_log", "log", "--per_env_rmp_reset"])

    d3_args = [
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
    ]
    d3, remaining = SETTLED.parse_args(d3_args)
    assert d3.per_env_rmp_reset is True
    assert d3.settle_reset_retries == 0
    assert remaining == []
    d3_retry, remaining = SETTLED.parse_args(
        [*d3_args, "--chunk_seed", "4", "--settle_reset_retries", "2"]
    )
    assert d3_retry.settle_reset_retries == 2
    assert d3_retry.chunk_seed == 4
    assert remaining == []
    h4, remaining = SETTLED.parse_args(
        [
            *d3_args,
            "--chunk_seed", "4",
            "--settle_reset_retries", "2",
            "--truck_pick_z_offset_m", "-0.005",
        ]
    )
    assert h4.truck_pick_z_offset_m == pytest.approx(-0.005)
    assert remaining == []
    for invalid_offset in ("-0.004", "0.001", "nan"):
        with pytest.raises(SystemExit):
            SETTLED.parse_args(
                [
                    *d3_args,
                    "--chunk_seed", "4",
                    "--settle_reset_retries", "2",
                    "--truck_pick_z_offset_m", invalid_offset,
                ]
            )
    with pytest.raises(SystemExit):
        SETTLED.parse_args(["--settling_log", "log", "--truck_pick_z_offset_m", "-0.005"])
    with pytest.raises(SystemExit):
        SETTLED.parse_args(
            [
                *d3_args,
                "--chunk_seed", "4",
                "--settle_reset_retries", "2",
                "--truck_yaw_fold",
                "--truck_yaw_fold_log", "fold.jsonl",
                "--truck_pick_z_offset_m", "-0.005",
            ]
        )
    with pytest.raises(SystemExit):
        SETTLED.parse_args([*d3_args, "--settle_reset_retries", "2"])
    for value in ("-1", "1", "3"):
        with pytest.raises(SystemExit):
            SETTLED.parse_args([*d3_args, "--chunk_seed", "4", "--settle_reset_retries", value])
    with pytest.raises(SystemExit):
        SETTLED.parse_args(["--settling_log", "log", "--settle_reset_retries", "2"])
    for remove in (
        "--box_robot_distance", "--enable_box_reselection", "--zero_action_noise",
        "--settle_arm_motion", "--grasp_offset_range", "--box_interpolation_steps",
    ):
        index = d3_args.index(remove)
        width = 3 if remove == "--grasp_offset_range" else 2 if remove == "--box_interpolation_steps" else 1
        with pytest.raises(SystemExit):
            SETTLED.parse_args(d3_args[:index] + d3_args[index + width:])
    for option, bad_value in (
        ("--settle_arm_joint_velocity", "0.14"),
        ("--settle_eef_velocity", "0.02"),
        ("--settle_max_steps", "19"),
        ("--box_interpolation_steps", "29"),
        ("--settle_linear_velocity", "0.1"),
        ("--settle_angular_velocity", "0.2"),
        ("--settle_stable_steps", "1"),
        ("--box_rotation_weight", "3"),
        ("--box_nn_k", "2"),
    ):
        invalid = list(d3_args)
        if option in invalid:
            invalid[invalid.index(option) + 1] = bad_value
        else:
            invalid.extend((option, bad_value))
        with pytest.raises(SystemExit):
            SETTLED.parse_args(invalid)

    selected, remaining = SETTLED.parse_args(
        [
            "--settling_log",
            "log",
            "--box_robot_distance",
            "--enable_box_reselection",
            "--zero_action_noise",
            "--chunk_seed",
            "2",
            "--settle_arm_motion",
            "--settle_arm_joint_velocity",
            "0.04",
            "--settle_eef_velocity",
            "0.009",
            "--grasp_offset_range",
            "50",
            "60",
            "--box_rotation_weight=3",
            "--box_nn_k",
            "2",
            "--box_interpolation_steps",
            "20",
            "--task",
            "stock",
        ]
    )
    assert selected.box_robot_distance is True
    assert selected.enable_box_reselection is True
    assert selected.zero_action_noise is True
    assert selected.chunk_seed == 2
    assert selected.settle_arm_motion is True
    assert selected.settle_arm_joint_velocity == pytest.approx(0.04)
    assert selected.settle_eef_velocity == pytest.approx(0.009)
    assert selected.grasp_offset_range == [50, 60]
    assert selected.box_rotation_weight == 3.0
    assert selected.box_nn_k == 2
    assert selected.box_interpolation_steps == 20
    assert remaining == ["--task", "stock"]
