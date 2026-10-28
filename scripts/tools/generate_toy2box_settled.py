#!/usr/bin/env python3
"""Run the stock Toy2Box Mimic generator after per-environment settling.

This file deliberately loads Isaac Lab only after validating every output path.
The original generator, environment loop, and data generation algorithm remain
authoritative. It injects a reset-queue proxy that advances a resetting
environment with open-gripper hold actions until the toy truck is stationary,
can retry only a capped environment reset under an explicit bounded policy, and
can apply an explicit runtime-only source-selection override to the box subtask.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import ExitStack
import copy
import datetime
import json
import math
from numbers import Integral
import os
from pathlib import Path
import re
import runpy
import sys
from typing import Any, Iterable, TextIO


EXPECTED_TASK = "Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0"
MAX_RESET_RETRY_RATIO = 0.02
GENERATOR_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "imitation_learning"
    / "isaaclab_mimic"
    / "generate_dataset.py"
)


class SettlingError(RuntimeError):
    """Raised when settling cannot safely complete."""


class SettlingCapError(SettlingError):
    """Raised for the sole settling failure eligible for a targeted reset retry."""

    def __init__(
        self,
        message: str,
        original_layout: dict[str, Any],
        discarded_layout: dict[str, Any],
    ):
        super().__init__(message)
        self.original_layout = original_layout
        self.discarded_layout = discarded_layout


class ConfigurationError(RuntimeError):
    """Raised when the runtime task config is not the approved Toy2Box shape."""


def _resolved(path: os.PathLike[str] | str) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _failed_dataset_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_failed{output.suffix}")


def _option_value(arguments: list[str], name: str) -> str | None:
    """Read a required single-value option without reordering arguments."""
    found: list[str] = []
    prefix = f"{name}="
    for index, argument in enumerate(arguments):
        if argument.startswith(prefix):
            found.append(argument[len(prefix) :])
        elif argument == name:
            if index + 1 >= len(arguments) or arguments[index + 1].startswith("--"):
                raise ValueError(f"{name} requires a value")
            found.append(arguments[index + 1])
    if len(found) > 1:
        raise ValueError(f"{name} must be provided exactly once")
    return found[0] if found else None


def _protected_output_paths() -> tuple[Path, ...]:
    """Reuse the evaluation observer's source-demo protection list."""
    try:
        from evaluate_toy2box_bc import PROTECTED_OUTPUT_PATHS
    except ImportError:
        # Import by file path is unnecessary during normal script execution, but
        # tests may load this module without putting scripts/tools on sys.path.
        import importlib.util

        evaluator = Path(__file__).with_name("evaluate_toy2box_bc.py")
        spec = importlib.util.spec_from_file_location("_toy2box_evaluator", evaluator)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load protected paths from {evaluator}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        PROTECTED_OUTPUT_PATHS = module.PROTECTED_OUTPUT_PATHS
    return tuple(Path(item) for item in PROTECTED_OUTPUT_PATHS)


def validate_invocation(
    original_args: list[str],
    settling_log: str,
    truck_yaw_fold_log: str | None = None,
) -> dict[str, Path]:
    """Validate inputs and reserve no paths; this runs before Kit is imported."""
    if "--headless" not in original_args:
        raise ValueError("the settled generator requires --headless")

    task = _option_value(original_args, "--task")
    input_value = _option_value(original_args, "--input_file")
    output_value = _option_value(original_args, "--output_file")
    if task != EXPECTED_TASK:
        raise ValueError(f"--task must be {EXPECTED_TASK}")
    if input_value is None or output_value is None:
        raise ValueError("explicit --input_file and --output_file are required")

    raw_input = Path(input_value).expanduser().absolute()
    raw_output = Path(output_value).expanduser().absolute()
    raw_failed = _failed_dataset_path(raw_output)
    raw_log = Path(settling_log).expanduser().absolute()
    raw_fold_log = (
        Path(truck_yaw_fold_log).expanduser().absolute()
        if truck_yaw_fold_log is not None
        else None
    )
    raw_outputs = tuple(
        path for path in (raw_output, raw_failed, raw_log, raw_fold_log) if path is not None
    )
    if len(set(raw_outputs)) != len(raw_outputs):
        raise ValueError("success, failed, and settling-log outputs must be distinct")
    if raw_input in raw_outputs:
        raise ValueError("an output must not alias the input dataset")
    for path in raw_outputs:
        # Check the path as written before resolving symlinks.  In particular,
        # resolve(strict=False) would otherwise hide an existing dangling link.
        if os.path.lexists(path):
            raise FileExistsError(f"output already exists: {path}")

    input_path = _resolved(raw_input)
    output_path = _resolved(raw_output)
    failed_path = _resolved(raw_failed)
    log_path = _resolved(raw_log)
    if input_path.suffix != ".hdf5" or output_path.suffix != ".hdf5":
        raise ValueError("--input_file and --output_file must end in lowercase .hdf5")
    if not input_path.is_file():
        raise FileNotFoundError(f"input dataset does not exist: {input_path}")

    paths = {
        "input": input_path,
        "output": output_path,
        "failed_output": failed_path,
        "settling_log": log_path,
    }
    fold_log_path = _resolved(raw_fold_log) if raw_fold_log is not None else None
    if fold_log_path is not None:
        paths["truck_yaw_fold_log"] = fold_log_path
    distinct_outputs = tuple(
        path for path in (output_path, failed_path, log_path, fold_log_path) if path is not None
    )
    if len(set(distinct_outputs)) != len(distinct_outputs):
        raise ValueError("success, failed, and settling-log outputs must be distinct")
    if input_path in distinct_outputs:
        raise ValueError("an output must not alias the input dataset")

    protected = {_resolved(item) for item in _protected_output_paths()}
    for path in distinct_outputs:
        if path in protected:
            raise ValueError(f"refusing protected source-demo path: {path}")
        if not path.parent.exists():
            raise FileNotFoundError(f"output parent does not exist: {path.parent}")
        if not path.parent.is_dir():
            raise NotADirectoryError(f"output parent is not a directory: {path.parent}")
    return paths


class JsonlLog:
    """Immediately flushed JSONL log shared by all generator coroutines."""

    def __init__(self, stream: TextIO):
        self.stream = stream
        self.settling_attempts = 0
        self.capped_attempts = 0
        self.generation_results = 0
        self.logical_initial_settling_starts = 0
        self.settling_retry_operations = 0
        self.discarded_layouts = 0

    def write(self, record: dict[str, Any]) -> None:
        self.stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self.stream.flush()


def _to_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def _truck_state(env: Any, env_id: int) -> tuple[list[float], list[float], float, float]:
    data = env.scene["toy_truck"].data
    position = _to_list(data.root_pos_w[env_id])[:3]
    quaternion = _to_list(data.root_quat_w[env_id])[:4]
    if hasattr(env.scene, "env_origins"):
        origin = _to_list(env.scene.env_origins[env_id])[:3]
        position = [value - offset for value, offset in zip(position, origin)]
    linear = _to_list(data.root_lin_vel_w[env_id])[:3]
    angular = _to_list(data.root_ang_vel_w[env_id])[:3]
    values = position + quaternion + linear + angular
    if len(position) != 3 or len(quaternion) != 4 or not all(math.isfinite(item) for item in values):
        raise SettlingError(f"non-finite or malformed toy-truck state for env {env_id}")
    linear_speed = math.sqrt(sum(component * component for component in linear))
    angular_speed = math.sqrt(sum(component * component for component in angular))
    return position, quaternion, linear_speed, angular_speed


def _body_pose(env: Any, env_id: int, body_name: str) -> dict[str, list[float]]:
    data = env.scene[body_name].data
    position_world = _to_list(data.root_pos_w[env_id])[:3]
    quaternion = _to_list(data.root_quat_w[env_id])[:4]
    origin = [0.0, 0.0, 0.0]
    if hasattr(env.scene, "env_origins"):
        origin = _to_list(env.scene.env_origins[env_id])[:3]
    position_env = [value - offset for value, offset in zip(position_world, origin)]
    values = position_world + position_env + quaternion
    if (len(position_world) != 3 or len(position_env) != 3 or len(quaternion) != 4
            or not all(math.isfinite(item) for item in values)):
        raise SettlingError(f"non-finite or malformed {body_name} pose for env {env_id}")
    return {
        "position_world": position_world,
        "position_env": position_env,
        "quaternion_wxyz": quaternion,
    }


def _layout_pose(env: Any, env_id: int) -> dict[str, Any]:
    return {
        "frame_semantics": {
            "position_world": "simulation world frame",
            "position_env": "environment-local frame (world minus env origin)",
            "quaternion_wxyz": "world-frame orientation, scalar first",
        },
        "toy_truck": _body_pose(env, env_id, "toy_truck"),
        "box": _body_pose(env, env_id, "box"),
    }


def _eef_position(env: Any, env_id: int) -> list[float]:
    pose = env.get_robot_eef_pose(env_ids=[env_id], eef_name="agibot")[0]
    position = _to_list(pose[:3, 3])
    if len(position) != 3 or not all(math.isfinite(item) for item in position):
        raise SettlingError(f"non-finite or malformed agibot EEF position for env {env_id}")
    return position


def _right_arm_joint_ids(env: Any) -> list[int]:
    robot = env.scene["robot"]
    found = robot.find_joints("right_arm_joint[1-7]")
    raw_ids = found[0] if isinstance(found, tuple) else found
    raw_values = [float(item) for item in raw_ids]
    ids = [int(item) for item in raw_values]
    joint_count = int(robot.data.joint_vel.shape[-1])
    if (len(ids) != 7 or len(set(ids)) != 7
            or any(not math.isfinite(raw) or raw != integer for raw, integer in zip(raw_values, ids))
            or any(item < 0 or item >= joint_count for item in ids)):
        raise SettlingError("right arm settling requires seven unique valid joint IDs")
    return ids


class SettlingCoordinator:
    """Coordinate fail-closed settling without changing the stock env loop."""

    def __init__(
        self,
        logger: JsonlLog,
        torch_module: Any,
        linear_threshold: float,
        angular_threshold: float,
        stable_steps: int,
        max_steps: int,
        settle_arm_motion: bool = False,
        arm_joint_velocity_threshold: float = 0.05,
        eef_velocity_threshold: float = 0.01,
        settle_reset_retries: int = 0,
        chunk_seed: int | None = None,
        output_label: str | None = None,
    ):
        self.logger = logger
        self.torch = torch_module
        self.linear_threshold = linear_threshold
        self.angular_threshold = angular_threshold
        self.stable_steps = stable_steps
        self.max_steps = max_steps
        self.settle_arm_motion = settle_arm_motion
        self.arm_joint_velocity_threshold = arm_joint_velocity_threshold
        self.eef_velocity_threshold = eef_velocity_threshold
        self.settle_reset_retries = settle_reset_retries
        self.chunk_seed = chunk_seed
        self.output_label = output_label
        self.attempts: dict[int, int] = {}
        self.action_queue: Any = None
        self._original_qsize: Any = None
        self.fatal_message: str | None = None

    def attach(self, action_queue: Any) -> None:
        if self.action_queue is action_queue:
            return
        if self.action_queue is not None:
            raise SettlingError("multiple action queues are not supported")
        self.action_queue = action_queue
        self._original_qsize = action_queue.qsize

        def checked_qsize() -> int:
            if self.fatal_message is not None:
                raise SettlingError(self.fatal_message)
            return self._original_qsize()

        action_queue.qsize = checked_qsize

    def fail(self, message: str) -> None:
        if self.fatal_message is None:
            self.fatal_message = message

    def begin_attempt(self, env_id: int) -> int:
        if self.fatal_message is not None:
            raise SettlingError(self.fatal_message)
        attempt = self.attempts.get(env_id, 0) + 1
        self.attempts[env_id] = attempt
        self.logger.logical_initial_settling_starts += 1
        if self.settle_reset_retries:
            self.logger.write(
                {
                    "record_type": "settling_attempt_start",
                    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "chunk_seed": self.chunk_seed,
                    "output_label": self.output_label,
                    "env_id": env_id,
                    "attempt": attempt,
                    "logical_initial_settling_starts": (
                        self.logger.logical_initial_settling_starts
                    ),
                }
            )
        return attempt

    def authorize_retry(
        self,
        env_id: int,
        attempt: int,
        retry_count: int,
        original_layout: dict[str, Any],
        discarded_layout: dict[str, Any],
    ) -> None:
        if self.attempts.get(env_id) != attempt:
            raise SettlingError(f"retry attempt identity mismatch for env {env_id}")
        if retry_count < 1 or retry_count > self.settle_reset_retries:
            raise SettlingError(f"invalid settling retry count {retry_count}")
        starts = self.logger.logical_initial_settling_starts
        next_operations = self.logger.settling_retry_operations + 1
        if starts <= 0 or next_operations / starts > MAX_RESET_RETRY_RATIO:
            message = (
                "settling reset retry ratio would exceed 2%: "
                f"{next_operations}/{starts} for env {env_id} attempt {attempt}"
            )
            self.fail(message)
            raise SettlingError(message)
        self.logger.settling_retry_operations = next_operations
        self.logger.discarded_layouts += 1
        self.logger.write(
            {
                "record_type": "settling_retry",
                "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "chunk_seed": self.chunk_seed,
                "output_label": self.output_label,
                "env_id": env_id,
                "attempt": attempt,
                "retry_count": retry_count,
                "logical_initial_settling_starts": starts,
                "retry_operations": next_operations,
                "retry_ratio": next_operations / starts,
                "original_layout": original_layout,
                "discarded_layout": discarded_layout,
            }
        )

    def restore(self) -> None:
        if self.action_queue is not None and self._original_qsize is not None:
            self.action_queue.qsize = self._original_qsize
        self.action_queue = None
        self._original_qsize = None

    async def settle(
        self,
        env: Any,
        env_id: int,
        action_queue: Any,
        attempt: int | None = None,
        retry_index: int = 0,
    ) -> None:
        self.attach(action_queue)
        if attempt is None:
            attempt = self.begin_attempt(env_id)
        elif self.attempts.get(env_id) != attempt:
            raise SettlingError(f"settling attempt identity mismatch for env {env_id}")
        if retry_index < 0 or retry_index > self.settle_reset_retries:
            raise SettlingError(f"invalid settling retry index {retry_index}")
        cycle_initial_layout = _layout_pose(env, env_id) if self.settle_reset_retries else None
        initial_position, initial_quaternion, _, _ = _truck_state(env, env_id)
        trace: list[dict[str, float | int]] = []
        consecutive = 0
        stable = False
        arm_joint_ids: list[int] | None = None
        step_dt: float | None = None
        previous_eef_position: list[float] | None = None
        previous_joint_positions: list[float] | None = None

        try:
            if self.settle_arm_motion:
                arm_joint_ids = _right_arm_joint_ids(env)
                step_dt = float(env.step_dt)
                if not math.isfinite(step_dt) or step_dt <= 0:
                    raise SettlingError("arm settling requires a finite positive env.step_dt")
                previous_eef_position = _eef_position(env, env_id)
                previous_joint_positions = _to_list(
                    env.scene["robot"].data.joint_pos[env_id, arm_joint_ids]
                )
                if len(previous_joint_positions) != 7 or not all(
                    math.isfinite(item) for item in previous_joint_positions
                ):
                    raise SettlingError(f"non-finite right arm joint position for env {env_id}")

            for step in range(1, self.max_steps + 1):
                action = self.torch.zeros(7, dtype=self.torch.float32, device="cpu")
                action[-1] = 1.0
                await action_queue.put((env_id, action))
                await action_queue.join()
                position, _, linear_speed, angular_speed = _truck_state(env, env_id)
                row = {
                    "step": step,
                    "linear_speed": linear_speed,
                    "angular_speed": angular_speed,
                    "z": position[2],
                }
                arm_stable = True
                if self.settle_arm_motion:
                    joint_velocities = _to_list(
                        env.scene["robot"].data.joint_vel[env_id, arm_joint_ids]
                    )
                    if len(joint_velocities) != 7 or not all(
                        math.isfinite(item) for item in joint_velocities
                    ):
                        raise SettlingError(f"non-finite right arm joint velocity for env {env_id}")
                    arm_joint_velocity_max = max(abs(item) for item in joint_velocities)
                    joint_positions = _to_list(
                        env.scene["robot"].data.joint_pos[env_id, arm_joint_ids]
                    )
                    if len(joint_positions) != 7 or not all(
                        math.isfinite(item) for item in joint_positions
                    ):
                        raise SettlingError(f"non-finite right arm joint position for env {env_id}")
                    arm_joint_position_delta_speed_max = max(
                        abs(current - previous) / step_dt
                        for current, previous in zip(joint_positions, previous_joint_positions)
                    )
                    previous_joint_positions = joint_positions
                    current_eef_position = _eef_position(env, env_id)
                    eef_linear_speed = math.sqrt(sum(
                        (current - previous) ** 2
                        for current, previous in zip(current_eef_position, previous_eef_position)
                    )) / step_dt
                    previous_eef_position = current_eef_position
                    row.update(
                        {
                            "arm_joint_velocity_max": arm_joint_velocity_max,
                            "arm_joint_velocities": joint_velocities,
                            "arm_joint_positions": joint_positions,
                            "arm_joint_position_delta_speed_max": arm_joint_position_delta_speed_max,
                            "eef_linear_speed": eef_linear_speed,
                        }
                    )
                    arm_stable = (
                        arm_joint_velocity_max < self.arm_joint_velocity_threshold
                        and eef_linear_speed < self.eef_velocity_threshold
                    )
                trace.append(row)
                if (linear_speed < self.linear_threshold
                        and angular_speed < self.angular_threshold and arm_stable):
                    consecutive += 1
                else:
                    consecutive = 0
                if consecutive >= self.stable_steps:
                    stable = True
                    break
        except Exception as exc:
            message = f"settling env {env_id} attempt {attempt} failed: {exc}"
            self.fail(message)
            raise SettlingError(message) from exc

        settled_position, settled_quaternion, _, _ = _truck_state(env, env_id)
        capped = not stable
        self.logger.settling_attempts += 1
        self.logger.capped_attempts += int(capped)
        self.logger.write(
            {
                "record_type": "settling_attempt",
                "env_id": env_id,
                "attempt": attempt,
                "retry_index": retry_index,
                "steps": len(trace),
                "stable": stable,
                "capped": capped,
                "initial_position_env": initial_position,
                "initial_quaternion_wxyz": initial_quaternion,
                "initial_layout": cycle_initial_layout,
                "settled_position_env": settled_position,
                "settled_quaternion_wxyz": settled_quaternion,
                "trace": trace,
                "hold_recorded": False,
                "arm_joint_ids": arm_joint_ids,
                "step_dt": step_dt,
            }
        )
        if capped:
            message = f"settling cap reached for env {env_id} attempt {attempt} after {self.max_steps} steps"
            if self.settle_reset_retries:
                raise SettlingCapError(message, cycle_initial_layout, _layout_pose(env, env_id))
            self.fail(message)
            raise SettlingError(message)

        # Discard the reset and hold-step recorder prefix, then establish the
        # settled state as the episode's sole post-reset initial state.
        env.recorder_manager.reset([env_id])
        env.recorder_manager.record_post_reset([env_id])
        if hasattr(env, "episode_length_buf"):
            env.episode_length_buf[env_id] = 0


class ResetQueueProxy:
    """Forward reset requests and settle only after the real reset completes."""

    def __init__(self, queue: Any, action_queue: Any, coordinator: SettlingCoordinator, env: Any, env_id: int):
        self.queue = queue
        self.action_queue = action_queue
        self.coordinator = coordinator
        self.env = env
        self.env_id = env_id
        self.attempt: int | None = None

    async def put(self, item: int) -> None:
        if item != self.env_id:
            raise SettlingError(f"reset env mismatch: expected {self.env_id}, got {item}")
        if self.attempt is not None:
            raise SettlingError(f"duplicate reset request for env {self.env_id}")
        self.attempt = self.coordinator.begin_attempt(self.env_id)
        await self.queue.put(item)

    async def join(self) -> None:
        if self.attempt is None:
            raise SettlingError(f"reset join before put for env {self.env_id}")
        await self.queue.join()
        retry_index = 0
        try:
            while True:
                try:
                    await self.coordinator.settle(
                        self.env,
                        self.env_id,
                        self.action_queue,
                        attempt=self.attempt,
                        retry_index=retry_index,
                    )
                    return
                except SettlingCapError as exc:
                    if retry_index >= self.coordinator.settle_reset_retries:
                        message = (
                            f"settling cap retries exhausted for env {self.env_id} "
                            f"attempt {self.attempt} after {retry_index} retries"
                        )
                        self.coordinator.fail(message)
                        raise SettlingError(message) from exc
                    retry_index += 1
                    self.coordinator.authorize_retry(
                        self.env_id,
                        self.attempt,
                        retry_index,
                        exc.original_layout,
                        exc.discarded_layout,
                    )
                    # The stock env.reset records/exports the current episode in its
                    # pre-reset hook. Clear only this capped env first so hold steps
                    # cannot become a phantom failed demonstration.
                    self.env.recorder_manager.reset([self.env_id])
                    await self.queue.put(self.env_id)
                    await self.queue.join()
        except SettlingError as exc:
            self.coordinator.fail(str(exc))
            raise
        except Exception as exc:
            message = f"settling env {self.env_id} failed: {exc}"
            self.coordinator.fail(message)
            raise SettlingError(message) from exc


def _pose_jump(torch_module: Any, start_pose: Any, target_pose: Any) -> tuple[float, float]:
    """Return translation metres and SO(3) geodesic degrees for two 4x4 poses."""
    if tuple(start_pose.shape) != (4, 4) or tuple(target_pose.shape) != (4, 4):
        raise ConfigurationError("transition audit requires 4x4 poses")
    if not bool(torch_module.isfinite(start_pose).all()) or not bool(torch_module.isfinite(target_pose).all()):
        raise ConfigurationError("transition audit encountered a non-finite pose")
    position_jump = float(torch_module.linalg.norm(start_pose[:3, 3] - target_pose[:3, 3]).item())
    delta_rotation = start_pose[:3, :3].transpose(0, 1) @ target_pose[:3, :3]
    cosine = torch_module.clamp((torch_module.trace(delta_rotation) - 1.0) / 2.0, -1.0, 1.0)
    rotation_jump = math.degrees(float(torch_module.acos(cosine).item()))
    if not math.isfinite(position_jump) or not math.isfinite(rotation_jump):
        raise ConfigurationError("transition audit produced a non-finite metric")
    return position_jump, rotation_jump


def _pose_yaw_radians(pose: Any) -> float:
    """Return a finite yaw in (-pi, pi] from a homogeneous pose."""
    if tuple(pose.shape)[-2:] != (4, 4):
        raise ConfigurationError("yaw folding requires 4x4 poses")
    while len(pose.shape) > 2:
        if int(pose.shape[0]) != 1:
            raise ConfigurationError("yaw folding requires exactly one target pose")
        pose = pose[0]
    yaw = math.atan2(float(pose[1, 0]), float(pose[0, 0]))
    if not math.isfinite(yaw):
        raise ConfigurationError("yaw folding encountered a non-finite pose")
    return math.pi if yaw <= -math.pi else yaw


def _yaw_requires_fold(yaw: float) -> bool:
    """Whether yaw lies outside the preregistered half-open interval (-90, 90]."""
    normalized = math.atan2(math.sin(yaw), math.cos(yaw))
    if normalized <= -math.pi:
        normalized = math.pi
    return normalized <= -math.pi / 2.0 or normalized > math.pi / 2.0


def _fold_pose_about_local_z(pose: Any) -> Any:
    """Return ``pose * Rz(pi)`` without mutating ``pose``."""
    folded = pose.clone() if hasattr(pose, "clone") else pose.copy()
    folded[..., :3, 0:2] *= -1
    return folded


def _fold_pose_to_truck_symmetry(pose: Any) -> tuple[Any, bool, float, float]:
    """Fold one pose into (-90, 90], returning pose, flag and before/after yaw."""
    original_yaw = _pose_yaw_radians(pose)
    should_fold = _yaw_requires_fold(original_yaw)
    folded = _fold_pose_about_local_z(pose) if should_fold else (
        pose.clone() if hasattr(pose, "clone") else pose.copy()
    )
    return folded, should_fold, original_yaw, _pose_yaw_radians(folded)


def _absolute_yaw_delta_degrees(first: float, second: float) -> float:
    return math.degrees(abs(math.atan2(math.sin(first - second), math.cos(first - second))))


def _fold_source_truck_poses(generator: Any) -> None:
    """Fold each source episode consistently using its first subtask start pose."""
    pool = generator.src_demo_datagen_info_pool
    boundaries = pool.subtask_boundaries.get("agibot")
    infos = pool.datagen_infos
    if boundaries is None or len(boundaries) != len(infos):
        raise ConfigurationError("yaw folding requires agibot boundaries for every source episode")
    originals: list[Any] = []
    metadata: list[dict[str, Any]] = []
    for index, (info, episode_boundaries) in enumerate(zip(infos, boundaries)):
        poses = info.object_poses.get("toy_truck")
        if poses is None or len(poses) == 0:
            raise ConfigurationError(f"source episode {index} has no toy_truck poses")
        try:
            start_index = int(episode_boundaries[0][0])
        except (IndexError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"source episode {index} has malformed subtask boundaries") from exc
        if start_index < 0 or start_index >= len(poses):
            raise ConfigurationError(f"source episode {index} has an invalid first subtask start")
        original = poses.clone() if hasattr(poses, "clone") else poses.copy()
        original_yaw = _pose_yaw_radians(original[start_index])
        folded_flag = _yaw_requires_fold(original_yaw)
        folded = _fold_pose_about_local_z(original) if folded_flag else (
            original.clone() if hasattr(original, "clone") else original.copy()
        )
        info.object_poses["toy_truck"] = folded
        originals.append(original)
        metadata.append(
            {
                "source_demo_index": index,
                "first_subtask_start_index": start_index,
                "original_yaw_deg": math.degrees(original_yaw),
                "folded_yaw_deg": math.degrees(_pose_yaw_radians(folded[start_index])),
                "folded": folded_flag,
            }
        )
    generator._truck_yaw_fold_source_original_poses = originals
    generator._truck_yaw_fold_source_metadata = metadata


def make_settled_generator(
    base_class: type,
    coordinator: SettlingCoordinator,
    enable_source_audit: bool = False,
    enable_truck_yaw_fold: bool = False,
    truck_yaw_fold_logger: JsonlLog | None = None,
    truck_pick_z_offset_m: float = 0.0,
) -> type:
    """Create the dynamic subclass installed into the stock generation module."""

    class SettledDataGenerator(base_class):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            if enable_truck_yaw_fold:
                if truck_yaw_fold_logger is None:
                    raise ConfigurationError("truck yaw folding requires its audit logger")
                _fold_source_truck_poses(self)

        def _apply_truck_pick_z_offset(
            self,
            trajectory: Any,
            *,
            env_id: int,
            eef_name: str,
            subtask_ind: int,
        ) -> Any:
            if truck_pick_z_offset_m == 0.0:
                return trajectory
            if eef_name != "agibot" or subtask_ind != 0:
                return trajectory
            object_ref = self.env_cfg.subtask_configs[eef_name][subtask_ind].object_ref
            if object_ref != "toy_truck":
                return trajectory

            before_z: list[float] = []
            after_z: list[float] = []
            # WaypointTrajectory asserts on out-of-range __getitem__ instead of
            # raising IndexError, so Python fallback iteration is not supported.
            for index in range(len(trajectory)):
                waypoint = trajectory[index]
                original_pose = waypoint.pose
                shifted_pose = original_pose.clone()
                before_z.append(float(original_pose[2, 3].item()))
                shifted_pose[2, 3] += truck_pick_z_offset_m
                waypoint.pose = shifted_pose
                after_z.append(float(shifted_pose[2, 3].item()))
            coordinator.logger.write(
                {
                    "record_type": "truck_pick_z_offset",
                    "env_id": env_id,
                    "attempt": coordinator.attempts.get(env_id),
                    "subtask": f"{eef_name}/{subtask_ind}",
                    "subtask_index": subtask_ind,
                    "eef_name": eef_name,
                    "object_ref": object_ref,
                    "shift_m": truck_pick_z_offset_m,
                    "waypoint_count": len(before_z),
                    "first_z_before_m": before_z[0] if before_z else None,
                    "first_z_after_m": after_z[0] if after_z else None,
                    "last_z_before_m": before_z[-1] if before_z else None,
                    "last_z_after_m": after_z[-1] if after_z else None,
                    "application_count": 1,
                    "applied_exactly_once": all(
                        math.isclose(after - before, truck_pick_z_offset_m, abs_tol=1e-6)
                        for before, after in zip(before_z, after_z)
                    ),
                }
            )
            return trajectory

        def generate_eef_subtask_trajectory(
            self,
            env_id: int,
            eef_name: str,
            subtask_ind: int,
            all_randomized_subtask_boundaries: dict,
            runtime_subtask_constraints_dict: dict,
            selected_src_demo_inds: dict,
        ) -> Any:
            audit_context_enabled = enable_source_audit or enable_truck_yaw_fold
            if not audit_context_enabled:
                trajectory = super().generate_eef_subtask_trajectory(
                    env_id=env_id,
                    eef_name=eef_name,
                    subtask_ind=subtask_ind,
                    all_randomized_subtask_boundaries=all_randomized_subtask_boundaries,
                    runtime_subtask_constraints_dict=runtime_subtask_constraints_dict,
                    selected_src_demo_inds=selected_src_demo_inds,
                )
                return self._apply_truck_pick_z_offset(
                    trajectory,
                    env_id=env_id,
                    eef_name=eef_name,
                    subtask_ind=subtask_ind,
                )
            if hasattr(self, "_source_selection_context"):
                raise ConfigurationError("nested source-selection audit context")
            object_ref = None
            if enable_truck_yaw_fold:
                object_ref = self.env_cfg.subtask_configs[eef_name][subtask_ind].object_ref
                context: dict[str, Any] | tuple[int, str, int] = {
                    "env_id": env_id,
                    "eef_name": eef_name,
                    "subtask_index": subtask_ind,
                    "object_ref": object_ref,
                    "target_original_yaw": None,
                    "target_folded_yaw": None,
                    "target_folded": False,
                }
            else:
                context = (env_id, eef_name, subtask_ind)
            self._source_selection_context = context
            original_get_object_poses = None
            original_instance_value = None
            had_instance_value = False
            if enable_truck_yaw_fold and object_ref == "toy_truck":
                original_get_object_poses = self.env.get_object_poses
                try:
                    instance_values = vars(self.env)
                except TypeError:
                    instance_values = {}
                had_instance_value = "get_object_poses" in instance_values
                original_instance_value = instance_values.get("get_object_poses")

                def folded_get_object_poses(*args: Any, **kwargs: Any) -> dict[str, Any]:
                    poses = original_get_object_poses(*args, **kwargs)
                    if "toy_truck" not in poses:
                        raise ConfigurationError("target object poses have no toy_truck")
                    folded_truck, folded_flag, original_yaw, folded_yaw = (
                        _fold_pose_to_truck_symmetry(poses["toy_truck"])
                    )
                    result = dict(poses)
                    result["toy_truck"] = folded_truck
                    if context["target_original_yaw"] is None:
                        context["target_original_yaw"] = original_yaw
                        context["target_folded_yaw"] = folded_yaw
                        context["target_folded"] = folded_flag
                    return result

                self.env.get_object_poses = folded_get_object_poses
            try:
                trajectory = super().generate_eef_subtask_trajectory(
                    env_id=env_id,
                    eef_name=eef_name,
                    subtask_ind=subtask_ind,
                    all_randomized_subtask_boundaries=all_randomized_subtask_boundaries,
                    runtime_subtask_constraints_dict=runtime_subtask_constraints_dict,
                    selected_src_demo_inds=selected_src_demo_inds,
                )
            finally:
                if original_get_object_poses is not None:
                    if had_instance_value:
                        self.env.get_object_poses = original_instance_value
                    else:
                        delattr(self.env, "get_object_poses")
                del self._source_selection_context
            return self._apply_truck_pick_z_offset(
                trajectory,
                env_id=env_id,
                eef_name=eef_name,
                subtask_ind=subtask_ind,
            )

        def select_source_demo(
            self,
            eef_name: str,
            eef_pose: Any,
            object_pose: Any,
            src_demo_current_subtask_boundaries: Any,
            subtask_object_name: str,
            selection_strategy_name: str,
            selection_strategy_kwargs: dict | None = None,
        ) -> int:
            selected_index = super().select_source_demo(
                eef_name=eef_name,
                eef_pose=eef_pose,
                object_pose=object_pose,
                src_demo_current_subtask_boundaries=src_demo_current_subtask_boundaries,
                subtask_object_name=subtask_object_name,
                selection_strategy_name=selection_strategy_name,
                selection_strategy_kwargs=selection_strategy_kwargs,
            )
            if enable_source_audit or enable_truck_yaw_fold:
                context = getattr(self, "_source_selection_context", None)
                if context is None:
                    raise ConfigurationError("source-selection call has no subtask audit context")
                if isinstance(context, tuple):
                    env_id, context_eef_name, subtask_index = context
                else:
                    env_id = context["env_id"]
                    context_eef_name = context["eef_name"]
                    subtask_index = context["subtask_index"]
                if context_eef_name != eef_name or env_id not in coordinator.attempts:
                    raise ConfigurationError("source-selection audit context is inconsistent")
                kwargs = {} if selection_strategy_kwargs is None else copy.deepcopy(selection_strategy_kwargs)
                if not isinstance(kwargs, dict):
                    raise ConfigurationError("source-selection audit requires dictionary kwargs")
                if enable_source_audit:
                    coordinator.logger.write(
                        {
                            "record_type": "source_selection",
                            "env_id": env_id,
                            "attempt": coordinator.attempts[env_id],
                            "eef_name": eef_name,
                            "subtask_index": subtask_index,
                            "object_ref": subtask_object_name,
                            "selection_strategy": selection_strategy_name,
                            "selection_strategy_kwargs": kwargs,
                            "selected_source_demo_index": int(selected_index),
                        }
                    )
                if enable_truck_yaw_fold:
                    boundary = src_demo_current_subtask_boundaries[int(selected_index)]
                    source_start = int(boundary[0])
                    source_info = self.src_demo_datagen_info_pool.datagen_infos[int(selected_index)]
                    source_pose = source_info.object_poses[subtask_object_name][source_start]
                    source_folded_yaw = _pose_yaw_radians(source_pose)
                    if subtask_object_name == "toy_truck":
                        source_original_pose = self._truck_yaw_fold_source_original_poses[
                            int(selected_index)
                        ][source_start]
                        source_original_yaw = _pose_yaw_radians(source_original_pose)
                        source_folded_flag = self._truck_yaw_fold_source_metadata[
                            int(selected_index)
                        ]["folded"]
                    else:
                        source_original_yaw = source_folded_yaw
                        source_folded_flag = False
                    target_original_yaw = context["target_original_yaw"]
                    target_folded_yaw = context["target_folded_yaw"]
                    if target_original_yaw is None:
                        target_original_yaw = _pose_yaw_radians(object_pose)
                        target_folded_yaw = target_original_yaw
                    truck_yaw_fold_logger.write(
                        {
                            "record_type": "truck_yaw_fold_selection",
                            "env_id": env_id,
                            "attempt": coordinator.attempts[env_id],
                            "subtask": f"{eef_name}/{subtask_index}",
                            "subtask_index": subtask_index,
                            "object_ref": subtask_object_name,
                            "target_original_yaw_deg": math.degrees(target_original_yaw),
                            "target_folded_yaw_deg": math.degrees(target_folded_yaw),
                            "target_folded": bool(context["target_folded"]),
                            "selected_source_demo_index": int(selected_index),
                            "source_original_yaw_deg": math.degrees(source_original_yaw),
                            "source_folded_yaw_deg": math.degrees(source_folded_yaw),
                            "source_folded": bool(source_folded_flag),
                            "abs_delta_yaw_deg": _absolute_yaw_delta_degrees(
                                target_folded_yaw, source_folded_yaw
                            ),
                        }
                    )
            return selected_index

        def merge_eef_subtask_trajectory(
            self,
            env_id: int,
            eef_name: str,
            subtask_index: int,
            prev_executed_traj: list[Any] | None,
            subtask_trajectory: Any,
        ) -> list[Any]:
            transition_record: dict[str, Any] | None = None
            if enable_source_audit and eef_name == "agibot" and subtask_index == 1:
                if env_id not in coordinator.attempts:
                    raise ConfigurationError(f"transition audit has no settling attempt for env {env_id}")
                actual_pose = self.env.get_robot_eef_pose(env_ids=[env_id], eef_name=eef_name)[0]
                first_target_pose = subtask_trajectory[0].pose
                interpolate_from_last = bool(
                    self.env_cfg.datagen_config.generation_interpolate_from_last_target_pose
                )
                if interpolate_from_last:
                    if not prev_executed_traj:
                        raise ConfigurationError("transition audit requires the previous executed trajectory")
                    bridge_start_pose = prev_executed_traj[-1].pose
                else:
                    bridge_start_pose = actual_pose
                position_jump, rotation_jump = _pose_jump(
                    coordinator.torch, actual_pose, first_target_pose
                )
                bridge_position_jump, bridge_rotation_jump = _pose_jump(
                    coordinator.torch, bridge_start_pose, first_target_pose
                )
                transition_record = {
                    "record_type": "box_transition",
                    "env_id": env_id,
                    "attempt": coordinator.attempts[env_id],
                    "position_jump_m": position_jump,
                    "rotation_jump_deg": rotation_jump,
                    "bridge_position_jump_m": bridge_position_jump,
                    "bridge_rotation_jump_deg": bridge_rotation_jump,
                    "num_interpolation_steps": int(
                        self.env_cfg.subtask_configs[eef_name][subtask_index].num_interpolation_steps
                    ),
                    "interpolate_from_last_target_pose": interpolate_from_last,
                }
            result = super().merge_eef_subtask_trajectory(
                env_id=env_id,
                eef_name=eef_name,
                subtask_index=subtask_index,
                prev_executed_traj=prev_executed_traj,
                subtask_trajectory=subtask_trajectory,
            )
            if transition_record is not None:
                coordinator.logger.write(transition_record)
            return result

        async def generate(
            self,
            env_id: int,
            success_term: Any,
            env_reset_queue: Any = None,
            env_action_queue: Any = None,
            pause_subtask: bool = False,
            export_demo: bool = True,
            motion_planner: Any = None,
        ) -> dict:
            if env_action_queue is not None:
                try:
                    coordinator.attach(env_action_queue)
                except Exception as exc:
                    message = f"cannot install settling action-queue guard: {exc}"
                    coordinator.fail(message)
                    raise SettlingError(message) from exc
            if env_reset_queue is None or env_action_queue is None:
                message = "settled generation requires both async queues"
                coordinator.fail(message)
                raise SettlingError(message)
            proxy = ResetQueueProxy(env_reset_queue, env_action_queue, coordinator, self.env, env_id)
            try:
                result = await super().generate(
                    env_id=env_id,
                    success_term=success_term,
                    env_reset_queue=proxy,
                    env_action_queue=env_action_queue,
                    pause_subtask=pause_subtask,
                    export_demo=export_demo,
                    motion_planner=motion_planner,
                )
            except SettlingError as exc:
                coordinator.fail(str(exc))
                raise
            except Exception as exc:
                message = f"generation env {env_id} failed after settling: {exc}"
                coordinator.fail(message)
                raise SettlingError(message) from exc
            coordinator.logger.generation_results += 1
            coordinator.logger.write(
                {
                    "record_type": "generation_result",
                    "env_id": env_id,
                    "attempt": coordinator.attempts[env_id],
                    "success": bool(result.get("success", False)),
                }
            )
            return result

    return SettledDataGenerator


def _subtask_snapshot(subtask: Any) -> dict[str, Any]:
    """Return the small, JSON-safe selection surface changed by this wrapper."""
    kwargs = getattr(subtask, "selection_strategy_kwargs", None)
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise ConfigurationError("box selection_strategy_kwargs must be a dictionary")
    return {
        "object_ref": getattr(subtask, "object_ref", None),
        "selection_strategy": getattr(subtask, "selection_strategy", None),
        "selection_strategy_kwargs": copy.deepcopy(kwargs),
        "num_interpolation_steps": getattr(subtask, "num_interpolation_steps", None),
    }


def _subtask_audit_snapshot(subtask: Any) -> dict[str, Any]:
    """Capture the JSON-safe behavior fields needed to prove subtask 1 stayed unchanged."""
    snapshot = _subtask_snapshot(subtask)
    for name in (
        "subtask_term_signal",
        "subtask_term_offset_range",
        "action_noise",
        "num_fixed_steps",
        "apply_noise_during_interpolation",
    ):
        snapshot[name] = copy.deepcopy(getattr(subtask, name, None))
    return snapshot


def _qualified_name(value: type) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _normalize_reset_ids(torch_module: Any, env_ids: Any, num_envs: int, device: Any) -> Any:
    """Return unique, ordered reset IDs on the controller device."""
    if env_ids is None:
        values = list(range(num_envs))
    elif isinstance(env_ids, slice):
        values = list(range(num_envs))[env_ids]
    elif isinstance(env_ids, Integral) and not isinstance(env_ids, bool):
        values = [int(env_ids)]
    elif torch_module.is_tensor(env_ids):
        if env_ids.ndim != 1 or env_ids.dtype == torch_module.bool or torch_module.is_floating_point(env_ids):
            raise ConfigurationError("RMP reset IDs tensor must be one-dimensional and integral")
        values = env_ids.detach().cpu().tolist()
    elif isinstance(env_ids, Sequence) and not isinstance(env_ids, (str, bytes)):
        values = list(env_ids)
    else:
        raise ConfigurationError("unsupported RMP reset ID selection")
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ConfigurationError("RMP reset IDs must be integral")
        value = int(value)
        if value < 0 or value >= num_envs:
            raise ConfigurationError(f"RMP reset ID {value} is outside [0, {num_envs})")
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ConfigurationError("RMP reset ID selection must not be empty")
    return torch_module.tensor(normalized, dtype=torch_module.long, device=device)


def make_per_env_rmpflow_action(base_class: type, logger: JsonlLog, torch_module: Any) -> type:
    """Create an opt-in RMPFlow action that resets only the selected environments."""

    class PerEnvRMPFlowAction(base_class):
        def _validate_initialized_controller(self) -> None:
            controller = self._rmpflow_controller
            if controller.num_robots != self.num_envs:
                raise ConfigurationError("RMP controller robot count does not match environment count")
            prim_paths = getattr(controller, "_prim_paths", None)
            if not isinstance(prim_paths, (list, tuple)) or len(prim_paths) != self.num_envs:
                raise ConfigurationError("RMP controller prim paths do not match environment count")
            mapped_ids: list[int] = []
            for path in prim_paths:
                match = re.search(r"/env_(\d+)(?:/|$)", str(path))
                if match is None:
                    raise ConfigurationError(f"RMP prim path has no numeric environment index: {path}")
                mapped_ids.append(int(match.group(1)))
            if mapped_ids != list(range(self.num_envs)):
                raise ConfigurationError("RMP controller prim-path order does not match environment indices")
            policies = getattr(controller, "articulation_policies", None)
            if not isinstance(policies, list) or len(policies) != self.num_envs:
                raise ConfigurationError("RMP controller policy count does not match environment count")
            for name in ("_command", "dof_pos_target", "dof_vel_target"):
                value = getattr(controller, name, None)
                if not torch_module.is_tensor(value) or value.ndim < 1 or value.shape[0] != self.num_envs:
                    raise ConfigurationError(f"RMP controller {name} buffer has an unexpected shape")

        def reset(self, env_ids: Any = None) -> None:
            controller = self._rmpflow_controller
            device = getattr(controller, "_device", self.device)
            reset_ids = _normalize_reset_ids(torch_module, env_ids, self.num_envs, device)
            initialized = hasattr(controller, "articulation_policies")
            if not initialized:
                if reset_ids.detach().cpu().tolist() != list(range(self.num_envs)):
                    raise ConfigurationError("initial RMP controller reset must include every environment")
                super().reset(reset_ids)
                self._validate_initialized_controller()
                logger.write(
                    {
                        "record_type": "per_env_rmp_initialization",
                        "env_ids": reset_ids.detach().cpu().tolist(),
                        "controller_id": id(controller),
                        "policy_ids": [id(policy) for policy in controller.articulation_policies],
                        "command_tensor_id": id(controller._command),
                        "position_target_tensor_id": id(controller.dof_pos_target),
                        "velocity_target_tensor_id": id(controller.dof_vel_target),
                        "raw_actions_tensor_id": id(self._raw_actions),
                    }
                )
                logger.write(
                    {
                        "record_type": "per_env_rmp_reset",
                        "env_ids": reset_ids.detach().cpu().tolist(),
                        "initialized_controller": True,
                        "controller_id": id(controller),
                        "policy_ids": [id(policy) for policy in controller.articulation_policies],
                        "command_tensor_id": id(controller._command),
                        "position_target_tensor_id": id(controller.dof_pos_target),
                        "velocity_target_tensor_id": id(controller.dof_vel_target),
                        "raw_actions_tensor_id": id(self._raw_actions),
                        "non_target_rows_unchanged": {
                            "raw_actions": True,
                            "command": True,
                            "position_target": True,
                            "velocity_target": True,
                        },
                    }
                )
                return

            self._validate_initialized_controller()
            selected = set(reset_ids.detach().cpu().tolist())
            other_ids = torch_module.tensor(
                [index for index in range(self.num_envs) if index not in selected],
                dtype=torch_module.long,
                device=device,
            )
            buffers = {
                "raw_actions": self._raw_actions,
                "command": controller._command,
                "position_target": controller.dof_pos_target,
                "velocity_target": controller.dof_vel_target,
            }
            identities_before = {name: id(value) for name, value in buffers.items()}
            controller_id = id(controller)
            policy_ids = [id(policy) for policy in controller.articulation_policies]
            untouched_before = {
                name: value[other_ids].clone() for name, value in buffers.items()
            }
            self._raw_actions[reset_ids] = 0.0
            controller.reset_idx(reset_ids)
            self._validate_initialized_controller()
            buffers_after = {
                "raw_actions": self._raw_actions,
                "command": controller._command,
                "position_target": controller.dof_pos_target,
                "velocity_target": controller.dof_vel_target,
            }
            unchanged = {
                name: identities_before[name] == id(value)
                and torch_module.equal(untouched_before[name], value[other_ids])
                for name, value in buffers_after.items()
            }
            if (controller_id != id(controller)
                    or policy_ids != [id(policy) for policy in controller.articulation_policies]
                    or not all(unchanged.values())):
                raise ConfigurationError("per-environment RMP reset changed non-target controller state")
            logger.write(
                {
                    "record_type": "per_env_rmp_reset",
                    "env_ids": reset_ids.detach().cpu().tolist(),
                    "initialized_controller": False,
                    "controller_id": controller_id,
                    "policy_ids": policy_ids,
                    "command_tensor_id": id(controller._command),
                    "position_target_tensor_id": id(controller.dof_pos_target),
                    "velocity_target_tensor_id": id(controller.dof_vel_target),
                    "raw_actions_tensor_id": id(self._raw_actions),
                    "non_target_rows_unchanged": unchanged,
                }
            )

    PerEnvRMPFlowAction.__name__ = "PerEnvRMPFlowAction"
    PerEnvRMPFlowAction.__qualname__ = "PerEnvRMPFlowAction"
    return PerEnvRMPFlowAction


def apply_box_selection_override(env_cfg: Any, settings: argparse.Namespace, logger: JsonlLog) -> None:
    """Apply the approved box-only source-selection override and log it."""
    datagen_config = getattr(env_cfg, "datagen_config", None)
    subtask_configs = getattr(env_cfg, "subtask_configs", None)
    if datagen_config is None or not isinstance(subtask_configs, dict):
        raise ConfigurationError("unexpected Toy2Box config: missing datagen_config or subtask_configs")
    if set(subtask_configs) != {"agibot"}:
        raise ConfigurationError("unexpected Toy2Box config: exactly one agibot end-effector is required")
    subtasks = subtask_configs.get("agibot")
    if not isinstance(subtasks, (list, tuple)) or len(subtasks) != 2:
        raise ConfigurationError("unexpected Toy2Box config: agibot must contain exactly two subtasks")
    if getattr(subtasks[0], "object_ref", None) != "toy_truck":
        raise ConfigurationError("unexpected Toy2Box config: subtask 1 must reference toy_truck")
    if getattr(subtasks[1], "object_ref", None) != "box":
        raise ConfigurationError("unexpected Toy2Box config: subtask 2 must reference box")
    for name in ("seed", "generation_select_src_per_subtask", "generation_select_src_per_arm"):
        if not hasattr(datagen_config, name):
            raise ConfigurationError(f"unexpected Toy2Box config: missing datagen {name}")
    if datagen_config.generation_select_src_per_subtask is not True:
        raise ConfigurationError("Toy2Box runtime override requires per-subtask source selection")
    if settings.zero_action_noise and any(not hasattr(subtask, "action_noise") for subtask in subtasks):
        raise ConfigurationError("zero-action-noise override requires subtask action_noise")

    box_subtask = subtasks[1]
    seed_before = datagen_config.seed
    per_arm_before = datagen_config.generation_select_src_per_arm
    per_subtask_before = datagen_config.generation_select_src_per_subtask
    subtask1_before = _subtask_audit_snapshot(subtasks[0])
    subtask2_before = _subtask_audit_snapshot(subtasks[1])
    original = _subtask_snapshot(box_subtask)
    if settings.box_robot_distance:
        box_subtask.selection_strategy = "nearest_neighbor_robot_distance"
        box_subtask.selection_strategy_kwargs = {
            "nn_k": settings.box_nn_k,
            "pos_weight": 1.0,
            "rot_weight": settings.box_rotation_weight,
        }
        if settings.box_interpolation_steps is not None:
            box_subtask.num_interpolation_steps = settings.box_interpolation_steps
    if settings.enable_box_reselection:
        datagen_config.generation_select_src_per_arm = True
    if settings.zero_action_noise:
        for subtask in subtasks:
            subtask.action_noise = 0.0
    if settings.grasp_offset_range is not None:
        subtasks[0].subtask_term_offset_range = tuple(settings.grasp_offset_range)
    effective = _subtask_snapshot(box_subtask)
    subtask1_after = _subtask_audit_snapshot(subtasks[0])
    subtask2_after = _subtask_audit_snapshot(subtasks[1])
    seed_after = datagen_config.seed
    if seed_after != seed_before:
        raise ConfigurationError("runtime box override changed the datagen seed")
    logger.write(
        {
            "record_type": "runtime_config_override",
            "subtask": "agibot/1",
            "eef_names": ["agibot"],
            "override_enabled": settings.box_robot_distance,
            "reselection_enabled": settings.enable_box_reselection,
            "zero_action_noise_enabled": settings.zero_action_noise,
            "grasp_offset_override_enabled": settings.grasp_offset_range is not None,
            "requested_grasp_offset_range": settings.grasp_offset_range,
            "datagen_seed_original": seed_before,
            "datagen_seed_effective": seed_after,
            "generation_select_src_per_arm_original": per_arm_before,
            "generation_select_src_per_arm_effective": datagen_config.generation_select_src_per_arm,
            "generation_select_src_per_subtask_original": per_subtask_before,
            "generation_select_src_per_subtask_effective": (
                datagen_config.generation_select_src_per_subtask
            ),
            "before": original,
            "after": effective,
            "subtask1_before": subtask1_before,
            "subtask1_after": subtask1_after,
            "subtask2_before": subtask2_before,
            "subtask2_after": subtask2_after,
        }
    )


def _action_config_behavior(action_cfg: Any) -> dict[str, str]:
    try:
        values = vars(action_cfg)
    except TypeError as exc:
        raise ConfigurationError("arm action config must expose its behavior fields") from exc
    return {
        name: repr(value)
        for name, value in sorted(values.items())
        if name != "class_type"
    }


def make_setup_env_config_wrapper(
    original_setup: Any,
    settings: argparse.Namespace,
    logger: JsonlLog,
    per_env_rmp_class: type | None = None,
) -> Any:
    """Wrap stock setup while preserving its arguments and success term."""

    def wrapped_setup(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        env_cfg, success_term = original_setup(*args, **kwargs)
        if settings.chunk_seed is not None:
            datagen_config = getattr(env_cfg, "datagen_config", None)
            if not hasattr(env_cfg, "seed") or datagen_config is None or not hasattr(datagen_config, "seed"):
                raise ConfigurationError("chunk seed override requires env and datagen seed fields")
            env_seed_before = env_cfg.seed
            datagen_seed_before = datagen_config.seed
            env_cfg.seed = settings.chunk_seed
            datagen_config.seed = settings.chunk_seed
            logger.write(
                {
                    "record_type": "chunk_seed_override",
                    "env_seed_original": env_seed_before,
                    "env_seed_effective": env_cfg.seed,
                    "datagen_seed_original": datagen_seed_before,
                    "datagen_seed_effective": datagen_config.seed,
                }
            )
        apply_box_selection_override(env_cfg, settings, logger)
        if settings.per_env_rmp_reset:
            actions = getattr(env_cfg, "actions", None)
            arm_action = getattr(actions, "arm_action", None)
            base_class = getattr(arm_action, "class_type", None)
            if per_env_rmp_class is None or not isinstance(base_class, type):
                raise ConfigurationError("per-environment RMP reset requires actions.arm_action.class_type")
            if not issubclass(per_env_rmp_class, base_class):
                raise ConfigurationError("per-environment RMP reset class does not wrap the configured action")
            behavior_before = _action_config_behavior(arm_action)
            arm_action.class_type = per_env_rmp_class
            behavior_after = _action_config_behavior(arm_action)
            if behavior_after != behavior_before:
                raise ConfigurationError("per-environment RMP reset changed arm action behavior config")
            logger.write(
                {
                    "record_type": "per_env_rmp_install",
                    "action_term": "arm_action",
                    "base_class": _qualified_name(base_class),
                    "effective_class": _qualified_name(per_env_rmp_class),
                    "config_behavior_unchanged": True,
                }
            )
        return env_cfg, success_term

    return wrapped_setup


def execute(
    original_args: list[str],
    paths: dict[str, Path],
    settings: argparse.Namespace,
    run_path: Any = runpy.run_path,
) -> None:
    """Load the stock script, inject settling, call its main, and close once."""
    original_argv = sys.argv
    namespace: dict[str, Any] | None = None
    simulation_app: Any = None
    coordinator: SettlingCoordinator | None = None
    status = "error"
    error: BaseException | None = None

    with ExitStack() as stack:
        stream = stack.enter_context(paths["settling_log"].open("x", encoding="utf-8"))
        fold_logger = None
        if getattr(settings, "truck_yaw_fold", False):
            fold_stream = stack.enter_context(
                paths["truck_yaw_fold_log"].open("x", encoding="utf-8")
            )
            fold_logger = JsonlLog(fold_stream)
        logger = JsonlLog(stream)
        header = {
                "record_type": "settling_header",
                "input_file": str(paths["input"]),
                "output_file": str(paths["output"]),
                "failed_output_file": str(paths["failed_output"]),
                "linear_threshold": settings.settle_linear_velocity,
                "angular_threshold": settings.settle_angular_velocity,
                "stable_steps": settings.settle_stable_steps,
                "max_steps": settings.settle_max_steps,
                "hold_recorded": False,
                "source_selection_audit_enabled": settings.enable_box_reselection,
                "zero_action_noise_enabled": settings.zero_action_noise,
                "chunk_seed": settings.chunk_seed,
                "settle_arm_motion": settings.settle_arm_motion,
                "settle_arm_joint_velocity": settings.settle_arm_joint_velocity,
                "settle_eef_velocity": settings.settle_eef_velocity,
                "grasp_offset_range": settings.grasp_offset_range,
                "grasp_offset_override_enabled": settings.grasp_offset_range is not None,
                "per_env_rmp_reset_enabled": settings.per_env_rmp_reset,
                "settle_reset_retries": settings.settle_reset_retries,
                "truck_pick_z_offset_m": settings.truck_pick_z_offset_m,
                "settle_retry_ratio_limit": MAX_RESET_RETRY_RATIO,
                "output_label": paths["output"].name,
                "box_transition_metrics": {
                    "position_jump": "current EEF to first transformed box target",
                    "bridge_jump": "actual interpolation start to first transformed box target",
                    "rotation": "SO(3) geodesic angle in degrees",
                },
            }
        if getattr(settings, "truck_yaw_fold", False):
            header.update(
                {
                    "truck_yaw_fold_enabled": True,
                    "truck_yaw_fold_log": str(paths["truck_yaw_fold_log"]),
                }
            )
        logger.write(header)
        try:
            sys.argv = [str(GENERATOR_SCRIPT), *original_args]
            namespace = run_path(str(GENERATOR_SCRIPT), run_name="__toy2box_settle_generate__")
            simulation_app = namespace["simulation_app"]

            import torch
            import isaaclab_mimic.datagen.generation as generation

            per_env_rmp_class = None
            if settings.per_env_rmp_reset:
                from isaaclab.envs.mdp.actions.rmpflow_task_space_actions import RMPFlowAction

                per_env_rmp_class = make_per_env_rmpflow_action(RMPFlowAction, logger, torch)

            coordinator = SettlingCoordinator(
                logger=logger,
                torch_module=torch,
                linear_threshold=settings.settle_linear_velocity,
                angular_threshold=settings.settle_angular_velocity,
                stable_steps=settings.settle_stable_steps,
                max_steps=settings.settle_max_steps,
                settle_arm_motion=settings.settle_arm_motion,
                arm_joint_velocity_threshold=settings.settle_arm_joint_velocity,
                eef_velocity_threshold=settings.settle_eef_velocity,
                settle_reset_retries=settings.settle_reset_retries,
                chunk_seed=settings.chunk_seed,
                output_label=paths["output"].name,
            )
            original_class = generation.DataGenerator
            generation.DataGenerator = make_settled_generator(
                original_class,
                coordinator,
                enable_source_audit=settings.enable_box_reselection,
                enable_truck_yaw_fold=getattr(settings, "truck_yaw_fold", False),
                truck_yaw_fold_logger=fold_logger,
                truck_pick_z_offset_m=settings.truck_pick_z_offset_m,
            )
            main_globals: dict[str, Any] | None = None
            original_setup_env_config: Any = None
            setup_patched = False
            try:
                main_globals = namespace["main"].__globals__
                if "setup_env_config" not in main_globals:
                    raise ConfigurationError("stock generator main has no setup_env_config global")
                original_setup_env_config = main_globals["setup_env_config"]
                if not callable(original_setup_env_config):
                    raise ConfigurationError("stock generator setup_env_config is not callable")
                main_globals["setup_env_config"] = make_setup_env_config_wrapper(
                    original_setup_env_config, settings, logger, per_env_rmp_class
                )
                setup_patched = True
                namespace["main"]()
                status = "complete"
            finally:
                if setup_patched:
                    main_globals["setup_env_config"] = original_setup_env_config
                generation.DataGenerator = original_class
        except BaseException as exc:
            error = exc
        finally:
            if coordinator is not None:
                coordinator.restore()
            summary = {
                "record_type": "settling_summary",
                "status": status,
                "settling_attempts": logger.settling_attempts,
                "capped_attempts": logger.capped_attempts,
                "generation_results": logger.generation_results,
                "logical_initial_settling_starts": logger.logical_initial_settling_starts,
                "settling_retry_operations": logger.settling_retry_operations,
                "discarded_layouts": logger.discarded_layouts,
                "settling_retry_ratio": (
                    logger.settling_retry_operations / logger.logical_initial_settling_starts
                    if logger.logical_initial_settling_starts else 0.0
                ),
                "settle_reset_retries": settings.settle_reset_retries,
            }
            try:
                import isaaclab_mimic.datagen.generation as generation

                summary.update(
                    {
                        "original_num_success": int(generation.num_success),
                        "original_num_failures": int(generation.num_failures),
                        "original_num_attempts": int(generation.num_attempts),
                    }
                )
            except ImportError:
                pass
            if error is not None:
                summary["error"] = f"{type(error).__name__}: {error}"
            logger.write(summary)
            try:
                if simulation_app is not None:
                    simulation_app.close()
            finally:
                sys.argv = original_argv
    if error is not None:
        raise error


def parse_args(argv: Iterable[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--settling_log", required=True)
    parser.add_argument("--settle_linear_velocity", type=float, default=0.001)
    parser.add_argument("--settle_angular_velocity", type=float, default=0.02)
    parser.add_argument("--settle_stable_steps", type=int, default=2)
    parser.add_argument("--settle_max_steps", type=int, default=10)
    parser.add_argument("--settle_arm_motion", action="store_true")
    parser.add_argument("--settle_arm_joint_velocity", type=float, default=0.05)
    parser.add_argument("--settle_eef_velocity", type=float, default=0.01)
    parser.add_argument("--box_robot_distance", action="store_true")
    parser.add_argument("--enable_box_reselection", action="store_true")
    parser.add_argument("--zero_action_noise", action="store_true")
    parser.add_argument("--chunk_seed", type=int, default=None)
    parser.add_argument("--box_rotation_weight", type=float, default=2.0)
    parser.add_argument("--box_nn_k", type=int, default=1)
    parser.add_argument("--box_interpolation_steps", type=int, default=None)
    parser.add_argument("--grasp_offset_range", type=int, nargs=2, default=None, metavar=("START", "END"))
    parser.add_argument("--per_env_rmp_reset", action="store_true")
    parser.add_argument("--settle_reset_retries", type=int, default=0)
    parser.add_argument("--truck_yaw_fold", action="store_true")
    parser.add_argument("--truck_yaw_fold_log", default=None)
    parser.add_argument("--truck_pick_z_offset_m", type=float, default=0.0)
    settings, original_args = parser.parse_known_args(arguments)
    for name in ("settle_linear_velocity", "settle_angular_velocity"):
        value = getattr(settings, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name} must be finite and positive")
    for name in ("settle_stable_steps", "settle_max_steps"):
        if getattr(settings, name) <= 0:
            parser.error(f"--{name} must be positive")
    if settings.settle_stable_steps > settings.settle_max_steps:
        parser.error("--settle_stable_steps must not exceed --settle_max_steps")
    for name in ("settle_arm_joint_velocity", "settle_eef_velocity"):
        value = getattr(settings, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name} must be finite and positive")
    arm_threshold_options = ("--settle_arm_joint_velocity", "--settle_eef_velocity")
    arm_threshold_requested = any(
        argument == option or argument.startswith(f"{option}=")
        for argument in arguments
        for option in arm_threshold_options
    )
    if arm_threshold_requested and not settings.settle_arm_motion:
        parser.error("arm settling thresholds require --settle_arm_motion")
    if not math.isfinite(settings.box_rotation_weight) or settings.box_rotation_weight <= 0:
        parser.error("--box_rotation_weight must be finite and positive")
    if settings.box_nn_k <= 0:
        parser.error("--box_nn_k must be positive")
    if settings.box_interpolation_steps is not None and settings.box_interpolation_steps <= 0:
        parser.error("--box_interpolation_steps must be positive")
    tuning_options = ("--box_rotation_weight", "--box_nn_k", "--box_interpolation_steps")
    tuning_requested = any(
        argument == option or argument.startswith(f"{option}=")
        for argument in arguments
        for option in tuning_options
    )
    if tuning_requested and not settings.box_robot_distance:
        parser.error("box selection tuning options require --box_robot_distance")
    if settings.enable_box_reselection and not settings.box_robot_distance:
        parser.error("--enable_box_reselection requires --box_robot_distance")
    if settings.zero_action_noise and not (
        settings.box_robot_distance and settings.enable_box_reselection
    ):
        parser.error(
            "--zero_action_noise requires --box_robot_distance and --enable_box_reselection"
        )
    if settings.chunk_seed is not None:
        if settings.chunk_seed <= 0:
            parser.error("--chunk_seed must be positive")
        if not (
            settings.box_robot_distance
            and settings.enable_box_reselection
            and settings.zero_action_noise
        ):
            parser.error(
                "--chunk_seed requires --box_robot_distance, --enable_box_reselection, "
                "and --zero_action_noise"
            )
    if settings.grasp_offset_range is not None:
        start, end = settings.grasp_offset_range
        if start < 0 or start > end:
            parser.error("--grasp_offset_range requires 0 <= START <= END")
        if not (
            settings.box_robot_distance
            and settings.enable_box_reselection
            and settings.zero_action_noise
        ):
            parser.error(
                "--grasp_offset_range requires --box_robot_distance, --enable_box_reselection, "
                "and --zero_action_noise"
            )
    if settings.per_env_rmp_reset:
        d2_regime = (
            settings.box_robot_distance
            and settings.enable_box_reselection
            and settings.zero_action_noise
            and settings.settle_arm_motion
            and settings.grasp_offset_range == [50, 60]
            and settings.box_interpolation_steps == 30
            and settings.settle_arm_joint_velocity == 0.15
            and settings.settle_eef_velocity == 0.01
            and settings.settle_max_steps == 20
            and settings.settle_linear_velocity == 0.001
            and settings.settle_angular_velocity == 0.02
            and settings.settle_stable_steps == 2
            and settings.box_rotation_weight == 2.0
            and settings.box_nn_k == 1
        )
        if not d2_regime:
            parser.error("--per_env_rmp_reset requires the approved D2 generation regime")
    if settings.settle_reset_retries not in (0, 2):
        parser.error("--settle_reset_retries must be either 0 or the approved value 2")
    if settings.settle_reset_retries and not settings.per_env_rmp_reset:
        parser.error("--settle_reset_retries requires the approved D3 per-environment RMP regime")
    if settings.settle_reset_retries and settings.chunk_seed is None:
        parser.error("--settle_reset_retries requires an explicit positive --chunk_seed")
    if settings.truck_yaw_fold and settings.truck_yaw_fold_log is None:
        parser.error("--truck_yaw_fold requires --truck_yaw_fold_log")
    if settings.truck_yaw_fold_log is not None and not settings.truck_yaw_fold:
        parser.error("--truck_yaw_fold_log requires --truck_yaw_fold")
    if settings.truck_pick_z_offset_m not in (0.0, -0.005):
        parser.error("--truck_pick_z_offset_m must be 0 or the registered H4 value -0.005")
    if settings.truck_pick_z_offset_m != 0.0:
        if not (
            settings.per_env_rmp_reset
            and settings.settle_reset_retries == 2
            and settings.chunk_seed is not None
            and not settings.truck_yaw_fold
        ):
            parser.error(
                "nonzero --truck_pick_z_offset_m requires the stock nonfold D3 regime, "
                "two settling retries, and an explicit chunk seed"
            )
    return settings, original_args


def main(argv: Iterable[str] | None = None) -> None:
    settings, original_args = parse_args(argv)
    paths = validate_invocation(
        original_args,
        settings.settling_log,
        settings.truck_yaw_fold_log,
    )
    execute(original_args, paths, settings)


if __name__ == "__main__":
    main()
