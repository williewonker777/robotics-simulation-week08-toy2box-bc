#!/usr/bin/env python3
"""Collect DAgger-style Toy2Box recovery demonstrations.

The learned policy is rolled out exactly like stock robomimic ``play.py`` until
the preregistered detector fires.  At that post-step state the recorder buffer
is replaced, making that state the correction episode's ``initial_state``.
Only the smooth scripted expert's subsequent actions and observations are
exported.  Existing outputs are never overwritten.

Real-simulator schema, success and quality compatibility are deliberately
validated again by the pipeline's S0 smoke; this module itself is importable for
simulation-free focused tests.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from collections import Counter
from pathlib import Path
import random
import sys
import traceback
from typing import Any

import numpy as np

from record_toy2box_scripted_demos import (
    MIN_QUALITY_SAMPLES,
    SMOOTH_QUALITY_LIMITS,
    SmoothPoseServo,
    Toy2BoxPolicy,
    evaluate_smooth_quality,
    read_state,
    rotation_error,
)
from toy2box_correction_logic import (
    FailureDetector,
    RecoveryObservation,
    skip_reason,
)


TASK = "Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0"
POLICY_HORIZON = 800
EXPERT_HORIZON = 600
REST_HEIGHT_SAMPLES = 30

REQUIRED_EPISODE_DATASETS = {
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
}
REQUIRED_STATE_DATASETS = (
    "articulation/robot/joint_position",
    "articulation/robot/joint_velocity",
    "articulation/robot/root_pose",
    "articulation/robot/root_velocity",
    "rigid_object/box/root_pose",
    "rigid_object/box/root_velocity",
    "rigid_object/toy_truck/root_pose",
    "rigid_object/toy_truck/root_velocity",
)


def fresh_output(path: os.PathLike[str] | str, suffix: str) -> Path:
    """Validate a create-only output without changing the filesystem."""
    output = Path(path).expanduser().resolve(strict=False)
    if output.suffix != suffix:
        raise ValueError(f"output must end in {suffix}: {output}")
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if not output.parent.exists():
        raise FileNotFoundError(f"output parent does not exist: {output.parent}")
    if not output.parent.is_dir():
        raise NotADirectoryError(f"output parent is not a directory: {output.parent}")
    return output


def validate_collection_counts(num_corrections: int, max_rollouts: int, smoke_three_rollouts: bool) -> None:
    """Validate the normal target contract and the exact three-rollout S0 mode."""
    if num_corrections < 1 or max_rollouts < 1:
        raise ValueError("--num_corrections and --max_rollouts must be positive")
    if num_corrections > max_rollouts:
        raise ValueError("--num_corrections cannot exceed --max_rollouts")
    if smoke_three_rollouts and max_rollouts != 3:
        raise ValueError("--smoke_three_rollouts requires --max_rollouts 3")


def reserve_outputs(dataset_file: Path, log_file: Path) -> None:
    """Atomically reserve distinct fresh outputs before launching Isaac Sim."""
    if dataset_file == log_file:
        raise ValueError("--dataset_file and --log_file must differ")
    created: list[Path] = []
    try:
        for path in (dataset_file, log_file):
            with path.open("x", encoding=None if path.suffix == ".hdf5" else "utf-8"):
                pass
            created.append(path)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_exception_summary(log_file: Path, error: BaseException, phase: str) -> None:
    """Persist a terminal exception record without masking the original error."""
    append_jsonl(
        log_file,
        {
            "event": "summary",
            "record_type": "summary",
            "status": "exception",
            "phase": phase,
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )


def close_app_with_failure(app: Any, exit_code: int = 1) -> None:
    """Clean up Kit while preserving a non-zero process outcome.

    SimulationApp.close tears down the Python runtime deeply enough that an
    exception raised afterwards can be lost.  Set Kit's own exit code first,
    then use ``os._exit`` as a fallback if graceful close returns.
    """
    if exit_code <= 0:
        raise ValueError("failure exit code must be positive")
    kit_app = getattr(app, "_app", None)
    post_quit = getattr(kit_app, "post_quit", None)
    try:
        if callable(post_quit):
            try:
                post_quit(exit_code)
            except BaseException:
                print("[ERROR] Failed to set Kit failure exit code; continuing cleanup:", flush=True)
                traceback.print_exc()
        app.close()
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(exit_code)


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def quaternion_rpy_degrees(quaternion_wxyz: Any) -> tuple[float, float, float]:
    """Return roll, pitch, yaw in degrees from a wxyz quaternion."""
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=float).reshape(4)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    pitch = math.asin(pitch_arg)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


def correction_skip_reason(
    state: dict[str, Any], truck_position_env: Any, settled_z: float
) -> str | None:
    """Apply the frozen preregistration-3 supplement after the safe rise."""
    return skip_reason(
        state["truck_quat"],
        truck_position_env,
        truck_height=float(state["truck"][2]),
        settled_truck_height=float(settled_z),
        truck_box_xy_distance=float(np.linalg.norm(state["truck"][:2] - state["box"][:2])),
        grasped=bool(state["grasp"]),
    )


def recovery_state_summary(state: dict[str, Any], truck_position_env: Any) -> dict[str, Any]:
    """Return compact state fields needed to audit detection and skip timing."""
    roll, pitch, yaw = quaternion_rpy_degrees(state["truck_quat"])
    return {
        "eef_position": np.asarray(state["eef"], dtype=float).tolist(),
        "truck_position": np.asarray(state["truck"], dtype=float).tolist(),
        "truck_position_env": np.asarray(truck_position_env, dtype=float).tolist(),
        "truck_rpy_degrees": [roll, pitch, yaw],
        "box_position": np.asarray(state["box"], dtype=float).tolist(),
        "grasped": bool(state["grasp"]),
    }


def object_position_in_env(env: Any, object_name: str) -> np.ndarray:
    """Read an object's xyz relative to env origin, not the robot-base observation frame."""
    world = _array(env.scene[object_name].data.root_pos_w[0]).reshape(-1)[:3]
    origin = _array(env.scene.env_origins[0]).reshape(-1)[:3]
    position = np.asarray(world - origin, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError(f"invalid env-origin position for {object_name}")
    return position


def reset_recorder_at_takeover(recorder_manager: Any) -> None:
    """Discard the policy prefix and capture the exact live takeover state."""
    recorder_manager.reset([0])
    recorder_manager.record_post_reset([0])


class RecoveryExpert:
    """Safe-open/rise preamble followed by the existing smooth top-down expert."""

    def __init__(self, state: dict[str, Any]):
        self.controller = Toy2BoxPolicy(top_down=True, control_mode="smooth")
        self.controller.phase = "recovery_open_and_rise"
        self.controller.phase_steps = 0
        self.controller.phase_history = ["recovery_open_and_rise"]
        self.controller.servo = SmoothPoseServo()
        self.hold_quat = np.asarray(state["quat"], dtype=float).copy()
        self.safe_target = np.asarray(state["eef"], dtype=float).copy()
        self.safe_target[2] = max(
            self.safe_target[2] + 0.18,
            float(state["truck"][2]) + 0.22,
            float(state["box"][2]) + 0.25,
        )
        self.stable_steps = 0
        self.safe_rise_ready = False

    @property
    def failure(self) -> str | None:
        return self.controller.failure

    @property
    def phase_history(self) -> list[str]:
        return self.controller.phase_history

    def advance(self, state: dict[str, Any]) -> np.ndarray:
        if self.controller.phase != "recovery_open_and_rise":
            return self.controller.advance(state)

        self.controller.phase_steps += 1
        action = np.zeros(7, dtype=np.float32)
        action[-1] = 1.0
        action[:6] = self.controller.servo.advance(
            self.safe_target - state["eef"], rotation_error(self.hold_quat, state["quat"])
        )
        if abs(float(state["eef"][2]) - float(self.safe_target[2])) < 0.015:
            self.stable_steps += 1
        else:
            self.stable_steps = 0
        if self.stable_steps >= 8:
            self.safe_rise_ready = True
        elif self.controller.phase_steps >= 100:
            self.controller.failure = "recovery_safe_rise_timeout"
        return action

    def finish_safe_rise(self, state: dict[str, Any]) -> None:
        """Enter scripted recovery only after the caller evaluates skip rules."""
        if not self.safe_rise_ready or self.controller.phase != "recovery_open_and_rise":
            raise RuntimeError("safe rise is not ready to finish")
        self.controller.initialize_from_settled_state(state)
        self.safe_rise_ready = False


def evaluate_skip_after_safe_rise(
    expert: RecoveryExpert,
    state: dict[str, Any],
    truck_position_env: Any,
    settled_z: float,
) -> tuple[bool, str | None]:
    """Evaluate the frozen skip contract only at the safe-rise handoff."""
    if not expert.safe_rise_ready:
        return False, None
    return True, correction_skip_reason(state, truck_position_env, settled_z)


def policy_observation(obs_dict: dict[str, Any], env: Any) -> dict[str, Any]:
    """Prepare observations byte-for-byte like stock robomimic play.py."""
    obs = copy.deepcopy(obs_dict["policy"])
    for name in obs:
        obs[name] = obs[name].squeeze()
    if hasattr(env.cfg, "image_obs_list"):
        for image_name in env.cfg.image_obs_list:
            if image_name in obs_dict["policy"]:
                image = obs_dict["policy"][image_name].squeeze()
                image = image.permute(2, 0, 1).clone().float()
                obs[image_name] = (image / 255.0).clip(0.0, 1.0)
    return obs


def detector_observation(step: int, state: dict[str, Any], action: Any, settled_z: float) -> RecoveryObservation:
    action_array = _array(action).reshape(-1)
    if action_array.shape != (7,):
        raise ValueError(f"policy action must have seven components, got {action_array.shape}")
    return RecoveryObservation(
        step=step,
        truck_pos_xyz=np.asarray(state["truck"], dtype=float),
        eef_pos_xyz=np.asarray(state["eef"], dtype=float),
        box_pos_xyz=np.asarray(state["box"], dtype=float),
        action_gripper_closed=bool(action_array[-1] < 0.0),
        settled_truck_z=float(settled_z),
    )


def settled_height_from_prefix(states: list[dict[str, Any]]) -> float:
    """Match the preregistered offline metric's median of the first 30 post-step z values."""
    if len(states) != REST_HEIGHT_SAMPLES:
        raise ValueError(f"settled-height prefix must contain exactly {REST_HEIGHT_SAMPLES} states")
    heights = np.asarray([state["truck"][2] for state in states], dtype=float)
    if heights.shape != (REST_HEIGHT_SAMPLES,) or not np.isfinite(heights).all():
        raise ValueError("settled-height prefix contains invalid truck z values")
    return float(np.median(heights))


def is_near_close_onset(previous_closed: bool, action: Any, state: dict[str, Any]) -> bool:
    """Detect an early close that makes a causal settled-height estimate impossible."""
    action_array = _array(action).reshape(-1)
    if action_array.shape != (7,):
        raise ValueError(f"policy action must have seven components, got {action_array.shape}")
    closed = bool(action_array[-1] < 0.0)
    distance = float(np.linalg.norm(np.asarray(state["eef"]) - np.asarray(state["truck"])))
    return closed and not previous_closed and distance <= 0.06


def validate_dataset_schema(path: os.PathLike[str] | str, expected_episodes: int | None = None) -> dict[str, Any]:
    """Validate the correction output against the E0 recorder/BC schema."""
    import h5py

    with h5py.File(path, "r") as dataset:
        if "data" not in dataset:
            raise ValueError("dataset has no /data group")
        data = dataset["data"]
        if "env_args" not in data.attrs:
            raise ValueError("dataset /data has no env_args attribute")
        episodes = sorted(data.keys())
        if expected_episodes is not None and len(episodes) != expected_episodes:
            raise ValueError(f"expected {expected_episodes} episodes, found {len(episodes)}")
        total = 0
        for name in episodes:
            episode = data[name]
            if not bool(episode.attrs.get("success", False)):
                raise ValueError(f"{name} is not marked successful")
            count = int(episode.attrs.get("num_samples", -1))
            if count < 1:
                raise ValueError(f"{name} has invalid num_samples={count}")
            for key, width in REQUIRED_EPISODE_DATASETS.items():
                if key not in episode:
                    raise ValueError(f"{name} is missing {key}")
                value = episode[key]
                if value.shape != (count, width):
                    raise ValueError(f"{name}/{key} has shape {value.shape}, expected {(count, width)}")
            for prefix in ("initial_state", "states"):
                for key in REQUIRED_STATE_DATASETS:
                    full_key = f"{prefix}/{key}"
                    if full_key not in episode:
                        raise ValueError(f"{name} is missing {full_key}")
                    value = episode[full_key]
                    expected_leading = 1 if prefix == "initial_state" else count
                    if value.shape[0] != expected_leading:
                        raise ValueError(
                            f"{name}/{full_key} has leading dimension {value.shape[0]}, expected {expected_leading}"
                        )
            total += count
        if "total" in data.attrs and int(data.attrs["total"]) != total:
            raise ValueError(f"data total {data.attrs['total']} does not equal episode total {total}")
    return {"episodes": len(episodes), "total_samples": total, "compatible": True}


def _layout(state: dict[str, Any]) -> dict[str, float]:
    _, _, yaw = quaternion_rpy_degrees(state["truck_quat"])
    return {
        "initial_truck_yaw_degrees": yaw,
        "initial_truck_x": float(state["truck"][0]),
        "initial_truck_y": float(state["truck"][1]),
    }


def _detection_kind(detection: Any) -> str:
    return str(getattr(detection, "kind", detection))


def collect(args: argparse.Namespace, simulation_app: Any) -> None:
    """Run stock policy rollouts and export successful recovery suffixes."""
    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import DatasetExportMode
    from isaaclab_tasks.utils import parse_env_cfg
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.torch_utils as TorchUtils
    from toy2box_quality import DEFAULT_LIMITS, quality_rejections, trajectory_quality

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=not args.disable_fabric)
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None
    env_cfg.env_name = args.task.split(":")[-1]
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = str(args.dataset_file.parent)
    env_cfg.recorders.dataset_filename = args.dataset_file.stem
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env = gym.make(args.task, cfg=env_cfg).unwrapped

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    env.seed(args.seed)
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    arm_joint_ids, arm_joint_names = env.scene["robot"].find_joints("right_arm_joint[1-7]")
    if len(arm_joint_ids) != 7:
        raise ValueError(f"expected seven right-arm joints, found {arm_joint_names}")
    quality_limits = dict(DEFAULT_LIMITS, **SMOOTH_QUALITY_LIMITS)
    detection_attempts: Counter[str] = Counter()
    detection_successes: Counter[str] = Counter()
    corrections = 0
    rollouts_completed = 0
    result_counts: Counter[str] = Counter()
    status = "max_rollouts"

    try:
        with torch.inference_mode():
            for rollout_index in range(args.max_rollouts):
                if not simulation_app.is_running():
                    status = "simulation_stopped"
                    break
                print(f"[INFO] Starting rollout {rollout_index}", flush=True)
                policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=str(args.checkpoint), device=device)
                policy.start_episode()
                obs_dict, _ = env.reset()
                state = read_state(env, obs_dict, arm_joint_ids)
                record: dict[str, Any] = {
                    "event": "rollout",
                    "record_type": "rollout",
                    "rollout": rollout_index,
                    "checkpoint": str(args.checkpoint),
                    "reset_seed": args.seed,
                    **_layout(state),
                    "stock_policy_success": False,
                    "detection_kind": None,
                    "detection_step": None,
                    "detection_details": None,
                    "detection_state_summary": None,
                    "skipped_reason": None,
                    "skip_evaluated_at": None,
                    "skip_evaluation_state_summary": None,
                    "expert_result": None,
                    "expert_steps": 0,
                    "quality": None,
                    "quality_rejections": [],
                    "exported": False,
                }
                detector = FailureDetector()
                detection = None
                takeover_obs = None
                takeover_state = None
                settled_z = None
                rest_prefix: list[tuple[int, dict[str, Any], np.ndarray]] = []
                previous_closed = False

                for step in range(1, POLICY_HORIZON + 1):
                    obs = policy_observation(obs_dict, env)
                    action_np = policy(obs)
                    action = torch.from_numpy(action_np).to(device=device).view(1, env.action_space.shape[1])
                    obs_dict, _, terminated, truncated, _ = env.step(action)
                    if bool(success_term.func(env, **success_term.params)[0]):
                        record["stock_policy_success"] = True
                        record["expert_result"] = "policy_success"
                        break
                    if bool(terminated[0]) or bool(truncated[0]):
                        record["expert_result"] = "policy_environment_termination"
                        break
                    state = read_state(env, obs_dict, arm_joint_ids)
                    action_array = _array(action).reshape(-1).copy()
                    if settled_z is None:
                        rest_prefix.append((step, state, action_array))
                        if len(rest_prefix) < REST_HEIGHT_SAMPLES:
                            if is_near_close_onset(previous_closed, action_array, state):
                                raise RuntimeError(
                                    "near-close onset occurred before the preregistered 30-step settled-height "
                                    "reference was available; stopping rather than applying a non-causal detector"
                                )
                            previous_closed = bool(action_array[-1] < 0.0)
                            continue
                        settled_z = settled_height_from_prefix([item[1] for item in rest_prefix])
                        for buffered_step, buffered_state, buffered_action in rest_prefix:
                            detection = detector.update(
                                detector_observation(buffered_step, buffered_state, buffered_action, settled_z)
                            )
                            if detection is not None:
                                break
                    else:
                        detection = detector.update(detector_observation(step, state, action_array, settled_z))
                    if detection is not None:
                        takeover_obs = obs_dict
                        takeover_state = state
                        break

                if detection is None:
                    if record["expert_result"] is None:
                        record["expert_result"] = "policy_horizon_no_detection"
                    env.recorder_manager.reset([0])
                    append_jsonl(args.log_file, record)
                    rollouts_completed += 1
                    result_counts[str(record["expert_result"])] += 1
                    continue

                kind = _detection_kind(detection)
                detection_attempts[kind] += 1
                record["detection_kind"] = kind
                record["detection_step"] = int(getattr(detection, "detected_step", step))
                details = getattr(detection, "details", None)
                record["detection_details"] = details if details is None or isinstance(details, dict) else vars(details)
                assert takeover_obs is not None and takeover_state is not None
                takeover_truck_env = object_position_in_env(env, "toy_truck")
                record["takeover_truck_position_env"] = takeover_truck_env.tolist()
                record["takeover_grasped"] = bool(takeover_state["grasp"])
                record["detection_state_summary"] = recovery_state_summary(
                    takeover_state, takeover_truck_env
                )
                record["settled_truck_z"] = settled_z

                reset_recorder_at_takeover(env.recorder_manager)
                expert = RecoveryExpert(takeover_state)
                obs_dict = takeover_obs
                state = takeover_state
                samples = {key: [] for key in ("actions", "eef", "quat", "arm_joints")}
                expert_result = "expert_step_limit"
                quality = None
                rejected_quality: list[str] = []
                for expert_step in range(1, EXPERT_HORIZON + 1):
                    action_np = expert.advance(state)
                    if expert.safe_rise_ready:
                        safe_rise_truck_env = object_position_in_env(env, "toy_truck")
                        evaluated, safe_rise_skip_reason = evaluate_skip_after_safe_rise(
                            expert, state, safe_rise_truck_env, settled_z
                        )
                        assert evaluated
                        record["skip_evaluated_at"] = "after_safe_rise"
                        record["skip_evaluation_state_summary"] = recovery_state_summary(
                            state, safe_rise_truck_env
                        )
                        if safe_rise_skip_reason is not None:
                            record["skipped_reason"] = safe_rise_skip_reason
                            expert_result = "skipped"
                            break
                        expert.finish_safe_rise(state)
                    samples["actions"].append(action_np.copy())
                    for key in ("eef", "quat", "arm_joints"):
                        samples[key].append(np.asarray(state[key]).copy())
                    action = torch.tensor(action_np, dtype=torch.float32, device=env.device).unsqueeze(0)
                    obs_dict, _, terminated, truncated, _ = env.step(action)
                    task_success = bool(success_term.func(env, **success_term.params)[0])
                    if task_success:
                        if len(samples["actions"]) < MIN_QUALITY_SAMPLES:
                            expert_result = "quality_insufficient_samples"
                        else:
                            quality, rejected_quality = evaluate_smooth_quality(
                                samples, trajectory_quality, quality_rejections, quality_limits
                            )
                            if rejected_quality:
                                expert_result = "quality_rejected"
                            else:
                                env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                                env.recorder_manager.set_success_to_episodes(
                                    [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                                )
                                env.recorder_manager.export_episodes([0])
                                expert_result = "success"
                                corrections += 1
                                detection_successes[kind] += 1
                                record["exported"] = True
                        break
                    if bool(terminated[0]) or bool(truncated[0]):
                        expert_result = "expert_environment_termination"
                        break
                    state = read_state(env, obs_dict, arm_joint_ids)
                    if expert.failure:
                        expert_result = expert.failure
                        break

                if not record["exported"]:
                    env.recorder_manager.reset([0])
                record.update(
                    expert_result=expert_result,
                    expert_steps=expert_step,
                    expert_phases=expert.phase_history,
                    quality=quality,
                    quality_rejections=rejected_quality,
                )
                append_jsonl(args.log_file, record)
                rollouts_completed += 1
                result_counts[expert_result] += 1
                if args.smoke_three_rollouts and expert_result == "quality_rejected":
                    status = "quality_contract_conflict"
                    break
                if corrections >= args.num_corrections and not args.smoke_three_rollouts:
                    status = "complete"
                    break

        if args.smoke_three_rollouts and rollouts_completed == 3 and corrections >= args.num_corrections:
            status = "complete"
        rates = {
            kind: {
                "attempts": count,
                "successes": detection_successes[kind],
                "success_rate": detection_successes[kind] / count,
            }
            for kind, count in sorted(detection_attempts.items())
        }
        schema = validate_dataset_schema(args.dataset_file, corrections) if corrections else None
        summary = {
            "event": "summary",
            "record_type": "summary",
            "status": status,
            "checkpoint": str(args.checkpoint),
            "reset_seed": args.seed,
            "requested_corrections": args.num_corrections,
            "max_rollouts": args.max_rollouts,
            "rollouts_completed": rollouts_completed,
            "exported_corrections": corrections,
            "exported": corrections,
            "result_counts": dict(sorted(result_counts.items())),
            "detection_metrics": rates,
            "dataset_schema": schema,
            "quality_limits": quality_limits,
        }
        append_jsonl(args.log_file, summary)
        print("CORRECTION_RESULT " + json.dumps(summary, sort_keys=True), flush=True)
        if status == "quality_contract_conflict":
            raise RuntimeError("S0 correction failed the unchanged quality contract")
        if corrections < args.num_corrections:
            raise RuntimeError(f"collected only {corrections}/{args.num_corrections} corrections")
    finally:
        env.close()


def add_collector_args(parser: argparse.ArgumentParser) -> None:
    """Add collector-specific CLI arguments before AppLauncher arguments."""
    parser.add_argument("--task", default=TASK, choices=[TASK])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_corrections", type=int, required=True)
    parser.add_argument("--max_rollouts", type=int, required=True)
    parser.add_argument("--dataset_file", type=Path, required=True)
    parser.add_argument("--log_file", type=Path, required=True)
    parser.add_argument(
        "--disable_fabric",
        action="store_true",
        default=False,
        help="Disable fabric and use USD I/O operations.",
    )
    parser.add_argument(
        "--smoke_three_rollouts",
        action="store_true",
        help="Run exactly three rollouts and require at least --num_corrections exports; S0 only.",
    )


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, Any]:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    add_collector_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    try:
        validate_collection_counts(args.num_corrections, args.max_rollouts, args.smoke_three_rollouts)
    except ValueError as error:
        parser.error(str(error))
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    try:
        args.dataset_file = fresh_output(args.dataset_file, ".hdf5")
        args.log_file = fresh_output(args.log_file, ".jsonl")
        reserve_outputs(args.dataset_file, args.log_file)
    except (ValueError, FileExistsError, FileNotFoundError, NotADirectoryError) as error:
        parser.error(str(error))
    return args, AppLauncher


def main() -> None:
    args, app_launcher_class = parse_args()
    app_launcher = None
    try:
        app_launcher = app_launcher_class(args)
        collect(args, app_launcher.app)
    except BaseException as error:
        traceback.print_exc()
        try:
            append_exception_summary(args.log_file, error, "app_launch" if app_launcher is None else "collect")
        except BaseException:
            print("[ERROR] Failed to append exception summary:", flush=True)
            traceback.print_exc()
        if app_launcher is not None:
            close_app_with_failure(app_launcher.app)
        raise
    else:
        app_launcher.app.close()


if __name__ == "__main__":
    main()
