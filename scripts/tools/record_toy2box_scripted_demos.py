# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Record Toy2Box demonstrations using state feedback and quality-gated smooth actions.

Smooth mode uses the documented continuous [-0.05, 0.05] fallback, NOT literal
keyboard outputs. Legacy keyboard mode remains available for comparison. Neither
mode uses GUI events, state teleports, or task/core changes. See README_RS.md.
"""

import argparse
import json
import os
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np

TASK = "Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0"
PROTECTED_DATASET = Path(__file__).resolve().parents[2] / "datasets/toy2box_agibot.hdf5"
GRASP_QUAT = np.array([-0.2635, 0.8834, 0.2154, 0.3223])
GRASP_QUAT /= np.linalg.norm(GRASP_QUAT)


def check_output_path(path: Path, overwrite: bool) -> Path:
    """Reject existing outputs by default, and always protect the user's source."""
    path = path.expanduser().resolve()
    protected = PROTECTED_DATASET.resolve()
    if path == protected or (path.exists() and protected.exists() and os.path.samefile(path, protected)):
        raise ValueError(f"The user demonstration file is read-only: {protected}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Choose a new path or explicitly use --overwrite.")
    return path


def discrete_command(error: np.ndarray, tolerance: float | np.ndarray) -> np.ndarray:
    """Keyboard-equivalent simultaneous key presses, with a positional dead band."""
    return np.where(np.abs(error) > tolerance, np.sign(error) * 0.05, 0.0)


def keyboard_rotation_command(error: np.ndarray, tolerance: float = 0.035) -> np.ndarray:
    """One rotation key at a time: Euler-to-rotvec preserves the exact keyboard scalar.

    Simultaneous rotation keys would couple the axes in Se3Keyboard.advance(),
    producing values outside the required {-0.05, 0, +0.05} component alphabet.
    """
    command = np.zeros(3, dtype=np.float32)
    axis = int(np.argmax(np.abs(error)))
    if abs(error[axis]) > tolerance:
        command[axis] = np.copysign(0.05, error[axis])
    return command


def rotation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Base-axis angle error: target * inverse(current), both wxyz quaternions."""
    w1, v1 = target[0], target[1:]
    current = current / np.linalg.norm(current)
    w2, v2 = current[0], -current[1:]
    quat = np.r_[w1 * w2 - np.dot(v1, v2), w1 * v2 + w2 * v1 + np.cross(v1, v2)]
    if quat[0] < 0:
        quat = -quat
    norm = np.linalg.norm(quat[1:])
    return quat[1:] * (2 * np.arctan2(norm, quat[0]) / norm) if norm > 1e-8 else np.zeros(3)


SMOOTH_QUALITY_LIMITS = {
    "eef_acceleration_norm_p95": 1.5,
    "eef_jerk_norm_p95": 40.0,
    "angular_acceleration_norm_p95": 2.0,
    "arm_joint_acceleration_max_component_p95": 8.0,
}
SMOOTH_WARMUP_STEPS = 25
SMOOTH_QUALITY_SKIP = 0
MIN_QUALITY_SAMPLES = 4
MAX_RNG_SKIP_DRAWS = 1_000_000
YAW_BIN_EDGES_DEGREES = (-90.0, -45.0, 0.0, 45.0, 90.0)


def rng_skip_draws_arg(value: str) -> int:
    """Parse a bounded count before Isaac Sim is launched."""
    try:
        draws = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--rng_skip_draws must be an integer") from error
    if not 0 <= draws <= MAX_RNG_SKIP_DRAWS:
        raise argparse.ArgumentTypeError(
            f"--rng_skip_draws must be in [0, {MAX_RNG_SKIP_DRAWS}]"
        )
    return draws


def consume_rng_skip_draws(draws: int, random_draw: Callable[[], float] = random.random) -> None:
    """Advance the seeded Python RNG used by task object-pose sampling."""
    for _ in range(draws):
        random_draw()


def quaternion_yaw_degrees(quaternion: np.ndarray) -> float:
    """Return wxyz quaternion yaw in the canonical (-180, 180] interval."""
    w, x, y, z = np.asarray(quaternion, dtype=float)
    yaw = float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))
    return 180.0 if np.isclose(yaw, -180.0) else yaw


def targeted_yaw_bin(yaw_degrees: float) -> int | None:
    """Map the targeted (-90, 90] interval to four preregistered 45-degree bins."""
    if not -90.0 < yaw_degrees <= 90.0:
        return None
    if yaw_degrees < -45.0:
        return 0
    if yaw_degrees < 0.0:
        return 1
    if yaw_degrees < 45.0:
        return 2
    return 3


def yaw_quota_decision(yaw_degrees: float, quotas, accepted_counts) -> tuple[bool, int | None, str | None]:
    """Pure decision surface for targeted collection; absent quotas preserve the old path."""
    yaw_bin = targeted_yaw_bin(yaw_degrees)
    if quotas is None:
        return True, yaw_bin, None
    if yaw_bin is None:
        return False, None, "outside_target_yaw"
    if accepted_counts[yaw_bin] >= quotas[yaw_bin]:
        return False, yaw_bin, "yaw_bin_quota_full"
    return True, yaw_bin, None


def validate_yaw_bin_quotas(values, num_demos: int) -> tuple[int, int, int, int] | None:
    """Validate the explicit four-bin collection contract before launching Isaac Sim."""
    if values is None:
        return None
    quotas = tuple(values)
    if len(quotas) != 4 or any(value < 0 for value in quotas):
        raise ValueError("--yaw_bin_quotas requires four nonnegative integers")
    if sum(quotas) != num_demos:
        raise ValueError("sum(--yaw_bin_quotas) must equal --num_demos")
    return quotas


def settled_truck_telemetry(state: dict) -> dict:
    """Build the common initial-layout fields for attempts and filtered skips."""
    position = np.asarray(state["truck"], dtype=float)
    return {
        "initial_truck_yaw_degrees": quaternion_yaw_degrees(state["truck_quat"]),
        "initial_truck_x": float(position[0]),
        "initial_truck_y": float(position[1]),
        "initial_truck_position": position.tolist(),
    }


def collection_layout_decision(state, warmup_complete, control_mode, quotas, accepted_counts):
    """Return honest initial telemetry plus the optional post-settle filter decision."""
    if control_mode == "smooth" and not warmup_complete:
        telemetry = {
            "initial_layout_settled": False,
            "initial_truck_yaw_degrees": None,
            "initial_truck_x": None,
            "initial_truck_y": None,
            "initial_truck_position": None,
        }
        return telemetry, quotas is None, None, (
            None if quotas is None else "warmup_environment_termination"
        )
    telemetry = {
        "initial_layout_settled": control_mode == "smooth",
        **settled_truck_telemetry(state),
    }
    allowed, yaw_bin, reason = yaw_quota_decision(
        telemetry["initial_truck_yaw_degrees"], quotas, accepted_counts
    )
    return telemetry, allowed, yaw_bin, reason


def append_jsonl(path: Path | None, record: dict) -> None:
    """Append one durable telemetry event when streaming output was requested."""
    if path is None:
        return
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def norm_clip(vector: np.ndarray, limit: float) -> np.ndarray:
    """Limit vector magnitude without making diagonal commands faster."""
    return vector * min(1.0, limit / max(float(np.linalg.norm(vector)), 1e-12))


class SmoothPoseServo:
    """Taper near the target, limit acceleration, and dwell at zero before reversing.

    RMPFlow's relative command is a target offset from the CURRENT pose, not a
    velocity. Proportional offsets therefore approach a fixed world target gently.
    Command state persists through planner transitions; only reset clears it.
    """

    def __init__(self):
        self.previous = np.zeros(6)
        self.active = np.ones(6, dtype=bool)
        self.dwell = np.zeros(6, dtype=int)

    def advance(self, position_error: np.ndarray, orientation_error: np.ndarray) -> np.ndarray:
        error = np.r_[position_error, orientation_error]
        stop = np.array([0.0015] * 3 + [0.015] * 3)
        restart = np.array([0.004] * 3 + [0.035] * 3)
        self.active |= np.abs(error) > restart
        self.active &= np.abs(error) > stop
        desired = np.r_[norm_clip(0.35 * position_error, 0.025), norm_clip(0.25 * orientation_error, 0.030)]
        desired[~self.active] = 0.0
        crossing = desired * self.previous < 0.0
        self.dwell[crossing] = 4
        desired[(self.dwell > 0) | crossing] = 0.0
        self.dwell = np.maximum(self.dwell - 1, 0)
        command = self.previous + np.r_[
            norm_clip(desired[:3] - self.previous[:3], 0.002),
            norm_clip(desired[3:] - self.previous[3:], 0.003),
        ]
        command[np.abs(command) < 1e-8] = 0.0
        self.previous = command
        return command.astype(np.float32)


class Toy2BoxPolicy:
    """Bounded pick-and-place state machine; all physical motion goes through env.step."""

    def __init__(self, top_down: bool = False, control_mode: str = "keyboard"):
        if control_mode not in ("keyboard", "smooth"):
            raise ValueError(f"Unknown control mode: {control_mode}")
        self.control_mode = control_mode
        self.servo = SmoothPoseServo()
        self.hold_quat = None
        self.hold_target = None
        self.release_target = None
        self.pause_target = None
        self.top_down = top_down
        self.phase = "settle"
        self.phase_steps = 0
        self.stable_steps = 0
        self.grasp_seen = False
        self.failure = None
        self.pickup = None
        self.lift_height = None
        self.hold_offset = np.zeros(3)
        self.transport_start = np.zeros(3)
        self.grasp_quat = GRASP_QUAT.copy()
        self.grasp_offset = np.array([0.020, 0.015, -0.010])
        self.carry_stage = 0
        self.carry_pause = 0
        self.release_xy = None
        self.phase_history = []
        self.hover_best_error = float("inf")
        self.hover_stale_steps = 0

    def transition(self, phase: str, state: dict):
        self.phase = phase
        self.phase_steps = 0
        self.stable_steps = 0
        self.phase_history.append(phase)
        if phase == "close":
            self.hold_quat = state["quat"].copy()
        if phase == "align":
            self.hold_target = state["box"] + self.hold_offset
            self.hold_target[2] = max(self.lift_height, state["box"][2] + self.hold_offset[2] + 0.22)
        if phase == "release":
            self.release_target = state["eef"].copy()
        print(
            f"PHASE {phase} eef={state['eef'].round(4).tolist()} truck={state['truck'].round(4).tolist()}"
            f" box={state['box'].round(4).tolist()} grasp={state['grasp']}",
            flush=True,
        )

    def initialize_from_settled_state(self, state: dict):
        """Initialize grasp targets from an already settled, freshly observed state."""
        w, x, y, z = state["truck_quat"]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        delta = (yaw - 0.95 + np.pi / 2) % np.pi - np.pi / 2
        c, si = np.cos(delta / 2), np.sin(delta / 2)
        w, x, y, z = GRASP_QUAT
        self.grasp_quat = np.array([c * w - si * z, c * x - si * y, c * y + si * x, c * z + si * w])
        c, si = np.cos(delta), np.sin(delta)
        self.grasp_offset[:2] = [c * 0.020 - si * 0.015, si * 0.020 + c * 0.015]
        if self.top_down:
            theta = (yaw + np.pi / 2) % np.pi - np.pi / 2
            self.grasp_quat = np.array([0.0, np.cos(theta / 2), np.sin(theta / 2), 0.0])
            self.grasp_offset = np.array([0.0, 0.0, 0.005])
        self.pickup = state["truck"] + self.grasp_offset
        self.transition("hover", state)

    def advance(self, state: dict) -> np.ndarray:
        self.phase_steps += 1
        self.grasp_seen |= state["grasp"]
        eef, truck, box = state["eef"], state["truck"], state["box"]
        action = np.zeros(7, dtype=np.float32)
        action[-1] = -1 if self.phase in ("close", "lift", "carry", "align", "lower") else 1
        target = eef.copy()
        rotate = self.phase in ("hover", "descend")
        if self.phase == "settle":
            if self.phase_steps >= 25:
                self.initialize_from_settled_state(state)
        elif self.phase == "hover":
            target = truck + self.grasp_offset
            target[2] = truck[2] + 0.18
            hover_error = (
                np.linalg.norm(rotation_error(self.grasp_quat, state["quat"]))
                + 2 * np.linalg.norm(target - eef)
            )
            if hover_error < self.hover_best_error - 0.01:
                self.hover_best_error = hover_error
                self.hover_stale_steps = 0
            else:
                self.hover_stale_steps += 1
            if self.phase_steps > 300 and self.hover_stale_steps > 120:
                self.failure = "hover_pose_unreachable"
            if (
                np.max(np.abs(target - eef)) < 0.015
                and np.linalg.norm(rotation_error(self.grasp_quat, state["quat"])) < 0.10
            ):
                self.stable_steps += 1
                if self.stable_steps >= 8:
                    self.transition("descend", state)
            else:
                self.stable_steps = 0
        elif self.phase == "descend":
            # Track small contact-induced truck movement while the open fingers descend.
            target = truck + self.grasp_offset
            if np.linalg.norm(eef - truck) < 0.035 and abs(eef[2] - truck[2]) < 0.020:
                self.stable_steps += 1
                if self.stable_steps >= 4:
                    self.pickup = eef.copy()
                    self.transition("close", state)
            else:
                self.stable_steps = 0
        elif self.phase == "close":
            target = self.pickup
            if self.phase_steps >= 35:
                self.lift_height = max(eef[2] + 0.22, box[2] + 0.23)
                self.transition("lift", state)
        elif self.phase == "lift":
            target = self.pickup.copy()
            target[2] = self.lift_height
            if self.phase_steps > 70 and truck[2] < self.pickup[2] + 0.07:
                self.failure = "pickup_did_not_lift_truck"
            elif abs(target[2] - eef[2]) < 0.015:
                if not self.grasp_seen or np.linalg.norm(eef - truck) > 0.05:
                    self.failure = "no_confirmed_grasp"
                else:
                    self.hold_offset = eef - truck
                    self.transport_start = eef.copy()
                    self.transition("carry", state)
        elif self.phase == "carry":
            # Match the human strategy: move out in x before crossing the body in y.
            # Freeze the grasp offset; a slipped object must not drag the wrist target away.
            if np.linalg.norm(eef - truck) > 0.09:
                self.failure = "lost_grasp_during_transport"
            target = box + self.hold_offset
            target[2] = max(self.lift_height, box[2] + self.hold_offset[2] + 0.22)
            if self.carry_stage == 0:
                target[0] = min(target[0], 0.80)
                target[1] = self.transport_start[1]
                if abs(eef[0] - target[0]) < 0.015:
                    self.carry_stage = 1
                    self.carry_pause = 8 if self.top_down else 0
                    self.pause_target = eef.copy()
            elif self.carry_stage == 1:
                target[0] = min(target[0], 0.80)
                if abs(eef[1] - target[1]) < 0.015:
                    self.carry_stage = 2
                    self.carry_pause = 8 if self.top_down else 0
                    self.pause_target = eef.copy()
            else:
                target[:2] = eef[:2] + box[:2] - truck[:2]
            align_radius = 0.025 if self.control_mode == "smooth" else (0.060 if self.top_down else 0.030)
            if np.linalg.norm(truck[:2] - box[:2]) < align_radius:
                self.transition("align", state)
        elif self.phase == "align":
            # Latch a zero-motion hold instead of dithering below one keyboard step.
            xy_error = np.linalg.norm(truck[:2] - box[:2])
            if np.linalg.norm(eef - truck) > 0.09:
                self.failure = "lost_grasp_during_transport"
            elif self.control_mode == "keyboard" and xy_error > (0.070 if self.top_down else 0.040):
                self.carry_stage = 2
                self.transition("carry", state)
            elif self.phase_steps >= 20 and xy_error < (
                0.030 if self.control_mode == "smooth" else (0.060 if self.top_down else 0.035)
            ):
                if self.top_down:
                    self.release_xy = float(xy_error)
                    self.transition("release", state)
                else:
                    self.transition("lower", state)
        elif self.phase == "lower":
            # Reduce ballistic rim strikes: lower the held truck before opening.
            if np.linalg.norm(eef - truck) > 0.09:
                self.failure = "lost_grasp_before_release"
            target[:2] = eef[:2] + box[:2] - truck[:2]
            target[2] = eef[2] + (box[2] + 0.13 - truck[2])
            if truck[2] < box[2] + 0.15 and np.linalg.norm(truck[:2] - box[:2]) < 0.025:
                self.stable_steps += 1
                if self.stable_steps >= 8:
                    self.release_xy = float(np.linalg.norm(truck[:2] - box[:2]))
                    self.transition("release", state)
            else:
                self.stable_steps = 0
        elif self.phase == "release":
            # A short keyboard nudge while opening counters outward finger-release motion.
            if self.top_down and self.phase_steps <= 4:
                action[:2] = discrete_command(box[:2] - truck[:2], 0.012)
            if self.phase_steps > 15 and self.phase_steps < 35:
                action[2] = 0.05
            if self.phase_steps > 120:
                self.failure = "placement_did_not_settle"
        if self.phase == "descend" and self.phase_steps > 120:
            self.failure = "pickup_pose_unreachable"
        if self.phase_steps > (650 if self.phase == "hover" else 240):
            self.failure = f"phase_timeout:{self.phase}"
        if self.control_mode == "smooth":
            if self.phase == "settle":
                return action
            if self.phase == "close":
                target = self.pickup.copy()
            elif self.phase == "align":
                target = self.hold_target.copy()
            elif self.phase == "release":
                target = self.release_target.copy()
                if self.phase_steps > 25:
                    target[2] += 0.06
            if self.phase == "carry" and self.carry_pause > 0:
                self.carry_pause -= 1
                target = self.pause_target.copy()
            orientation = self.grasp_quat if self.phase in ("hover", "descend") else self.hold_quat
            if orientation is None:
                orientation = state["quat"]
            action[:6] = self.servo.advance(target - eef, rotation_error(orientation, state["quat"]))
            return action
        if self.phase not in ("settle", "release", "done"):
            action[:3] = discrete_command(target - eef, np.array([0.006, 0.006, 0.004]))
        if self.phase == "close":
            action[:6] = 0
        elif self.phase == "lift":
            action[:2] = 0
        if self.phase == "carry" and self.carry_pause > 0:
            self.carry_pause -= 1
            action[:6] = 0
        if self.phase == "align":
            action[:6] = 0
        if rotate:
            action[3:6] = keyboard_rotation_command(rotation_error(self.grasp_quat, state["quat"]))
        return action


def prepare_smooth_recording(env, warmup_action, warmup_steps: int = SMOOTH_WARMUP_STEPS):
    """Settle physics, discard warmup recording, and open a fresh recorder episode.

    This deliberately does not reset physics, the scene, events, or commands after
    settling. Manager resets are followed immediately by observation computation;
    the caller performs the first subsequent physics step with an explicit action.
    """
    observation = env.obs_buf
    for warmup_step in range(1, warmup_steps + 1):
        observation, _, terminated, truncated, _ = env.step(warmup_action)
        if bool(terminated[0]) or bool(truncated[0]):
            env.recorder_manager.reset([0])
            return observation, warmup_step, False

    env_ids = [0]
    env.observation_manager.reset(env_ids)
    env.action_manager.reset(env_ids)
    env.episode_length_buf[env_ids] = 0
    env.recorder_manager.reset(env_ids)
    env.recorder_manager.record_post_reset(env_ids)
    env.obs_buf = env.observation_manager.compute(update_history=True)
    return env.obs_buf, warmup_steps, True


def evaluate_smooth_quality(samples, trajectory_quality, quality_rejections, quality_limits):
    """Evaluate only recorded post-warmup samples; no reset-settle prefix is present."""
    quality = trajectory_quality(
        np.asarray(samples["actions"]),
        np.asarray(samples["eef"]),
        np.asarray(samples["quat"]),
        np.asarray(samples["arm_joints"]),
        skip=SMOOTH_QUALITY_SKIP,
    )
    return quality, quality_rejections(quality, quality_limits)


def read_state(env, observation: dict, arm_joint_ids) -> dict:
    """Read task observations (base frame), and live contact diagnostics without state writes."""
    policy = observation["policy"]

    def array(key):
        return policy[key][0].detach().cpu().numpy().copy()

    forces = env.scene["contact_grasp"].data.net_forces_w[0].norm(dim=-1).cpu().tolist()
    return {
        "eef": array("eef_pos"),
        "arm_joints": env.scene["robot"].data.joint_pos[0, arm_joint_ids].cpu().numpy().copy(),
        "quat": array("eef_quat"),
        "truck": array("toy_truck_positions"),
        "truck_quat": array("toy_truck_orientations"),
        "box": array("box_positions"),
        "truck_vel": env.scene["toy_truck"].data.root_lin_vel_w[0].cpu().numpy().copy(),
        "grasp": bool(observation["subtask_terms"]["grasp"][0]),
        "forces": forces,
    }


def collect(args, simulation_app):
    # AppLauncher must be running before importing simulation-dependent modules.
    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import DatasetExportMode
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.env_name = args.task.split(":")[-1]
    cfg.seed = args.seed
    if not cfg.actions.arm_action.use_relative_mode:
        raise ValueError("Relative RMPFlow mode is required. Unset USE_RELATIVE_MODE or set it to True.")
    success_term = cfg.terminations.success
    cfg.terminations.success = None
    cfg.terminations.time_out = None
    cfg.observations.policy.concatenate_terms = False
    cfg.recorders = ActionStateRecorderManagerCfg()
    cfg.recorders.dataset_export_dir_path = str(args.dataset_file.parent)
    cfg.recorders.dataset_filename = args.dataset_file.stem
    cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env = cast(ManagerBasedRLEnv, gym.make(args.task, cfg=cfg).unwrapped)
    consume_rng_skip_draws(args.rng_skip_draws)
    from toy2box_quality import DEFAULT_LIMITS, quality_rejections, trajectory_quality

    arm_joint_ids, arm_joint_names = env.scene["robot"].find_joints("right_arm_joint[1-7]")
    if len(arm_joint_ids) != 7:
        raise ValueError(f"Expected seven right-arm joints, found {arm_joint_names}")
    quality_limits = dict(DEFAULT_LIMITS, **SMOOTH_QUALITY_LIMITS)
    results = {
        "task": args.task,
        "seed": args.seed,
        "rng_skip_draws": args.rng_skip_draws,
        "rng_skip_source": "python.random.random",
        "rng_skip_rationale": (
            f"Advance the Python RNG seeded with {args.seed}, which the task's object-pose sampler uses, "
            "to obtain different placements; "
            "no simulator state or action is teleported."
            if args.rng_skip_draws
            else "Disabled; the seeded randomization stream is unchanged."
        ),
        "top_down": args.top_down,
        "control_mode": args.control_mode,
        "continuous_fallback_reason": "Avoid fixed-key bang-bang chatter; user requested smooth-only data."
        if args.control_mode == "smooth" else None,
        "quality_limits": quality_limits if args.control_mode == "smooth" else None,
        "right_arm_joint_ids": arm_joint_ids,
        "right_arm_joint_names": arm_joint_names,
        "dataset_file": str(args.dataset_file),
        "keyboard_pose_values": [-0.05, 0.0, 0.05] if args.control_mode == "keyboard" else None,
        "pose_action_range": [-0.05, 0.05],
        "keyboard_gripper_values": [-1, 1],
        "num_success_steps": 10,
        "warmup_steps": SMOOTH_WARMUP_STEPS if args.control_mode == "smooth" else 0,
        "warmup_recorded": False if args.control_mode == "smooth" else None,
        "quality_skip": SMOOTH_QUALITY_SKIP if args.control_mode == "smooth" else None,
        "requested_demos": args.num_demos,
        "max_final_xy": args.max_final_xy,
        "max_steps": args.max_steps,
        "max_attempts": args.max_attempts,
        "yaw_bin_edges_degrees": list(YAW_BIN_EDGES_DEGREES),
        "yaw_target_interval": "(-90, 90]",
        "yaw_bin_quotas": list(args.yaw_bin_quotas) if args.yaw_bin_quotas is not None else None,
        "accepted_yaw_bin_counts": [0, 0, 0, 0],
        "candidate_layouts": 0,
        "attempted_layouts": 0,
        "skipped_layouts": 0,
        "skip_reason_counts": {},
        "attempts": [],
        "skips": [],
    }
    start = time.monotonic()
    try:
        with torch.inference_mode():
            attempt = 0
            while attempt < args.max_attempts:
                if not simulation_app.is_running():
                    break
                results["candidate_layouts"] += 1
                candidate_index = results["candidate_layouts"]
                # Same explicit reset order as record_demos.handle_reset().
                env.sim.reset()
                env.recorder_manager.reset()
                observation, _ = env.reset()
                controller = Toy2BoxPolicy(top_down=args.top_down, control_mode=args.control_mode)
                samples = {key: [] for key in ("actions", "eef", "quat", "arm_joints")}
                quality = None
                rejected_quality = []
                success_steps = 0
                min_grasp_distance = float("inf")
                grasp_steps = 0
                reason = "step_limit"
                step = 0
                state = read_state(env, observation, arm_joint_ids)
                warmup_steps = 0
                warmup_complete = True
                if args.control_mode == "smooth":
                    warmup_action = torch.zeros((1, 7), dtype=torch.float32, device=env.device)
                    warmup_action[0, -1] = 1.0
                    observation, warmup_steps, warmup_complete = prepare_smooth_recording(
                        env, warmup_action
                    )
                    if warmup_complete:
                        state = read_state(env, observation, arm_joint_ids)
                        controller.initialize_from_settled_state(state)
                    else:
                        state = read_state(env, observation, arm_joint_ids)
                        reason = "warmup_environment_termination"
                initial_layout, allowed, initial_yaw_bin, skip_reason = collection_layout_decision(
                    state,
                    warmup_complete,
                    args.control_mode,
                    args.yaw_bin_quotas,
                    results["accepted_yaw_bin_counts"],
                )
                initial_truck_yaw_degrees = initial_layout["initial_truck_yaw_degrees"]
                if not allowed:
                    results["skipped_layouts"] += 1
                    results["skip_reason_counts"][skip_reason] = (
                        results["skip_reason_counts"].get(skip_reason, 0) + 1
                    )
                    skip = {
                        "event": "layout_skip",
                        "candidate_index": candidate_index,
                        "skip_index": results["skipped_layouts"],
                        "attempt": None,
                        "reason": skip_reason,
                        **initial_layout,
                        "yaw_bin": initial_yaw_bin,
                        "yaw_bin_quotas": list(args.yaw_bin_quotas),
                        "accepted_yaw_bin_counts": results["accepted_yaw_bin_counts"].copy(),
                    }
                    results["skips"].append(skip)
                    results["elapsed_seconds"] = time.monotonic() - start
                    args.stats_file.write_text(json.dumps(results, indent=2) + "\n")
                    append_jsonl(args.attempt_log, skip)
                    print("LAYOUT_SKIP " + json.dumps(skip), flush=True)
                    continue
                attempt += 1
                results["attempted_layouts"] = attempt
                print(
                    f"ATTEMPT {attempt} candidate={candidate_index} "
                    f"initial_settled_truck={initial_layout['initial_truck_position']} "
                    f"initial_settled_yaw_degrees={initial_truck_yaw_degrees} "
                    f"box={state['box'].tolist()}",
                    flush=True,
                )
                successes_before = env.recorder_manager.exported_successful_episode_count
                for step in range(1, args.max_steps + 1) if warmup_complete else ():
                    if not simulation_app.is_running() or env.sim.is_stopped():
                        reason = "simulation_stopped"
                        break
                    actions = controller.advance(state)
                    samples["actions"].append(actions.copy())
                    for key in ("eef", "quat", "arm_joints"):
                        samples[key].append(state[key].copy())
                    # Raw actions are recorded unmodified; smooth mode is explicitly not keyboard-valued.
                    observation, _, terminated, truncated, _ = env.step(
                        torch.tensor(actions, device=env.device).unsqueeze(0)
                    )
                    if bool(terminated[0] or truncated[0]):
                        reason = "environment_termination"
                        break  # env.step has auto-reset; never splice that new state into this attempt.
                    state = read_state(env, observation, arm_joint_ids)
                    controller.grasp_seen |= state["grasp"]
                    distance = float(np.linalg.norm(state["eef"] - state["truck"]))
                    if state["grasp"]:
                        grasp_steps += 1
                        min_grasp_distance = min(min_grasp_distance, distance)
                    task_success = bool(success_term.func(env, **success_term.params)[0])
                    success_steps = success_steps + 1 if task_success else 0
                    if args.trace_interval and step % args.trace_interval == 0:
                        print(
                            f"TRACE attempt={attempt} step={step} phase={controller.phase} "
                            f"eef={state['eef'].round(4).tolist()} truck={state['truck'].round(4).tolist()} "
                            f"quat={state['quat'].round(3).tolist()} "
                            f"forces={np.round(state['forces'], 2).tolist()} "
                            f"dist={distance:.4f} grasp={state['grasp']}",
                            flush=True,
                        )
                    if success_steps >= 10:
                        final_xy = float(np.linalg.norm(state["truck"][:2] - state["box"][:2]))
                        if not controller.grasp_seen or final_xy > args.max_final_xy:
                            reason = "success_without_grasp_or_centered_placement"
                            break
                        if args.control_mode == "smooth":
                            if len(samples["actions"]) < MIN_QUALITY_SAMPLES:
                                reason = "quality_insufficient_samples"
                                break
                            quality, rejected_quality = evaluate_smooth_quality(
                                samples, trajectory_quality, quality_rejections, quality_limits
                            )
                            if rejected_quality:
                                reason = "quality_rejected"
                                break
                        # Same success finalization as record_demos.process_success_condition().
                        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                        env.recorder_manager.set_success_to_episodes(
                            [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                        )
                        env.recorder_manager.export_episodes([0])
                        reason = "success"
                        break
                    if controller.failure:
                        reason = controller.failure
                        break
                if (
                    args.control_mode == "smooth"
                    and quality is None
                    and len(samples["actions"]) >= MIN_QUALITY_SAMPLES
                ):
                    quality, rejected_quality = evaluate_smooth_quality(
                        samples, trajectory_quality, quality_rejections, quality_limits
                    )
                result = {
                    "event": "attempt",
                    "candidate_index": candidate_index,
                    "attempt": attempt,
                    **initial_layout,
                    "yaw_bin": initial_yaw_bin,
                    "steps": step,
                    "warmup_steps": warmup_steps,
                    "warmup_recorded": False if args.control_mode == "smooth" else None,
                    "quality_skip": SMOOTH_QUALITY_SKIP if args.control_mode == "smooth" else None,
                    "result": reason,
                    "quality": quality,
                    "quality_rejections": rejected_quality,
                    "phases": controller.phase_history,
                    "grasp_steps": grasp_steps,
                    "min_grasp_distance": min_grasp_distance if grasp_steps else None,
                    "release_xy": controller.release_xy,
                    "final_xy": float(np.linalg.norm(state["truck"][:2] - state["box"][:2])),
                    "final_truck": state["truck"].tolist(),
                    "final_eef": state["eef"].tolist(),
                }
                successes_after = env.recorder_manager.exported_successful_episode_count
                accepted = successes_after == successes_before + 1
                if accepted and initial_yaw_bin is not None:
                    results["accepted_yaw_bin_counts"][initial_yaw_bin] += 1
                result["accepted"] = accepted
                result["accepted_demo_index"] = successes_before if accepted else None
                result["accepted_demo_key"] = f"demo_{successes_before}" if accepted else None
                result["yaw_bin_quotas"] = (
                    list(args.yaw_bin_quotas) if args.yaw_bin_quotas is not None else None
                )
                result["accepted_yaw_bin_counts"] = results["accepted_yaw_bin_counts"].copy()
                results["attempts"].append(result)
                results["successes"] = successes_after
                results["success_rate"] = results["successes"] / attempt
                results["elapsed_seconds"] = time.monotonic() - start
                args.stats_file.write_text(json.dumps(results, indent=2) + "\n")
                append_jsonl(args.attempt_log, result)
                print("ATTEMPT_RESULT " + json.dumps(result), flush=True)
                if results["successes"] >= args.num_demos or reason == "simulation_stopped":
                    break
        print("COLLECTION_RESULT " + json.dumps({k: v for k, v in results.items() if k != "attempts"}), flush=True)
        if results.get("successes", 0) < args.num_demos:
            raise RuntimeError(f"Collected only {results.get('successes', 0)}/{args.num_demos}; see {args.stats_file}")
    finally:
        env.close()


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=TASK, choices=[TASK])
    parser.add_argument("--dataset_file", type=Path, default=Path("datasets/toy2box_agibot_codex_smooth.hdf5"))
    parser.add_argument("--control_mode", choices=["smooth", "keyboard"], default="smooth")
    parser.add_argument("--stats_file", type=Path, help="Attempt log JSON (default: next to the dataset).")
    parser.add_argument(
        "--attempt_log",
        type=Path,
        help="Optional fresh JSONL stream with one final event per attempted or skipped layout.",
    )
    parser.add_argument(
        "--yaw_bin_quotas",
        type=int,
        nargs=4,
        metavar=("Q0", "Q1", "Q2", "Q3"),
        help="Accepted-demo quotas for (-90,-45), [-45,0), [0,45), [45,90] degrees; absent disables filtering.",
    )
    parser.add_argument("--num_demos", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rng_skip_draws",
        type=rng_skip_draws_arg,
        default=0,
        help="Advance the task's seeded Python RNG stream by 0..1000000 draws before the first reset attempt.",
    )
    parser.add_argument("--max_attempts", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=1400)
    parser.add_argument(
        "--max_final_xy", type=float, default=0.06,
        help="Additional XY quality limit (task success is <0.10 m).",
    )
    parser.add_argument("--trace_interval", type=int, default=0)
    grasp_mode = parser.add_mutually_exclusive_group()
    grasp_mode.add_argument("--top_down", dest="top_down", action="store_true", help="Vertical grasp (default).")
    grasp_mode.add_argument(
        "--reference_grasp", dest="top_down", action="store_false", help="Use the tilted human-reference grasp."
    )
    parser.set_defaults(top_down=True)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing outputs, NEVER the user's source.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.num_demos, args.max_attempts, args.max_steps) < 1:
        parser.error("--num_demos, --max_attempts and --max_steps must be positive")
    if not 0 < args.max_final_xy <= 0.10:
        parser.error("--max_final_xy must be in (0, 0.10]")
    if args.dataset_file.suffix != ".hdf5":
        parser.error("--dataset_file must have the .hdf5 extension")
    try:
        args.yaw_bin_quotas = validate_yaw_bin_quotas(args.yaw_bin_quotas, args.num_demos)
        if args.yaw_bin_quotas is not None and args.control_mode != "smooth":
            raise ValueError("--yaw_bin_quotas requires --control_mode smooth for the 25-step settled pose")
        args.dataset_file = check_output_path(args.dataset_file, args.overwrite)
        args.stats_file = check_output_path(
            args.stats_file or args.dataset_file.with_suffix(".stats.json"), args.overwrite
        )
        if args.attempt_log is not None:
            args.attempt_log = check_output_path(args.attempt_log, args.overwrite)
        output_paths = [args.dataset_file, args.stats_file]
        if args.attempt_log is not None:
            output_paths.append(args.attempt_log)
        if len(set(output_paths)) != len(output_paths):
            raise ValueError("Dataset, statistics, and attempt-log paths must differ")
        for index, path in enumerate(output_paths):
            for other in output_paths[:index]:
                if path.exists() and other.exists() and os.path.samefile(path, other):
                    raise ValueError("Dataset, statistics, and attempt-log paths must differ")
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    args.dataset_file.parent.mkdir(parents=True, exist_ok=True)
    args.stats_file.parent.mkdir(parents=True, exist_ok=True)
    if args.attempt_log is not None:
        args.attempt_log.parent.mkdir(parents=True, exist_ok=True)
    # Reserve new paths before Isaac Sim starts. The recorder later opens ONLY our chosen output in mode 'w'.
    for path in output_paths:
        with path.open("w" if args.overwrite else "x"):
            pass
    app_launcher = AppLauncher(args)
    try:
        collect(args, app_launcher.app)
    except BaseException:
        # Kit fast shutdown may exit the interpreter: print errors before closing it.
        import traceback

        traceback.print_exc()
        raise
    finally:
        app_launcher.app.close()


if __name__ == "__main__":
    main()
