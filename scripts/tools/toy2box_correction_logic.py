"""Pure, simulator-free logic for Toy2Box DAgger correction collection.

Step indices are one-based post-action observations, matching a stock rollout
loop after its first ``env.step``.  Distances are measured in the robot-base
frame.  The thresholds in this module are the frozen preregistration-3 values
(report/PREREGISTRATION.md); callers must not tune them from rollout outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


NEAR_CLOSE_DISTANCE_M = 0.06
ACQUISITION_LIFT_M = 0.03
ACQUISITION_EEF_DISTANCE_M = 0.06
ACQUISITION_CONSECUTIVE_STEPS = 10
GRASP_FAILURE_DEADLINE_STEPS = 60
DROP_BOX_XY_DISTANCE_M = 0.12
DROP_EEF_DISTANCE_M = 0.06
NO_NEAR_CLOSE_DEADLINE_STEP = 450
MAX_TRUCK_TILT_DEG = 25.0
TRUCK_HEIGHT_TOLERANCE_M = 0.03
TRUCK_ENV_X_BOUNDS_M = (-0.20, 0.25)
TRUCK_ENV_Y_BOUNDS_M = (-0.35, -0.10)
BOX_EXCLUSION_XY_DISTANCE_M = 0.12


@dataclass(frozen=True)
class RecoveryObservation:
    """One post-action policy rollout observation."""

    step: int
    truck_pos_xyz: Any
    eef_pos_xyz: Any
    box_pos_xyz: Any
    action_gripper_closed: bool
    settled_truck_z: float


@dataclass(frozen=True)
class FailureDetection:
    """A preregistered takeover trigger and its auditable measurements."""

    kind: str
    detected_step: int
    details: Mapping[str, Any]


def _vector3(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite length-three vector")
    return array


def _normalized_quaternion_wxyz(value: Any) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("truck_quat_wxyz must be a finite length-four vector")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("truck_quat_wxyz must have nonzero norm")
    return quaternion / norm


def truck_roll_pitch_degrees(truck_quat_wxyz: Any) -> tuple[float, float]:
    """Return intrinsic XYZ roll and pitch for a wxyz quaternion."""

    w, x, y, z = _normalized_quaternion_wxyz(truck_quat_wxyz)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_sine = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_sine)
    return math.degrees(roll), math.degrees(pitch)


def skip_reason(
    truck_quat_wxyz: Any,
    truck_position_env: Any,
    *,
    truck_height: float,
    settled_truck_height: float,
    truck_box_xy_distance: float,
    grasped: bool,
) -> str | None:
    """Return the fixed preregistration-3 takeover exclusion, or ``None``.

    ``truck_position_env`` is explicitly relative to the environment origin;
    its Z component is not used.  ``truck_height`` and
    ``settled_truck_height`` must share any translation-invariant frame.
    ``truck_box_xy_distance`` is likewise frame-translation invariant.
    """

    roll_deg, pitch_deg = truck_roll_pitch_degrees(truck_quat_wxyz)
    if abs(roll_deg) > MAX_TRUCK_TILT_DEG or abs(pitch_deg) > MAX_TRUCK_TILT_DEG:
        return "truck_tilt_exceeds_25_deg"
    position_env = _vector3(truck_position_env, "truck_position_env")
    heights = (float(truck_height), float(settled_truck_height))
    if not all(math.isfinite(value) for value in heights):
        raise ValueError("truck heights must be finite")
    if not isinstance(grasped, (bool, np.bool_)):
        raise ValueError("grasped must be an observed boolean")
    box_distance = float(truck_box_xy_distance)
    if not math.isfinite(box_distance) or box_distance < 0.0:
        raise ValueError("truck_box_xy_distance must be finite and nonnegative")

    if heights[0] < heights[1] - TRUCK_HEIGHT_TOLERANCE_M:
        return "truck_below_settled_height"
    if heights[0] > heights[1] + TRUCK_HEIGHT_TOLERANCE_M and not bool(grasped):
        return "truck_above_settled_ungrasped"
    if not (
        TRUCK_ENV_X_BOUNDS_M[0] <= position_env[0] <= TRUCK_ENV_X_BOUNDS_M[1]
        and TRUCK_ENV_Y_BOUNDS_M[0] <= position_env[1] <= TRUCK_ENV_Y_BOUNDS_M[1]
    ):
        return "truck_xy_outside_bounds"
    if box_distance <= BOX_EXCLUSION_XY_DISTANCE_M:
        return "truck_within_box_exclusion_radius"
    return None


class FailureDetector:
    """State machine implementing the preregistration-3 triggers exactly."""

    def __init__(self) -> None:
        self.previous_step = 0
        self.previous_gripper_closed = False
        self.near_close_seen = False
        self.pending_near_close_step: int | None = None
        self.acquisition_run = 0
        self.acquired = False
        self.acquired_step: int | None = None
        self.detection: FailureDetection | None = None

    def update(self, observation: RecoveryObservation) -> FailureDetection | None:
        """Consume exactly one consecutive, one-based post-action observation.

        A close at step ``c`` has through step ``c + 60`` to complete the tenth
        consecutive acquisition observation.  If it does not, trigger (a) at
        ``c + 60``.  At step 450, a near-close onset on that same step counts and
        therefore suppresses (c).  Once a detection is returned, further calls
        are rejected because the policy must stop immediately.
        """

        if self.detection is not None:
            raise RuntimeError("failure detector updated after takeover trigger")
        if not isinstance(observation.step, int) or isinstance(observation.step, bool):
            raise ValueError("step must be an integer")
        expected_step = self.previous_step + 1
        if observation.step != expected_step:
            raise ValueError(f"expected consecutive step {expected_step}, got {observation.step}")
        truck = _vector3(observation.truck_pos_xyz, "truck_pos_xyz")
        eef = _vector3(observation.eef_pos_xyz, "eef_pos_xyz")
        box = _vector3(observation.box_pos_xyz, "box_pos_xyz")
        if not math.isfinite(float(observation.settled_truck_z)):
            raise ValueError("settled_truck_z must be finite")

        step = observation.step
        gripper_closed = bool(observation.action_gripper_closed)
        eef_distance = float(np.linalg.norm(eef - truck))
        box_xy_distance = float(np.linalg.norm(truck[:2] - box[:2]))
        close_onset = gripper_closed and not self.previous_gripper_closed
        if close_onset and eef_distance <= NEAR_CLOSE_DISTANCE_M:
            self.near_close_seen = True
            if not self.acquired and self.pending_near_close_step is None:
                self.pending_near_close_step = step

        if self.near_close_seen and not self.acquired:
            acquisition_sample = (
                truck[2] >= float(observation.settled_truck_z) + ACQUISITION_LIFT_M
                and eef_distance <= ACQUISITION_EEF_DISTANCE_M
            )
            self.acquisition_run = self.acquisition_run + 1 if acquisition_sample else 0
            if self.acquisition_run >= ACQUISITION_CONSECUTIVE_STEPS:
                self.acquired = True
                self.acquired_step = step
                self.pending_near_close_step = None

        detection: FailureDetection | None = None
        if (
            self.acquired
            and box_xy_distance > DROP_BOX_XY_DISTANCE_M
            and eef_distance > DROP_EEF_DISTANCE_M
        ):
            detection = FailureDetection(
                kind="drop",
                detected_step=step,
                details={
                    "acquired_step": self.acquired_step,
                    "truck_box_xy_distance_m": box_xy_distance,
                    "truck_eef_distance_m": eef_distance,
                },
            )
        elif (
            self.pending_near_close_step is not None
            and step - self.pending_near_close_step >= GRASP_FAILURE_DEADLINE_STEPS
        ):
            detection = FailureDetection(
                kind="grasp_failure",
                detected_step=step,
                details={
                    "near_close_step": self.pending_near_close_step,
                    "deadline_steps": GRASP_FAILURE_DEADLINE_STEPS,
                    "acquisition_consecutive_steps": self.acquisition_run,
                    "truck_eef_distance_m": eef_distance,
                    "truck_lift_m": float(truck[2] - observation.settled_truck_z),
                },
            )
        elif step >= NO_NEAR_CLOSE_DEADLINE_STEP and not self.near_close_seen:
            detection = FailureDetection(
                kind="no_near_close_by_450",
                detected_step=step,
                details={"deadline_step": NO_NEAR_CLOSE_DEADLINE_STEP},
            )

        self.previous_step = step
        self.previous_gripper_closed = gripper_closed
        self.detection = detection
        return detection


def slice_aligned_episode(
    sample_trees: Mapping[str, Any],
    takeover_index: int,
    initial_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Slice aligned action/observation/state trees at a pre-action index.

    ``initial_state`` is supplied explicitly from the simulator snapshot at the
    takeover boundary; it is never inferred from a post-action sample.  Every
    leaf in ``sample_trees`` must share the same leading sample dimension.
    """

    lengths: list[int] = []

    def inspect(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            if not value:
                raise ValueError(f"empty sample mapping at {path}")
            for key, child in value.items():
                inspect(child, f"{path}/{key}")
            return
        array = np.asarray(value)
        if array.ndim < 1:
            raise ValueError(f"sample leaf {path} has no leading sample dimension")
        lengths.append(int(array.shape[0]))

    if not sample_trees:
        raise ValueError("sample_trees must not be empty")
    inspect(sample_trees, "samples")
    if len(set(lengths)) != 1:
        raise ValueError(f"sample trees are misaligned: lengths={sorted(set(lengths))}")
    sample_count = lengths[0]
    if not isinstance(takeover_index, int) or isinstance(takeover_index, bool):
        raise ValueError("takeover_index must be an integer")
    if not 0 <= takeover_index < sample_count:
        raise ValueError("takeover_index is outside the recorded samples")

    def sliced(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: sliced(child) for key, child in value.items()}
        return np.array(np.asarray(value)[takeover_index:], copy=True)

    def copied(value: Any) -> Any:
        if isinstance(value, Mapping):
            if not value:
                raise ValueError("initial_state mappings must not be empty")
            return {key: copied(child) for key, child in value.items()}
        array = np.asarray(value)
        if array.ndim < 1 or array.shape[0] != 1:
            raise ValueError("initial_state leaves must have leading dimension one")
        return np.array(array, copy=True)

    result = sliced(sample_trees)
    result["initial_state"] = copied(initial_state)
    result["num_samples"] = sample_count - takeover_index
    return result
