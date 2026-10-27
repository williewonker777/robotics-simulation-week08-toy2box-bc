"""Pure-NumPy trajectory smoothness metrics for Toy2Box demonstrations.

Physical derivatives exclude reset-settling intervals, retaining the preceding
context needed to measure the first controlled response. Action flip metrics use
the complete action sequence so reset settling cannot hide command chatter.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


DEFAULT_LIMITS = {
    "eef_velocity_norm_max": 0.25,
    "eef_acceleration_norm_p95": 4.2,
    "eef_acceleration_norm_max": 2.0,
    "eef_jerk_norm_p95": 95.0,
    "eef_jerk_norm_max": 60.0,
    "angular_velocity_norm_p95": 0.32,
    "angular_velocity_norm_max": 0.5,
    "angular_acceleration_norm_p95": 4.7,
    "angular_acceleration_norm_max": 5.0,
    "arm_joint_acceleration_max_component_p95": 17.0,
    "arm_joint_acceleration_max_component_max": 15.0,
    "action_immediate_flip_count": 0,
    "visible_rapid_eef_reversal_count": 3,
    "visible_rapid_arm_joint_reversal_count": 6,
}


def _as_trajectory(name: str, value: np.ndarray, width: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or (width is not None and array.shape[1] != width):
        suffix = f" with shape (T, {width})" if width is not None else " with shape (T, N)"
        raise ValueError(f"{name} must be a two-dimensional array{suffix}; got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _summary(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    return float(np.percentile(values, 95)), float(np.max(values))


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply batches of wxyz quaternions."""
    lw, lv = left[..., :1], left[..., 1:]
    rw, rv = right[..., :1], right[..., 1:]
    scalar = lw * rw - np.sum(lv * rv, axis=-1, keepdims=True)
    vector = lw * rv + rw * lv + np.cross(lv, rv)
    return np.concatenate((scalar, vector), axis=-1)


def _angular_velocity(quaternions: np.ndarray, dt: float) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("eef_quaternions_wxyz contains a zero-norm quaternion")
    quaternions = quaternions / norms[:, None]

    # q and -q encode the same orientation.  Choose adjacent signs with a
    # non-negative dot product, yielding the shortest relative rotation.
    current = quaternions[:-1]
    following = quaternions[1:].copy()
    following[np.sum(current * following, axis=1) < 0.0] *= -1.0
    inverse = current.copy()
    inverse[:, 1:] *= -1.0
    relative = _quaternion_multiply(inverse, following)
    relative /= np.linalg.norm(relative, axis=1)[:, None]
    relative[relative[:, 0] < 0.0] *= -1.0

    vector_norm = np.linalg.norm(relative[:, 1:], axis=1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(relative[:, 0], 0.0, 1.0))
    scale = np.divide(angle, vector_norm, out=np.full_like(angle, 2.0), where=vector_norm > 1e-12)
    return relative[:, 1:] * scale[:, None] / dt


def _rapid_reversals(
    velocity: np.ndarray, threshold: float, max_gap: int, *, inclusive: bool = False
) -> int:
    """Count per-axis significant sign reversals separated by at most max_gap samples.

    Sub-threshold samples are ignored rather than treated as a sign.  Raw
    velocity diagnostics use a strict floor; visible displacement metrics set
    ``inclusive`` so motion exactly at their stated threshold is significant.
    """
    count = 0
    for component in velocity.T:
        amplitude = np.abs(component)
        significant = np.flatnonzero(amplitude >= threshold if inclusive else amplitude > threshold)
        if significant.size < 2:
            continue
        gaps = np.diff(significant)
        signs = np.sign(component[significant])
        count += int(np.count_nonzero((gaps <= max_gap) & (signs[1:] != signs[:-1])))
    return count


def trajectory_quality(
    actions: np.ndarray,
    eef_positions: np.ndarray,
    eef_quaternions_wxyz: np.ndarray,
    arm_joint_positions: np.ndarray,
    dt: float = 0.05,
    skip: int = 25,
) -> dict:
    """Return JSON-serializable command and physical smoothness metrics.

    Raw rapid-velocity reversals use component amplitudes above 0.004 m/s for
    the EEF and 0.03 rad/s for arm joints and are diagnostic only: those small
    floors also detect sub-millimetre mechanical noise.  Gated visible reversal
    metrics instead require per-step component displacements of at least 1 mm
    or 0.005 rad.  A reversal is rapid when the next significant opposite-sign
    sample occurs within four samples (0.2 seconds with the default ``dt``);
    sub-threshold samples between them are ignored.
    """
    actions = _as_trajectory("actions", actions)
    positions = _as_trajectory("eef_positions", eef_positions, 3)
    quaternions = _as_trajectory("eef_quaternions_wxyz", eef_quaternions_wxyz, 4)
    joints = _as_trajectory("arm_joint_positions", arm_joint_positions)
    lengths = {actions.shape[0], positions.shape[0], quaternions.shape[0], joints.shape[0]}
    if len(lengths) != 1:
        raise ValueError(
            "trajectory arrays must have the same number of samples; got "
            f"actions={len(actions)}, eef_positions={len(positions)}, "
            f"eef_quaternions_wxyz={len(quaternions)}, arm_joint_positions={len(joints)}"
        )
    if actions.shape[1] == 0 or joints.shape[1] == 0:
        raise ValueError("actions and arm_joint_positions must have at least one component")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive; got {dt}")
    if isinstance(skip, bool) or not isinstance(skip, (int, np.integer)) or skip < 0:
        raise ValueError(f"skip must be a non-negative integer; got {skip!r}")
    if len(actions) < skip + 4:
        raise ValueError(f"at least skip + 4 samples are required; got {len(actions)} with skip={skip}")

    # The last action component is the gripper in Toy2Box.  Opening/closing is
    # intentionally excluded from arm-motion chatter counts.
    motion_actions = actions[:, :-1] if actions.shape[1] > 1 else actions
    previous, following = motion_actions[:-1], motion_actions[1:]
    component_flips = (previous * following) < 0.0
    component_sign_reversal_counts = []
    for component in motion_actions.T:
        nonzero_signs = np.sign(component[component != 0.0])
        component_sign_reversal_counts.append(int(np.count_nonzero(np.diff(nonzero_signs))))
    active = np.any(previous != 0.0, axis=1)
    direct_opposite = active & np.all(np.isclose(following, -previous, rtol=0.0, atol=1e-12), axis=1)

    # Differentiate the full trajectory before slicing so the first response
    # after reset settling retains the pre-command state needed by finite
    # differences.  Derivative order N needs N preceding context samples.
    position_deltas = np.diff(positions, axis=0)
    eef_velocity = position_deltas / dt
    eef_acceleration = np.diff(eef_velocity, axis=0) / dt
    eef_jerk = np.diff(eef_acceleration, axis=0) / dt
    angular_velocity = _angular_velocity(quaternions, dt)
    angular_acceleration = np.diff(angular_velocity, axis=0) / dt
    joint_deltas = np.diff(joints, axis=0)
    joint_velocity = joint_deltas / dt
    joint_acceleration = np.diff(joint_velocity, axis=0) / dt

    velocity_start = max(skip - 1, 0)
    acceleration_start = max(skip - 2, 0)
    jerk_start = max(skip - 3, 0)
    position_deltas = position_deltas[velocity_start:]
    joint_deltas = joint_deltas[velocity_start:]
    eef_velocity = eef_velocity[velocity_start:]
    eef_acceleration = eef_acceleration[acceleration_start:]
    eef_jerk = eef_jerk[jerk_start:]
    angular_velocity = angular_velocity[velocity_start:]
    angular_acceleration = angular_acceleration[acceleration_start:]
    joint_velocity = joint_velocity[velocity_start:]
    joint_acceleration = joint_acceleration[acceleration_start:]

    eef_velocity_p95, eef_velocity_max = _summary(np.linalg.norm(eef_velocity, axis=1))
    eef_acceleration_p95, eef_acceleration_max = _summary(np.linalg.norm(eef_acceleration, axis=1))
    eef_jerk_p95, eef_jerk_max = _summary(np.linalg.norm(eef_jerk, axis=1))
    angular_velocity_p95, angular_velocity_max = _summary(np.linalg.norm(angular_velocity, axis=1))
    angular_acceleration_p95, angular_acceleration_max = _summary(np.linalg.norm(angular_acceleration, axis=1))
    joint_velocity_p95, joint_velocity_max = _summary(np.max(np.abs(joint_velocity), axis=1))
    joint_acceleration_p95, joint_acceleration_max = _summary(np.max(np.abs(joint_acceleration), axis=1))

    max_gap = 4
    eef_reversal_threshold = 0.004
    arm_reversal_threshold = 0.03
    visible_eef_delta_threshold = 0.001
    visible_arm_delta_threshold = 0.005
    return {
        "metric_definition_version": 2,
        "samples_total": int(len(actions)),
        "samples_physical_metrics": int(len(actions) - skip),
        "dt_seconds": float(dt),
        "initial_samples_skipped": int(skip),
        "action_direct_opposite_consecutive_count": int(np.count_nonzero(direct_opposite)),
        "action_immediate_flip_count": int(np.count_nonzero(component_flips)),
        "action_nonzero_sign_reversal_counts": component_sign_reversal_counts,
        "eef_velocity_norm_p95": eef_velocity_p95,
        "eef_velocity_norm_max": eef_velocity_max,
        "eef_acceleration_norm_p95": eef_acceleration_p95,
        "eef_acceleration_norm_max": eef_acceleration_max,
        "eef_jerk_norm_p95": eef_jerk_p95,
        "eef_jerk_norm_max": eef_jerk_max,
        "angular_velocity_norm_p95": angular_velocity_p95,
        "angular_velocity_norm_max": angular_velocity_max,
        "angular_acceleration_norm_p95": angular_acceleration_p95,
        "angular_acceleration_norm_max": angular_acceleration_max,
        "arm_joint_velocity_max_component_p95": joint_velocity_p95,
        "arm_joint_velocity_max_component_max": joint_velocity_max,
        "arm_joint_acceleration_max_component_p95": joint_acceleration_p95,
        "arm_joint_acceleration_max_component_max": joint_acceleration_max,
        "rapid_eef_velocity_reversal_count": _rapid_reversals(
            eef_velocity, eef_reversal_threshold, max_gap
        ),
        "rapid_arm_joint_velocity_reversal_count": _rapid_reversals(
            joint_velocity, arm_reversal_threshold, max_gap
        ),
        "visible_rapid_eef_reversal_count": _rapid_reversals(
            position_deltas, visible_eef_delta_threshold, max_gap, inclusive=True
        ),
        "visible_rapid_arm_joint_reversal_count": _rapid_reversals(
            joint_deltas, visible_arm_delta_threshold, max_gap, inclusive=True
        ),
        "rapid_reversal_definition": {
            "eef_component_velocity_threshold_m_s": eef_reversal_threshold,
            "arm_component_velocity_threshold_rad_s": arm_reversal_threshold,
            "raw_velocity_metrics_are_diagnostic_only": True,
            "visible_eef_component_step_threshold_m": visible_eef_delta_threshold,
            "visible_arm_component_step_threshold_rad": visible_arm_delta_threshold,
            "visible_metrics_are_quality_gated": True,
            "maximum_sample_gap": max_gap,
            "maximum_gap_seconds": float(max_gap * dt),
            "subthreshold_samples_ignored": True,
        },
    }


def quality_rejections(metrics: Mapping[str, object], limits: Mapping[str, float] | None = None) -> list[str]:
    """Describe metrics exceeding upper bounds without modifying ``metrics``.

    ``limits`` updates individual defaults, allowing trial-derived tightening
    while keeping a stable set of default safety checks.
    """
    effective_limits = dict(DEFAULT_LIMITS)
    if limits is not None:
        unknown = set(limits) - set(effective_limits)
        if unknown:
            raise ValueError(f"unknown quality limits: {sorted(unknown)}")
        for name, limit in limits.items():
            if not np.isfinite(limit) or limit < 0.0:
                raise ValueError(f"quality limit {name} must be finite and non-negative; got {limit}")
            effective_limits[name] = float(limit)

    rejections = []
    for name, limit in effective_limits.items():
        if name not in metrics:
            raise ValueError(f"required quality metric is missing: {name}")
        value = metrics[name]
        if not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value):
            raise ValueError(f"quality metric {name} must be a finite number; got {value!r}")
        if float(value) > limit:
            rejections.append(f"{name}={value} exceeds {limit}")
    return rejections
