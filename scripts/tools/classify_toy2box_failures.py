#!/usr/bin/env python3
"""Offline Toy2Box milestone calibration and rollout classification.

The classifier is deliberately simulator-free.  Thresholds are calibrated from
the immutable successful source demonstrations and serialized before they are
used on policy rollouts.  Classifications are operational measurements, not
manual ground-truth labels.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = 2
REQUIRED_ARRAYS = {
    "eef_pos": (3,),
    "eef_quat": (4,),
    "gripper_pos": None,
    "action": None,
    "truck_pos": (3,),
    "truck_quat": (4,),
    "box_pos": (3,),
    "box_quat": (4,),
    "success_value": (),
}
STAGES = ("approach", "grasp", "transport", "placement", "success")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_json_ready(value), stream, indent=2, sort_keys=True)
        stream.write("\n")


def _percentiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {}
    keys = ("min", "median", "p90", "p95", "p99", "max")
    points = (0, 50, 90, 95, 99, 100)
    return dict(zip(keys, np.percentile(array, points).tolist(), strict=True))


def _round_up(value: float, quantum: float) -> float:
    return float(math.ceil((value - 1e-12) / quantum) * quantum)


def _first_run(mask: np.ndarray, length: int, start: int = 0) -> int | None:
    """Return the first index of ``length`` consecutive true values."""
    if length <= 0:
        raise ValueError("run length must be positive")
    count = 0
    for index in range(max(0, start), len(mask)):
        count = count + 1 if bool(mask[index]) else 0
        if count >= length:
            return index - length + 1
    return None


def _max_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _quat_x_heading_deg(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """XY heading of a quaternion's body-x axis, for wxyz quaternions."""
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    axis_x_x = 1.0 - 2.0 * (y * y + z * z)
    axis_x_y = 2.0 * (x * y + w * z)
    return np.degrees(np.arctan2(axis_x_y, axis_x_x))


def folded_heading_error_deg(eef_quat: np.ndarray, truck_quat: np.ndarray) -> tuple[float, float]:
    """Return signed and absolute heading error modulo parallel-gripper 180° symmetry."""
    difference = float(_quat_x_heading_deg(eef_quat) - _quat_x_heading_deg(truck_quat))
    signed = (difference + 90.0) % 180.0 - 90.0
    return signed, abs(signed)


def _source_success_mask(truck: np.ndarray, box: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """Reproduce the checked-in environment's actual Toy2Box success term."""
    xy = np.linalg.norm(truck[:, :2] - box[:, :2], axis=1)
    height = np.abs(truck[:, 2] - box[:, 2])
    opened = np.all(np.abs(np.abs(gripper[:, :2]) - 0.994) < 0.2, axis=1)
    return (xy < 0.10) & ((height - 0.06) < 0.04) & opened


def _source_rows(dataset: Path) -> list[dict[str, Any]]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised by CLI environment
        raise RuntimeError("calibration and dataset-summary require h5py") from exc

    rows: list[dict[str, Any]] = []
    with h5py.File(dataset, "r") as hdf:
        names = sorted(hdf["data"], key=lambda name: int(name.rsplit("_", 1)[1]))
        for name in names:
            demo = hdf["data"][name]
            obs = demo["obs"]
            action = np.asarray(demo["actions"], dtype=np.float64)
            eef = np.asarray(obs["eef_pos"], dtype=np.float64)
            eef_quat = np.asarray(obs["eef_quat"], dtype=np.float64)
            truck = np.asarray(obs["toy_truck_positions"], dtype=np.float64)
            truck_quat = np.asarray(obs["toy_truck_orientations"], dtype=np.float64)
            box = np.asarray(obs["box_positions"], dtype=np.float64)
            gripper = np.asarray(obs["gripper_pos"], dtype=np.float64)
            close_indices = np.flatnonzero(action[:, -1] < 0.0)
            if not close_indices.size:
                raise ValueError(f"{name} has no close command")
            close = int(close_indices[0])
            release_indices = np.flatnonzero((np.arange(len(action)) > close) & (action[:, -1] > 0.0))
            release = int(release_indices[0]) if release_indices.size else len(action)
            baseline_z = float(np.median(truck[: min(30, len(truck)), 2]))
            lifted = truck[:, 2] >= baseline_z + 0.03
            lift_indices = np.flatnonzero(lifted & (np.arange(len(action)) >= close))
            if not lift_indices.size:
                raise ValueError(f"{name} never reaches the specified 3 cm lift")
            lift = int(lift_indices[0])
            distance = np.linalg.norm(eef - truck, axis=1)
            open_before_close = (np.arange(len(action)) < close) & (action[:, -1] >= 0.0)
            held = lifted & (np.arange(len(action)) >= close) & (np.arange(len(action)) < release)
            success = _source_success_mask(truck, box, gripper)
            success_indices = np.flatnonzero(success)
            if not success_indices.size:
                raise ValueError(f"{name} does not reproduce environment success")
            success_index = int(success_indices[0])
            window = slice(lift, min(lift + 10, len(action)))
            relative_span = np.ptp(eef[window] - truck[window], axis=0)
            delta_eef = np.linalg.norm(np.diff(eef, axis=0), axis=1)
            low_action = np.linalg.norm(action[1:, :6], axis=1) <= 0.005
            low_motion = delta_eef <= 0.0005
            signed_heading, absolute_heading = folded_heading_error_deg(eef_quat[close], truck_quat[close])
            raw_relative_heading = float(
                (_quat_x_heading_deg(eef_quat[close]) - _quat_x_heading_deg(truck_quat[close]) + 180.0)
                % 360.0
                - 180.0
            )
            rows.append(
                {
                    "name": name,
                    "steps": len(action),
                    "close_index": close,
                    "lift_index": lift,
                    "success_index": success_index,
                    "baseline_z": baseline_z,
                    "baseline_z_range_30": float(np.ptp(truck[: min(30, len(truck)), 2])),
                    "min_open_distance": float(np.min(distance[open_before_close])),
                    "close_distance": float(distance[close]),
                    "lift_distance": float(distance[lift]),
                    "relative_span_10_max_axis": float(np.max(relative_span)),
                    "min_held_box_xy": float(np.min(np.linalg.norm(truck[held, :2] - box[held, :2], axis=1))),
                    "success_box_xy": float(np.linalg.norm(truck[success_index, :2] - box[success_index, :2])),
                    "max_low_action_motion_run": _max_run(low_action & low_motion),
                    "close_heading_error_signed_deg": signed_heading,
                    "close_heading_error_abs_deg": absolute_heading,
                    "close_relative_heading_deg": raw_relative_heading,
                    "close_grasp_mode_deg": 0 if abs(raw_relative_heading) <= 90.0 else 180,
                    "close_offset_xyz": (eef[close] - truck[close]).tolist(),
                }
            )
    return rows


def calibrate(dataset: Path) -> dict[str, Any]:
    rows = _source_rows(dataset)
    max_open_distance = max(row["min_open_distance"] for row in rows)
    max_lift_distance = max(row["lift_distance"] for row in rows)
    max_relative_span = max(row["relative_span_10_max_axis"] for row in rows)
    max_stationary_run = max(row["max_low_action_motion_run"] for row in rows)
    max_held_xy = max(row["min_held_box_xy"] for row in rows)

    approach_distance = _round_up(max_open_distance + 0.001, 0.005)
    grasp_follow_distance = _round_up(max_lift_distance + 0.001, 0.005)
    grasp_relative_span = _round_up(max_relative_span + 0.001, 0.005)
    stall_window = int(math.ceil((max_stationary_run + 10) / 10.0) * 10)
    transport_allowance = max(0.02, _round_up(max_held_xy, 0.01))

    evidence_fields = {
        "min_open_distance_m": _percentiles(row["min_open_distance"] for row in rows),
        "close_distance_m": _percentiles(row["close_distance"] for row in rows),
        "lift_distance_m": _percentiles(row["lift_distance"] for row in rows),
        "relative_span_first_10_lifted_steps_m": _percentiles(
            row["relative_span_10_max_axis"] for row in rows
        ),
        "min_held_box_xy_m": _percentiles(row["min_held_box_xy"] for row in rows),
        "success_box_xy_m": _percentiles(row["success_box_xy"] for row in rows),
        "baseline_z_range_first_30_steps_m": _percentiles(row["baseline_z_range_30"] for row in rows),
        "low_action_low_motion_max_run_steps": _percentiles(row["max_low_action_motion_run"] for row in rows),
        "close_heading_error_mod180_abs_deg": _percentiles(
            row["close_heading_error_abs_deg"] for row in rows
        ),
    }
    thresholds = {
        "baseline_steps": 30,
        "approach_distance_m": approach_distance,
        "close_action_threshold": 0.0,
        "lift_height_m": 0.03,
        "grasp_consecutive_steps": 10,
        "grasp_follow_distance_m": grasp_follow_distance,
        "grasp_relative_span_m": grasp_relative_span,
        "transport_xy_threshold_m": 0.10 + transport_allowance,
        "environment_success_xy_threshold_m": 0.10,
        "environment_success_height_diff_m": 0.06,
        "environment_success_height_threshold_m": 0.04,
        "environment_gripper_open_target": 0.994,
        "environment_gripper_open_tolerance": 0.2,
        "stall_action_norm_threshold": 0.005,
        "stall_eef_step_m": 0.0005,
        "stall_consecutive_steps": stall_window,
        "timeout_horizon_steps": 800,
        "initial_truck_x_edges_m": [0.45, 0.55, 0.65, 0.75, 0.85],
        "initial_truck_y_edges_m": [-0.30, -0.25, -0.20, -0.15],
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "toy2box_stage_classifier_thresholds",
        "source_dataset": str(dataset.resolve()),
        "source_demo_count": len(rows),
        "timestep_semantics": "source HDF observations aligned with actions at the same index; rollout NPZ may be post-action",
        "thresholds": thresholds,
        "selection_rules": {
            "baseline_steps": "1.5 s at the recorded 20 Hz; source truck-z range over this window is reported below",
            "approach_distance_m": "ceil((source max minimum open-command EEF/truck distance + 1 mm) / 5 mm) * 5 mm",
            "lift_height_m": "fixed by the pre-registered milestone definition: settled z + 3 cm",
            "grasp_consecutive_steps": "10 steps (0.5 s at 20 Hz), matching the source recorder's continuous success confirmation count",
            "grasp_follow_distance_m": "ceil((source max separation at first 3 cm lift + 1 mm) / 5 mm) * 5 mm",
            "grasp_relative_span_m": "ceil((source max EEF-minus-truck span over first 10 lifted steps + 1 mm) / 5 mm) * 5 mm",
            "transport_xy_threshold_m": "environment 10 cm success XY boundary plus max(2 cm, source held-transport envelope rounded to 1 cm)",
            "stall_thresholds": "low action <=0.005 and EEF step <=0.5 mm; duration is source maximum joint run + >=10 steps, rounded to 10",
            "timeout_horizon_steps": "assignment evaluation horizon",
            "initial_location_bins": "fixed 10 cm x and 5 cm y bins covering source/configured reset support; values outside edge ranges get explicit underflow/overflow bins",
        },
        "transport_proxy": {
            "basis": "circular XY proxy: checked-in environment success threshold 0.10 m plus a source-derived 0.02 m allowance",
            "physical_box_opening_verified": False,
            "asset_inspection": (
                "A cached local USD wrapper/mesh candidate was inspected without Kit, but it exposes no semantic opening "
                "dimensions and its scan mesh/unit/xform composition does not justify a physical aperture measurement."
            ),
            "limitation": "The 0.12 m transport radius is intentionally approximate and must not be reported as measured box geometry.",
        },
        "numeric_evidence": evidence_fields,
        "uncertainty": [
            "The 40 demonstrations are successful scripted-agent trajectories, not independent manual ground truth.",
            "Thresholds describe this controller/dataset at 20 Hz and may not transfer to another robot, control rate, or pose convention.",
            "Approach uses truck-center distance as the observable grasp-point proxy; close offsets are reported separately.",
            "Transport uses the checked-in success-region XY boundary with a positive empirical allowance, not measured box mesh dimensions.",
            "Stall is a conservative low-command/low-motion operational flag and does not detect every oscillatory or semantic stall.",
        ],
    }
    return summary


def load_thresholds(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("kind") != "toy2box_stage_classifier_thresholds":
        raise ValueError("not a Toy2Box stage-classifier threshold document")
    return document["thresholds"]


def _load_trial(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    missing = sorted(set(REQUIRED_ARRAYS) - set(arrays))
    if missing:
        raise ValueError(f"{path}: missing arrays: {', '.join(missing)}")
    length = len(arrays["eef_pos"])
    if length == 0:
        raise ValueError(f"{path}: empty trial")
    for name, suffix in REQUIRED_ARRAYS.items():
        value = arrays[name]
        if value.shape[0] != length:
            raise ValueError(f"{path}: {name} length {value.shape[0]} != {length}")
        if suffix is not None and value.shape[1:] != suffix:
            raise ValueError(f"{path}: {name} shape {value.shape}, expected [T,{','.join(map(str, suffix))}]")
    if arrays["gripper_pos"].ndim != 2 or arrays["gripper_pos"].shape[1] < 2:
        raise ValueError(f"{path}: gripper_pos must be [T,>=2]")
    if arrays["action"].ndim != 2 or arrays["action"].shape[1] < 7:
        raise ValueError(f"{path}: action must be [T,>=7]")
    for name in REQUIRED_ARRAYS:
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{path}: {name} contains non-finite values")
    return arrays


def _grasp_index(
    lifted: np.ndarray,
    distance: np.ndarray,
    relative: np.ndarray,
    close_index: int,
    thresholds: dict[str, Any],
) -> int | None:
    k = int(thresholds["grasp_consecutive_steps"])
    eligible = lifted & (distance <= float(thresholds["grasp_follow_distance_m"]))
    for start in range(close_index, len(eligible) - k + 1):
        stop = start + k
        if eligible[start:stop].all() and np.ptp(relative[start:stop], axis=0).max() <= float(
            thresholds["grasp_relative_span_m"]
        ):
            return start
    return None


def _reset_initial(arrays: dict[str, np.ndarray], name: str, fallback: np.ndarray) -> tuple[np.ndarray, str]:
    key = f"reset_initial_{name}"
    if key in arrays:
        value = np.asarray(arrays[key], dtype=np.float64).reshape(-1)
        if value.shape != fallback.shape or not np.isfinite(value).all():
            raise ValueError(f"{key} shape/value is invalid")
        return value, "observer_reset_initial"
    return np.asarray(fallback, dtype=np.float64), "post_first_step_fallback"


def _close_onsets(close_command: np.ndarray) -> np.ndarray:
    previous = np.r_[False, close_command[:-1]]
    return np.flatnonzero(close_command & ~previous)


def classify_trial(arrays: dict[str, np.ndarray], thresholds: dict[str, Any], *, path: str = "") -> dict[str, Any]:
    eef = np.asarray(arrays["eef_pos"], dtype=np.float64)
    truck = np.asarray(arrays["truck_pos"], dtype=np.float64)
    box = np.asarray(arrays["box_pos"], dtype=np.float64)
    action = np.asarray(arrays["action"], dtype=np.float64)
    success = np.asarray(arrays["success_value"], dtype=bool).reshape(-1)
    distance = np.linalg.norm(eef - truck, axis=1)
    relative = eef - truck
    baseline_steps = min(int(thresholds["baseline_steps"]), len(truck))
    settled_z = float(np.median(truck[:baseline_steps, 2]))
    close_command = action[:, -1] < float(thresholds["close_action_threshold"])
    close_onsets = _close_onsets(close_command)
    gripper = np.asarray(arrays["gripper_pos"], dtype=np.float64)
    physically_open = np.all(
        np.abs(np.abs(gripper[:, :2]) - float(thresholds["environment_gripper_open_target"]))
        < float(thresholds["environment_gripper_open_tolerance"]),
        axis=1,
    )
    approach_candidates = np.flatnonzero(
        physically_open & (distance <= float(thresholds["approach_distance_m"]))
    )
    paired_closes = [int(index) for index in close_onsets if np.any(approach_candidates < index)]
    close_index = paired_closes[0] if paired_closes else None
    if close_index is not None:
        approach_index = int(approach_candidates[approach_candidates < close_index][-1])
    elif close_onsets.size == 0 and approach_candidates.size:
        approach_index = int(approach_candidates[0])
    else:
        approach_index = None
    lifted = truck[:, 2] >= settled_z + float(thresholds["lift_height_m"])
    grasp_index = None
    if close_index is not None:
        for candidate_close in paired_closes:
            candidate_grasp = _grasp_index(
                lifted & close_command, distance, relative, candidate_close, thresholds
            )
            if candidate_grasp is not None:
                grasp_index = candidate_grasp
                eligible_closes = [index for index in paired_closes if index <= grasp_index]
                close_index = eligible_closes[-1]
                approach_index = int(approach_candidates[approach_candidates < close_index][-1])
                break
    transport_index = None
    if grasp_index is not None:
        xy = np.linalg.norm(truck[:, :2] - box[:, :2], axis=1)
        indices = np.flatnonzero((np.arange(len(xy)) >= grasp_index) & lifted & (xy <= thresholds["transport_xy_threshold_m"]))
        transport_index = int(indices[0]) if indices.size else None
    placement_indices = np.flatnonzero(success)
    placement_index = int(placement_indices[0]) if placement_indices.size else None

    proxy_inconsistencies = []
    for name, index in (
        ("approach", approach_index),
        ("grasp", grasp_index),
        ("transport", transport_index),
    ):
        if index is None:
            proxy_inconsistencies.append(name)
        elif placement_index is not None and index > placement_index:
            proxy_inconsistencies.append(f"{name}_after_actual_success")
    if placement_index is not None:
        stage = "success"
    elif approach_index is None:
        stage = "approach"
    elif grasp_index is None:
        stage = "grasp"
    elif transport_index is None:
        stage = "transport"
    else:
        stage = "placement"

    stall_mask = np.zeros(len(action), dtype=bool)
    if len(action) > 1:
        action_norm = np.linalg.norm(action[1:, :6], axis=1)
        eef_step = np.linalg.norm(np.diff(eef, axis=0), axis=1)
        stall_mask[1:] = (action_norm <= thresholds["stall_action_norm_threshold"]) & (
            eef_step <= thresholds["stall_eef_step_m"]
        )
    stall_run = _max_run(stall_mask)
    truncated = bool(np.any(np.asarray(arrays.get("truncated", []), dtype=bool)))
    timeout = truncated or len(action) >= int(thresholds["timeout_horizon_steps"])

    diagnostic_close_index = close_index
    diagnostic_close_basis = "selected chronological close after open approach"
    if diagnostic_close_index is None and close_onsets.size:
        diagnostic_close_index = int(close_onsets[0])
        diagnostic_close_basis = "first close onset; no qualifying chronological approach-close pair"
    initial_truck_pos, truck_pos_basis = _reset_initial(arrays, "truck_pos", truck[0])
    initial_truck_quat, truck_quat_basis = _reset_initial(arrays, "truck_quat", arrays["truck_quat"][0])
    initial_box_pos, box_pos_basis = _reset_initial(arrays, "box_pos", box[0])
    diagnostics: dict[str, Any] = {
        "close_index": close_index,
        "diagnostic_close_index": diagnostic_close_index,
        "diagnostic_close_basis": diagnostic_close_basis,
        "close_heading_error_signed_mod180_deg": None,
        "close_heading_error_abs_mod180_deg": None,
        "close_offset_xyz_m": None,
        "close_offset_norm_m": None,
        "initial_truck_position": initial_truck_pos.tolist(),
        "initial_truck_heading_deg": float(_quat_x_heading_deg(initial_truck_quat)),
        "initial_box_position": initial_box_pos.tolist(),
        "initial_state_basis": {
            "truck_pos": truck_pos_basis,
            "truck_quat": truck_quat_basis,
            "box_pos": box_pos_basis,
        },
        "stall": stall_run >= int(thresholds["stall_consecutive_steps"]),
        "max_stall_run_steps": stall_run,
        "timeout": timeout,
        "timeout_basis": "truncated" if truncated else ("horizon_length" if timeout else "none"),
    }
    if diagnostic_close_index is not None:
        signed, absolute = folded_heading_error_deg(
            arrays["eef_quat"][diagnostic_close_index], arrays["truck_quat"][diagnostic_close_index]
        )
        offset = relative[diagnostic_close_index]
        diagnostics.update(
            {
                "close_heading_error_signed_mod180_deg": signed,
                "close_heading_error_abs_mod180_deg": absolute,
                "close_offset_xyz_m": offset.tolist(),
                "close_offset_norm_m": float(np.linalg.norm(offset)),
            }
        )
    return {
        "path": path,
        "num_steps": len(action),
        "stage": stage,
        "success": placement_index is not None,
        "actual_success": placement_index is not None,
        "prior_milestone_proxy_inconsistencies": proxy_inconsistencies if placement_index is not None else [],
        "milestone_indices": {
            "approach": approach_index,
            "grasp": grasp_index,
            "transport": transport_index,
            "placement": placement_index,
        },
        "diagnostics": diagnostics,
    }


def _failure_diagnostics(trials: list[dict[str, Any]]) -> dict[str, Any]:
    heading = [
        trial["diagnostics"]["close_heading_error_abs_mod180_deg"]
        for trial in trials
        if trial["diagnostics"]["close_heading_error_abs_mod180_deg"] is not None
    ]
    offsets = [
        trial["diagnostics"]["close_offset_norm_m"]
        for trial in trials
        if trial["diagnostics"]["close_offset_norm_m"] is not None
    ]
    return {
        "count": len(trials),
        "paths": [trial["path"] for trial in trials],
        "close_heading_error_abs_mod180_deg": _percentiles(heading),
        "close_offset_norm_m": _percentiles(offsets),
        "stall_count": sum(bool(trial["diagnostics"]["stall"]) for trial in trials),
        "stall_rate": sum(bool(trial["diagnostics"]["stall"]) for trial in trials) / len(trials) if trials else None,
        "timeout_count": sum(bool(trial["diagnostics"]["timeout"]) for trial in trials),
        "timeout_rate": sum(bool(trial["diagnostics"]["timeout"]) for trial in trials) / len(trials) if trials else None,
    }


def _axis_bins(trials: list[dict[str, Any]], axis: int, edges: list[float]) -> list[dict[str, Any]]:
    ranges: list[tuple[float | None, float | None]] = [(None, edges[0])]
    ranges.extend(zip(edges[:-1], edges[1:], strict=True))
    ranges.append((edges[-1], None))
    output = []
    for low, high in ranges:
        selected = []
        for trial in trials:
            value = float(trial["diagnostics"]["initial_truck_position"][axis])
            if (low is None and value < high) or (high is None and value >= low) or (
                low is not None and high is not None and low <= value < high
            ):
                selected.append(trial)
        failures = sum(not bool(trial["actual_success"]) for trial in selected)
        output.append(
            {
                "range_m": [low, high],
                "attempts": len(selected),
                "failures": failures,
                "failure_rate": failures / len(selected) if selected else None,
            }
        )
    return output


def summarize_trials(trials: list[dict[str, Any]], thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    counts = Counter(trial["stage"] for trial in trials)
    failures = [trial for trial in trials if not bool(trial["actual_success"])]
    grasp_failures = [trial for trial in failures if trial["stage"] == "grasp"]
    actual_successes = sum(bool(trial["actual_success"]) for trial in trials)
    yaw_bins = []
    for low in range(-180, 180, 45):
        selected = []
        for trial in trials:
            yaw = (float(trial["diagnostics"]["initial_truck_heading_deg"]) + 180.0) % 360.0 - 180.0
            if low <= yaw < low + 45:
                selected.append(trial)
        selected_counts = Counter(trial["stage"] for trial in selected)
        selected_successes = sum(bool(trial["actual_success"]) for trial in selected)
        yaw_bins.append(
            {
                "range_deg": [low, low + 45],
                "attempts": len(selected),
                "successes": selected_successes,
                "success_rate": selected_successes / len(selected) if selected else None,
                "stage_counts": {stage: selected_counts[stage] for stage in STAGES},
            }
        )
    summary = {
        "attempts": len(trials),
        "successes": actual_successes,
        "success_rate": actual_successes / len(trials) if trials else None,
        "stage_counts": {stage: counts[stage] for stage in STAGES},
        "stage_rates": {stage: counts[stage] / len(trials) if trials else None for stage in STAGES},
        "failure_paths_by_stage": {
            stage: [trial["path"] for trial in failures if trial["stage"] == stage] for stage in STAGES[:-1]
        },
        "all_failure_diagnostics": _failure_diagnostics(failures),
        "grasp_failure_diagnostics": _failure_diagnostics(grasp_failures),
        "actual_success_with_proxy_inconsistency_count": sum(
            bool(trial["prior_milestone_proxy_inconsistencies"]) for trial in trials if trial["actual_success"]
        ),
        "initial_truck_yaw_45deg": yaw_bins,
        "initial_state_basis_counts": dict(
            Counter(trial["diagnostics"]["initial_state_basis"]["truck_pos"] for trial in trials)
        ),
    }
    if thresholds is not None:
        summary["initial_truck_x_bins_m"] = _axis_bins(
            trials, 0, [float(value) for value in thresholds["initial_truck_x_edges_m"]]
        )
        summary["initial_truck_y_bins_m"] = _axis_bins(
            trials, 1, [float(value) for value in thresholds["initial_truck_y_edges_m"]]
        )
        summary["initial_location_bin_contract"] = {
            "edge_rule": "left-inclusive, right-exclusive",
            "out_of_range": "explicit first underflow and last overflow bins",
            "position_basis": "observer reset_initial_truck_pos when present; otherwise labeled post-first-step fallback",
        }
    return summary


def aggregate_classifications(documents: list[dict[str, Any]]) -> dict[str, Any]:
    for document in documents:
        if document.get("kind") != "toy2box_stage_classification":
            raise ValueError("aggregate inputs must be Toy2Box stage-classification JSON files")
    pooled = [trial for document in documents for trial in document["trials"]]
    threshold_sets = [document.get("threshold_values") for document in documents]
    pooled_thresholds = threshold_sets[0] if threshold_sets and all(item == threshold_sets[0] for item in threshold_sets) else None
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "toy2box_stage_classification_aggregate",
        "by_label": {document["label"]: document["summary"] for document in documents},
        "pooled": summarize_trials(pooled, pooled_thresholds),
        "source_files": [document.get("source_file") for document in documents],
        "note": "Pooled trials are an unpaired descriptive aggregation across model conditions.",
    }


def _dataset_axis_bins(
    positions: np.ndarray, modes: np.ndarray, axis: int, edges: list[float]
) -> list[dict[str, Any]]:
    ranges: list[tuple[float | None, float | None]] = [(None, edges[0])]
    ranges.extend(zip(edges[:-1], edges[1:], strict=True))
    ranges.append((edges[-1], None))
    output = []
    for low, high in ranges:
        values = positions[:, axis]
        selected = (
            (values < high)
            if low is None
            else ((values >= low) if high is None else ((values >= low) & (values < high)))
        )
        output.append(
            {
                "range_m": [low, high],
                "count": int(selected.sum()),
                "grasp_mode_0_count": int(np.sum(modes[selected] == 0)),
                "grasp_mode_180_count": int(np.sum(modes[selected] == 180)),
            }
        )
    return output


def dataset_summary(dataset: Path, thresholds: dict[str, Any]) -> dict[str, Any]:
    rows = _source_rows(dataset)
    yaw_edges = np.arange(-180.0, 181.0, 45.0)
    bins: list[dict[str, Any]] = []
    truck_yaw = []
    truck_position = []
    signed_errors = np.asarray([row["close_heading_error_signed_deg"] for row in rows])
    raw_relative = np.asarray([row["close_relative_heading_deg"] for row in rows])
    modes = np.asarray([row["close_grasp_mode_deg"] for row in rows])
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("dataset-summary requires h5py") from exc
    with h5py.File(dataset, "r") as hdf:
        names = sorted(hdf["data"], key=lambda name: int(name.rsplit("_", 1)[1]))
        for name in names:
            truck_yaw.append(float(_quat_x_heading_deg(hdf["data"][name]["obs"]["toy_truck_orientations"][0])))
            truck_position.append(np.asarray(hdf["data"][name]["obs"]["toy_truck_positions"][0], dtype=np.float64))
    truck_yaw_array = (np.asarray(truck_yaw) + 180.0) % 360.0 - 180.0
    truck_position_array = np.asarray(truck_position)
    error_array = signed_errors
    # A parallel gripper is geometrically symmetric modulo 180 degrees, but the
    # raw relative heading still exposes which of the two controller modes was used.
    for low, high in zip(yaw_edges[:-1], yaw_edges[1:], strict=True):
        selected = (truck_yaw_array >= low) & (truck_yaw_array < high)
        values = error_array[selected]
        bins.append(
            {
                "range_deg": [float(low), float(high)],
                "count": int(selected.sum()),
                "folded_heading_error_abs_deg": _percentiles(np.abs(values)),
                "grasp_mode_0_count": int(np.sum(modes[selected] == 0)),
                "grasp_mode_180_count": int(np.sum(modes[selected] == 180)),
                "raw_relative_heading_deg": _percentiles(raw_relative[selected]),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "toy2box_dataset_grasp_summary",
        "dataset": str(dataset.resolve()),
        "demos": len(rows),
        "truck_start_yaw_45deg": bins,
        "truck_start_x_bins_m": _dataset_axis_bins(
            truck_position_array, modes, 0, [float(value) for value in thresholds["initial_truck_x_edges_m"]]
        ),
        "truck_start_y_bins_m": _dataset_axis_bins(
            truck_position_array, modes, 1, [float(value) for value in thresholds["initial_truck_y_edges_m"]]
        ),
        "initial_location_bin_contract": {
            "edge_rule": "left-inclusive, right-exclusive",
            "out_of_range": "explicit first underflow and last overflow bins",
            "edges_source": "frozen stage-classifier threshold document",
            "state_basis": (
                "HDF obs/toy_truck_positions index 0 (first recorded pre-action observation); "
                "unlike observer rollout NPZ, source/generated HDF has no separate reset_initial_truck_pos"
            ),
        },
        "close_heading_error_mod180_abs_deg": _percentiles(np.abs(error_array)),
        "close_offset_norm_m": _percentiles(np.linalg.norm([row["close_offset_xyz"] for row in rows], axis=1)),
        "note": "Operational HDF measurements; no manual labels or ground-truth grasp-mode claims.",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibrate_parser = subparsers.add_parser("calibrate", help="freeze thresholds from successful source HDF5")
    calibrate_parser.add_argument("--source", type=Path, required=True)
    calibrate_parser.add_argument("--output", type=Path, required=True)
    summary_parser = subparsers.add_parser("dataset-summary", help="summarize HDF5 yaw and grasp offsets")
    summary_parser.add_argument("--dataset", type=Path, required=True)
    summary_parser.add_argument("--thresholds", type=Path, required=True)
    summary_parser.add_argument("--output", type=Path, required=True)
    classify_parser = subparsers.add_parser("classify", help="classify one or more observer NPZ trials")
    classify_parser.add_argument("--thresholds", type=Path, required=True)
    classify_parser.add_argument("--input", type=Path, nargs="+", required=True)
    classify_parser.add_argument("--output", type=Path, required=True)
    classify_parser.add_argument("--label", default="unlabeled")
    aggregate_parser = subparsers.add_parser("aggregate", help="pool completed per-seed classification JSON files")
    aggregate_parser.add_argument("--input", type=Path, nargs="+", required=True)
    aggregate_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "calibrate":
        result = calibrate(args.source)
    elif args.command == "dataset-summary":
        result = dataset_summary(args.dataset, load_thresholds(args.thresholds))
    elif args.command == "classify":
        thresholds = load_thresholds(args.thresholds)
        paths: list[Path] = []
        for item in args.input:
            paths.extend(sorted(item.glob("*.npz")) if item.is_dir() else [item])
        if not paths:
            raise ValueError("no NPZ trials found")
        trials = [classify_trial(_load_trial(path), thresholds, path=str(path.resolve())) for path in paths]
        result = {
            "schema_version": SCHEMA_VERSION,
            "kind": "toy2box_stage_classification",
            "label": args.label,
            "thresholds_file": str(args.thresholds.resolve()),
            "threshold_values": thresholds,
            "summary": summarize_trials(trials, thresholds),
            "trials": trials,
            "limitations": [
                "Stage assignments are threshold-based operational classifications, not manual ground truth.",
                "Actual success_value overrides proxy milestone misses; those misses remain explicit inconsistencies.",
                "Initial-layout summaries prefer reset_initial_* and label post-first-step fallback when reset arrays are absent.",
                "Timeout inferred from length requires the frozen 800-step evaluation horizon unless truncated is recorded.",
            ],
        }
    else:
        documents = []
        for path in args.input:
            with path.open(encoding="utf-8") as stream:
                document = json.load(stream)
            document["source_file"] = str(path.resolve())
            documents.append(document)
        result = aggregate_classifications(documents)
    _write_new_json(args.output, result)
    print(json.dumps(_json_ready(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
