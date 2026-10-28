#!/usr/bin/env python3
"""Recompute the preregistered H4 verdict from public metrics or full artifacts.

The verdict thresholds and numerical functions are a path-portable copy of the
frozen pre-result analysis. ``--recorded-result`` needs only the published H4
metrics; the full mode can recompute those metrics from the generated dataset
and observer trajectories. The contract and baseline default to the canonical
pre-result registration under ``artifacts/h4_registration``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import h5py
import numpy as np
from scipy.stats import norm


HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
REPOSITORY = HERE.parents[2]
DEFAULT_CONTRACT = REPOSITORY / "artifacts/h4_registration/preregistration.json"
DEFAULT_BASELINE = REPOSITORY / "artifacts/h4_registration/baseline_metrics.json"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TOOLS))

from close_precision import rotmat  # noqa: E402
from recovery_metrics import trial as recovery_trial  # noqa: E402
from transport_slip import secured_rows  # noqa: E402
from toy2box_quality import trajectory_quality  # noqa: E402


ARM_JOINT_IDS = [5, 7, 9, 11, 13, 15, 17]


def first_close_index(actions: np.ndarray) -> int | None:
    closed = np.asarray(actions)[:, -1] < 0
    onsets = np.flatnonzero(closed & ~np.concatenate([[False], closed[:-1]]))
    return int(onsets[0]) if onsets.size else None


def truck_frame_height_mm(
    eef_position: np.ndarray, truck_position: np.ndarray, truck_quaternion: np.ndarray
) -> float:
    return float((rotmat(truck_quaternion).T @ (eef_position - truck_position))[2] * 1000.0)


def lift_delay_steps(eef_positions: np.ndarray, close_index: int) -> int | None:
    lifted = np.flatnonzero(
        np.asarray(eef_positions)[close_index:, 2] - eef_positions[close_index, 2] > 0.01
    )
    return int(lifted[0]) if lifted.size else None


def generated_metrics(dataset: Path, *, dt: float = 0.05, skip: int = 0) -> dict[str, Any]:
    heights: list[float] = []
    delays: list[int] = []
    acceleration_p95: list[float] = []
    jerk_p95: list[float] = []
    close_counts: list[int] = []
    with h5py.File(dataset, "r") as h5_file:
        demos = h5_file["data"]
        names = sorted(demos, key=lambda value: int(value.removeprefix("demo_")))
        for name in names:
            demo = demos[name]
            if not bool(demo.attrs.get("success", False)):
                raise ValueError(f"generated cohort contains unsuccessful episode: {name}")
            actions = demo["actions"][:]
            eef = demo["obs/eef_pos"][:]
            eef_quat = demo["obs/eef_quat"][:]
            truck = demo["obs/toy_truck_positions"][:]
            truck_quat = demo["obs/toy_truck_orientations"][:]
            closed = actions[:, -1] < 0
            onsets = np.flatnonzero(closed & ~np.concatenate([[False], closed[:-1]]))
            close_counts.append(int(onsets.size))
            if not onsets.size:
                raise ValueError(f"successful generated episode has no close onset: {name}")
            close = int(onsets[0])
            heights.append(truck_frame_height_mm(eef[close], truck[close], truck_quat[close]))
            delay = lift_delay_steps(eef, close)
            if delay is None:
                raise ValueError(f"successful generated episode never lifts 1 cm after close: {name}")
            delays.append(delay)
            initial = demo["initial_state/articulation/robot/joint_position"][:]
            states = demo["states/articulation/robot/joint_position"][:]
            joints = np.vstack([initial[0], states])[:, ARM_JOINT_IDS][:-1]
            quality = trajectory_quality(actions, eef, eef_quat, joints, dt=dt, skip=skip)
            acceleration_p95.append(float(quality["eef_acceleration_norm_p95"]))
            jerk_p95.append(float(quality["eef_jerk_norm_p95"]))
    return {
        "episodes": len(heights),
        "first_close_height_mm_median": float(np.median(heights)),
        "lift_delay_median_steps": float(np.median(delays)),
        "lift_delay_minmax": [int(min(delays)), int(max(delays))],
        "one_close_onset_each": all(value == 1 for value in close_counts),
        "quality_skip": skip,
        "quality_dt": dt,
        "cohort_median_eef_acceleration_norm_p95": float(np.median(acceleration_p95)),
        "cohort_median_eef_jerk_norm_p95": float(np.median(jerk_p95)),
        "per_episode_filter_applied": False,
    }


def rollout_metrics(npz_directories: Iterable[Path]) -> dict[str, Any]:
    paths = [path for directory in npz_directories for path in sorted(directory.glob("trial_*.npz"))]
    heights: list[float] = []
    no_near_close = 0
    for path in paths:
        recovery = recovery_trial(str(path))
        if not recovery["near_closes"]:
            no_near_close += 1
            continue
        close = int(recovery["near_closes"][0])
        with np.load(path, allow_pickle=True) as episode:
            heights.append(
                truck_frame_height_mm(
                    episode["eef_pos"][close],
                    episode["truck_pos"][close],
                    episode["truck_quat"][close],
                )
            )
    secured = [row for directory in npz_directories for row in secured_rows(str(directory))]
    return {
        "trials": len(paths),
        "first_near_close_count": len(heights),
        "no_near_close_count": no_near_close,
        "first_near_close_height_mm_median": float(np.median(heights)) if heights else None,
        "secured_first": len(secured),
        "slipped": sum(bool(row["slipped"]) for row in secured),
    }


def one_sided_two_proportion_pvalue(
    candidate_successes: int,
    candidate_trials: int,
    baseline_successes: int,
    baseline_trials: int,
) -> float:
    if not (
        candidate_trials > 0
        and baseline_trials > 0
        and 0 <= candidate_successes <= candidate_trials
        and 0 <= baseline_successes <= baseline_trials
    ):
        raise ValueError("invalid binomial counts")
    pooled = (candidate_successes + baseline_successes) / (candidate_trials + baseline_trials)
    variance = pooled * (1.0 - pooled) * (1.0 / candidate_trials + 1.0 / baseline_trials)
    if variance == 0.0:
        return 1.0
    z_score = (candidate_successes / candidate_trials - baseline_successes / baseline_trials) / math.sqrt(variance)
    return float(norm.sf(z_score))


def evaluate(contract: dict[str, Any], baseline: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    generated = observed["generated"]
    rollouts = observed["rollouts"]
    successes = {str(key): int(value) for key, value in observed["successes_by_seed"].items()}
    pooled_successes = sum(successes.values())
    selection = contract["performance_selection"]
    alpha = float(selection["alpha_one_sided"])
    seed_pass = {
        "101": successes["101"] > int(selection["seed101_successes_gt"]),
        "111": successes["111"] > int(selection["seed111_successes_gt"]),
    }
    comparisons = {}
    for label, counts in selection["pooled_successes_p_less_than_alpha_vs"].items():
        pvalue = one_sided_two_proportion_pvalue(pooled_successes, 200, int(counts[0]), int(counts[1]))
        comparisons[label] = {"baseline": counts, "p_one_sided": pvalue, "passes": pvalue < alpha}
    performance_pass = all(seed_pass.values()) and all(row["passes"] for row in comparisons.values())

    mechanism = contract["mechanism"]
    if not baseline.get("created_from_existing_e2_only"):
        raise ValueError("baseline is not the frozen E2-only measurement")
    if baseline["generated_500"]["first_close_height_mm_median"] != mechanism["baseline_generated_close_height_mm"]:
        raise ValueError("generated close-height baseline differs from the preregistration")
    if baseline["e2_rollouts_200"]["first_near_close_height_mm_median"] != mechanism["baseline_policy_close_height_mm"]:
        raise ValueError("policy close-height baseline differs from the preregistration")
    if [baseline["e2_rollouts_200"]["slipped"], baseline["e2_rollouts_200"]["secured_first"]] != mechanism["baseline_slip"]:
        raise ValueError("slip baseline differs from the preregistration")
    generated_baseline = float(mechanism["baseline_generated_close_height_mm"])
    policy_baseline = float(mechanism["baseline_policy_close_height_mm"])
    m1 = (
        generated["first_close_height_mm_median"] <= generated_baseline - 3.0
        and generated["first_close_height_mm_median"] < 10.0
    )
    policy_height = rollouts["first_near_close_height_mm_median"]
    m2 = policy_height is not None and policy_height <= policy_baseline - 3.0
    baseline_slip, baseline_secured = (int(value) for value in mechanism["baseline_slip"])
    slip_p = (
        one_sided_two_proportion_pvalue(
            baseline_slip,
            baseline_secured,
            int(rollouts["slipped"]),
            int(rollouts["secured_first"]),
        )
        if rollouts["secured_first"] > 0
        else None
    )
    # The generic helper tests whether its first proportion is greater.  Here that
    # is baseline > E6, exactly the registered lower-slip alternative.
    m3 = (
        rollouts["secured_first"] > 0
        and rollouts["slipped"] / rollouts["secured_first"] < baseline_slip / baseline_secured
        and slip_p is not None
        and slip_p < 0.05
    )
    measurement_unavailable = []
    if policy_height is None:
        measurement_unavailable.append("M2")
    if rollouts["secured_first"] == 0:
        measurement_unavailable.append("M3")
    supported = performance_pass and m1 and m2 and m3
    rejected = (
        pooled_successes <= 40
        or not m1
        or (policy_height is not None and not m2)
        or (
            rollouts["secured_first"] > 0
            and rollouts["slipped"] / rollouts["secured_first"] >= baseline_slip / baseline_secured
        )
    )
    verdict = "supported" if supported else "rejected" if rejected else "inconclusive"
    return {
        "hypothesis": "H4",
        "condition": "E6",
        "generated": generated,
        "rollouts": rollouts,
        "successes_by_seed": successes,
        "pooled_successes": pooled_successes,
        "performance": {
            "seed_thresholds_pass": seed_pass,
            "pooled_comparisons": comparisons,
            "passes": performance_pass,
            "selected_condition": selection["if_pass"] if performance_pass else selection["otherwise"],
        },
        "mechanism": {
            "M1": m1,
            "M2": m2,
            "M3": m3,
            "slip_p_one_sided": slip_p,
            "measurement_unavailable": measurement_unavailable,
            "baseline_generated_close_height_mm": generated_baseline,
            "baseline_policy_close_height_mm": policy_baseline,
            "baseline_slip": [baseline_slip, baseline_secured],
        },
        "verdict": verdict,
    }


def recorded_observed(recorded: dict[str, Any]) -> dict[str, Any]:
    """Extract only the frozen verdict inputs from a recorded H4 result."""
    required = ("generated", "rollouts", "successes_by_seed")
    missing = [key for key in required if key not in recorded]
    if missing:
        raise ValueError(f"recorded result is missing verdict inputs: {missing}")
    successes = {str(key): int(value) for key, value in recorded["successes_by_seed"].items()}
    if set(successes) != {"101", "111"}:
        raise ValueError("recorded result must contain exactly seeds 101 and 111")
    return {
        "generated": recorded["generated"],
        "rollouts": recorded["rollouts"],
        "successes_by_seed": successes,
    }


def observed_from_full_artifacts(contract: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset is None or args.observer_npz is None or args.observer_results is None:
        raise ValueError(
            "full mode requires --dataset, two --observer_npz directories, and two --observer_results files"
        )
    successes = {}
    for seed, path in zip(contract["training_seeds"], args.observer_results, strict=True):
        result = json.loads(path.read_text())
        if result.get("rollouts") != 100 or result.get("evaluation_seed") != 42 or result.get("horizon") != 800:
            raise ValueError(f"observer result violates registered evaluation: {path}")
        successes[str(seed)] = int(result["successes"])
    return {
        "generated": generated_metrics(
            args.dataset,
            dt=float(contract["pilot"]["quality_dt"]),
            skip=int(contract["pilot"]["quality_skip"]),
        ),
        "rollouts": rollout_metrics(args.observer_npz),
        "successes_by_seed": successes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--recorded-result", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--observer_npz", type=Path, nargs=2)
    parser.add_argument("--observer_results", type=Path, nargs=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    baseline = json.loads(args.baseline.read_text())
    has_full_argument = any(
        value is not None for value in (args.dataset, args.observer_npz, args.observer_results)
    )
    if args.recorded_result is not None and has_full_argument:
        parser.error("--recorded-result cannot be combined with full-artifact arguments")
    if args.recorded_result is not None:
        observed = recorded_observed(json.loads(args.recorded_result.read_text()))
    else:
        try:
            observed = observed_from_full_artifacts(contract, args)
        except ValueError as error:
            parser.error(str(error))
    result = evaluate(contract, baseline, observed)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
