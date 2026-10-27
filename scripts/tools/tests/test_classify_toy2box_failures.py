import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "classify_toy2box_failures.py"
SPEC = importlib.util.spec_from_file_location("classify_toy2box_failures", SCRIPT)
CLASSIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLASSIFIER)


THRESHOLDS = {
    "baseline_steps": 3,
    "approach_distance_m": 0.025,
    "close_action_threshold": 0.0,
    "lift_height_m": 0.03,
    "grasp_consecutive_steps": 3,
    "grasp_follow_distance_m": 0.03,
    "grasp_relative_span_m": 0.005,
    "transport_xy_threshold_m": 0.12,
    "environment_gripper_open_target": 0.994,
    "environment_gripper_open_tolerance": 0.2,
    "stall_action_norm_threshold": 0.005,
    "stall_eef_step_m": 0.0005,
    "stall_consecutive_steps": 4,
    "timeout_horizon_steps": 20,
    "initial_truck_x_edges_m": [-0.1, 0.1],
    "initial_truck_y_edges_m": [-0.1, 0.1],
}


def trial(stage="success"):
    length = 16
    truck = np.zeros((length, 3), dtype=float)
    truck[:, 2] = 0.38
    eef = truck + [0.2, 0.0, 0.02]
    box = np.tile([0.30, 0.0, 0.38], (length, 1)).astype(float)
    action = np.zeros((length, 7), dtype=float)
    action[:, -1] = 1.0
    success = np.zeros(length, dtype=bool)
    if stage != "approach":
        eef[3:] = truck[3:] + [0.015, 0.0, 0.0]
        action[4:, -1] = -1.0
    if stage not in ("approach", "grasp"):
        truck[6:12, 2] += 0.04
        eef[6:12] = truck[6:12] + [0.015, 0.0, 0.0]
    if stage not in ("approach", "grasp", "transport"):
        truck[9:, :2] = [0.25, 0.0]
        eef[9:] = truck[9:] + [0.015, 0.0, 0.0]
    if stage == "success":
        success[13:] = True
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (length, 1))
    return {
        "eef_pos": eef,
        "eef_quat": quat.copy(),
        "gripper_pos": np.full((length, 2), 0.994),
        "action": action,
        "truck_pos": truck,
        "truck_quat": quat.copy(),
        "box_pos": box,
        "box_quat": quat.copy(),
        "success_value": success,
    }


@pytest.mark.parametrize("stage", ["approach", "grasp", "transport", "placement", "success"])
def test_first_missing_milestone(stage):
    result = CLASSIFIER.classify_trial(trial(stage), THRESHOLDS)
    assert result["stage"] == stage
    assert result["success"] is (stage == "success")


def test_heading_is_folded_modulo_180_and_close_offset_is_reported():
    arrays = trial("grasp")
    angle = np.deg2rad(170.0)
    arrays["eef_quat"][4] = [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]
    result = CLASSIFIER.classify_trial(arrays, THRESHOLDS)
    assert result["diagnostics"]["close_heading_error_signed_mod180_deg"] == pytest.approx(-10.0)
    assert result["diagnostics"]["close_heading_error_abs_mod180_deg"] == pytest.approx(10.0)
    assert result["diagnostics"]["close_offset_norm_m"] == pytest.approx(0.015)


def test_npz_validation_and_create_only_output(tmp_path):
    arrays = trial("success")
    npz = tmp_path / "trial.npz"
    np.savez(npz, **arrays)
    loaded = CLASSIFIER._load_trial(npz)
    assert loaded["eef_pos"].shape == (16, 3)
    output = tmp_path / "result.json"
    CLASSIFIER._write_new_json(output, {"ok": True})
    assert json.loads(output.read_text()) == {"ok": True}
    with pytest.raises(FileExistsError):
        CLASSIFIER._write_new_json(output, {"ok": False})


def test_stall_and_timeout_are_explicit_operational_flags():
    arrays = trial("grasp")
    arrays["action"][-5:, :6] = 0.0
    arrays["eef_pos"][-5:] = arrays["eef_pos"][-5]
    arrays["truncated"] = np.r_[np.zeros(15, dtype=bool), True]
    result = CLASSIFIER.classify_trial(arrays, THRESHOLDS)
    assert result["diagnostics"]["stall"] is True
    assert result["diagnostics"]["timeout"] is True
    assert result["diagnostics"]["timeout_basis"] == "truncated"


def test_summary_has_yaw45_stage_counts_and_aggregate_pools_labels():
    success = CLASSIFIER.classify_trial(trial("success"), THRESHOLDS, path="a.npz")
    failure = CLASSIFIER.classify_trial(trial("grasp"), THRESHOLDS, path="b.npz")
    one = {"kind": "toy2box_stage_classification", "label": "seed101", "trials": [success], "summary": {}}
    two = {"kind": "toy2box_stage_classification", "label": "seed111", "trials": [failure], "summary": {}}
    aggregate = CLASSIFIER.aggregate_classifications([one, two])
    assert aggregate["pooled"]["stage_counts"]["success"] == 1
    assert aggregate["pooled"]["stage_counts"]["grasp"] == 1
    zero_yaw = aggregate["pooled"]["initial_truck_yaw_45deg"][4]
    assert zero_yaw["attempts"] == 2
    assert zero_yaw["stage_counts"]["success"] == 1


def test_actual_success_overrides_missed_proxy_and_rate_uses_actual_success():
    arrays = trial("success")
    arrays["eef_pos"][:] = arrays["truck_pos"] + [0.2, 0.0, 0.0]
    result = CLASSIFIER.classify_trial(arrays, THRESHOLDS)
    assert result["stage"] == "success"
    assert result["actual_success"] is True
    assert result["prior_milestone_proxy_inconsistencies"] == ["approach", "grasp", "transport"]
    summary = CLASSIFIER.summarize_trials([result], THRESHOLDS)
    assert summary["successes"] == 1
    assert summary["actual_success_with_proxy_inconsistency_count"] == 1


def test_approach_after_close_is_not_retroactive_and_reset_initial_is_preferred():
    arrays = trial("approach")
    arrays["action"][:, -1] = -1.0
    arrays["eef_pos"][5:] = arrays["truck_pos"][5:] + [0.015, 0.0, 0.0]
    arrays["reset_initial_truck_pos"] = np.array([0.07, -0.08, 0.38])
    arrays["reset_initial_truck_quat"] = np.array([1.0, 0.0, 0.0, 0.0])
    arrays["reset_initial_box_pos"] = np.array([0.3, 0.0, 0.38])
    result = CLASSIFIER.classify_trial(arrays, THRESHOLDS)
    assert result["stage"] == "approach"
    assert result["milestone_indices"]["approach"] is None
    assert result["diagnostics"]["diagnostic_close_index"] == 0
    assert result["diagnostics"]["initial_truck_position"] == pytest.approx([0.07, -0.08, 0.38])
    assert result["diagnostics"]["initial_state_basis"]["truck_pos"] == "observer_reset_initial"


def test_grasp_only_and_location_bin_summaries_are_separate():
    grasp = CLASSIFIER.classify_trial(trial("grasp"), THRESHOLDS, path="grasp.npz")
    transport = CLASSIFIER.classify_trial(trial("transport"), THRESHOLDS, path="transport.npz")
    summary = CLASSIFIER.summarize_trials([grasp, transport], THRESHOLDS)
    assert summary["all_failure_diagnostics"]["count"] == 2
    assert summary["grasp_failure_diagnostics"]["paths"] == ["grasp.npz"]
    assert sum(item["attempts"] for item in summary["initial_truck_x_bins_m"]) == 2
    assert summary["initial_location_bin_contract"]["out_of_range"].startswith("explicit")


def test_dataset_position_bins_match_classifier_edge_and_overflow_contract():
    positions = np.array(
        [
            [-0.2, -0.2, 0.0],
            [-0.1, -0.1, 0.0],
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.0],
        ]
    )
    modes = np.array([0, 180, 0, 180])
    bins = CLASSIFIER._dataset_axis_bins(positions, modes, 0, [-0.1, 0.1])
    assert [item["count"] for item in bins] == [1, 2, 1]
    assert bins[0]["grasp_mode_0_count"] == 1
    assert bins[1]["grasp_mode_0_count"] == 1
    assert bins[1]["grasp_mode_180_count"] == 1
    assert bins[2]["grasp_mode_180_count"] == 1
