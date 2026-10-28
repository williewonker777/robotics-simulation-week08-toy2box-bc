import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest


TOOLS = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H5 = load_module("public_h5_verdict", TOOLS / "toy2box_analysis/h5_verdict.py")


def generated(*, speed=0.03, xy=5.0, height=12.5, heading=2.0, audit=True):
    result = {
        "episodes": 500,
        "preclose10_mean_eef_speed_m_s_median": speed,
        "first_close_xy_offset_mm_median": xy,
        "first_close_height_mm_median": height,
        "first_close_heading_error_abs_deg_median": heading,
    }
    if audit is not None:
        result["waypoint_audit"] = {
            "all_invariants_pass": audit,
            "inserted_waypoint_count_minmax": [9, 9],
            "minimum_expanded_translation_interval_m_min": 0.001,
        }
    return result


def observed(successes=(40, 40), *, xy=8.0, secured=110, near=167, generated_metrics=None):
    return {
        "generated": generated_metrics or generated(),
        "rollouts": {
            "trials": 200,
            "first_near_close_count": near,
            "no_near_close_count": 200 - near,
            "first_near_close_xy_offset_mm_median": xy if near else None,
            "secured_first": secured,
            "slipped": 0,
        },
        "successes_by_seed": {"101": successes[0], "111": successes[1]},
    }


@pytest.fixture
def registration():
    return json.loads(H5.DEFAULT_CONTRACT.read_text()), json.loads(H5.DEFAULT_BASELINE.read_text())


def test_registered_source_function_digests_are_frozen():
    expected = {
        "first_close_index": "ab1730aff238afbfcada4ca1ddfa5a2a63928ed2171bfc1606a358067ef0979e",
        "close_geometry": "30d705bf245b3df1d62004162a308f34fde449c18550e8e9830b556631645cb6",
        "preclose10_mean_speed": "b946f5987d668c0b5c3a879ff72a585c275bef7c130436d8b2d59c12f206c1f1",
        "lift_delay_steps": "79a62cb14c0d2c3b283b714a10185ea7bb6efd47f3fc1935d7996d2531007097",
        "generated_metrics": "843ea27ed806eb00e5bc26e68008a294f5737e77e9b9cd56f8b8583f0c911074",
        "rollout_metrics": "8b44ae81df6782408bd2a6e5f38f84c1ffc7d5aa1ad71de6ce5ef228ae9e0174",
        "one_sided_two_proportion_pvalue": "15a8125b2a1b9a6d4635b5c2c3fc456797e4c91a46ca03b19055435de208fdcb",
        "evaluate": "9e66fd500bcd4ed2fe1eb740f6b7954e4b1ee16af84babe48a2645a554d4c6a3",
    }

    actual = {
        name: hashlib.sha256(inspect.getsource(getattr(H5, name)).encode()).hexdigest()
        for name in expected
    }

    assert actual == expected


def test_canonical_baseline_matches_preregistered_mechanism(registration):
    contract, baseline = registration

    assert baseline["generated"] == contract["mechanism"]["baseline_generated"]
    assert baseline["policy"]["xy_mm_median"] == contract["mechanism"]["baseline_policy_xy_mm_median"]
    assert [baseline["policy"]["secured"], baseline["policy"]["near"]] == contract["mechanism"]["baseline_acquisition"]


def test_strict_waypoint_interval_rejects_equality(registration):
    contract, baseline = registration
    metrics = generated()
    metrics["waypoint_audit"]["minimum_expanded_translation_interval_m_min"] = contract["pilot"]["min_expanded_translation_interval_m"]

    result = H5.evaluate(contract, baseline, observed(generated_metrics=metrics))

    assert result["mechanism"]["M1"] is False
    assert result["verdict"] == "rejected"


def test_strict_waypoint_interval_accepts_value_above_boundary(registration):
    contract, baseline = registration
    metrics = generated()
    boundary = float(contract["pilot"]["min_expanded_translation_interval_m"])
    metrics["waypoint_audit"]["minimum_expanded_translation_interval_m_min"] = np.nextafter(boundary, np.inf)

    result = H5.evaluate(contract, baseline, observed(generated_metrics=metrics))

    assert result["mechanism"]["M1"] is True


def test_missing_mechanism_measurements_are_inconclusive(registration):
    contract, baseline = registration

    result = H5.evaluate(
        contract,
        baseline,
        observed((19, 22), near=0, secured=0, generated_metrics=generated(audit=None)),
    )

    assert result["mechanism"]["measurement_unavailable"] == ["M1_waypoint_audit", "M2", "M3"]
    assert result["verdict"] == "inconclusive"


def test_performance_and_all_mechanisms_supported_selects_e7(registration):
    contract, baseline = registration

    result = H5.evaluate(contract, baseline, observed())

    assert result["performance"]["passes"] is True
    assert [result["mechanism"][key] for key in ("M1", "M2", "M3")] == [True, True, True]
    assert result["verdict"] == "supported"
    assert result["performance"]["selected_condition"] == "E7"


def test_pooled_success_rejection_boundary_selects_e2(registration):
    contract, baseline = registration

    result = H5.evaluate(contract, baseline, observed((18, 22)))

    assert result["pooled_successes"] == 40
    assert result["verdict"] == "rejected"
    assert result["performance"]["selected_condition"] == "E2"


def test_recorded_result_uses_only_frozen_verdict_inputs():
    source = observed()
    source.update({"verdict": "untrusted", "extra": {"ignored": True}})

    extracted = H5.recorded_observed(source)

    assert extracted == {
        "generated": source["generated"],
        "rollouts": source["rollouts"],
        "successes_by_seed": {"101": 40, "111": 40},
    }


def test_recorded_result_cli_uses_canonical_registration_defaults(tmp_path, monkeypatch):
    recorded_path = tmp_path / "recorded.json"
    output_path = tmp_path / "verdict.json"
    recorded_path.write_text(json.dumps(observed()))
    monkeypatch.setattr(
        "sys.argv",
        ["h5_verdict.py", "--recorded-result", str(recorded_path), "--output", str(output_path)],
    )

    assert H5.main() == 0
    assert json.loads(output_path.read_text())["verdict"] == "supported"
