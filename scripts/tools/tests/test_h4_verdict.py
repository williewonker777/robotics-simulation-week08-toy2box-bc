import importlib.util
import json
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H4 = load_module("public_h4_verdict", TOOLS / "toy2box_analysis/h4_verdict.py")


@pytest.fixture
def contract():
    return {
        "training_seeds": [101, 111],
        "pilot": {"quality_dt": 0.05, "quality_skip": 0},
        "performance_selection": {
            "alpha_one_sided": 0.025,
            "seed101_successes_gt": 18,
            "seed111_successes_gt": 22,
            "pooled_successes_p_less_than_alpha_vs": {
                "E2": [40, 200],
                "E0": [39, 200],
            },
            "if_pass": "E6",
            "otherwise": "E2",
        },
        "mechanism": {
            "baseline_generated_close_height_mm": 12.547045802109217,
            "baseline_policy_close_height_mm": 13.630639760921525,
            "baseline_slip": [42, 78],
        },
    }


@pytest.fixture
def baseline():
    return {
        "created_from_existing_e2_only": True,
        "generated_500": {"first_close_height_mm_median": 12.547045802109217},
        "e2_rollouts_200": {
            "first_near_close_height_mm_median": 13.630639760921525,
            "slipped": 42,
            "secured_first": 78,
        },
    }


def observed(successes=(40, 40), *, generated_height=7.5, policy_height=9.5, slipped=10, secured=80):
    return {
        "generated": {"first_close_height_mm_median": generated_height},
        "rollouts": {
            "trials": 200,
            "first_near_close_count": 170,
            "no_near_close_count": 30,
            "first_near_close_height_mm_median": policy_height,
            "secured_first": secured,
            "slipped": slipped,
        },
        "successes_by_seed": {"101": successes[0], "111": successes[1]},
    }


def test_supported_vector_matches_frozen_reference(contract, baseline):
    result = H4.evaluate(contract, baseline, observed())

    assert result["verdict"] == "supported"
    assert result["performance"]["pooled_comparisons"]["E2"]["p_one_sided"] == 6.374837460548514e-06
    assert result["performance"]["pooled_comparisons"]["E0"]["p_one_sided"] == 3.6590032797846017e-06
    assert result["mechanism"]["slip_p_one_sided"] == 1.6039201003660718e-08


def test_pooled_successes_at_rejection_boundary_are_rejected(contract, baseline):
    result = H4.evaluate(contract, baseline, observed((18, 22)))

    assert result["pooled_successes"] == 40
    assert result["verdict"] == "rejected"
    assert result["performance"]["selected_condition"] == "E2"


def test_missing_mechanism_measurements_remain_inconclusive(contract, baseline):
    result = H4.evaluate(
        contract,
        baseline,
        observed((19, 22), policy_height=None, slipped=0, secured=0),
    )

    assert result["pooled_successes"] == 41
    assert result["mechanism"]["measurement_unavailable"] == ["M2", "M3"]
    assert result["verdict"] == "inconclusive"


def test_proportion_test_rejects_invalid_counts():
    with pytest.raises(ValueError, match="invalid binomial counts"):
        H4.one_sided_two_proportion_pvalue(201, 200, 40, 200)


def test_recorded_result_uses_only_frozen_verdict_inputs():
    source = observed()
    source.update({"verdict": "untrusted", "extra": {"ignored": True}})

    extracted = H4.recorded_observed(source)

    assert extracted == {
        "generated": source["generated"],
        "rollouts": source["rollouts"],
        "successes_by_seed": {"101": 40, "111": 40},
    }


def test_recorded_result_requires_both_registered_seeds():
    source = observed()
    source["successes_by_seed"] = {"101": 40}

    with pytest.raises(ValueError, match="exactly seeds 101 and 111"):
        H4.recorded_observed(source)


def test_recorded_result_cli_uses_canonical_registration_defaults(tmp_path, monkeypatch):
    recorded_path = tmp_path / "recorded.json"
    output_path = tmp_path / "verdict.json"
    recorded_path.write_text(json.dumps(observed()))
    monkeypatch.setattr(
        "sys.argv",
        [
            "h4_verdict.py",
            "--recorded-result",
            str(recorded_path),
            "--output",
            str(output_path),
        ],
    )

    assert H4.main() == 0
    assert json.loads(output_path.read_text())["verdict"] == "supported"
