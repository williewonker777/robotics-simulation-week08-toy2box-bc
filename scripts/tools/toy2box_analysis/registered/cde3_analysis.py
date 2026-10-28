#!/usr/bin/env python3
"""Fail-closed offline analysis for preregistrations 2 and 3."""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent
BASE_PATH = HERE / "cde2_analysis.py"
RECOVERY_PATH = EVIDENCE / "recovery_metrics.py"
SEEDS = ("101", "111")
STAGES = ("approach", "grasp", "transport", "placement", "success")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load("toy2box_cde2_analysis", BASE_PATH)
recovery = load("toy2box_recovery", RECOVERY_PATH)


class AnalysisError(RuntimeError):
    pass


def analyze_condition(name: str, cfg: dict[str, Any], parent: Path):
    dataset = base.resolve(cfg["dataset"]["combined"], parent)
    expected = int(cfg["dataset"]["expected_demos"])
    conv = base.convention(dataset)
    if conv["total_demos"] != expected:
        raise AnalysisError(f"{name}: expected {expected} demos, got {conv['total_demos']}")
    source = cfg["dataset"].get("source")
    models: dict[str, Any] = {}
    cohorts: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    recover_rows = []
    counts = []
    for seed in SEEDS:
        model = cfg["models"][seed]
        npz = base.resolve(model["trajectory_dir"], parent)
        rows, verify = base.observer_rows(npz, base.resolve(model["results_json"], parent))
        rec_rows = [recovery.trial(str(p)) for p in sorted(npz.glob("trial_*.npz"))]
        if len(rec_rows) != 100:
            raise AnalysisError(f"{name}/{seed}: recovery cohort is not 100")
        cohorts[seed] = rows
        all_rows.extend(rows)
        recover_rows.extend(rec_rows)
        success = sum(r["success"] for r in rows)
        counts.append(success)
        stages = Counter(r["stage"] for r in rows)
        models[seed] = {
            "successes": success, "trials": 100, "rate": success / 100,
            "wilson95": base.wilson95(success, 100),
            "v3_1_stages": {s: stages[s] for s in STAGES},
            "close_by_abs_yaw": base.close_bins(rows),
            "recovery": recovery.summarize(rec_rows),
            "observer_verification": verify,
        }
    stages = Counter(r["stage"] for r in all_rows)
    result = {
        "pooled_successes": sum(counts), "pooled_trials": 200,
        "models": models,
        "condition_success": {
            "seed_success_counts": counts,
            "mean_rate": statistics.mean(counts) / 100,
            "sample_sd_rate_ddof1": statistics.stdev(counts) / 100,
        },
        "v3_1_pooled_stages": {s: stages[s] for s in STAGES},
        "close_by_abs_yaw_pooled": base.close_bins(all_rows),
        "recovery_pooled": recovery.summarize(recover_rows),
        "dataset_convention": conv,
        "provenance": {"combined_dataset": str(dataset)},
    }
    if source:
        result["source_yaw_histogram"] = base.yaw_histogram(base.resolve(source, parent))
    generation = cfg.get("generation")
    if generation:
        result["generation_yield"] = base.generation_yield(
            [base.resolve(p, parent) for p in generation["success_files"]],
            [base.resolve(p, parent) for p in generation["failed_files"]],
        )
    return result, cohorts


def h1_decision(conditions, comparisons):
    e0, e1, e2 = (conditions[n] for n in ("E0", "E1", "E2"))
    target = e1["close_by_abs_yaw_pooled"]["[45,90)"]
    closed = target["closed"]
    if type(closed) is not int or closed <= 0:
        raise AnalysisError("H1 prediction 2 has no closed trials in |yaw| [45,90)")
    mode_total = target["mode0"] + target["mode180"] + target["between"]
    if mode_total != closed:
        raise AnalysisError("H1 prediction 2 mode counts disagree with closed-trial denominator")
    between_fraction = target["between"] / closed
    between_fraction_limit = 0.5 * 25 / 55
    prediction2 = (
        between_fraction <= between_fraction_limit
        and target["heading_error_abs_median_deg"] is not None
        and target["heading_error_abs_median_deg"] < 10
    )
    both = all(e1["models"][s]["successes"] > e0["models"][s]["successes"] for s in SEEDS)
    abs_test = comparisons["E1_vs_E0"]["pooled_abs_yaw_le_90"]
    abs_up = abs_test["first"]["successes"] > 7
    significant = abs_test["p_one_sided"] < 0.05
    if both and abs_up and significant and prediction2:
        verdict = "supported"
    elif not abs_up or not prediction2:
        verdict = "rejected"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "rules": {"both_seeds_whole_strictly_improve": both,
                  "abs_yaw_le_90_success_above_E0_7_of_99": abs_up,
                  "abs_yaw_le_90_p_one_sided_lt_0_05": significant,
                  "prediction_2": prediction2},
        "prediction_2_observed": {
            "between": target["between"],
            "closed": closed,
            "between_fraction": between_fraction,
            "registered_baseline_fraction": 25 / 55,
            "half_baseline_fraction_limit": between_fraction_limit,
            "heading_error_median_deg": target["heading_error_abs_median_deg"],
            "heading_error_lt_10_deg": (
                target["heading_error_abs_median_deg"] is not None
                and target["heading_error_abs_median_deg"] < 10
            ),
        },
        "E2_specificity": {
            "verdict": "not_mechanically_adjudicable",
            "reason": "Preregistration uses 'similar' without a numeric tolerance.",
            "E1_pooled": e1["pooled_successes"], "E2_pooled": e2["pooled_successes"],
        },
    }


def h3_decision(conditions, comparisons):
    e0, e4 = conditions["E0"], conditions["E4"]
    both = all(e4["models"][s]["successes"] > e0["models"][s]["successes"] for s in SEEDS)
    whole = comparisons["E4_vs_E0"]["pooled_whole"]
    rec = e4["recovery_pooled"]
    recovery_test = base.ztest(rec["recoveries"], rec["first_failed"], 5, 102)
    recovery_up = rec["recoveries"] / rec["first_failed"] > 5 / 102 if rec["first_failed"] else False
    mean_up = e4["condition_success"]["mean_rate"] > 0.195
    if both and whole["p_one_sided"] < 0.05 and recovery_test["p_one_sided"] < 0.05:
        verdict = "supported"
    elif not mean_up or not recovery_up:
        verdict = "rejected"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "rules": {"both_seeds_whole_strictly_improve": both,
                  "whole_p_one_sided_lt_0_05": whole["p_one_sided"] < 0.05,
                  "recovery_rate_above_E0_5_of_102": recovery_up,
                  "recovery_p_one_sided_lt_0_05": recovery_test["p_one_sided"] < 0.05,
                  "mean_above_E0_19_5pct": mean_up},
        "recovery_test_vs_E0_5_of_102": recovery_test,
    }


def markdown(result):
    lines = ["# preregistration 2+3 results", "", "| condition | seed101 | seed111 | pooled |", "|---|---:|---:|---:|"]
    for name, condition in result["conditions"].items():
        lines.append(f"| {name} | {condition['models']['101']['successes']}/100 | {condition['models']['111']['successes']}/100 | {condition['pooled_successes']}/200 |")
    d = result["preregistered_decisions"]
    lines += ["", f"- H1″: **{d['H1_double_prime']['verdict']}**", f"- H3: **{d['H3']['verdict']}**", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--output", required=True); ap.add_argument("--markdown", required=True)
    args = ap.parse_args()
    input_path, output, md = map(lambda p: Path(p).expanduser().resolve(), (args.input, args.output, args.markdown))
    if output.exists() or md.exists():
        raise AnalysisError("refusing to overwrite analysis output")
    manifest = json.loads(input_path.read_text())
    names = tuple(manifest.get("conditions", {}))
    if not {"E0", "E1", "E2", "E4"}.issubset(names) or set(names) - {"E0", "E1", "E2", "E4", "E5"}:
        raise AnalysisError("conditions must be E0/E1/E2/E4 and optional E5")
    conditions, cohorts = {}, {}
    for name in names:
        conditions[name], cohorts[name] = analyze_condition(name, manifest["conditions"][name], input_path.parent)
    e0_abs = [base.subset_count(cohorts["E0"][s]) for s in SEEDS]
    if (sum(x[0] for x in e0_abs), sum(x[1] for x in e0_abs)) != (7, 99) or conditions["E0"]["pooled_successes"] != 39:
        raise AnalysisError("registered E0 anchors disagree with observer data")
    comparisons = {
        f"{name}_vs_E0": base.comparison(conditions[name], cohorts[name], conditions["E0"], cohorts["E0"])
        for name in names if name != "E0"
    }
    decisions = {"H1_double_prime": h1_decision(conditions, comparisons), "H3": h3_decision(conditions, comparisons)}
    result = {"schema_version": 3, "created_utc": datetime.now(timezone.utc).isoformat(),
              "input_manifest": str(input_path), "conditions": conditions,
              "comparisons": comparisons, "preregistered_decisions": decisions,
              "statistical_contract": {"tests": "unpaired one-sided pooled two-proportion z; first > second",
                                           "E0_whole": "39/200", "E0_abs_yaw_le_90": "7/99",
                                           "E0_recovery": "5/102", "condition_sd_ddof": 1}}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True); stream.write("\n")
    try:
        with md.open("x") as stream: stream.write(markdown(result))
    except Exception:
        output.unlink(missing_ok=True); raise


if __name__ == "__main__":
    main()
