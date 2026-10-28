#!/usr/bin/env python3
"""Fail-closed offline analysis for preregistration 2.

Consumes only completed observer NPZ cohorts and generation HDF5 manifests. It
writes create-only JSON/Markdown and mechanically applies every numeric rule in
preregistration 2. The word "similar" in the E2 specificity clause has no
registered numeric tolerance, so that clause is explicitly not adjudicated.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent
CONDITIONS = ("E0", "E1", "E2", "E3")
SEEDS = ("101", "111")
STAGES = ("approach", "grasp", "transport", "placement", "success")
SIGNED_BINS = tuple((lo, lo + 45) for lo in range(-180, 180, 45))
ABS_BINS = ((0, 45), (45, 90), (90, 135), (135, 181))


class AnalysisError(RuntimeError):
    pass


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AnalysisError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE_TOOL = load_module("toy2box_stage_v3", EVIDENCE / "stage_v3.py")
CLOSE_TOOL = load_module("toy2box_close_precision", EVIDENCE / "close_precision.py")


def resolve(value: str, parent: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else parent / path).resolve()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise AnalysisError(f"required file missing: {path}")


def heading_x(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    w, x, y, z = (q[..., i] for i in range(4))
    return np.degrees(np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z)))


def wrap(value: float | np.ndarray) -> float | np.ndarray:
    return (value + 180.0) % 360.0 - 180.0


def signed_label(yaw: float) -> str:
    value = float(wrap(yaw))
    for lo, hi in SIGNED_BINS:
        if lo <= value < hi:
            return f"[{lo},{hi})"
    raise AnalysisError(f"unbinnable yaw {yaw}")


def wilson95(successes: int, trials: int) -> dict[str, float]:
    if not 0 <= successes <= trials or trials <= 0:
        raise AnalysisError(f"invalid binomial count {successes}/{trials}")
    z = 1.959963984540054
    p = successes / trials
    den = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / den
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / den
    return {"low": center - half, "high": center + half}


def ztest(first_success: int, first_n: int, second_success: int, second_n: int) -> dict[str, Any]:
    if min(first_n, second_n) <= 0:
        raise AnalysisError("empty two-proportion test sample")
    pooled = (first_success + second_success) / (first_n + second_n)
    variance = pooled * (1 - pooled) * (1 / first_n + 1 / second_n)
    if variance <= 0:
        raise AnalysisError("zero-variance two-proportion test")
    z = (first_success / first_n - second_success / second_n) / math.sqrt(variance)
    return {
        "alternative": "first_rate_greater_than_second_rate",
        "first": {"successes": first_success, "trials": first_n},
        "second": {"successes": second_success, "trials": second_n},
        "z": z,
        "p_one_sided": 0.5 * math.erfc(z / math.sqrt(2)),
    }


def observer_rows(npz_dir: Path, results_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = sorted(npz_dir.glob("trial_*.npz"))
    if [p.name for p in paths] != [f"trial_{i:05d}.npz" for i in range(100)]:
        raise AnalysisError(f"observer cohort is not exactly contiguous 0..99: {npz_dir}")
    rows: list[dict[str, Any]] = []
    vector: list[bool] = []
    for index, path in enumerate(paths):
        stage = STAGE_TOOL.classify(str(path))["stage"]
        close = CLOSE_TOOL.trial(str(path))
        with np.load(path, allow_pickle=True) as data:
            success = bool(np.asarray(data["final_success"]).item())
        if success != (stage == "success"):
            raise AnalysisError(f"v3.1 success disagreement: {path}")
        close.update(trial=index, stage=stage, success=success, path=str(path))
        rows.append(close)
        vector.append(success)
    result = json.loads(results_path.read_text())
    if [bool(v) for v in result.get("trial_results", [])] != vector:
        raise AnalysisError(f"observer/result vector mismatch: {npz_dir}")
    return rows, {"npz_count": 100, "results_json": str(results_path), "vector_match": True}


def close_bins(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for lo, hi in ABS_BINS:
        group = [r for r in rows if lo <= abs(float(wrap(r["yaw0"]))) < hi]
        closed = [r for r in group if r["closed"]]
        errors = [abs(float(r["herr"])) for r in closed]
        modes = Counter(r["mode"] for r in closed)
        output[f"[{lo},{hi})"] = {
            "trials": len(group),
            "successes": sum(r["success"] for r in group),
            "closed": len(closed),
            "heading_error_abs_median_deg": None if not errors else float(np.median(errors)),
            "heading_error_abs_p75_deg": None if not errors else float(np.percentile(errors, 75)),
            "mode0": modes["mode0"], "mode180": modes["mode180"], "between": modes["between"],
        }
    return output


def convention(path: Path) -> dict[str, Any]:
    rows = []
    with h5py.File(path, "r") as h5:
        for name in sorted(h5["data"]):
            demo = h5["data"][name]
            actions = demo["actions"][:]
            close = np.flatnonzero(actions[:, -1] < 0)
            if not len(close):
                continue
            k = int(close[0])
            obs = demo["obs"]
            eef = heading_x(obs["eef_quat"][:])
            yaw = float(heading_x(obs["toy_truck_orientations"][0]))
            grasp = float(eef[k])
            rotation = float(np.degrees(np.unwrap(np.radians(eef[: k + 1])))[-1] - eef[0])
            rows.append({
                "yaw": yaw,
                "follows": -95.0 < grasp <= 95.0,
                "mode": "mode0" if abs(float(wrap(grasp - yaw))) < 90 else "mode180",
                "wrist_rotation_deg": rotation,
            })
        total = len(h5["data"])
    if not rows:
        raise AnalysisError(f"dataset has no close command: {path}")
    bins = {}
    for lo, hi in SIGNED_BINS:
        group = [r for r in rows if signed_label(r["yaw"]) == f"[{lo},{hi})"]
        bins[f"[{lo},{hi})"] = {
            "n": len(group),
            "convention_fraction": None if not group else sum(r["follows"] for r in group) / len(group),
            "mode0": sum(r["mode"] == "mode0" for r in group),
            "mode180": sum(r["mode"] == "mode180" for r in group),
        }
    return {
        "tool_contract": "dataset_convention.py; first close; (-90,90] +/-5deg",
        "path": str(path), "total_demos": total, "demos_with_close": len(rows),
        "convention_fraction": sum(r["follows"] for r in rows) / len(rows), "signed_yaw_bins": bins,
    }


def dataset_yaws(paths: list[Path]) -> list[float]:
    values = []
    for path in paths:
        require_file(path)
        with h5py.File(path, "r") as h5:
            for name in sorted(h5["data"]):
                values.append(float(heading_x(h5["data"][name]["obs"]["toy_truck_orientations"][0])))
    return values


def yaw_histogram(path: Path) -> dict[str, int]:
    values = dataset_yaws([path])
    return {f"[{lo},{hi})": sum(signed_label(v) == f"[{lo},{hi})" for v in values) for lo, hi in SIGNED_BINS}


def generation_yield(success_files: list[Path], failed_files: list[Path]) -> dict[str, Any]:
    if not success_files or not failed_files:
        raise AnalysisError("generation yield needs successful and failed HDF5 lists")
    successes, failures = dataset_yaws(success_files), dataset_yaws(failed_files)
    rows = []
    for lo, hi in SIGNED_BINS:
        label = f"[{lo},{hi})"
        s, f = sum(signed_label(v) == label for v in successes), sum(signed_label(v) == label for v in failures)
        rows.append({"yaw_bin": label, "successes": s, "failures": f, "attempts": s + f,
                     "yield": None if s + f == 0 else s / (s + f)})
    return {"successes": len(successes), "failures": len(failures), "attempts": len(successes) + len(failures),
            "yield": len(successes) / (len(successes) + len(failures)), "signed_yaw_bins": rows,
            "denominator": "completed success+failed demos; settling-reset discarded layouts excluded"}


def analyze_condition(name: str, cfg: dict[str, Any], parent: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    dataset = resolve(cfg["dataset"]["combined"], parent)
    source = resolve(cfg["dataset"]["source"], parent)
    additional = resolve(cfg["dataset"].get("additional_500", cfg["dataset"]["combined"]), parent)
    expected = 1000 if name == "E3" else 500
    conv = convention(dataset)
    if conv["total_demos"] != expected:
        raise AnalysisError(f"{name} dataset count {conv['total_demos']} != {expected}")
    add_conv = convention(additional)
    if name != "E0" and add_conv["total_demos"] != 500:
        raise AnalysisError(f"{name} additional dataset is not 500")
    models, cohorts, counts = {}, {}, []
    all_rows = []
    for seed in SEEDS:
        model = cfg["models"][seed]
        rows, verify = observer_rows(resolve(model["trajectory_dir"], parent), resolve(model["results_json"], parent))
        cohorts[seed] = rows
        all_rows.extend(rows)
        successes = sum(r["success"] for r in rows)
        counts.append(successes)
        stages = Counter(r["stage"] for r in rows)
        models[seed] = {"successes": successes, "trials": 100, "rate": successes / 100,
                        "wilson95": wilson95(successes, 100),
                        "v3_1_stages": {s: stages[s] for s in STAGES},
                        "close_by_abs_yaw": close_bins(rows), "observer_verification": verify}
    stages = Counter(r["stage"] for r in all_rows)
    generation = cfg["generation"]
    return ({
        "pooled_successes": sum(counts), "pooled_trials": 200,
        "models": models,
        "condition_success": {"seed_success_counts": counts, "mean_rate": statistics.mean(counts) / 100,
                              "sample_sd_rate_ddof1": statistics.stdev(counts) / 100,
                              "pooled_successes": sum(counts), "pooled_trials": 200},
        "v3_1_pooled_stages": {s: stages[s] for s in STAGES},
        "close_by_abs_yaw_pooled": close_bins(all_rows),
        "source_yaw_histogram": yaw_histogram(source),
        "dataset_convention": {"combined": conv, "additional_500": add_conv},
        "generation_yield": generation_yield(
            [resolve(p, parent) for p in generation["success_files"]],
            [resolve(p, parent) for p in generation["failed_files"]]),
        "provenance": {"combined_dataset": str(dataset), "source_dataset": str(source), "additional_500": str(additional)},
    }, cohorts)


def subset_count(rows: list[dict[str, Any]]) -> tuple[int, int]:
    group = [r for r in rows if abs(float(wrap(r["yaw0"]))) <= 90.0]
    return sum(r["success"] for r in group), len(group)


def comparison(first: dict[str, Any], first_rows: dict[str, list[dict[str, Any]]], second: dict[str, Any], second_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    per_seed = {}
    for seed in SEEDS:
        fs, fn = first["models"][seed]["successes"], 100
        ss, sn = second["models"][seed]["successes"], 100
        fa, fan = subset_count(first_rows[seed]); sa, san = subset_count(second_rows[seed])
        per_seed[seed] = {"whole": ztest(fs, fn, ss, sn), "abs_yaw_le_90": ztest(fa, fan, sa, san)}
    f_abs = [subset_count(first_rows[s]) for s in SEEDS]
    s_abs = [subset_count(second_rows[s]) for s in SEEDS]
    return {
        "per_seed_unpaired": per_seed,
        "pooled_whole": ztest(first["pooled_successes"], 200, second["pooled_successes"], 200),
        "pooled_abs_yaw_le_90": ztest(sum(x[0] for x in f_abs), sum(x[1] for x in f_abs),
                                           sum(x[0] for x in s_abs), sum(x[1] for x in s_abs)),
    }


def decisions(conditions: dict[str, Any], comparisons: dict[str, Any]) -> dict[str, Any]:
    e0, e1, e2, e3 = (conditions[n] for n in CONDITIONS)
    target = e1["close_by_abs_yaw_pooled"]["[45,90)"]
    prediction2 = target["between"] <= 12 and target["heading_error_abs_median_deg"] is not None and target["heading_error_abs_median_deg"] < 10
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
    scores = {n: conditions[n]["pooled_successes"] for n in ("E1", "E2", "E3")}
    selected = "E3" if scores["E3"] >= scores["E1"] else "E1"
    if scores["E2"] > scores["E1"] and scores["E2"] > scores["E3"]:
        selected = "E2"
    return {
        "H1_double_prime": {"verdict": verdict,
            "rules": {"both_seeds_whole_strictly_improve": both,
                      "abs_yaw_le_90_success_above_E0_7_of_99": abs_up,
                      "abs_yaw_le_90_p_one_sided_lt_0_05": significant,
                      "prediction_2": prediction2},
            "prediction_2_observed": {"between": target["between"], "registered_baseline": "25/55",
                                      "half_or_less_integer_cutoff": 12,
                                      "heading_error_median_deg": target["heading_error_abs_median_deg"],
                                      "heading_error_lt_10_deg": target["heading_error_abs_median_deg"] is not None and target["heading_error_abs_median_deg"] < 10}},
        "E2_specificity_clause": {"verdict": "not_mechanically_adjudicable",
            "reason": "Preregistration says E2 rises 'similarly' to E1 but registers no numeric similarity threshold; none is inferred post hoc.",
            "observed": {"E1_pooled_successes": scores["E1"], "E2_pooled_successes": scores["E2"]}},
        "final_condition": {"selected": selected, "pooled_successes": scores,
            "rule": "higher of E1/E3; tie E3; E2 only if strictly greater than both"},
    }


def markdown(result: dict[str, Any]) -> str:
    lines = ["# preregistration-2 C/D results", "", "## Success", "",
             "| condition | seed101 | seed111 | mean ± sample SD (ddof=1) | pooled |", "|---|---:|---:|---:|---:|"]
    for name in CONDITIONS:
        c = result["conditions"][name]; s = c["condition_success"]
        lines.append(f"| {name} | {c['models']['101']['successes']}/100 | {c['models']['111']['successes']}/100 | {100*s['mean_rate']:.1f}% ± {100*s['sample_sd_rate_ddof1']:.2f}%p | {c['pooled_successes']}/200 |")
    lines += ["", "## Preregistered decision", "", f"- H1″: **{result['preregistered_decisions']['H1_double_prime']['verdict']}**",
              f"- E2 specificity: **not mechanically adjudicable** (`similar` has no registered numeric threshold).",
              f"- Final condition: **{result['preregistered_decisions']['final_condition']['selected']}**", "",
              "## v3.1 stage distribution", "", "| condition | approach | grasp | transport | placement | success |", "|---|---:|---:|---:|---:|---:|"]
    for name in CONDITIONS:
        st = result["conditions"][name]["v3_1_pooled_stages"]
        lines.append("| " + name + " | " + " | ".join(str(st[s]) for s in STAGES) + " |")
    lines += ["", "## Limits", "", "- All condition comparisons are unpaired.",
              "- E2 'similar' cannot be converted into a post-hoc cutoff; observed values are reported without a specificity verdict.",
              "- Yield excludes discarded pre-attempt settling layouts.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True); parser.add_argument("--output", required=True); parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    input_path, output, md = map(lambda p: Path(p).expanduser().resolve(), (args.input, args.output, args.markdown))
    require_file(input_path)
    if output.exists() or md.exists():
        raise AnalysisError("refusing to overwrite analysis output")
    manifest = json.loads(input_path.read_text())
    if manifest.get("schema_version") != 1 or set(manifest.get("conditions", {})) != set(CONDITIONS):
        raise AnalysisError("analysis input must have schema 1 and exactly E0/E1/E2/E3")
    conditions, cohorts = {}, {}
    for name in CONDITIONS:
        conditions[name], cohorts[name] = analyze_condition(name, manifest["conditions"][name], input_path.parent)
    e0_abs = [subset_count(cohorts["E0"][s]) for s in SEEDS]
    if (sum(x[0] for x in e0_abs), sum(x[1] for x in e0_abs)) != (7, 99) or conditions["E0"]["pooled_successes"] != 39:
        raise AnalysisError("registered E0 anchors do not match observer data (whole 39/200, |yaw|<=90 7/99)")
    comparisons = {f"{name}_vs_E0": comparison(conditions[name], cohorts[name], conditions["E0"], cohorts["E0"]) for name in ("E1", "E2", "E3")}
    comparisons["E3_vs_E1"] = comparison(conditions["E3"], cohorts["E3"], conditions["E1"], cohorts["E1"])
    result = {"schema_version": 2, "created_utc": datetime.now(timezone.utc).isoformat(), "input_manifest": str(input_path),
              "statistical_contract": {"Wilson_CI": 0.95, "condition_sd_ddof": 1, "tests": "unpaired one-sided pooled two-proportion z; first > second", "E0_anchors": {"whole": "39/200", "abs_yaw_le_90": "7/99"}},
              "conditions": conditions, "comparisons": comparisons}
    result["preregistered_decisions"] = decisions(conditions, comparisons)
    with output.open("x") as stream: json.dump(result, stream, indent=2, sort_keys=True); stream.write("\n")
    try:
        with md.open("x") as stream: stream.write(markdown(result))
    except Exception:
        output.unlink(missing_ok=True); raise


if __name__ == "__main__":
    main()
