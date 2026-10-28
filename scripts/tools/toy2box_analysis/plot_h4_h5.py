#!/usr/bin/env python3
"""Create compact descriptive figures for the registered H4 and H5 results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPOSITORY = Path(__file__).resolve().parents[3]


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials**2)) / denominator
    return center - half_width, center + half_width


def plot_success_ci(h4: dict, h5: dict, output: Path) -> None:
    conditions = ["E0\nbaseline", "E2\nselected", "E6\nH4", "E7\nH5"]
    successes = [
        int(h4["performance"]["pooled_comparisons"]["E0"]["baseline"][0]),
        int(h4["performance"]["pooled_comparisons"]["E2"]["baseline"][0]),
        int(h4["pooled_successes"]),
        int(h5["pooled_successes"]),
    ]
    trials = [200] * len(successes)
    rates = [success / trial for success, trial in zip(successes, trials)]
    intervals = [wilson_interval(success, trial) for success, trial in zip(successes, trials)]
    errors = [[rate - low for rate, (low, _) in zip(rates, intervals)], [high - rate for rate, (_, high) in zip(rates, intervals)]]

    figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    colors = ["#9aa0a6", "#2f6f9f", "#d9792b", "#8c62aa"]
    bars = axis.bar(conditions, [100.0 * rate for rate in rates], color=colors, width=0.66)
    axis.errorbar(range(len(rates)), [100.0 * rate for rate in rates], yerr=[[100.0 * value for value in errors[0]], [100.0 * value for value in errors[1]]], fmt="none", ecolor="#202124", capsize=5, linewidth=1.3)
    for bar, success, trial, (_, high) in zip(bars, successes, trials, intervals):
        axis.text(bar.get_x() + bar.get_width() / 2.0, 100.0 * high + 0.8, f"{success}/{trial}", ha="center", va="bottom", fontsize=10)
    axis.set_ylim(0, max(32, 100.0 * max(high for _, high in intervals) + 6))
    axis.set_ylabel("Successful rollouts (%)")
    axis.set_title("H4/H5 rollout outcomes with descriptive Wilson 95% intervals")
    axis.grid(axis="y", alpha=0.22)
    axis.text(0.5, -0.18, "Registered decisions use the preregistered one-sided tests; intervals here are descriptive.", transform=axis.transAxes, ha="center", va="top", fontsize=8.5, color="#4a4a4a")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_mechanisms(h4: dict, h5: dict, contract: dict, output: Path) -> None:
    h4_generated_limit = float(h4["mechanism"]["baseline_generated_close_height_mm"]) - 3.0
    h4_policy_limit = float(h4["mechanism"]["baseline_policy_close_height_mm"]) - 3.0
    h4_slip, h4_secured = h4["mechanism"]["baseline_slip"]
    h5_generated = h5["mechanism"]["baseline_generated"]
    h5_speed_limit = 0.75 * float(h5_generated["preclose10_mean_eef_speed_m_s_median"])
    h5_xy_limit = float(contract["mechanism"]["policy_xy_mm_median_max"])
    h5_secured, h5_near = contract["mechanism"]["baseline_acquisition"]

    rows = [
        ["H4 M1", f"generated close z ≤ {h4_generated_limit:.2f} mm", f"{h4['generated']['first_close_height_mm_median']:.2f} mm", "PASS" if h4["mechanism"]["M1"] else "FAIL"],
        ["H4 M2", f"policy near-close z ≤ {h4_policy_limit:.2f} mm", f"{h4['rollouts']['first_near_close_height_mm_median']:.2f} mm", "PASS" if h4["mechanism"]["M2"] else "FAIL"],
        ["H4 M3", f"slip < {100*h4_slip/h4_secured:.1f}% and p < .05", f"{100*h4['rollouts']['slipped']/h4['rollouts']['secured_first']:.1f}%, p={h4['mechanism']['slip_p_one_sided']:.3f}", "PASS" if h4["mechanism"]["M3"] else "FAIL"],
        ["H5 M1", f"pre-close speed ≤ {h5_speed_limit:.4f} m/s + audit/geometry", f"{h5['generated']['preclose10_mean_eef_speed_m_s_median']:.4f} m/s; audit pass", "PASS" if h5["mechanism"]["M1"] else "FAIL"],
        ["H5 M2", f"policy near-close XY ≤ {h5_xy_limit:.2f} mm", f"{h5['rollouts']['first_near_close_xy_offset_mm_median']:.2f} mm", "PASS" if h5["mechanism"]["M2"] else "FAIL"],
        ["H5 M3", f"acquisition > {100*h5_secured/h5_near:.1f}% and p < .05", f"{100*h5['rollouts']['secured_first']/h5['rollouts']['first_near_close_count']:.1f}%, p={h5['mechanism']['acquisition_p_one_sided']:.3f}", "PASS" if h5["mechanism"]["M3"] else "FAIL"],
    ]
    figure, axis = plt.subplots(figsize=(10.8, 4.1), constrained_layout=True)
    axis.axis("off")
    table = axis.table(cellText=rows, colLabels=["Registered test", "Prediction", "Observed", "Outcome"], cellLoc="left", colLoc="left", loc="center", colWidths=[0.12, 0.40, 0.34, 0.10])
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    table.scale(1.0, 1.55)
    for column in range(4):
        table[(0, column)].set_facecolor("#dce6ef")
        table[(0, column)].set_text_props(weight="bold")
    for row_index, row in enumerate(rows, start=1):
        table[(row_index, 3)].set_facecolor("#d7ead3" if row[3] == "PASS" else "#f3d2ce")
        table[(row_index, 3)].set_text_props(weight="bold")
    axis.set_title("Registered H4/H5 mechanism predictions and observed measurements", fontsize=13, pad=12)
    axis.text(0.5, 0.01, "M1 rows show the principal quantitative component; outcome colors use the complete frozen mechanism rule.", transform=axis.transAxes, ha="center", va="bottom", fontsize=8.5, color="#4a4a4a")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h4-result", type=Path, default=REPOSITORY / "artifacts/experiments/h4/h4_results.json")
    parser.add_argument("--h5-result", type=Path, default=REPOSITORY / "artifacts/experiments/h5/h5_results.json")
    parser.add_argument("--h5-contract", type=Path, default=REPOSITORY / "artifacts/h5_registration/preregistration.json")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY / "artifacts/figures")
    args = parser.parse_args()
    outputs = [args.output_dir / "h4_h5_success_ci.png", args.output_dir / "h4_h5_mechanisms.png"]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        parser.error(f"refusing to overwrite existing output: {existing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    h4 = json.loads(args.h4_result.read_text())
    h5 = json.loads(args.h5_result.read_text())
    contract = json.loads(args.h5_contract.read_text())
    plot_success_ci(h4, h5, outputs[0])
    plot_mechanisms(h4, h5, contract, outputs[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
