"""Recompute the preregistered H1'' and H3 verdicts and the final-model choice from the rollout records in artifacts/.

No simulator and no generated datasets are needed. The rules are those of report/PREREGISTRATION.md
(preregistrations 2 and 3 with supplement 3) and match the registered analyzer whose output is
artifacts/experiments/registered_results.json:
  stage     stage_v3.classify (v3.1)
  close     close_precision.trial: decisive close, heading error, grasp mode
  recovery  recovery_metrics.trial / summarize
  tests     unpaired one-sided pooled two-proportion z test (first rate > second rate)

Usage: python3 scripts/tools/toy2box_analysis/verdicts.py [--root <repo root>] [--json out.json]
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from close_precision import trial as close_trial  # noqa: E402
from recovery_metrics import summarize, trial as recovery_trial  # noqa: E402
from stage_v3 import classify  # noqa: E402

SEEDS = ("101", "111")
ABS_BINS = ((0, 45), (45, 90), (90, 135), (135, 181))
ROLLOUTS = {
    "E0": "artifacts/e0_baseline/rollouts/seed{seed}",
    "E1": "artifacts/experiments/e1/seed{seed}/rollouts",
    "E2": "artifacts/experiments/e2/seed{seed}/rollouts",
    "E4": "artifacts/experiments/e4/seed{seed}/rollouts",
}
E0_ABS, E0_WHOLE, E0_RECOVERY = (7, 99), 39, (5, 102)


def wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def ztest(fs, fn, ss, sn):
    pooled = (fs + ss) / (fn + sn)
    z = (fs / fn - ss / sn) / math.sqrt(pooled * (1 - pooled) * (1 / fn + 1 / sn))
    return {"first": [fs, fn], "second": [ss, sn], "z": z, "p_one_sided": 0.5 * math.erfc(z / math.sqrt(2))}


def model_rows(npz_dir):
    paths = sorted(glob.glob(os.path.join(npz_dir, "trial_*.npz")))
    if [os.path.basename(p) for p in paths] != [f"trial_{i:05d}.npz" for i in range(100)]:
        raise SystemExit(f"{npz_dir}: expected trial_00000..trial_00099")
    rows = []
    for p in paths:
        row = close_trial(p)
        with np.load(p, allow_pickle=True) as z:
            row["success"] = bool(np.asarray(z["final_success"]).item())
        if row["success"] != (classify(p)["stage"] == "success"):
            raise SystemExit(f"{p}: stage/success disagreement")
        rows.append(row)
    return rows, [recovery_trial(p) for p in paths]


def close_bins(rows):
    out = {}
    for lo, hi in ABS_BINS:
        group = [r for r in rows if lo <= abs(wrap(r["yaw0"])) < hi]
        closed = [r for r in group if r["closed"]]
        modes = Counter(r["mode"] for r in closed)
        err = [abs(r["herr"]) for r in closed]
        out[f"[{lo},{hi})"] = {"trials": len(group), "successes": sum(r["success"] for r in group), "closed": len(closed),
                               "heading_error_abs_median_deg": float(np.median(err)) if err else None,
                               "mode0": modes["mode0"], "mode180": modes["mode180"], "between": modes["between"]}
    return out


def abs_le_90(rows):
    group = [r for r in rows if abs(wrap(r["yaw0"])) <= 90.0]
    return sum(r["success"] for r in group), len(group)


def condition(root, name):
    seeds, all_rows, all_rec = {}, [], []
    for s in SEEDS:
        rows, rec = model_rows(os.path.join(root, ROLLOUTS[name].format(seed=s)))
        seeds[s] = {"successes": sum(r["success"] for r in rows), "abs_le_90": abs_le_90(rows)}
        all_rows += rows
        all_rec += rec
    counts = [seeds[s]["successes"] for s in SEEDS]
    return {"seeds": seeds, "pooled": sum(counts), "mean": float(np.mean(counts)) / 100,
            "sd_ddof1": float(np.std(counts, ddof=1)) / 100, "close_bins": close_bins(all_rows),
            "recovery": summarize(all_rec),
            "abs_le_90": [sum(seeds[s]["abs_le_90"][i] for s in SEEDS) for i in (0, 1)]}


def h1(c):
    e0, e1 = c["E0"], c["E1"]
    t = c["E1"]["close_bins"]["[45,90)"]
    between_fraction = t["between"] / t["closed"]
    prediction2 = between_fraction <= 0.5 * 25 / 55 and t["heading_error_abs_median_deg"] < 10
    both = all(e1["seeds"][s]["successes"] > e0["seeds"][s]["successes"] for s in SEEDS)
    test = ztest(*e1["abs_le_90"], *e0["abs_le_90"])
    abs_up = e1["abs_le_90"][0] > E0_ABS[0]
    verdict = ("supported" if both and abs_up and test["p_one_sided"] < 0.05 and prediction2
               else "rejected" if not abs_up or not prediction2 else "inconclusive")
    return {"verdict": verdict, "both_seeds_improve": both, "abs_le_90_up": abs_up, "abs_le_90_test": test,
            "prediction2": prediction2, "between_fraction_45_90": between_fraction,
            "heading_error_median_45_90": t["heading_error_abs_median_deg"]}


def h3(c):
    e0, e4 = c["E0"], c["E4"]
    both = all(e4["seeds"][s]["successes"] > e0["seeds"][s]["successes"] for s in SEEDS)
    whole = ztest(e4["pooled"], 200, e0["pooled"], 200)
    rec = e4["recovery"]
    rec_test = ztest(rec["recoveries"], rec["first_failed"], *E0_RECOVERY)
    rec_up = rec["recoveries"] / rec["first_failed"] > E0_RECOVERY[0] / E0_RECOVERY[1]
    mean_up = e4["mean"] > 0.195
    verdict = ("supported" if both and whole["p_one_sided"] < 0.05 and rec_test["p_one_sided"] < 0.05
               else "rejected" if not mean_up or not rec_up else "inconclusive")
    return {"verdict": verdict, "both_seeds_improve": both, "whole_test": whole, "recovery_test": rec_test,
            "recovery_up": rec_up, "mean_up": mean_up}


def final_choice(c, v1, v3):
    if v1 == "supported" and v3 == "supported":
        return "E5 (required; not run because this branch did not occur)"
    if v1 == "supported":
        return "E1"
    if v3 == "supported":
        return "E4"
    score = {n: c[n]["pooled"] for n in ("E0", "E1", "E2", "E4")}
    best = [n for n, v in score.items() if v == max(score.values())]
    if len(best) == 1 or "E0" in best:
        return best[0] if len(best) == 1 else "E0"
    low = {n: min(c[n]["seeds"][s]["successes"] for s in SEEDS) for n in best}
    best = [n for n in best if low[n] == max(low.values())]
    return min(best, key={"E1": 0, "E4": 1, "E2": 2}.__getitem__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--json")
    a = ap.parse_args()
    c = {n: condition(a.root, n) for n in ROLLOUTS}
    if tuple(c["E0"]["abs_le_90"]) != E0_ABS or c["E0"]["pooled"] != E0_WHOLE:
        raise SystemExit("E0 anchors disagree with the preregistered values")
    for n, x in c.items():
        print(f"{n}: seed101 {x['seeds']['101']['successes']}/100, seed111 {x['seeds']['111']['successes']}/100, "
              f"pooled {x['pooled']}/200, mean {100 * x['mean']:.1f}% +/- {100 * x['sd_ddof1']:.2f}%p (sample SD)")
    v1, v3 = h1(c), h3(c)
    t = v1["abs_le_90_test"]
    print(f"H1'': {v1['verdict']}  (both seeds improve {v1['both_seeds_improve']}; |yaw|<=90 {t['first']} vs {t['second']} "
          f"p={t['p_one_sided']:.3f}; prediction 2 {v1['prediction2']}: between {v1['between_fraction_45_90']:.3f} <= 0.227, "
          f"heading error median {v1['heading_error_median_45_90']:.1f} deg < 10)")
    print(f"H3: {v3['verdict']}  (both seeds improve {v3['both_seeds_improve']}; whole p={v3['whole_test']['p_one_sided']:.3f}; "
          f"recovery {v3['recovery_test']['first']} vs {v3['recovery_test']['second']} p={v3['recovery_test']['p_one_sided']:.3f})")
    final = final_choice(c, v1["verdict"], v3["verdict"])
    print(f"final condition: {final}")
    if a.json:
        json.dump({"conditions": c, "H1_double_prime": v1, "H3": v3, "final_condition": final}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
