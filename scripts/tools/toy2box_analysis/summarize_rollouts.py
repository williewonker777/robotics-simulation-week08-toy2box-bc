"""Success summary of one condition from observer rollout records (one trial_*.npz directory per training seed).

Reports per-seed successes with Wilson 95% CIs, the across-seed mean and sample standard deviation (ddof=1),
and pooled counts for |initial truck yaw| <= 90 deg and > 90 deg.

Usage: python3 summarize_rollouts.py --label E0 seed101=<npz_dir> seed111=<npz_dir> [--json out.json]
"""

import argparse
import glob
import json
import math
import os

import numpy as np


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def yaw_deg(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--json")
    ap.add_argument("runs", nargs="+", help="name=npz_dir, one per training seed")
    a = ap.parse_args()
    out = {"condition": a.label, "per_seed": [], "source": "observer trial_*.npz final_success (stock play.py success predicate)"}
    small, large = [0, 0], [0, 0]
    for item in a.runs:
        name, d = item.split("=", 1)
        files = sorted(glob.glob(os.path.join(d, "trial_*.npz")))
        vec, yaws = [], []
        for p in files:
            z = np.load(p, allow_pickle=True)
            vec.append(bool(z["final_success"]))
            yaws.append(yaw_deg(z["reset_initial_truck_quat"]))
        k, n = sum(vec), len(vec)
        for ok, y in zip(vec, yaws):
            bucket = small if abs(y) <= 90.0 else large
            bucket[0] += ok
            bucket[1] += 1
        out["per_seed"].append({"run": name, "successes": k, "rollouts": n, "rate": k / n, "wilson_95_ci": wilson(k, n), "trial_success": vec})
        print(f"{a.label} {name}: {k}/{n} = {k / n:.1%}  Wilson95 [{wilson(k, n)[0]:.1%}, {wilson(k, n)[1]:.1%}]")
    rates = [r["rate"] for r in out["per_seed"]]
    out["mean"] = float(np.mean(rates))
    out["std_ddof1"] = float(np.std(rates, ddof=1)) if len(rates) > 1 else None
    out["pooled_abs_yaw_le_90"] = {"successes": small[0], "rollouts": small[1]}
    out["pooled_abs_yaw_gt_90"] = {"successes": large[0], "rollouts": large[1]}
    print(f"{a.label}: mean {out['mean']:.1%} +/- {out['std_ddof1'] * 100 if out['std_ddof1'] is not None else float('nan'):.2f}%p (ddof=1)"
          f"  |yaw|<=90 {small[0]}/{small[1]}  |yaw|>90 {large[0]}/{large[1]}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
