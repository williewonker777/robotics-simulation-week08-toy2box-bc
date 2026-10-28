"""Toy2Box H3 mechanism metrics (fixed before any H3 data, 2026-10-27 16:55 KST): does the policy recover after a failed grasp?

Per trial (observer NPZ):
  near close   a close onset (action[-1] crosses below 0) with the EEF within 6 cm of the truck
  secured      a held run (truck >= 3 cm above its rest height, EEF within 6 cm, gripper closed, >= 40 consecutive steps)
               that starts before the gripper next opens
Outcome groups: first near close secured / first near close failed / never closed near the truck.
Recovery rate = final successes among trials whose first near close failed.
Retry rate    = trials whose first near close failed and that closed near the truck again.
Drop re-grasp = trials with a secured first close that ended in failure and closed near the truck again.

Usage: python3 recovery_metrics.py <npz_dir> [...] [--json out.json]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_v3 import runs  # noqa: E402

LIFT_M, FOLLOW_M, NEAR_M, HOLD_STEPS, REST_STEPS = 0.03, 0.06, 0.06, 40, 30


def trial(path):
    z = np.load(path, allow_pickle=True)
    eef, tr, act = z["eef_pos"], z["truck_pos"], z["action"]
    rest = float(np.median(tr[:REST_STEPS, 2]))
    closed = act[:, -1] < 0
    dist = np.linalg.norm(eef - tr, axis=1)
    held = runs((tr[:, 2] - rest >= LIFT_M) & (dist <= FOLLOW_M) & closed, HOLD_STEPS)
    onsets = np.flatnonzero(closed & ~np.concatenate([[False], closed[:-1]]))
    near = [int(k) for k in onsets if dist[k] <= NEAR_M]
    secured = []
    for k in near:
        opened = np.flatnonzero(~closed[k:])
        end = k + (int(opened[0]) if opened.size else len(closed) - k)
        secured.append(bool(held[k:end].any()))
    return {"path": os.path.basename(path), "success": bool(z["final_success"]), "near_closes": near, "secured": secured}


def summarize(rows):
    first_ok = [r for r in rows if r["secured"] and r["secured"][0]]
    first_fail = [r for r in rows if r["secured"] and not r["secured"][0]]
    never = [r for r in rows if not r["secured"]]
    lost = [r for r in first_ok if not r["success"]]
    return {
        "trials": len(rows),
        "first_secured": len(first_ok), "success_given_first_secured": sum(r["success"] for r in first_ok),
        "first_failed": len(first_fail), "recoveries": sum(r["success"] for r in first_fail),
        "first_failed_retried_near": sum(len(r["secured"]) > 1 for r in first_fail),
        "first_failed_later_secured": sum(any(r["secured"][1:]) for r in first_fail),
        "never_near": len(never), "success_given_never_near": sum(r["success"] for r in never),
        "secured_then_failed": len(lost), "secured_then_failed_regrasped_near": sum(len(r["secured"]) > 1 for r in lost),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_dirs", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    out = {}
    allrows = []
    for d in a.npz_dirs:
        rows = [trial(p) for p in sorted(glob.glob(os.path.join(d, "trial_*.npz")))]
        allrows += rows
        out[d] = summarize(rows)
    out["pooled"] = summarize(allrows)
    for k, s in out.items():
        print(f"{k}: first near close secured {s['first_secured']} (success {s['success_given_first_secured']}), "
              f"failed {s['first_failed']} (recovered {s['recoveries']}, retried {s['first_failed_retried_near']}), "
              f"never near {s['never_near']}; secured-then-failed {s['secured_then_failed']} (re-grasped {s['secured_then_failed_regrasped_near']})")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
