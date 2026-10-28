"""Toy2Box post-hoc metric (written 2026-10-28 after the E0/E1/E2/E4 observers; exploratory, not preregistered):
why do secured grasps fail in transport?

Per trial whose first near close was secured (recovery_metrics.py definitions):
  slipped      the truck left the gripper (EEF-to-truck > 6 cm) before the gripper next opened
  grasp height EEF height above the truck center in the truck frame at that close onset [cm]
  lift delay   steps from that close onset until the EEF has risen 1 cm (the demos wait exactly 40 steps)

Usage: python3 transport_slip.py <label>=<npz_dir>[,<npz_dir>...] [...] [--json out.json]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from close_precision import rotmat  # noqa: E402
from recovery_metrics import FOLLOW_M, trial  # noqa: E402

HEIGHT_BINS = ((-9.0, 0.8), (0.8, 1.1), (1.1, 1.3), (1.3, 1.5), (1.5, 9.0))
DELAY_BINS = ((0, 15), (15, 25), (25, 35), (35, 45), (45, 10_000))


def secured_rows(npz_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(npz_dir, "trial_*.npz"))):
        t = trial(path)
        if not (t["secured"] and t["secured"][0]):
            continue
        z = np.load(path, allow_pickle=True)
        eef, tr, closed = z["eef_pos"], z["truck_pos"], z["action"][:, -1] < 0
        k = t["near_closes"][0]
        height = float((rotmat(z["truck_quat"][k]).T @ (eef[k] - tr[k]))[2] * 100.0)
        up = np.flatnonzero(eef[k:, 2] - eef[k, 2] > 0.01)
        opened = np.flatnonzero(~closed[k:])
        end = k + int(opened[0]) if opened.size else len(closed)
        dist = np.linalg.norm(eef - tr, axis=1)
        rows.append({"path": os.path.basename(path), "success": t["success"],
                     "slipped": bool(np.any(dist[k:end] > FOLLOW_M)), "grasp_height_cm": height,
                     "lift_delay_steps": int(up[0]) if up.size else None})
    return rows


def rate_table(rows, key, bins):
    out = []
    for lo, hi in bins:
        sel = [r for r in rows if r[key] is not None and lo <= r[key] < hi]
        out.append({"bin": [lo, hi], "n": len(sel), "slipped": sum(r["slipped"] for r in sel)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="+", help="label=dir[,dir...]")
    ap.add_argument("--json")
    a = ap.parse_args()
    result, pooled = {}, []
    for group in a.groups:
        label, dirs = group.split("=", 1)
        rows = [r for d in dirs.split(",") for r in secured_rows(d)]
        pooled += rows
        result[label] = {"secured_first": len(rows), "slipped": sum(r["slipped"] for r in rows),
                         "grasp_height_median_cm": float(np.median([r["grasp_height_cm"] for r in rows])) if rows else None}
        print(f"{label}: secured first {len(rows)}, slipped {result[label]['slipped']}, "
              f"grasp height median {result[label]['grasp_height_median_cm']:.2f} cm")
    result["pooled_by_grasp_height_cm"] = rate_table(pooled, "grasp_height_cm", HEIGHT_BINS)
    result["pooled_by_lift_delay_steps"] = rate_table(pooled, "lift_delay_steps", DELAY_BINS)
    for name in ("pooled_by_grasp_height_cm", "pooled_by_lift_delay_steps"):
        print(name)
        for b in result[name]:
            rate = b["slipped"] / b["n"] if b["n"] else float("nan")
            print(f"  {b['bin'][0]:>6} .. {b['bin'][1]:<6} n={b['n']:3d} slip {rate:.2f}")
    if a.json:
        json.dump(result, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
