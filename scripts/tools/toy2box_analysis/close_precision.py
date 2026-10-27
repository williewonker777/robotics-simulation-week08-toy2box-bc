"""Toy2Box mechanism metric: wrist alignment at the decisive gripper close, per v3.1 stage and |truck yaw| bin.

Decisive close = the close onset (action[-1] crosses below 0) with the smallest EEF-to-truck distance.
Heading error = gripper x-axis heading minus truck x-axis heading, wrapped to (-90, 90] (parallel-gripper symmetry).
Offsets are the EEF position in the truck frame at that step.

Usage: python3 close_precision.py <npz_dir> [<npz_dir> ...] [--json out.json]
Stages come from stage_v3.py (imported from this directory).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_v3 import classify  # noqa: E402


def rotmat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def heading(R):
    return np.degrees(np.arctan2(R[..., 1, 0], R[..., 0, 0]))


def trial(path):
    z = np.load(path, allow_pickle=True)
    eef, tr, tq, act = z["eef_pos"], z["truck_pos"], z["truck_quat"], z["action"]
    yaw0 = float(heading(rotmat(z["reset_initial_truck_quat"])))
    row = dict(path=path, yaw0=yaw0, stage=classify(path)["stage"], closed=False)
    closed = act[:, -1] < 0
    onsets = np.flatnonzero(closed & ~np.concatenate([[False], closed[:-1]]))
    if onsets.size:
        k = int(onsets[np.argmin(np.linalg.norm(eef[onsets] - tr[onsets], axis=1))])
        Rt = rotmat(tq[k])
        off = Rt.T @ (eef[k] - tr[k]) * 100.0
        herr = (heading(rotmat(z["eef_quat"][k])) - heading(Rt) + 90.0) % 180.0 - 90.0
        # grasp mode relative to the truck: ~0 or ~180 deg, or in between (neither mode)
        rel = (heading(rotmat(z["eef_quat"][k])) - heading(Rt) + 180.0) % 360.0 - 180.0
        mode = "mode0" if abs(rel) < 30.0 else ("mode180" if abs(rel) > 150.0 else "between")
        row.update(closed=True, k=k, ox=float(off[0]), oy=float(off[1]), oz=float(off[2]), herr=float(herr), mode=mode)
    return row


def summary(rows):
    g = [r for r in rows if r["closed"]]
    if not g:
        return f"n={len(rows):3d} (no close)"
    h = np.abs([r["herr"] for r in g])
    return (
        f"n={len(rows):3d} closed={len(g):3d} |herr| med {np.median(h):5.1f} p75 {np.percentile(h, 75):5.1f}"
        f"  |ox| med {np.median(np.abs([r['ox'] for r in g])):4.1f} |oy| med {np.median(np.abs([r['oy'] for r in g])):4.1f} cm"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_dirs", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = [trial(p) for d in a.npz_dirs for p in sorted(glob.glob(os.path.join(d, "trial_*.npz")))]
    print(f"{len(rows)} trials from {len(a.npz_dirs)} dir(s)")
    print("by v3.1 stage:")
    for st in ("success", "placement", "transport", "grasp", "approach"):
        print(f"  {st:9s}", summary([r for r in rows if r["stage"] == st]))
    print("by |initial truck yaw| (all trials, decisive close):")
    for lo, hi in ((0, 45), (45, 90), (90, 135), (135, 181)):
        g = [r for r in rows if lo <= abs(r["yaw0"]) < hi]
        s = sum(r["stage"] == "success" for r in g)
        print(f"  [{lo:3d},{hi:3d})", summary(g), f"  success {s}/{len(g)}")
    print("grasp mode at the decisive close (truck-relative heading ~0 / ~180 / between):")
    for lo, hi in ((0, 45), (45, 90), (90, 135), (135, 181)):
        g = [r for r in rows if lo <= abs(r["yaw0"]) < hi and r["closed"]]
        c = {m: sum(r["mode"] == m for r in g) for m in ("mode0", "mode180", "between")}
        print(f"  [{lo:3d},{hi:3d}) n={len(g):3d}  ~0 {c['mode0']:3d}  ~180 {c['mode180']:3d}  between {c['between']:3d}")
    if a.json:
        json.dump(rows, open(a.json, "w"))


if __name__ == "__main__":
    main()
