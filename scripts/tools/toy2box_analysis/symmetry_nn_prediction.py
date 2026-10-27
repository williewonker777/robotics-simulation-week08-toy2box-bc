"""Toy2Box: offline prediction of Mimic subtask-1 source selection, stock vs 180-degree truck symmetry.

Mimic's nearest_neighbor_object picks uniformly among the nn_k=3 sources minimising
|dpos| [m] + rotation angle [rad]. Targets are the 200 E0 evaluation layouts (robot base frame).
Each source's grasp heading is folded into (-90, 90] (as recorded), so the generated grasp heading
for a target is h_src + (target_frame_yaw - src_yaw); its mode relative to the target truck is 0 or 180.
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np


def yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def heading_x(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.degrees(np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z)))


ap = argparse.ArgumentParser(description="Top-3 nearest-source |delta yaw| per target yaw bin (stock, symmetric, folded, mode-consistent).")
ap.add_argument("--source", default="datasets/toy2box_agibot_codex_smooth.hdf5")
ap.add_argument("--json", default=None, help="write per-target rows here")
ap.add_argument("npz_dirs", nargs="+", help="observer rollout dirs (trial_*.npz) whose reset layouts are the targets")
args = ap.parse_args()
src = h5py.File(args.source, "r")
S = []
for n in src["data"]:
    o = src["data"][n]["obs"]
    a = src["data"][n]["actions"][:]
    k = int(np.flatnonzero(a[:, -1] < 0)[0])
    S.append((o["toy_truck_positions"][0], float(yaw(o["toy_truck_orientations"][0])), float(heading_x(o["eef_quat"][k]))))
spos = np.array([s[0] for s in S])
syaw = np.array([s[1] for s in S])
sh = np.array([s[2] for s in S])

T = []
for d in args.npz_dirs:
    for p in sorted(glob.glob(os.path.join(d, "trial_*.npz"))):
        z = np.load(p)
        T.append((z["reset_initial_truck_pos"], float(yaw(z["reset_initial_truck_quat"])), bool(z["final_success"])))


def top3(tpos, tyaw, symmetric):
    cands = []
    for flip in ((0.0, 180.0) if symmetric else (0.0,)):
        fyaw = tyaw + flip
        dyaw = wrap(fyaw - syaw)
        cost = np.linalg.norm(spos[:, :2] - tpos[:2], axis=1) + np.radians(np.abs(dyaw))
        for i in range(len(S)):
            cands.append((cost[i], i, dyaw[i]))
    cands.sort()
    out = []
    for c, i, dy in cands[:3]:
        gen_h = sh[i] + dy  # generated grasp heading after the rigid yaw transform
        mode = int(round(abs(wrap(gen_h - tyaw)) / 180.0)) * 180
        out.append((abs(dy), mode, wrap(gen_h)))
    return out


rows = []
for tpos, tyaw, ok in T:
    for sym in (False, True):
        t3 = top3(tpos, tyaw, sym)
        rows.append(dict(sym=sym, yaw=tyaw, success=ok, dyaw=[x[0] for x in t3], modes=[x[1] for x in t3]))
bins = [(0, 45), (45, 90), (90, 135), (135, 181)]
print("per |target yaw| bin: median |dyaw| over the top-3 candidates, share of targets whose top-3 mixes grasp modes")
for sym in (False, True):
    print("symmetric" if sym else "stock    ")
    for lo, hi in bins:
        g = [r for r in rows if r["sym"] == sym and lo <= abs(r["yaw"]) < hi]
        dy = np.concatenate([r["dyaw"] for r in g])
        mixed = np.mean([len(set(r["modes"])) > 1 for r in g])
        e0 = sum(r["success"] for r in g)
        print(f"  |yaw| [{lo:3d},{hi:3d}) n={len(g):3d} median|dyaw| {np.median(dy):5.1f}  p90 {np.percentile(dy, 90):5.1f}  mixed-mode targets {mixed:5.1%}   E0 success {e0}/{len(g)}")
if args.json:
    json.dump(rows, open(args.json, "w"))


# Canonical variant: fold every truck frame (sources and targets) to yaw in (-90, 90] by a 180-degree flip,
# then run the stock top-3 nearest neighbour in the folded frames. The generated grasp keeps the demonstrators'
# convention (gripper x heading in (-90, 90]) by construction.
def fold(a):
    a = wrap(a)
    return np.where(a > 90.0, a - 180.0, np.where(a <= -90.0, a + 180.0, a))


syaw_c = fold(syaw)
print("canonical (fold to (-90, 90])")
canon = []
for tpos, tyaw, ok in T:
    dy = fold(tyaw) - syaw_c  # both in (-90, 90], no wrap: the fold boundary is not crossed
    cost = np.linalg.norm(spos[:, :2] - tpos[:2], axis=1) + np.radians(np.abs(dy))
    idx = np.argsort(cost)[:3]
    canon.append(dict(yaw=tyaw, success=ok, dyaw=np.abs(dy[idx]).tolist()))
for lo, hi in bins:
    g = [r for r in canon if lo <= abs(r["yaw"]) < hi]
    dyv = np.concatenate([r["dyaw"] for r in g])
    print(f"  |yaw| [{lo:3d},{hi:3d}) n={len(g):3d} median|dyaw| {np.median(dyv):5.1f}  p90 {np.percentile(dyv, 90):5.1f}  mixed-mode targets   0.0% (by construction)")


# Mode-consistent variant (no flips): the target's grasp mode follows the demonstrators' convention
# (mode 0 if its yaw is in (-90, 90], else mode 180); only sources of that same mode are candidates,
# then the stock top-3 nearest neighbour on the original (unfolded) poses.
smode = np.where((syaw > -90.0) & (syaw <= 90.0), 0, 180)
print("mode-consistent (same-mode sources only, no flip)")
mc = []
for tpos, tyaw, ok in T:
    tm = 0 if -90.0 < tyaw <= 90.0 else 180
    idx_all = np.flatnonzero(smode == tm)
    dy = wrap(tyaw - syaw[idx_all])
    cost = np.linalg.norm(spos[idx_all, :2] - tpos[:2], axis=1) + np.radians(np.abs(dy))
    sel = np.argsort(cost)[:3]
    mc.append(dict(yaw=tyaw, dyaw=np.abs(dy[sel]).tolist()))
for lo, hi in bins:
    g = [r for r in mc if lo <= abs(r["yaw"]) < hi]
    dyv = np.concatenate([r["dyaw"] for r in g])
    print(f"  |yaw| [{lo:3d},{hi:3d}) n={len(g):3d} median|dyaw| {np.median(dyv):5.1f}  p90 {np.percentile(dyv, 90):5.1f}  mixed-mode targets   0.0% (by construction)")
