"""Toy2Box generated-data check: does the grasp keep the demonstrators' convention, per truck-yaw bin?

Convention: at the first close command the gripper x-axis heading lies in (-90, 90] (the 40 sources all do).
Also reports the wrist rotation start -> grasp (unwrapped) and the truck-relative grasp mode (0 or 180).

Usage: python3 dataset_convention.py <dataset.hdf5> [--tol_deg 5]
"""

import argparse

import h5py
import numpy as np


def heading_x(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.degrees(np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z)))


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--tol_deg", type=float, default=5.0)
a = ap.parse_args()

rows = []
with h5py.File(a.dataset, "r") as f:
    for name in f["data"]:
        d = f["data"][name]
        act = d["actions"][:]
        close = np.flatnonzero(act[:, -1] < 0)
        if close.size == 0:
            continue
        k = int(close[0])
        o = d["obs"]
        eh = heading_x(o["eef_quat"][:])
        ty = float(heading_x(o["toy_truck_orientations"][0]))
        hk = float(eh[k])
        rot = float(np.degrees(np.unwrap(np.radians(eh[: k + 1])))[-1] - eh[0])
        conv = (-90.0 - a.tol_deg) < hk <= (90.0 + a.tol_deg)
        mode = 0 if abs(wrap(hk - ty)) < 90.0 else 180
        rows.append((ty, hk, rot, conv, mode))
r = np.array(rows, dtype=float)
print(f"{a.dataset}: {len(rows)} demos with a close; convention (-90,90] +/-{a.tol_deg:g} deg: {r[:, 3].mean():.1%}")
print("  truck-yaw bin      n  convention  mode0:mode180  |wrist rot| median")
for lo in range(-180, 180, 45):
    m = (r[:, 0] >= lo) & (r[:, 0] < lo + 45)
    if m.any():
        g = r[m]
        print(
            f"  [{lo:+4d},{lo + 45:+4d})  {m.sum():4d}  {g[:, 3].mean():9.1%}  {int((g[:, 4] == 0).sum()):5d}:{int((g[:, 4] == 180).sum()):<5d}"
            f"  {np.median(np.abs(g[:, 2])):8.1f}"
        )
