"""Toy2Box: wrist yaw at grasp vs truck yaw in source / Mimic-generated demos (checks 180-degree grasp symmetry use)."""

import argparse

import h5py
import numpy as np


def rotmat(q):
    """wxyz quaternion(s) -> rotation matrices."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def heading(R, axis):
    v = R[..., :, axis]
    return np.degrees(np.arctan2(v[..., 1], v[..., 0]))


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


parser = argparse.ArgumentParser()
parser.add_argument("dataset_file")
parser.add_argument("--eef_axis", type=int, default=0, help="EEF body axis whose xy heading is used")
parser.add_argument("--per_demo", action="store_true")
args = parser.parse_args()

rows = []
with h5py.File(args.dataset_file, "r") as f:
    names = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
    for name in names:
        d = f["data"][name]
        a = d["actions"][:]
        close = np.flatnonzero(a[:, -1] < 0)
        if close.size == 0:
            continue
        k = int(close[0])
        Re = rotmat(d["obs"]["eef_quat"][:])
        Rt = rotmat(d["obs"]["toy_truck_orientations"][:])
        truck0 = heading(Rt[0], 0)
        eef0 = heading(Re[0], args.eef_axis)
        eefk = heading(Re[k], args.eef_axis)
        rows.append((name, truck0, eef0, eefk, wrap(eefk - truck0), wrap(eefk - eef0), Re[k][2, args.eef_axis]))
rows_arr = np.array([r[1:] for r in rows])
truck0, eef0, eefk, rel, rot, axis_z = rows_arr.T
print(f"{args.dataset_file}: {len(rows)} demos with a close command; eef axis {args.eef_axis} z-component at grasp mean {axis_z.mean():+.2f}")
print(f"  EEF heading at start: mean {eef0.mean():+.1f} deg, std {eef0.std():.1f}")
print(f"  wrist yaw change start->grasp |deg|: median {np.median(np.abs(rot)):.1f}, p90 {np.percentile(np.abs(rot), 90):.1f}, max {np.abs(rot).max():.1f}")
hist, edges = np.histogram(rel, bins=np.linspace(-180, 180, 13))
print("  grasp yaw relative to truck (eef - truck) histogram, 30-deg bins from -180:", hist.tolist())
print("  by truck start yaw bin: n | median |wrist yaw change| | relative-yaw mode bins")
for lo in range(-180, 180, 45):
    m = (truck0 >= lo) & (truck0 < lo + 45)
    if m.any():
        rh, _ = np.histogram(rel[m], bins=np.linspace(-180, 180, 13))
        print(f"    [{lo:+4d},{lo + 45:+4d}) n={m.sum():3d} |rot| median {np.median(np.abs(rot[m])):6.1f}  rel-hist {rh.tolist()}")
if args.per_demo:
    for r in rows:
        print(f"    {r[0]:9s} truck {r[1]:+7.1f} eef0 {r[2]:+7.1f} eef@grasp {r[3]:+7.1f} rel {r[4]:+7.1f} rot {r[5]:+7.1f}")
