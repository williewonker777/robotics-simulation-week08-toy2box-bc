"""Toy2Box: per-trial mechanism metrics from observer NPZs (close timing/offset, wrist heading mode, truck motion).

Usage: python3 rollout_mechanism.py <npz_dir> [<npz_dir> ...]
"""

import glob
import os
import sys

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


def heading(R, axis=0):
    v = R[..., :, axis]
    return np.degrees(np.arctan2(v[..., 1], v[..., 0]))


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def trial_metrics(path):
    z = np.load(path, allow_pickle=True)
    eef, truck, act = z["eef_pos"], z["truck_pos"], z["action"]
    truck0 = z["reset_initial_truck_pos"]
    yaw0 = float(heading(rotmat(z["reset_initial_truck_quat"])))
    eef_h = heading(rotmat(z["eef_quat"]))
    h0 = float(heading(rotmat(z["reset_initial_eef_quat"])))
    close = np.flatnonzero(act[:, -1] < 0)
    k = int(close[0]) if close.size else None
    m = dict(
        success=bool(z["final_success"]),
        steps=len(act),
        yaw0=yaw0,
        x0=float(truck0[0]),
        y0=float(truck0[1]),
        eef_h0=h0,
        close=k,
        n_close_runs=int(np.sum(np.diff((act[:, -1] < 0).astype(int)) == 1) + (act[0, -1] < 0)),
        lift=float(truck[:, 2].max() - truck0[2]),
        truck_xy_moved=float(np.linalg.norm(truck[:, :2] - truck0[:2], axis=1).max()),
        min_d3=float(np.linalg.norm(eef - truck, axis=1).min()),
    )
    if k is not None:
        d = eef[k] - truck[k]
        m.update(
            close_dxy=float(np.linalg.norm(d[:2])),
            close_dz=float(d[2]),
            close_h=float(eef_h[k]),
            # wrist rotation actually performed start -> close, and grasp mode relative to the truck
            close_rot=float(wrap(eef_h[k] - h0)),
            close_rel=float(wrap(eef_h[k] - yaw0)),
            min_d3_before_close=float(np.linalg.norm(eef[: k + 1] - truck[: k + 1], axis=1).min()),
        )
    # cumulative wrist heading travel (unwrapped), to see back-and-forth rotation
    hu = np.degrees(np.unwrap(np.radians(eef_h)))
    upto = (k + 1) if k is not None else len(hu)
    m["h_travel"] = float(np.abs(np.diff(hu[:upto])).sum())
    m["h_net"] = float(hu[upto - 1] - hu[0])
    return m


def main():
    rows = []
    for d in sys.argv[1:]:
        for p in sorted(glob.glob(os.path.join(d, "trial_*.npz"))):
            m = trial_metrics(p)
            m["src"] = os.path.basename(d.rstrip("/"))
            m["trial"] = int(os.path.basename(p)[6:11])
            rows.append(m)
    return rows


if __name__ == "__main__":
    import json

    rows = main()
    json.dump(rows, sys.stdout)
