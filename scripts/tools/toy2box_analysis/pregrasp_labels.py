"""Toy2Box post-hoc metric (2026-10-28, exploratory, not preregistered): "open and rise" actions next to the truck.

Pre-grasp zone: EEF within 6 cm of the truck in XY and less than 6 cm above it, truck still on the table
(less than 1 cm above its rest height). Per demo file (HDF5) or rollout dir (observer NPZ), counts the zone steps whose
action rises (dz > 0.5 mm per step) with the gripper commanded open, against descending-open and closed steps.

Usage: python3 pregrasp_labels.py [--hdf5 label=path ...] [--npz label=dir[,dir...] ...] [--json out.json]
"""

import argparse
import glob
import json
import os

import h5py
import numpy as np

XY_M, HEIGHT_M, ON_TABLE_M, DZ_M = 0.06, 0.06, 0.01, 0.0005


def count(eef, truck, action, rest):
    height = eef[:, 2] - truck[:, 2]
    xy = np.linalg.norm(eef[:, :2] - truck[:, :2], axis=1)
    zone = (xy < XY_M) & (height < HEIGHT_M) & (truck[:, 2] - rest < ON_TABLE_M)
    opened = action[:, -1] > 0
    return {"zone_steps": int(zone.sum()),
            "rise_open": int(np.sum(zone & opened & (action[:, 2] > DZ_M))),
            "descend_open": int(np.sum(zone & opened & (action[:, 2] < -DZ_M))),
            "closed": int(np.sum(zone & ~opened))}


def add(total, part):
    for k, v in part.items():
        total[k] = total.get(k, 0) + v
    return total


def from_hdf5(path):
    total, episodes = {}, 0
    with h5py.File(path, "r") as f:
        for name in f["data"]:
            g = f["data"][name]
            truck = g["obs/toy_truck_positions"][:]
            add(total, count(g["obs/eef_pos"][:], truck, g["actions"][:], float(np.median(truck[:5, 2]))))
            episodes += 1
    return {**total, "episodes": episodes}


def from_npz(dirs):
    total, trials, trials_with_rise = {}, 0, 0
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "trial_*.npz"))):
            z = np.load(path, allow_pickle=True)
            part = count(z["eef_pos"], z["truck_pos"], z["action"], float(np.median(z["truck_pos"][:30, 2])))
            add(total, part)
            trials += 1
            trials_with_rise += part["rise_open"] >= 5
    return {**total, "trials": trials, "trials_with_5plus_rise_open": trials_with_rise}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", action="append", default=[])
    ap.add_argument("--npz", action="append", default=[])
    ap.add_argument("--json")
    a = ap.parse_args()
    out = {}
    for item in a.hdf5:
        label, path = item.split("=", 1)
        out[label] = from_hdf5(path)
    for item in a.npz:
        label, dirs = item.split("=", 1)
        out[label] = from_npz(dirs.split(","))
    for label, r in out.items():
        share = 100.0 * r["rise_open"] / r["zone_steps"] if r["zone_steps"] else float("nan")
        print(f"{label}: zone steps {r['zone_steps']}, rise+open {r['rise_open']} ({share:.1f}%), "
              f"descend+open {r['descend_open']}, closed {r['closed']}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
