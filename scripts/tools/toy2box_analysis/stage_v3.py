"""Toy2Box stage classifier v3: furthest stage reached, judged from the truck's own state.

v2 gated "approach" on EEF-to-truck-center <= 25 mm, which mislabels off-center grasps
(manual check: 7 of 8 sampled v2 "approach" failures were closed on the truck in video).
v3 rules, fixed before the second validation sample:
  success    stock success predicate at episode end
  placement  truck lifted and following the gripper while within 0.12 m XY of the box center
  transport  truck secured: lifted >= 3 cm above rest while within 6 cm of the EEF for >= 40 consecutive
             steps (2 s; the sources lift for 39-44 steps after the grasp signal) but never held near the box
  grasp      gripper closed with the EEF within 6 cm of the truck, or the truck was disturbed
             (moved >= 1 cm in XY, rotated >= 5 deg, or rose >= 1 cm); includes lifts lost within 2 s
  approach   none of the above
v3.1 (after the 20-trial development check, before the fresh validation sample): the transport
run length went from 10 to 40 steps, so a lift lost within 2 s counts as an insecure grasp.

Usage: python3 stage_v3.py <npz_dir> [...] > out.json
"""

import glob
import json
import os
import sys

import numpy as np

LIFT_M = 0.03
FOLLOW_M = 0.06
FOLLOW_STEPS = 40
BOX_XY_M = 0.12
CLOSE_NEAR_M = 0.06
DISTURB_XY_M = 0.01
DISTURB_YAW_DEG = 5.0
DISTURB_Z_M = 0.01
REST_STEPS = 30


def yaw_deg(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def runs(mask, min_len):
    """Boolean mask of steps that belong to a True run of at least min_len."""
    out = np.zeros_like(mask)
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def classify(path):
    z = np.load(path, allow_pickle=True)
    eef, truck, box, act = z["eef_pos"], z["truck_pos"], z["box_pos"], z["action"]
    rest = truck[: min(REST_STEPS, len(truck))]
    rest_z = float(np.median(rest[:, 2]))
    d_eef = np.linalg.norm(eef - truck, axis=1)
    closed = act[:, -1] < 0
    lifted = (truck[:, 2] - rest_z >= LIFT_M) & (d_eef <= FOLLOW_M) & closed
    held = runs(lifted, FOLLOW_STEPS)
    near_box = np.linalg.norm(truck[:, :2] - box[:, :2], axis=1) <= BOX_XY_M
    onsets = np.flatnonzero(closed & ~np.concatenate([[False], closed[:-1]]))
    close_near = bool(np.any(d_eef[onsets] <= CLOSE_NEAR_M)) if onsets.size else False
    yaw = yaw_deg(z["truck_quat"])
    dyaw = np.abs((yaw - yaw[0] + 180.0) % 360.0 - 180.0)
    disturbed = bool(
        np.linalg.norm(truck[:, :2] - truck[0, :2], axis=1).max() >= DISTURB_XY_M
        or dyaw.max() >= DISTURB_YAW_DEG
        or (truck[:, 2] - rest_z).max() >= DISTURB_Z_M
    )
    if bool(z["final_success"]):
        stage = "success"
    elif np.any(held & near_box):
        stage = "placement"
    elif held.any():
        stage = "transport"
    elif close_near or disturbed:
        stage = "grasp"
    else:
        stage = "approach"
    return dict(
        stage=stage,
        held_steps=int(held.sum()),
        close_near=close_near,
        disturbed=disturbed,
        n_close_onsets=int(onsets.size),
        first_hold=int(np.argmax(held)) if held.any() else None,
        steps=int(len(act)),
    )


if __name__ == "__main__":
    out = []
    for d in sys.argv[1:]:
        for p in sorted(glob.glob(os.path.join(d, "trial_*.npz"))):
            r = classify(p)
            r["src"] = os.path.basename(d.rstrip("/"))
            r["trial"] = int(os.path.basename(p)[6:11])
            out.append(r)
    json.dump(out, sys.stdout)
