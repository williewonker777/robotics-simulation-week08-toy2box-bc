"""Summarize the teleoperated demos stored in an Isaac Lab HDF5 dataset.

One line is printed per demo:

    len   number of steps (seconds at --hz in parentheses)
    sw    number of gripper command switches (open <-> close)
    idle  share of steps without any arm motion command
    ok    success flag stored in the HDF5 file (record_demos.py only saves successful demos)

Usage:
    ./isaaclab.sh -p summarize_demos.py [--dataset_file ./datasets/toy2box_agibot.hdf5] [--hz 20]
"""

import argparse
import re

import h5py
import numpy as np


def demo_index(name: str) -> int:
    """Sort key so that demo_10 comes after demo_9."""
    digits = re.sub(r"\D", "", name)
    return int(digits) if digits else -1


def main():
    parser = argparse.ArgumentParser(description="Summarize demos in an Isaac Lab HDF5 dataset.")
    parser.add_argument("--dataset_file", type=str, default="./datasets/toy2box_agibot.hdf5", help="HDF5 dataset.")
    parser.add_argument(
        "--hz", type=float, default=20.0, help="Environment step rate used to convert steps to seconds."
    )
    parser.add_argument(
        "--idle_eps", type=float, default=1e-6, help="Arm command magnitude at or below which a step counts as idle."
    )
    args = parser.parse_args()

    with h5py.File(args.dataset_file, "r") as f:
        demos = sorted(f["data"].keys(), key=demo_index)
        actions = [np.asarray(f["data"][name]["actions"]) for name in demos]
        print(f"episodes: {len(demos)} | total: {sum(len(a) for a in actions)}")

        for name, demo_actions in zip(demos, actions):
            length = len(demo_actions)
            # the last action dimension is the binary gripper command (negative closes the gripper)
            gripper_closed = demo_actions[:, -1] < 0
            switches = int(np.count_nonzero(gripper_closed[1:] != gripper_closed[:-1]))
            # keyboard teleoperation sends an all-zero arm command while no key is pressed
            idle_steps = np.all(np.abs(demo_actions[:, :-1]) <= args.idle_eps, axis=1)
            idle = float(np.mean(idle_steps)) if length else 0.0
            attrs = f["data"][name].attrs
            ok = int(bool(attrs["success"])) if "success" in attrs else "?"
            print(f"{name:<9}len={length:4d} ({length / args.hz:.1f}s) sw={switches:2d} idle={idle:.0%} ok={ok}")


if __name__ == "__main__":
    main()
