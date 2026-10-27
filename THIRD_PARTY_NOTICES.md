# Third-party notices

## Isaac Lab v2.3.0 (BSD-3-Clause)

This repository is a snapshot of [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) at tag `v2.3.0`,
with the Toy2Box imitation-learning additions described in the README. The upstream history is not included.
Isaac Lab files keep their upstream copyright headers. The license is in [`LICENSE`](LICENSE).
The upstream README is kept as [`README_ISAACLAB.md`](README_ISAACLAB.md).
The upstream CI workflows (`.github/workflows`, `.github/actions`) are left out so that they do not run in this repository.

## isaaclab_mimic (Apache-2.0)

`source/isaaclab_mimic` and the Mimic scripts are distributed under the license in [`LICENSE-mimic`](LICENSE-mimic).
`scripts/tools/generate_toy2box_settled.py` wraps the Isaac Lab Mimic generation loop at runtime and does not modify it.

## robomimic 0.4.0 (MIT)

Training and evaluation use the installed robomimic package through the stock Isaac Lab scripts.
robomimic is not redistributed here.

## Simulator and assets

Isaac Sim, the Agibot robot, the toy truck and the box assets are loaded from the NVIDIA asset server at run time.
They are not redistributed (USD files are excluded by `.gitignore`), and their own licenses apply.

## Fonts in media

Labels in `artifacts/media/` were drawn with NanumGothic (SIL Open Font License 1.1). Font files are not redistributed.
