"""Toy2Box E0 figures: v3.1 failure stages per training seed, and source coverage vs BC outcome per truck-yaw bin.

Usage: python3 plot_e0.py <stage_v31.json> <close_precision.json> <source.hdf5> <out_dir>
Inputs come from stage_v3.py and close_precision.py (artifacts/e0_baseline/).
"""

import json
import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

for path in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    if os.path.exists(path):
        font_manager.fontManager.addfont(path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

STAGES = [("approach", "접근 실패", "#9e9e9e"), ("grasp", "파지 실패", "#e45756"), ("transport", "운반 실패", "#f2a541"),
          ("placement", "배치 실패", "#b279a2"), ("success", "성공", "#54a24b")]
BINS = [(-180, -135), (-135, -90), (-90, -45), (-45, 0), (0, 45), (45, 90), (90, 135), (135, 180)]


def yaw_deg(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def in_bin(y, lo, hi):
    return lo <= y < hi or (hi == 180 and y == 180)


def main():
    stage_rows, close_rows, source, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    st = json.load(open(stage_rows))
    rows = json.load(open(close_rows))

    # Figure 1: stage counts per training seed.
    fig, ax = plt.subplots(figsize=(8, 2.6))
    groups = [("seed 101", [r for r in st if "seed101" in r["src"]]), ("seed 111", [r for r in st if "seed111" in r["src"]]), ("합계", st)]
    for gi, (name, g) in enumerate(groups):
        left = 0
        for key, label, color in STAGES:
            n = sum(r["stage"] == key for r in g)
            share = 100.0 * n / len(g)
            ax.barh(gi, share, left=left, color=color, edgecolor="white", label=label if gi == 0 else None)
            if n:
                ax.text(left + share / 2, gi, str(n), ha="center", va="center", fontsize=9, color="white" if key != "placement" else "black")
            left += share
    ax.set_yticks(range(len(groups)), [g[0] for g in groups])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("rollout 비율 (%) · 막대 안 숫자는 횟수")
    ax.set_title("E0 기준선: 가장 멀리 간 단계 (v3.1 분류, 평가 seed 42 × 100회)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.35), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "e0_stages.png"), dpi=150)
    plt.close(fig)

    # Figure 2: per signed truck-yaw bin, source demos, E0 success rate and grasp mode at the decisive close.
    f = h5py.File(source, "r")
    syaw = [float(yaw_deg(f["data"][k]["obs/toy_truck_orientations"][0])) for k in f["data"]]
    labels = [f"[{lo},{hi})" for lo, hi in BINS]
    src_n = [sum(in_bin(y, lo, hi) for y in syaw) for lo, hi in BINS]
    succ, tot, m0, m180, mid = [], [], [], [], []
    for lo, hi in BINS:
        g = [r for r in rows if in_bin(r["yaw0"], lo, hi)]
        tot.append(len(g))
        succ.append(sum(r["stage"] == "success" for r in g))
        c = [r for r in g if r["closed"]]
        m0.append(sum(r["mode"] == "mode0" for r in c))
        m180.append(sum(r["mode"] == "mode180" for r in c))
        mid.append(sum(r["mode"] == "between" for r in c))
    x = np.arange(len(BINS))
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.2), sharex=True)
    axes[0].bar(x, src_n, color="#4c78a8")
    for i, n in enumerate(src_n):
        axes[0].text(i, n + 0.3, str(n), ha="center", fontsize=9)
    axes[0].set_ylabel("원본 시범 수")
    axes[0].set_ylim(0, max(src_n) + 2.5)
    axes[0].set_title("트럭 초기 yaw 구간별: 원본 시범 40개의 분포와 E0 정책의 결과")
    rate = [100.0 * s / t if t else 0 for s, t in zip(succ, tot)]
    axes[1].bar(x, rate, color="#54a24b")
    for i, (s, t) in enumerate(zip(succ, tot)):
        axes[1].text(i, rate[i] + 1, f"{s}/{t}", ha="center", fontsize=9)
    axes[1].set_ylabel("E0 성공률 (%)")
    axes[1].set_ylim(0, max(rate) + 12)
    axes[2].bar(x, m0, color="#72b7b2", label="~0° 자세")
    axes[2].bar(x, m180, bottom=m0, color="#4c78a8", label="~180° 자세")
    axes[2].bar(x, mid, bottom=np.add(m0, m180), color="#e45756", label="그 사이 (어긋남)")
    axes[2].set_ylabel("닫는 순간 손목 방향 (회)")
    axes[2].set_ylim(0, max(np.add(np.add(m0, m180), mid)) + 9)
    axes[2].legend(ncol=3, fontsize=9, loc="upper center")
    axes[2].set_xticks(x, labels)
    axes[2].set_xlabel("트럭 초기 yaw 구간 (°)")
    for a in axes:
        a.axvspan(1.5, 5.5, color="#fff3cd", zorder=-1)
    fig.text(0.99, 0.005, "노란 배경: |yaw| ≤ 90° (원본 5개)", ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "e0_yaw_bins.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
