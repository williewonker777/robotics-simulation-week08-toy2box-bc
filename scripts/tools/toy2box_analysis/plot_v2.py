"""Toy2Box experiment figures: success per condition, failure stages, grasp mode in |yaw| 45-90 deg, transport slip vs grasp height.

Usage: python3 plot_v2.py <registered_results.json> <transport_slip.json> <out_dir>
Inputs are artifacts/experiments/registered_results.json and artifacts/mechanism/transport_slip.json.
"""

import json
import os
import sys

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

CONDS = [("E0", "E0 기준선"), ("E1", "E1 방향 공백 메움"), ("E2", "E2 개수 대조"), ("E4", "E4 교정 데이터")]
STAGES = [("approach", "접근 실패", "#9e9e9e"), ("grasp", "파지 실패", "#e45756"), ("transport", "운반 실패", "#f2a541"),
          ("placement", "배치 실패", "#b279a2"), ("success", "성공", "#54a24b")]


def success(res, out):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for i, (key, label) in enumerate(CONDS):
        c = res["conditions"][key]
        seeds = [c["models"][s]["successes"] for s in ("101", "111")]
        mean, sd = 100 * c["condition_success"]["mean_rate"], 100 * c["condition_success"]["sample_sd_rate_ddof1"]
        ax.bar(i, mean, color="#4c78a8" if key != "E0" else "#9e9e9e", width=0.6)
        ax.errorbar(i, mean, yerr=sd, color="black", capsize=5, lw=1)
        for s, v in zip(("101", "111"), seeds):
            ax.plot(i + (-0.12 if s == "101" else 0.12), v, "o", color="black", ms=4)
        ax.text(i, mean + sd + 1.2, f"{mean:.1f}%\n({c['pooled_successes']}/200)", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(CONDS)), [l for _, l in CONDS])
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 34)
    ax.set_title("조건별 성공률: 막대 = 두 학습 seed 평균, 오차 막대 = 표본 표준편차, 점 = seed 101 / 111", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "v2_success_by_condition.png"), dpi=150)
    plt.close(fig)


def stages(res, out):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    left = np.zeros(len(CONDS))
    for key, label, color in STAGES:
        vals = np.array([res["conditions"][c]["v3_1_pooled_stages"][key] for c, _ in CONDS])
        ax.barh(range(len(CONDS)), vals, left=left, color=color, label=label)
        for i, v in enumerate(vals):
            if v >= 9:
                ax.text(left[i] + v / 2, i, str(v), ha="center", va="center", fontsize=8, color="white")
        left += vals
    ax.set_yticks(range(len(CONDS)), [l for _, l in CONDS])
    ax.invert_yaxis()
    ax.set_xlabel("rollout 수 (두 seed 합계 200회)")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "v2_stages_by_condition.png"), dpi=150)
    plt.close(fig)


def grasp_mode(res, out):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    parts = [("mode0", "~0° 자세", "#54a24b"), ("mode180", "~180° 자세", "#4c78a8"), ("between", "그 사이 (어긋남)", "#e45756")]
    left = np.zeros(len(CONDS))
    for key, label, color in parts:
        vals = np.array([res["conditions"][c]["close_by_abs_yaw_pooled"]["[45,90)"][key] /
                         res["conditions"][c]["close_by_abs_yaw_pooled"]["[45,90)"]["closed"] for c, _ in CONDS])
        ax.barh(range(len(CONDS)), 100 * vals, left=100 * left, color=color, label=label)
        left += vals
    for i, (c, _) in enumerate(CONDS):
        b = res["conditions"][c]["close_by_abs_yaw_pooled"]["[45,90)"]
        ax.text(101, i, f"방향 오차 중앙값 {b['heading_error_abs_median_deg']:.1f}°\n성공 {b['successes']}/{b['trials']}",
                va="center", fontsize=8)
    ax.axvline(100 * (1 - 0.5 * 25 / 55), color="black", lw=0.8, ls=":")
    ax.set_yticks(range(len(CONDS)), [l for _, l in CONDS])
    ax.invert_yaxis()
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("|트럭 yaw| 45~90°에서 닫는 순간의 잡는 자세 (%)  ·  빨간 구간이 점선 오른쪽에서 시작하면 '그 사이' ≤ 22.7% (예측 2)")
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.4, 1.18), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "v2_grasp_mode_45_90.png"), dpi=150)
    plt.close(fig)


def slip(sl, out):
    rows = sl["pooled_by_grasp_height_cm"]
    labels = ["< 0.8", "0.8~1.1", "1.1~1.3", "1.3~1.5", "≥ 1.5"]
    rates = [100 * r["slipped"] / r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(range(len(rows)), rates, color="#f2a541", width=0.6)
    for i, r in enumerate(rows):
        ax.text(i, rates[i] + 1.5, f"{r['slipped']}/{r['n']}", ha="center", fontsize=9)
    ax.set_xticks(range(len(rows)), labels)
    ax.set_xlabel("닫는 순간 그리퍼가 트럭 중심보다 높은 정도 (cm)")
    ax.set_ylabel("운반 중 놓친 비율 (%)")
    ax.set_ylim(0, 90)
    ax.set_title("첫 집기로 트럭을 확보한 rollout (E0·E1·E2·E4 합계)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "v2_slip_vs_grasp_height.png"), dpi=150)
    plt.close(fig)


def main():
    res = json.load(open(sys.argv[1]))
    sl = json.load(open(sys.argv[2]))
    out = sys.argv[3]
    os.makedirs(out, exist_ok=True)
    success(res, out)
    stages(res, out)
    grasp_mode(res, out)
    slip(sl, out)


if __name__ == "__main__":
    main()
