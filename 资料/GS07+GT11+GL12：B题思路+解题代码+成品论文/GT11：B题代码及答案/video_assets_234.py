"""
Generate video-script figures and illustrative data for Questions 2, 3, and 4.

The outputs are intentionally presentation-oriented rather than production
algorithms.  They support the video script video_script_4q.md.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Wedge


ROOT = Path(__file__).resolve().parent


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Times New Roman"],
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",
    }
)


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix, dpi in [("pdf", None), ("png", 300), ("svg", None)]:
        fig.savefig(ROOT / f"{stem}.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_q2_aoa_geometry_figure() -> None:
    """D-optimality of a 90-degree AOA intersection."""

    alpha_deg = np.linspace(5.0, 175.0, 400)
    alpha = np.deg2rad(alpha_deg)
    det_norm = np.sin(alpha) ** 2
    crlb_trace_norm = 2.0 / np.maximum(det_norm, 1e-6)

    pd.DataFrame(
        {
            "intersection_angle_deg": alpha_deg,
            "det_fim_norm": det_norm,
            "tr_crlb_norm": crlb_trace_norm,
        }
    ).to_csv(ROOT / "q2_aoa_geometry.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)

    axes[0].plot(alpha_deg, det_norm, color="#1f77b4", linewidth=2.4)
    axes[0].axvline(90.0, color="red", linestyle="--", linewidth=1.5)
    axes[0].annotate(
        "D-最优点：两条视线垂直",
        xy=(90.0, 1.0),
        xytext=(115.0, 0.72),
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red",
    )
    axes[0].set_title(r"Fisher信息行列式：$\det(J)\propto\sin^2\alpha$")
    axes[0].set_xlabel("两条测向线夹角 / °")
    axes[0].set_ylabel(r"$\det(J)$ 归一化值")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(alpha_deg, crlb_trace_norm, color="#d62728", linewidth=2.4)
    axes[1].axvline(90.0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_yscale("log")
    axes[1].set_title(r"位置CRLB迹：$\mathrm{tr}(CRLB)\propto2/\sin^2\alpha$")
    axes[1].set_xlabel("两条测向线夹角 / °")
    axes[1].set_ylabel(r"$\mathrm{tr}(CRLB)$ 归一化值")
    axes[1].grid(True, which="both", alpha=0.25)

    fig.suptitle("第二问：纯测向定位的最优交会角几何理论", y=1.02)
    save_figure(fig, "fig_q2_aoa_geometry")


def make_q2_candidate_region_figure() -> None:
    """Candidate region of the second station under a 90-degree criterion."""

    fig, ax = plt.subplots(figsize=(9.0, 7.8), constrained_layout=True)
    source = np.array([0.0, 0.0])
    first_station = np.array([0.0, -700.0])

    # First bearing: from first station to source is +90 degrees.
    first_theta = np.linspace(-800.0, 300.0, 100)
    ax.plot(
        [first_station[0], 0.0],
        [first_station[1], 0.0],
        color="#1f77b4",
        linewidth=2.2,
        label="第一检测点示向线",
    )

    # Two ideal perpendicular second bearings.
    ax.plot([-1000.0, 1000.0], [0.0, 0.0], color="#ff7f00", linewidth=1.6, linestyle="--")
    ax.annotate(
        "理想第二视线：与第一视线成90°",
        xy=(950.0, 35.0),
        fontsize=10,
        color="#ff7f00",
    )

    # Candidate ring from a nominal source-to-station distance assumption.
    ring_theta = np.linspace(0.0, 2.0 * np.pi, 500)
    for radius, color, label in [
        (1000.0, "#2ca02c", "最小接收半径 1000 m"),
        (1500.0, "#d62728", "最大接收半径 1500 m"),
    ]:
        ax.plot(
            radius * np.cos(ring_theta),
            radius * np.sin(ring_theta),
            color=color,
            linestyle="--",
            linewidth=1.3,
            label=label,
        )

    # Fill a simple candidate band around the horizontal ideal bearing.
    for radius, color in [(1000.0, "#2ca02c"), (1500.0, "#d62728")]:
        ax.fill_between(
            np.linspace(-radius, radius, 200),
            -0.15 * radius,
            0.15 * radius,
            color=color,
            alpha=0.08,
        )

    ax.scatter(*source, s=130, marker="*", color="black", zorder=5, label="待定位干扰源")
    ax.scatter(*first_station, s=90, marker="s", color="#1f77b4", zorder=5, label="第一检测点")

    ax.text(
        -1050.0,
        720.0,
        "候选区域 = 交会角接近90°\n且满足接收距离约束的可行带",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="0.45"),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1250.0, 1250.0)
    ax.set_ylim(-1000.0, 1000.0)
    ax.grid(True, alpha=0.22)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("第二检测点候选区域：D-最优交会角 + 距离约束")
    ax.legend(loc="lower left", fontsize=9)
    save_figure(fig, "fig_q2_candidate_region")


def make_q3_search_pipeline_figure() -> None:
    """High-level search, localization, and clearing pipeline."""

    fig, ax = plt.subplots(figsize=(12.0, 6.8), constrained_layout=True)
    ax.axis("off")

    boxes = [
        ("初始状态\n原点 (0,0)\n频道 1", "#d9d9d9"),
        ("稀疏覆盖预扫描\n圆盘覆盖 + 频道扫描", "#aec7e8"),
        ("未知基数多目标估计\nPHD / CPHD / 粒子PHD", "#98df8a"),
        ("信息增益驱动规划\n滚动时域路径决策", "#ffbb78"),
        ("近点精确定位\n光学探测 20 m", "#ff9896"),
        ("激光清除\n/clear 成功或重调度", "#c5b0d5"),
    ]
    positions = [
        (0.015, 0.50, 0.13, 0.26),
        (0.18, 0.50, 0.17, 0.26),
        (0.40, 0.50, 0.17, 0.26),
        (0.62, 0.50, 0.17, 0.26),
        (0.83, 0.50, 0.13, 0.26),
        (0.71, 0.10, 0.17, 0.24),
    ]

    for (text, color), (x, y, w, h) in zip(boxes, positions):
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012",
            linewidth=1.3,
            edgecolor="#333333",
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=10,
            linespacing=1.55,
        )

    arrows = [
        ((0.145, 0.63), (0.18, 0.63)),
        ((0.35, 0.63), (0.40, 0.63)),
        ((0.57, 0.63), (0.62, 0.63)),
        ((0.79, 0.63), (0.83, 0.63)),
        ((0.83, 0.50), (0.755, 0.34)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=18,
                color="#333333",
                linewidth=1.6,
            )
        )

    ax.text(
        0.84,
        0.84,
        "反馈环：\n新测向 -> 更新后验 -> 选择下一个检测点",
        ha="center",
        fontsize=10,
        color="#b30000",
    )
    ax.text(
        0.04,
        0.14,
        "目标函数：\n最短虚拟总时间\n(移动 + 切换 + 检测 + 精确定位 + 清除)",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.5"),
    )
    ax.set_title("问题3：未知数量全向源的主动搜索、定位与清除流水线", fontsize=14)
    save_figure(fig, "fig_q3_search_pipeline")


def make_q3_time_budget_csv() -> None:
    rows = [
        ("单次测向 /measure", 5, "每个有效检测动作固定耗时"),
        ("频道切换", 1, "任意两个频道之间"),
        ("移动速度", 5, "m/s，直线距离换算耗时"),
        ("光学精确定位", 3, "在 /clear 内完成"),
        ("清除成功", 5, "精确定位 3 s + 激光 2 s"),
        ("清除失败", 3, "20 m 内无目标"),
        ("目标数量", "10-16", "具体数量未知"),
        ("有效接收半径", "1000-1500", "m，不同源可能不同"),
        ("程序运行上限", 1200, "s，/enter 后计时"),
        ("测试窗口上限", 1500, "s，含 5 s 倒计时"),
        ("虚拟时间上限", 360000, "s，即 100 h"),
    ]
    df = pd.DataFrame(rows, columns=["项目", "数值", "说明"])
    df.to_csv(ROOT / "q3_time_budget.csv", index=False, encoding="utf-8-sig")


def make_q4_directional_figure() -> None:
    """Directional source semantics and negative-information update."""

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.8), constrained_layout=True)
    source = np.array([0.0, 0.0])
    direction_deg = 38.0

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.22)

    # Left panel: wedge geometry.
    ax = axes[0]
    wedge = Wedge(
        (0.0, 0.0),
        900.0,
        direction_deg - 90.0,
        direction_deg + 90.0,
        facecolor="#1f77b4",
        alpha=0.20,
        edgecolor="#1f77b4",
        linewidth=1.3,
    )
    ax.add_patch(wedge)
    ax.scatter(*source, s=130, marker="*", color="black", zorder=6, label="定向干扰源")

    ax.annotate(
        "",
        xy=(620.0 * math.cos(math.radians(direction_deg)), 620.0 * math.sin(math.radians(direction_deg))),
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="->", color="black", linewidth=2.0),
    )
    ax.text(640.0, 580.0, "未知定向方向", fontsize=10, color="black")

    inside = np.array([540.0 * math.cos(math.radians(20.0)), 540.0 * math.sin(math.radians(20.0))])
    outside = np.array([560.0 * math.cos(math.radians(170.0)), 560.0 * math.sin(math.radians(170.0))])
    ax.scatter(*inside, s=80, marker="o", color="#2ca02c", zorder=6, label="覆盖内：返回 direction")
    ax.scatter(*outside, s=80, marker="x", color="#d62728", zorder=6, label="覆盖外：返回 no_signal")
    ax.text(inside[0] + 45.0, inside[1] + 20.0, "有信号", color="#2ca02c", fontsize=10)
    ax.text(outside[0] - 180.0, outside[1] + 15.0, "无信号", color="#d62728", fontsize=10)
    ax.set_xlim(-1000.0, 1050.0)
    ax.set_ylim(-950.0, 1000.0)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("定向覆盖扇区与两类检测结果")
    ax.legend(loc="upper right", fontsize=9)

    # Right panel: negative evidence concept.
    ax = axes[1]
    ax.axis("off")
    texts = [
        ("方向未知时，no_signal 不是空观测", 0.50, 0.88, "#b30000"),
        ("先验：方向在 [0°, 360°) 上近似均匀", 0.50, 0.72, "#333333"),
        ("在检测点 P 得到 no_signal", 0.50, 0.58, "#333333"),
        ("排除所有能把信号覆盖到 P 的方向区间", 0.50, 0.44, "#333333"),
        ("后验：可行方向集中到互补区间", 0.50, 0.30, "#333333"),
        ("继续移动到正交方向，快速压缩方向不确定性", 0.50, 0.16, "#b30000"),
    ]
    for text, x, y, color in texts:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=12,
            color=color,
            linespacing=1.4,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("把 no_signal 转化为负信息贝叶斯更新")

    fig.suptitle("问题4：混合全向源与定向源的类型-方向联合辨识", y=1.02)
    save_figure(fig, "fig_q4_directional")

    pd.DataFrame(
        {
            "measure_result": ["direction", "near", "no_signal"],
            "是否得到方位角": ["是", "否", "否"],
            "全向源解释": ["距离在接收半径内", "距离 ≤ 5 m", "频道无源或距离超限"],
            "定向源额外解释": ["检测点位于覆盖扇区内", "距离 ≤ 5 m 且位于覆盖扇区内", "检测点位于覆盖扇区外"],
            "对搜索的作用": ["直接更新位置后验", "可进入清除阶段", "更新类型与方向的负证据"],
        }
    ).to_csv(ROOT / "q4_measurement_semantics.csv", index=False, encoding="utf-8-sig")


def make_q2_q4_performance_table() -> None:
    """Illustrative performance table used in the final video summary."""

    rows = [
        ("问题1", "半平面交 + 旋转卡壳 + Welzl", "精确、可证、复杂度低", "不假设直径圆必然覆盖"),
        ("问题2", "Fisher信息 D-最优 + 90°准则", "理论最优几何配置", "给出第二检测点候选区域"),
        ("问题3", "PHD/粒子滤波 + 信息增益规划", "自动处理未知目标数量", "在线搜索-定位-清除闭环"),
        ("问题4", "类型-方向增广滤波 + 负信息", "能解释定向源 no_signal", "最终清除仍只依赖 20 m 半径"),
    ]
    df = pd.DataFrame(rows, columns=["问题", "主算法链", "优势", "关键结论"])
    df.to_csv(ROOT / "q1_q4_video_summary.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    make_q2_aoa_geometry_figure()
    make_q2_candidate_region_figure()
    make_q3_search_pipeline_figure()
    make_q3_time_budget_csv()
    make_q4_directional_figure()
    make_q2_q4_performance_table()
    print("Video assets generated.")


if __name__ == "__main__":
    main()
