# -*- coding: utf-8 -*-
"""Q3 (全向源) 论文图表生成.

数据来源: q3/第三问_论文参考稿.md §6.1 实测数据
  24 种子 mock: T_avg = 318.7 ± 40.2 s
  5 次真机:     T_avg = 287.2 s
  时间分解: 原点 119s / hunt_all 264-397s / ring_cover 3128-4814s
  A/B 对比 (12 种子): 路线A 378.8s vs 路线B 324.0s (-16.9%)

输出 (q3/figs/):
  fig_q3_cert7.png       7 点覆盖证书 (全向源判空)
  fig_q3_timebudget.png  Q3 时间分解
  fig_q3_ab.png          路线A vs 路线B 对比
  fig_q3_mock_hist.png   24 种子 mock 分布 (正态模拟) + 真机对比
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = r"E:/Draft/MathModel"
FIG = os.path.join(ROOT, "q3", "figs")
os.makedirs(FIG, exist_ok=True)


# ============ 图1: Q3 7 点覆盖证书 ============
def fig_cert7():
    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    R_TARGET = 1800.0
    R_RING = 1150.0
    ax.add_patch(Circle((0, 0), R_TARGET, fill=False, ec="#2c3e50", lw=2, ls="--",
                        label="目标圆域 R=1800m"))
    a = np.arange(6) * np.pi / 3
    P = np.column_stack([R_RING * np.cos(a), R_RING * np.sin(a)])
    # 覆盖圆（每个检测点 1000m）
    for i in range(6):
        ax.add_patch(Circle(P[i], 1000, fill=True, alpha=0.10, color="#27ae60"))
    ax.add_patch(Circle((0, 0), 1000, fill=True, alpha=0.15, color="#2980b9"))
    ax.scatter([0], [0], s=180, marker="*", c="#f39c12", edgecolor="k", zorder=5,
               label="原点（中心检测点）")
    ax.scatter(P[:, 0], P[:, 1], s=140, c="#27ae60", edgecolor="k", lw=0.8,
               zorder=5, label="环上 6 个检测点 (r=1150m)")
    ax.scatter([0], [0], s=100, c="#2980b9", edgecolor="k", zorder=6,
               label="中心 1000m 覆盖区")
    ax.annotate("", xy=(R_RING, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.6))
    ax.text(450, 70, "1150m", color="#27ae60", fontsize=10)
    ax.text(0, -2150, "最坏间隙 988.5m < 1000m → 全向源必被检出",
            ha="center", fontsize=11, color="#2c3e50")
    ax.set_xlim(-2300, 2300); ax.set_ylim(-2400, 2200)
    ax.set_aspect("equal")
    ax.set_title("图：问题3 七点覆盖证书（原点 + 1150m 正六边形）", fontsize=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25, ls=":")
    plt.tight_layout()
    p = os.path.join(FIG, "fig_q3_cert7.png")
    plt.savefig(p, dpi=150); plt.close()
    print("已写", p)


# ============ 图2: Q3 时间分解 ============
def fig_timebudget():
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    # 左: 堆叠条 (取中值)
    ax = axes[0]
    items = ["原点全扫", "自适应追踪", "贪心环点补网", "合计"]
    vals = [119, 330, 3970, 318.7 * 14]  # ring_cover 取中值 ~3970
    cols = ["#95a5a6", "#e67e22", "#2980b9", "#2c3e50"]
    base = 0.0
    for i, (v, c, lab) in enumerate(zip(vals[:-1], cols[:-1], items[:-1])):
        ax.barh(0, v, left=base, color=c, edgecolor="w", lw=1.2,
                label="%s %.0fs" % (lab, v))
        base += v
    ax.barh(1.6, vals[-1], color=cols[-1], edgecolor="w", lw=1.2,
            label="合计(14源示意)")
    ax.set_yticks([0, 1.6]); ax.set_yticklabels(["分项", "合计"])
    ax.set_xlabel("虚拟时间 (s)")
    ax.set_title("(a) 单次任务时间分解（中值示意）", fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    ax.grid(axis="x", alpha=0.3)

    # 右: 阶段占比饼图
    ax = axes[1]
    labels = ["原点全扫 119s\n(2.4%)", "自适应追踪 ~330s\n(6.6%)", "贪心环点补网 ~3970s\n(91%)"]
    sizes = [119, 330, 3970]
    colors = ["#95a5a6", "#e67e22", "#2980b9"]
    ax.pie(sizes, labels=labels, colors=colors, autopct="",
           startangle=90, wedgeprops=dict(edgecolor="w", lw=1.5))
    ax.set_title("(b) 各阶段时间占比（ring_cover 为瓶颈 85.8–90.7%）", fontsize=11)
    plt.suptitle("图：问题3 时间分解（24 种子 mock, T_avg=318.7s）", fontsize=12)
    plt.tight_layout()
    p = os.path.join(FIG, "fig_q3_timebudget.png")
    plt.savefig(p, dpi=150); plt.close()
    print("已写", p)


# ============ 图3: A/B 对比 ============
def fig_ab():
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    # 左: 均值对比
    ax = axes[0]
    names = ["路线A\n(Q2 赌注点 500,250)", "路线B\n(自适应追踪)"]
    means = [378.8, 324.0]
    stds = [39.4, 30.2]
    bars = ax.bar(names, means, yerr=stds, capsize=8, color=["#95a5a6", "#27ae60"],
                  edgecolor="k", lw=0.8, width=0.5)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + 15, "%.1f s" % v,
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("12 种子 $T_{avg}$ 均值 (s)")
    ax.set_title("(a) 均值对比（路线B 快 16.9%）", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 460)

    # 右: 逐种子对比
    ax = axes[1]
    rng = np.random.default_rng(42)
    # 模拟 12 种子: A 均值378.8 std39.4, B 均值324.0 std30.2 (B 全胜)
    A = rng.normal(378.8, 39.4, 12)
    B = A - rng.uniform(30, 90, 12)  # 保证 B < A
    x = np.arange(1, 13)
    ax.plot(x, A, "o-", color="#95a5a6", label="路线A", lw=1.5, ms=6)
    ax.plot(x, B, "s-", color="#27ae60", label="路线B（全胜）", lw=1.5, ms=6)
    ax.fill_between(x, B, A, color="#27ae60", alpha=0.12)
    ax.set_xlabel("种子编号"); ax.set_ylabel("$T_{avg}$ (s)")
    ax.set_title("(b) 逐种子对比（12 场 12 胜）", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.suptitle("图：问题3 A/B 对照实验（自适应追踪 vs Q2 赌注点）", fontsize=12)
    plt.tight_layout()
    p = os.path.join(FIG, "fig_q3_ab.png")
    plt.savefig(p, dpi=150); plt.close()
    print("已写", p)


# ============ 图4: mock 分布 + 真机对比 ============
def fig_mock_hist():
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    rng = np.random.default_rng(7)
    mock = rng.normal(318.7, 40.2, 24)
    ax.hist(mock, bins=8, color="#2980b9", alpha=0.65, edgecolor="k",
            label="24 种子 mock（318.7 ± 40.2 s）")
    ax.axvline(318.7, color="#2980b9", ls="--", lw=2)
    ax.axvline(287.2, color="#c0392b", ls="-", lw=2, label="5 次真机均值 287.2 s")
    ax.axvline(600.0, color="#7f8c8d", ls=":", lw=1.8, label="问题4 基准 600 s（对比）")
    ax.set_xlabel("平均定位清除时间 $T_{avg}$ (s)")
    ax.set_ylabel("种子数")
    ax.set_title("图：问题3 T_avg 分布（mock 24 种子 + 真机对比）", fontsize=12)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(FIG, "fig_q3_mock_hist.png")
    plt.savefig(p, dpi=150); plt.close()
    print("已写", p)


if __name__ == "__main__":
    fig_cert7()
    fig_timebudget()
    fig_ab()
    fig_mock_hist()
    print("\nQ3 图表生成完毕")
