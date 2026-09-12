# -*- coding: utf-8 -*-
"""Q4 理论示意图生成 (环抱证书核心创新).

输出到 q4/figs/:
  fig_cert_skeleton.png    26 点证书骨架 (三环)
  fig_encircle.png         环抱证书原理 (探针环抱 G, maxgap<=180)
  fig_n3_edgeaway.png      N3 强命题: 圆缘朝外源, 圆内点零检出
  fig_detect_semantics.png 检测语义: 全向 360 vs 定向 180 半平面
  fig_timebudget.png       时间账分解 (堆叠条)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = r"E:/Draft/MathModel"
sys.path.insert(0, os.path.join(ROOT, "q4"))
sys.path.insert(0, os.path.join(ROOT, "q2"))
FIG = os.path.join(ROOT, "q4", "figs")
os.makedirs(FIG, exist_ok=True)

from ledger2 import SKELETON


def ring(r, k, ph=0.0):
    a = ph + np.arange(k) * 2 * np.pi / k
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


# ============ 图1: 26 点证书骨架 ============
def fig_skeleton():
    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    ax.add_patch(Circle((0, 0), 1800, fill=False, ec="#2c3e50", lw=2, ls="--", zorder=1))
    rings = [(0, 6, "#e74c3c", "内环 r=400, 6 点"),
             (6, 14, "#2980b9", "中环 r=1300, 8 点"),
             (14, 26, "#27ae60", "外环 r=1900, 12 点")]
    for a, b, c, lab in rings:
        P = SKELETON[a:b]
        ax.scatter(P[:, 0], P[:, 1], s=110, c=c, edgecolor="k", linewidth=0.7,
                   zorder=3, label="%s" % lab)
    # 圆心
    ax.scatter([0], [0], s=140, marker="*", c="#f39c12", edgecolor="k", zorder=4,
               label="原点（初始扫描点）")
    # 标注半径
    ax.annotate("", xy=(400, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#e74c3c"))
    ax.annotate("", xy=(1300, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#2980b9"))
    ax.annotate("", xy=(1900, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#27ae60"))
    ax.text(200, 60, "400", color="#e74c3c", fontsize=10)
    ax.text(650, 60, "1300", color="#2980b9", fontsize=10)
    ax.text(950, 60, "1900", color="#27ae60", fontsize=10)
    ax.text(0, -2150, "目标圆域 R=1800m", ha="center", fontsize=11, color="#2c3e50")
    ax.set_xlim(-2500, 2500); ax.set_ylim(-2500, 2500)
    ax.set_aspect("equal")
    ax.set_title("图：26 点环抱证书骨架\n（判空下界：maxgap = 171.8° < 180°）", fontsize=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25, ls=":")
    plt.tight_layout()
    p = os.path.join(FIG, "fig_cert_skeleton.png")
    plt.savefig(p, dpi=150); plt.close()
    print("已写", p)


# ============ 图2: 环抱证书原理 ============
def fig_encircle():
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.0))
    G = np.array([0.0, 200.0])
    R = 1000.0

    # 左: 违规 (maxgap > 180)
    ax = axes[0]
    ax.add_patch(Circle(G, R, fill=False, ec="#95a5a6", lw=1.2, ls=":"))
    angs = np.array([20, 40, 70, 100])  # 探针方向(度)
    P = G + R * 0.85 * np.column_stack([np.cos(np.radians(angs)), np.sin(np.radians(angs))])
    ax.scatter(P[:, 0], P[:, 1], s=120, c="#c0392b", edgecolor="k", zorder=3, label="探针")
    ax.scatter([G[0]], [G[1]], s=180, marker="*", c="#2c3e50", zorder=4, label="候选位置 G")
    # 画最大空隙楔形 (仅空隙区, 从最大角到最小角+360)
    ax.add_patch(Wedge(G, R * 0.85, 100, 380, alpha=0.30, color="#e74c3c",
                       label="最大空隙 280°"))
    ax.text(G[0] - 100, G[1] - 700, "maxgap = 280° > 180°\n存在朝向背对全部探针\n→ 可能漏检（证书失效）",
            ha="center", fontsize=11, color="#c0392b")
    ax.set_xlim(-1400, 1400); ax.set_ylim(-1200, 1400); ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.25, ls=":")
    ax.set_title("(a) 不满足环抱条件（有漏洞）", fontsize=11)

    # 右: 合规 (maxgap <= 180)
    ax = axes[1]
    ax.add_patch(Circle(G, R, fill=False, ec="#95a5a6", lw=1.2, ls=":"))
    angs2 = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    P2 = G + R * 0.85 * np.column_stack([np.cos(np.radians(angs2)), np.sin(np.radians(angs2))])
    ax.scatter(P2[:, 0], P2[:, 1], s=120, c="#27ae60", edgecolor="k", zorder=3, label="探针")
    ax.scatter([G[0]], [G[1]], s=180, marker="*", c="#2c3e50", zorder=4, label="候选位置 G")
    # 画 180 度扇区示意
    ax.add_patch(Wedge(G, R * 0.85, 20, 200, alpha=0.20, color="#2980b9"))
    ax.text(G[0], G[1] - 700,
            "maxgap = 45° ≤ 180°\n任意闭半平面必含探针\n→ 任何朝向都被检出",
            ha="center", fontsize=11, color="#27ae60")
    ax.set_xlim(-1400, 1400); ax.set_ylim(-1200, 1400); ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.25, ls=":")
    ax.set_title("(b) 满足环抱条件（充要）", fontsize=11)
    plt.suptitle("图：环抱证书判空原理（蓝色扇区=任一朝向的 180° 覆盖半平面）", fontsize=12)
    plt.tight_layout()
    p = os.path.join(FIG, "fig_encircle.png")
    plt.savefig(p, dpi=150); plt.close()
    print("已写", p)


# ============ 图3: N3 强命题 ============
def fig_n3():
    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    ax.add_patch(Circle((0, 0), 1800, fill=False, ec="#2c3e50", lw=2, ls="--"))
    # 圆缘朝外源 G=(1800,0), 朝向 phi=0 (朝外)
    G = np.array([1800.0, 0.0])
    ax.scatter([G[0]], [G[1]], s=200, marker="*", c="#c0392b", edgecolor="k", zorder=5)
    ax.text(1750, 200, "源 G（圆缘）", fontsize=11, color="#c0392b", ha="right")
    # 覆盖半平面 (朝外): 半平面 x >= 1800
    ax.add_patch(Polygon([[1800, -1000], [2600, -1000], [2600, 1000], [1800, 1000]],
                         closed=True, alpha=0.25, color="#e74c3c"))
    ax.text(2150, 400, "覆盖半平面\n（朝外, x≥1800）", fontsize=10, color="#c0392b", ha="center")
    # 1000m 接收圆
    ax.add_patch(Circle(G, 1000, fill=False, ec="#e67e22", lw=1.5, ls=":"))
    # 有效的检出点集合 = 半平面 ∩ 圆 ∩ 圆域 ≈ {G}
    ax.scatter([G[0]], [G[1]], s=400, facecolors="none", edgecolors="#e67e22", lw=2, zorder=4)
    # 圆内任意点 p
    for p in [(1150, 0), (1200, 600), (300, 300)]:
        ax.scatter([p[0]], [p[1]], s=80, c="#2980b9", edgecolor="k", zorder=3)
        ax.plot([G[0], p[0]], [G[1], p[1]], "b--", lw=0.8, alpha=0.6)
    ax.text(600, 900, "圆内任意检测点 p\n(S-G)·u_φ < 0\n→ 位于源盲侧, 零检出",
            fontsize=11, color="#2980b9", ha="left")
    ax.set_xlim(-2200, 2900); ax.set_ylim(-2200, 2200)
    ax.set_aspect("equal")
    ax.set_title("图：N3 强命题——圆缘朝外定向源\n（纯圆内检测点集对此类源零检出 → 证书必须含圆外点）", fontsize=12)
    ax.grid(alpha=0.25, ls=":")
    plt.tight_layout()
    p = os.path.join(FIG, "fig_n3_edgeaway.png")
    plt.savefig(p, dpi=150); plt.close()
    print("已写", p)


# ============ 图4: 检测语义 ============
def fig_semantics():
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.2))
    R = 1000.0

    ax = axes[0]
    ax.add_patch(Circle((0, 0), R, fill=True, alpha=0.25, color="#27ae60"))
    ax.add_patch(Circle((0, 0), R, fill=False, ec="#27ae60", lw=2))
    ax.scatter([0], [0], s=200, marker="*", c="#c0392b", edgecolor="k", zorder=4)
    ax.text(0, -R - 180, "全向源：360° 均匀辐射\n任意方向检测点（≤1000m）必检出",
            ha="center", fontsize=11, color="#1e8449")
    ax.annotate("", xy=(R * 0.95, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#27ae60", lw=2))
    ax.text(R * 0.5, 60, "R∈[1000,1500]", fontsize=9, color="#1e8449")
    ax.set_xlim(-1300, 1300); ax.set_ylim(-1300, 1300); ax.set_aspect("equal")
    ax.grid(alpha=0.25, ls=":"); ax.set_title("(a) 全向源检测语义", fontsize=11)

    ax = axes[1]
    # 定向源: 朝右 180 度半平面
    ax.add_patch(Wedge((0, 0), R, -90, 90, alpha=0.25, color="#e74c3c"))
    ax.add_patch(Circle((0, 0), R, fill=False, ec="#7f8c8d", lw=1.2, ls=":"))
    ax.plot([0, 0], [-R, R], "k-", lw=1.2)
    ax.scatter([0], [0], s=200, marker="*", c="#c0392b", edgecolor="k", zorder=4)
    # 盲侧点
    ax.scatter([-600, -400], [300, -400], s=90, c="#2980b9", edgecolor="k", zorder=3)
    ax.text(-950, 500, "盲侧（无信号）\nno_signal ≠ 距离远", fontsize=10, color="#2980b9")
    # 点亮侧点
    ax.scatter([600, 400], [300, -400], s=90, c="#27ae60", edgecolor="k", zorder=3)
    ax.text(500, 800, "点亮侧\n（可测得示向度）", fontsize=10, color="#1e8449")
    ax.annotate("", xy=(R * 0.95, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2))
    ax.text(R * 0.5, 80, "定向方向 φ", fontsize=9, color="#c0392b")
    ax.set_xlim(-1300, 1300); ax.set_ylim(-1300, 1300); ax.set_aspect("equal")
    ax.grid(alpha=0.25, ls=":")
    ax.set_title("(b) 定向源检测语义（180° 闭半平面）", fontsize=11)
    plt.suptitle("图：两类干扰源的检测语义对比（问题4 的核心变化）", fontsize=12)
    plt.tight_layout()
    p = os.path.join(FIG, "fig_detect_semantics.png")
    plt.savefig(p, dpi=150); plt.close()
    print("已写", p)


# ============ 图5: 时间账 ============
def fig_timebudget():
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    items = ["初始扫描", "证书巡回移动", "绕行清除/接近源", "末段 finalize",
             "测量", "换频", "合计"]
    vals = [119, 3752, 1250, 460, 1615, 300, 7496]
    cols = ["#95a5a6", "#2980b9", "#e67e22", "#8e44ad", "#27ae60", "#f39c12", "#2c3e50"]
    left = 0.0
    bars = []
    for i, (v, c) in enumerate(zip(vals[:-1], cols[:-1])):
        b = ax.barh(0, v, left=left, color=c, edgecolor="w", linewidth=1.2,
                    label="%s %.0fs" % (items[i], v))
        bars.append(b)
        left += v
    ax.barh(1.6, vals[-1], color="#2c3e50", edgecolor="w", linewidth=1.2,
            label="合计 %.0fs" % vals[-1])
    ax.set_yticks([0, 1.6])
    ax.set_yticklabels(["分项", "合计"])
    ax.set_xlabel("虚拟时间 (s)")
    ax.set_title("图：问题4 单种子时间账分解（24 种子平均，T_avg≈592s）", fontsize=12)
    ax.legend(loc="lower right", fontsize=8.5, ncol=1, framealpha=0.95)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, 8300)
    plt.tight_layout()
    p = os.path.join(FIG, "fig_timebudget.png")
    plt.savefig(p, dpi=150); plt.close()
    print("已写", p)


if __name__ == "__main__":
    fig_skeleton()
    fig_encircle()
    fig_n3()
    fig_semantics()
    fig_timebudget()
    print("\n理论示意图全部生成完毕")
