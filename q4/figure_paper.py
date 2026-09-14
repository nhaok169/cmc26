# -*- coding: utf-8 -*-
"""问题四论文图：环抱判据、26 点证书、圆外停点、策略对照。"""
from __future__ import annotations

import os
import sys

import matplotlib
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Wedge

matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "paper"))
import figstyle as fs  # noqa: E402

fs.apply()
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)


def ring(r, k, ph=0.0):
    a = ph + np.arange(k) * 2 * np.pi / k
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


def fig_encircle():
    fig, axes = fs.new_fig(1, 2)
    ax, bx = axes
    G = np.array([0.0, 0.0])
    R = 1000.0
    th = np.linspace(0, 2 * np.pi, 360)

    ax.plot(R * np.cos(th), R * np.sin(th), color=fs.GRID, lw=1.0, ls=":")
    angs = np.deg2rad([20, 40, 70, 100])
    P = R * 0.82 * np.column_stack([np.cos(angs), np.sin(angs)])
    ax.add_patch(Wedge(G, R * 0.82, 100, 380, facecolor=fs.FILL_BAD, edgecolor="none", alpha=0.85, zorder=1))
    ax.plot(P[:, 0], P[:, 1], "o", color=fs.ROSE, ms=8, zorder=4)
    ax.plot(*G, "*", color=fs.INK, ms=14, zorder=5)
    ax.annotate(r"$G$", G, textcoords="offset points", xytext=(8, -16), fontsize=10)
    ax.text(0.50, 0.08, r"$\mathrm{maxgap}=280^\circ>180^\circ$" + "\n背向空隙可漏检",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color=fs.ROSE)
    ax.set_xlim(-1150, 1150)
    ax.set_ylim(-1150, 1150)
    fs.style_map(ax, title=r"(a) 不满足环抱")

    bx.plot(R * np.cos(th), R * np.sin(th), color=fs.GRID, lw=1.0, ls=":")
    angs2 = np.deg2rad(np.arange(0, 360, 45))
    P2 = R * 0.82 * np.column_stack([np.cos(angs2), np.sin(angs2)])
    bx.add_patch(Wedge(G, R * 0.82, 20, 200, facecolor=fs.FILL_IO, edgecolor="none", alpha=0.80, zorder=1))
    bx.plot(P2[:, 0], P2[:, 1], "o", color=fs.GREEN, ms=8, zorder=4)
    bx.plot(*G, "*", color=fs.INK, ms=14, zorder=5)
    bx.annotate(r"$G$", G, textcoords="offset points", xytext=(8, -16), fontsize=10)
    bx.text(0.50, 0.08, r"$\mathrm{maxgap}=45^\circ\leq 180^\circ$" + "\n任意朝向必含探针",
            transform=bx.transAxes, ha="center", va="bottom", fontsize=8.5, color=fs.GREEN)
    bx.set_xlim(-1150, 1150)
    bx.set_ylim(-1150, 1150)
    fs.style_map(bx, title=r"(b) 满足环抱")
    fs.save(fig, os.path.join(OUT, "fig_q4_encircle.png"))


def fig_cert26():
    fig, ax = fs.new_fig()
    th = np.linspace(0, 2 * np.pi, 360)
    ax.plot(1800 * np.cos(th), 1800 * np.sin(th), color=fs.ARENA, lw=1.2, zorder=2)
    ax.plot(1000 * np.cos(th), 1000 * np.sin(th), color=fs.BLUE, lw=0.9, ls="--", zorder=2)
    rings = [
        (ring(400, 6), fs.BLUE, r"$400\,\mathrm{m}\times 6$"),
        (ring(1300, 8), fs.TEAL, r"$1300\,\mathrm{m}\times 8$"),
        (ring(1900, 12), fs.ROSE, r"$1900\,\mathrm{m}\times 12$（圆外）"),
    ]
    for P, col, lab in rings:
        ax.plot(P[:, 0], P[:, 1], "o", color=col, ms=7, zorder=4, label=lab, ls="none")
    ax.add_patch(Rectangle((1800, -420), 520, 840, facecolor=fs.FILL_WARN, edgecolor="none", alpha=0.55, zorder=1))
    ax.plot(1800, 0, "*", color=fs.INK, ms=13, zorder=6)
    ax.annotate(
        "圆缘源 $G$",
        xy=(1800, 0),
        xytext=(520, 1520),
        textcoords="data",
        fontsize=8.5,
        ha="center",
        arrowprops=dict(arrowstyle="-|>", color=fs.INK, lw=0.8, mutation_scale=8),
    )
    ax.text(2140, 180, "外向\n半平面", fontsize=8, color=fs.MUTED, ha="center", va="center")
    ax.set_xlim(-2350, 2550)
    ax.set_ylim(-2350, 2350)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=col, markeredgecolor=col, markersize=7, label=lab)
        for (_, col, lab) in rings
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    fs.style_map(ax, title=r"26 点环抱证书（外环必须出圆）")
    fs.save(fig, os.path.join(OUT, "fig_q4_cert26.png"))


def fig_edgeaway():
    fig, ax = fs.new_fig()
    th = np.linspace(0, 2 * np.pi, 360)
    ax.plot(1800 * np.cos(th), 1800 * np.sin(th), color=fs.ARENA, lw=1.25, ls="--", zorder=2)
    G = np.array([1800.0, 0.0])
    ax.add_patch(Rectangle((1800, -1100), 700, 2200, facecolor=fs.FILL_BAD, edgecolor="none", alpha=0.55, zorder=1))
    ax.add_patch(Circle(G, 1000, fill=False, ec=fs.ROSE, lw=1.1, ls=":", zorder=3))
    inside = np.array([[400.0, 300.0], [1200.0, 580.0], [1150.0, 40.0]])
    ax.plot(inside[:, 0], inside[:, 1], "o", color=fs.BLUE, ms=8, zorder=5)
    for p in inside:
        ax.plot([p[0], G[0]], [p[1], G[1]], color=fs.BLUE, lw=0.8, ls=":", zorder=3)
    ax.plot(*G, "*", color=fs.ROSE, ms=14, zorder=6)
    ax.annotate(r"源 $G$（圆缘外向）", G, textcoords="offset points", xytext=(-150, 18), fontsize=9, color=fs.ROSE)
    ax.annotate("可检半平面 $x\\geq 1800$", (2150, 720), fontsize=8.5, color=fs.ROSE, ha="center")
    ax.annotate("圆内探针均在盲侧", (200, 900), fontsize=8.5, color=fs.BLUE)
    ax.set_xlim(-2100, 2600)
    ax.set_ylim(-2000, 2100)
    fs.style_map(ax, title=r"圆缘外向源：圆内探针 maxgap $=360^\circ$")
    fs.save(fig, os.path.join(OUT, "fig_q4_edgeaway.png"))


def fig_strat_rate():
    labels = ["七点覆盖", "实用环绕\n+早停", "实用环绕\n走满", "26 点环抱\n（采用）"]
    mix = np.array([1, 6, 6, 7]) / 8 * 100
    out = np.array([0, 0, 0, 5]) / 6 * 100
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = fs.new_fig()
    ax.bar(x - w / 2, mix, w, color=fs.BLUE, label="混合随机（8 局）")
    ax.bar(x + w / 2, out, w, color=fs.ROSE, label="圆缘外向应力（6 局）")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("全清比例 / %")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper left", fontsize=8.5)
    fs.style_xy(ax, title="问题四本地全清比例")
    fs.save(fig, os.path.join(OUT, "fig_q4_strat_rate.png"))


def fig_compare():
    q3 = np.array([327, 380, 329, 339, 355, 290, 368, 402, 319, 328, 397, 366, 384, 382, 340, 385, 306, 351, 337])
    q4 = np.array([583.47, 518.78, 698.52])  # 正式测试三局；演练 17 局另见表
    fig, ax = fs.new_fig()
    data = [q3, q4]
    bp = ax.boxplot(data, tick_labels=["问题三演练\n19 局", "问题四正式测试\n3 局"], widths=0.45,
                    patch_artist=True, medianprops=dict(color=fs.SAND, lw=1.6))
    for patch, col in zip(bp["boxes"], [fs.FILL_OK, fs.FILL_BAD]):
        patch.set_facecolor(col)
        patch.set_edgecolor(fs.INK)
    ax.scatter(np.ones(len(q3)), q3, color=fs.INK, s=12, zorder=4, alpha=0.75)
    ax.scatter(2 * np.ones(len(q4)), q4, color=fs.INK, s=22, zorder=4)
    ax.set_ylabel(r"平均定位清除时间 / s")
    fs.style_xy(ax, title="问题三演练与问题四正式测试")
    fs.save(fig, os.path.join(OUT, "fig_q4_compare.png"))


def main():
    fig_encircle()
    fig_cert26()
    fig_edgeaway()
    fig_strat_rate()
    fig_compare()


if __name__ == "__main__":
    main()
