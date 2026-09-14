# -*- coding: utf-8 -*-
"""问题1流程：求顶点后，直径圆覆盖与最小包围圆并行。"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, str(os.path.join(HERE, "..", "paper")))
import figstyle as fs  # noqa: E402

fs.apply()
OUT_DIR = os.path.join(HERE, "figs")

STYLES = {
    "start": (fs.GRID, fs.INK, 0.28),
    "proc": ("#FFFFFF", fs.INK, 0.04),
    "io": (fs.FILL_IO, fs.BLUE, 0.04),
    "ok": (fs.FILL_OK, fs.GREEN, 0.10),
    "fail": (fs.FILL_BAD, fs.ROSE, 0.10),
}


def box(ax, x, y, w, h, text, kind="proc", fontsize=10):
    fc, ec, rad = STYLES[kind]
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle=f"round,pad=0.02,rounding_size={rad}",
            linewidth=1.05,
            facecolor=fc,
            edgecolor=ec,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=fs.INK, linespacing=1.25)
    return x, y, w, h


def diamond(ax, x, y, w, h, text, fontsize=10):
    verts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
    ax.add_patch(Polygon(verts, closed=True, facecolor=fs.FILL_WARN, edgecolor=fs.INK, lw=1.05))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=fs.INK, linespacing=1.15)
    return x, y, w, h


def arrow(ax, x1, y1, x2, y2, text=None, dy=0.14, color=None):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color or fs.INK, lw=1.0, mutation_scale=9),
    )
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, text, fontsize=9, color=fs.MUTED, ha="center", va="bottom")


def main():
    fig, ax = plt.subplots(figsize=(fs.W_IN, fs.H))
    ax.set_xlim(0.05, 10.55)
    ax.set_ylim(0.08, 6.55)
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.03)

    y1, y0, yf = 5.45, 3.20, 1.15

    a = box(ax, 0.95, y1, 1.30, 0.66, "开始", "start", 11)
    b = box(ax, 2.90, y1, 2.00, 0.78, "输入 $S_i$、$\\theta_i$", "io", 10.5)
    c = box(ax, 5.25, y1, 2.15, 0.86, "半平面求交\n得到顶点 $V$", "proc", 10.5)
    d = diamond(ax, 7.55, y1, 1.70, 1.05, "$V$ 有界\n且非空？", 10)
    fail = box(ax, 9.40, y1, 1.20, 0.70, "直径\n无定义", "fail", 9.5)

    e = box(ax, 1.55, y0, 2.10, 0.92, "枚举全部顶点对\n最长连线即直径 $AB$", "proc", 9.5)
    f = box(ax, 3.95, y0, 1.95, 0.88, "圆心取中点\n$O=(A+B)/2$", "proc", 9.5)
    g = diamond(ax, 6.25, y0, 1.75, 1.08, "各顶点到 $O$\n均不超过 $R$？", 9.5)
    mecb = box(ax, 8.95, y0, 2.20, 0.92, "求最小包围圆\n得到 $R_{\\mathrm{MEC}}$", "io", 9)
    h = box(ax, 4.90, yf, 1.70, 0.64, "直径圆能覆盖", "ok", 9)
    no = box(ax, 7.40, yf, 1.90, 0.64, "直径圆不能覆盖", "fail", 9)

    arrow(ax, a[0] + a[2] / 2, y1, b[0] - b[2] / 2, y1)
    arrow(ax, b[0] + b[2] / 2, y1, c[0] - c[2] / 2, y1)
    arrow(ax, c[0] + c[2] / 2, y1, d[0] - d[2] / 2, y1)
    arrow(ax, d[0] + d[2] / 2, y1, fail[0] - fail[2] / 2, y1, "否")

    ymid = 4.32
    ax.plot([d[0], d[0]], [y1 - d[3] / 2, ymid], color=fs.INK, lw=1.0)
    ax.plot([e[0], mecb[0]], [ymid, ymid], color=fs.INK, lw=1.0)
    arrow(ax, e[0], ymid, e[0], e[1] + e[3] / 2)
    arrow(ax, mecb[0], ymid, mecb[0], mecb[1] + mecb[3] / 2)
    ax.text(3.85, ymid + 0.08, "是", fontsize=9, color=fs.MUTED, ha="center")
    ax.text(e[0], ymid - 0.22, "覆盖判定", fontsize=8, color=fs.MUTED, ha="center")
    ax.text(mecb[0], ymid - 0.22, "并行", fontsize=8, color=fs.BLUE, ha="center")

    arrow(ax, e[0] + e[2] / 2, y0, f[0] - f[2] / 2, y0)
    arrow(ax, f[0] + f[2] / 2, y0, g[0] - g[2] / 2, y0)
    ax.plot([g[0], g[0]], [y0 - g[3] / 2, (y0 + yf) / 2], color=fs.INK, lw=1.0)
    ax.plot([h[0], no[0]], [(y0 + yf) / 2, (y0 + yf) / 2], color=fs.INK, lw=1.0)
    arrow(ax, h[0], (y0 + yf) / 2, h[0], h[1] + h[3] / 2)
    arrow(ax, no[0], (y0 + yf) / 2, no[0], no[1] + no[3] / 2)
    ax.text(h[0] - 0.15, (y0 + yf) / 2 + 0.08, "是", fontsize=9, color=fs.MUTED, ha="right")
    ax.text(no[0] + 0.15, (y0 + yf) / 2 + 0.08, "否", fontsize=9, color=fs.ROSE, ha="left")

    fs.save(fig, os.path.join(OUT_DIR, "fig_algorithm_flow.png"))


if __name__ == "__main__":
    main()
