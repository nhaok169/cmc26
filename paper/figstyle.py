# -*- coding: utf-8 -*-
"""论文插图统一体例：温和配色、清晰字体、固定版心。

所有图按同一画布输出（约 14.2 cm × 10.7 cm），Word 里等宽插入后高度一致。
同类对照才用 1×2 / 1×3；地图与曲线、热力图与折线不再拼在同一张图里。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Paul Tol muted
BLUE = "#4C78A8"
TEAL = "#72B7B2"
GREEN = "#54A24B"
SAND = "#ECA957"
ROSE = "#C44E52"
PURPLE = "#8B6BB1"
INK = "#2B2B2B"
MUTED = "#6B6B6B"
GRID = "#D9D9D9"
FILL_P = "#E6C5C6"
FILL_OK = "#C5DCC0"
FILL_WARN = "#F3E4C4"
FILL_BAD = "#F0D0D0"
FILL_IO = "#D5E3F0"
FILL_TEAL = "#C9E4E1"
ARENA = "#8A8A8A"

W_IN = 5.59  # 14.2 cm
H = 4.20
H_MAP = H
H_WIDE2 = H
H_LINE = H
H_FLOW = H

DPI = 300

CMAP_DIAM = LinearSegmentedColormap.from_list(
    "paper_diam",
    ["#F4F1EA", "#D8E4EE", "#9BB8D3", "#4C78A8", "#2F4A6A"],
)
CMAP_CAND = LinearSegmentedColormap.from_list(
    "paper_cand",
    ["#D5E3F0", "#C5DCC0"],
)


def apply():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Times New Roman", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            "legend.frameon": True,
            "legend.edgecolor": GRID,
            "legend.fancybox": False,
            "axes.unicode_minus": False,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.linewidth": 0.8,
            "grid.color": GRID,
            "grid.linestyle": ":",
            "grid.linewidth": 0.7,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "mathtext.fontset": "cm",
        }
    )


def new_fig(nrows=1, ncols=1, **kwargs):
    kwargs.setdefault("figsize", (W_IN, H))
    kwargs.setdefault("constrained_layout", True)
    return plt.subplots(nrows, ncols, **kwargs)


def save(fig, path, dpi=DPI, size=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if size is None:
        size = (W_IN, H)
    fig.set_size_inches(size[0], size[1], forward=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    fig.savefig(str(path.with_suffix(".pdf")), bbox_inches="tight", facecolor="white", pad_inches=0.04)
    plt.close(fig)
    print("saved", path)


def style_map(ax, title=None, xlabel="x / m", ylabel="y / m"):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=6, color=INK)
    ax.grid(True, zorder=0)
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.8)


def style_xy(ax, title=None):
    if title:
        ax.set_title(title, pad=6, color=INK)
    ax.grid(True, axis="both", zorder=0)
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.8)
