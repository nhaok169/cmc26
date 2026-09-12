# -*- coding: utf-8 -*-
"""问题4 示意图：环绕网、轨迹、清除进度，各出一张。"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from matplotlib.patches import Circle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "paper"))
from run_q4 import listen_net, ring_points  # noqa: E402
import figstyle as fs  # noqa: E402

fs.apply()

OUT = os.path.join(HERE, "figs")
LOG = os.path.join(HERE, "logs", "q4_20260911_223928.json")
ARENA = 1800.0


def load_log(path):
    d = json.loads(open(path, encoding="utf-8").read())
    xy, clears, t_clear = [], [], []
    for r in d["log"]:
        if r["path"] not in ("/measure", "/clear"):
            continue
        p = r["payload"]["position"]
        x, y = float(p["x"]), float(p["y"])
        xy.append((x, y))
        if r["path"] == "/clear" and r["response"].get("clear_result") == "success":
            clears.append((x, y, int(r["payload"]["channel"])))
            t_clear.append(float(r["response"]["virtual_time_s"]))
    return np.array(xy), clears, t_clear


def fig_net(pts):
    fig, ax = fs.new_fig()
    ax.add_patch(Circle((0, 0), ARENA, fill=False, ec=fs.ARENA, lw=1.15))
    ax.add_patch(Circle((0, 0), 1000, fill=False, ec=fs.BLUE, lw=0.9, ls="--"))
    ax.plot(pts[:, 0], pts[:, 1], "o", ms=6, color=fs.ROSE, zorder=5)
    ax.plot(pts[0, 0], pts[0, 1], "s", ms=8, color=fs.BLUE, zorder=6)
    ax.annotate("O", (80, 80), color=fs.BLUE, fontsize=10)
    ax.set_xlim(-2100, 2100)
    ax.set_ylim(-2100, 2100)
    fs.style_map(ax, title="实用环绕网（圆内）")
    fs.save(fig, os.path.join(OUT, "fig_q4_net.png"))


def fig_cert():
    fig, ax = fs.new_fig()
    th = np.linspace(0, 2 * np.pi, 360)
    ax.plot(ARENA * np.cos(th), ARENA * np.sin(th), color=fs.ARENA, lw=1.15)
    ax.plot(1000 * np.cos(th), 1000 * np.sin(th), color=fs.BLUE, lw=0.8, ls="--")
    layers = [
        (ring_points(400.0, 6), fs.BLUE, "400 m×6"),
        (ring_points(1300.0, 8), fs.TEAL, "1300 m×8"),
        (ring_points(1900.0, 12), fs.ROSE, "1900 m×12（圆外）"),
    ]
    for pts, c, lab in layers:
        xy = np.array(pts)
        ax.plot(xy[:, 0], xy[:, 1], "o", ms=5.5, color=c, zorder=5, label=lab)
    G = np.array([1800.0, 0.0])
    ax.plot(G[0], G[1], "*", ms=12, color=fs.INK, zorder=6)
    ax.annotate("圆缘源", G, textcoords="offset points", xytext=(-18, 12), fontsize=9)
    ax.fill_betweenx([-400, 400], 1800, 2300, color=fs.FILL_WARN, alpha=0.45, zorder=1)
    ax.annotate("外向半平面", (2050, 220), fontsize=8.5, color=fs.MUTED)
    ax.set_xlim(-2300, 2300)
    ax.set_ylim(-2300, 2300)
    ax.legend(loc="lower left", fontsize=8)
    fs.style_map(ax, title="26 点环抱证书（外环必须出圆）")
    fs.save(fig, os.path.join(OUT, "fig_q4_cert.png"))


def fig_traj(pts, xy, clears):
    fig, ax = fs.new_fig()
    ax.add_patch(Circle((0, 0), ARENA, fill=False, ec=fs.ARENA, lw=1.05))
    ax.plot(xy[:, 0], xy[:, 1], color=fs.TEAL, lw=0.8, zorder=2, alpha=0.85)
    ax.plot(pts[:, 0], pts[:, 1], "o", ms=4, color=fs.ROSE, alpha=0.75, zorder=3)
    if clears:
        C = np.array([(c[0], c[1]) for c in clears])
        ax.plot(C[:, 0], C[:, 1], "*", ms=11, color=fs.GREEN, zorder=5)
        for x, y, ch in clears:
            ax.annotate(str(ch), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8, color=fs.GREEN)
    ax.set_xlim(-1950, 1950)
    ax.set_ylim(-1950, 1950)
    fs.style_map(ax, title="演练轨迹与清除点")
    fs.save(fig, os.path.join(OUT, "fig_q4_traj.png"))


def fig_time(t_clear):
    fig, ax = fs.new_fig()
    ax.plot(t_clear, np.arange(1, len(t_clear) + 1), "-o", ms=5, color=fs.BLUE, lw=1.7)
    ax.set_xlabel("虚拟时间 / s")
    ax.set_ylabel("累计清除数")
    ax.set_ylim(0, 17)
    fs.style_xy(ax, title="清除进度")
    fs.save(fig, os.path.join(OUT, "fig_q4_time.png"))


def main():
    os.makedirs(OUT, exist_ok=True)
    pts = np.array(listen_net())
    xy, clears, t_clear = load_log(LOG)
    fig_net(pts)
    fig_traj(pts, xy, clears)
    fig_time(t_clear)
    fig_cert()


if __name__ == "__main__":
    main()
