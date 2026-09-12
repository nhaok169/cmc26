# -*- coding: utf-8 -*-
"""问题3 论文图：覆盖网、轨迹、清除进度、本地对照，各出一张。"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "paper"))
import figstyle as fs  # noqa: E402

fs.apply()

OUT = os.path.join(HERE, "figs")
LOG = os.path.join(HERE, "logs", "q3_20260911_161523.json")
RING_R = 1150.0
ARENA_R = 1800.0
COVER_R = 1000.0


def ring_pts():
    return [
        RING_R * np.array([math.cos(2 * math.pi * k / 6), math.sin(2 * math.pi * k / 6)])
        for k in range(6)
    ]


def parse_log(path):
    d = json.loads(open(path, encoding="utf-8").read())
    rows = d["log"]
    path_xy = []
    events = []
    prev = None
    last_ch = 1
    move_m = 0.0
    n_switch = 0
    n_meas = 0
    n_clear_ok = 0
    for r in rows:
        pth, pl, rs = r["path"], r["payload"], r["response"]
        t = float(rs.get("virtual_time_s") or 0.0)
        if pth in ("/measure", "/clear"):
            x, y = float(pl["position"]["x"]), float(pl["position"]["y"])
            ch = int(pl["channel"])
            if prev is not None:
                move_m += math.hypot(x - prev[0], y - prev[1])
            prev = (x, y)
            path_xy.append((x, y, t, ch, pth))
            if pth == "/measure":
                n_meas += 1
                if ch != last_ch:
                    n_switch += 1
                last_ch = ch
                events.append((x, y, rs.get("measure_result"), ch, t))
            else:
                if rs.get("clear_result") == "success":
                    n_clear_ok += 1
                events.append((x, y, "clear_" + str(rs.get("clear_result")), ch, t))
    summary = d["summary"]
    T = float(summary["virtual_time_s"])
    move_s = move_m / 5.0
    meas_s = n_meas * 5.0
    sw_s = n_switch * 1.0
    clr_s = n_clear_ok * 5.0
    return {
        "summary": summary,
        "path_xy": path_xy,
        "events": events,
        "move_m": move_m,
        "parts": {"移动": move_s, "检测": meas_s, "切频道": sw_s, "清除": clr_s},
        "T": T,
        "n_meas": n_meas,
        "n_switch": n_switch,
    }


def fig_coverage():
    fig, ax = fs.new_fig()
    th = np.linspace(0, 2 * np.pi, 360)
    ax.plot(ARENA_R * np.cos(th), ARENA_R * np.sin(th), color=fs.INK, lw=1.2, label="目标圆 1800 m")
    ax.add_patch(Circle((0, 0), COVER_R, fc=fs.BLUE, ec="none", alpha=0.08, lw=0))
    ax.plot(COVER_R * np.cos(th), COVER_R * np.sin(th), color=fs.BLUE, lw=1.0, ls="--")
    rings = ring_pts()
    for i, p in enumerate(rings):
        ax.add_patch(Circle(p, COVER_R, fc=fs.SAND, ec="none", alpha=0.07, lw=0))
        ax.plot(p[0], p[1], "s", color=fs.SAND, ms=7, zorder=5)
        ax.annotate(f"$A_{i}$", p, textcoords="offset points", xytext=(7, 7), fontsize=9, color=fs.SAND)
    ax.plot(0, 0, "o", color=fs.BLUE, ms=8, zorder=6)
    ax.annotate("$O$", (0, 0), textcoords="offset points", xytext=(-16, -14), fontsize=11, color=fs.BLUE)
    ang = math.pi / 6
    w = ARENA_R * np.array([math.cos(ang), math.sin(ang)])
    a0 = rings[0]
    ax.plot(w[0], w[1], "*", color=fs.ROSE, ms=12, zorder=7)
    ax.plot([w[0], a0[0]], [w[1], a0[1]], color=fs.ROSE, lw=1.0, ls=":")
    ax.annotate("最差点", w, textcoords="offset points", xytext=(-52, 10), fontsize=9, color=fs.ROSE)
    ax.set_xlim(-1950, 1950)
    ax.set_ylim(-1950, 1950)
    ax.legend(loc="lower left", fontsize=8.5)
    fs.style_map(ax, title="覆盖网：原点 + 1150 m 六等分点")
    fs.save(fig, os.path.join(OUT, "fig_q3_cover.png"))


def fig_traj(data):
    fig, ax = fs.new_fig()
    th = np.linspace(0, 2 * np.pi, 360)
    ax.plot(ARENA_R * np.cos(th), ARENA_R * np.sin(th), color=fs.ARENA, lw=1.05)
    for p in ring_pts():
        ax.plot(p[0], p[1], "s", color=fs.SAND, ms=6, zorder=4)
    ax.plot(0, 0, "o", color=fs.BLUE, ms=7, zorder=5)
    xy = np.array([[x, y] for x, y, *_ in data["path_xy"]])
    ax.plot(xy[:, 0], xy[:, 1], color=fs.MUTED, lw=0.9, alpha=0.8, zorder=2)
    clears = [(x, y, ch) for x, y, k, ch, t in data["events"] if k == "clear_success"]
    heard = [(x, y) for x, y, k, ch, t in data["events"] if k == "direction"]
    if heard:
        h = np.array(heard)
        ax.plot(h[:, 0], h[:, 1], "+", color=fs.BLUE, ms=7, zorder=3, label="测到示向度")
    for x, y, ch in clears:
        ax.plot(x, y, "*", color=fs.GREEN, ms=11, zorder=6)
        ax.annotate(str(ch), (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8, color=fs.GREEN)
    ax.set_xlim(-1950, 1950)
    ax.set_ylim(-1950, 1950)
    if heard:
        ax.legend(loc="lower left", fontsize=8.5)
    fs.style_map(ax, title="官方演练轨迹（星号为清除点）")
    fs.save(fig, os.path.join(OUT, "fig_q3_traj.png"))


def fig_time(data):
    fig, ax = fs.new_fig()
    times = data["summary"]["clear_virtual_times"]
    items = sorted((float(t), int(ch)) for ch, t in times.items())
    ts = [0.0] + [t for t, _ in items]
    ns = list(range(0, len(items) + 1))
    ax.step(ts, ns, where="post", color=fs.BLUE, lw=2.0)
    ax.axvline(119, color=fs.MUTED, ls="--", lw=1.0)
    ax.text(220, 1.15, "原点扫频结束", fontsize=9, color=fs.MUTED)
    ax.set_xlabel("虚拟时间 / s")
    ax.set_ylabel("已清除个数")
    ax.set_ylim(0, 16)
    fs.style_xy(ax, title="清除进度")
    fs.save(fig, os.path.join(OUT, "fig_q3_time.png"))
    parts = data["parts"]
    tot = sum(parts.values())
    print("parts", {k: round(v, 1) for k, v in parts.items()}, "T", data["T"])
    print("pct", {k: round(100 * v / tot, 1) for k, v in parts.items()})


def fig_mc():
    mc_path = os.path.join(HERE, "logs", "mc_local.json")
    if not os.path.isfile(mc_path):
        print("skip mc: missing", mc_path)
        return
    mc = json.loads(open(mc_path, encoding="utf-8").read())
    fig, ax = fs.new_fig()
    avgs = [r["avg_s"] for r in mc["rows"]]
    ax.hist(avgs, bins=8, color=fs.BLUE, edgecolor="white", alpha=0.92)
    ax.axvline(370.7, color=fs.ROSE, ls="--", lw=1.3, label="官方演练 371 s/个")
    ax.axvline(
        mc["mean_avg_s"], color=fs.INK, ls="-", lw=1.2,
        label=f"本地 {mc['n_trials']} 局均值 {mc['mean_avg_s']:.0f} s/个",
    )
    ax.set_xlabel("平均定位清除时间 / s")
    ax.set_ylabel("局数")
    ax.legend(loc="upper right")
    fs.style_xy(ax, title="本地对照：平均定位清除时间")
    fs.save(fig, os.path.join(OUT, "fig_q3_mc.png"))


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_coverage()
    data = parse_log(LOG)
    fig_traj(data)
    fig_time(data)
    kinds = Counter(k for _, _, k, _, _ in data["events"])
    print("events", dict(kinds))
    fig_mc()


if __name__ == "__main__":
    main()
