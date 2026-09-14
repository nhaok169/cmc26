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
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Arc, Circle

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


def cover_d(r, n=6, R=ARENA_R):
    r = np.asarray(r, float)
    return np.sqrt(R ** 2 + r ** 2 - 2.0 * R * r * np.cos(np.pi / n))


def fig_coverage():
    """图18：(a) 覆盖网；(b) 最坏点余弦三角形 + d(r)–r。"""
    cosn = math.cos(math.pi / 6.0)
    bcoef = 2.0 * ARENA_R * cosn
    disc = bcoef ** 2 - 4.0 * (ARENA_R ** 2 - COVER_R ** 2)
    r_star = 0.5 * (bcoef - math.sqrt(disc))
    d_ring = float(cover_d(RING_R))
    print("coverage: r*={:.2f} d(1150)={:.2f}".format(r_star, d_ring))

    fig = plt.figure(figsize=(fs.W_IN, fs.H), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.18, 1.0], height_ratios=[1.05, 1.0])
    ax = fig.add_subplot(gs[:, 0])
    bx = fig.add_subplot(gs[0, 1])
    cx = fig.add_subplot(gs[1, 1])

    th = np.linspace(0, 2 * np.pi, 360)
    ax.plot(ARENA_R * np.cos(th), ARENA_R * np.sin(th), color=fs.INK, lw=1.2, label="目标圆 1800 m")
    ax.add_patch(Circle((0, 0), COVER_R, fc=fs.BLUE, ec="none", alpha=0.08, lw=0))
    ax.plot(COVER_R * np.cos(th), COVER_R * np.sin(th), color=fs.BLUE, lw=1.0, ls="--")
    rings = ring_pts()
    for i, p in enumerate(rings):
        ax.add_patch(Circle(p, COVER_R, fc=fs.SAND, ec="none", alpha=0.07, lw=0))
        ax.plot(p[0], p[1], "s", color=fs.SAND, ms=7, zorder=5)
        ax.annotate(f"$A_{i}$", p, textcoords="offset points", xytext=(7, 7), fontsize=8, color=fs.SAND)
    ax.plot(0, 0, "o", color=fs.BLUE, ms=8, zorder=6)
    ax.annotate("$O$", (0, 0), textcoords="offset points", xytext=(-16, -14), fontsize=10, color=fs.BLUE)
    ang = math.pi / 6
    w = ARENA_R * np.array([math.cos(ang), math.sin(ang)])
    a0 = rings[0]
    ax.plot(w[0], w[1], "*", color=fs.ROSE, ms=11, zorder=7)
    ax.plot([w[0], a0[0]], [w[1], a0[1]], color=fs.ROSE, lw=1.0, ls=":")
    ax.annotate("最差点", w, textcoords="offset points", xytext=(12, 8), fontsize=8, color=fs.ROSE)
    ax.set_xlim(-1950, 1950)
    ax.set_ylim(-1950, 1950)
    ax.legend(loc="lower left", fontsize=7.5)
    fs.style_map(ax, title=r"(a) 原点 + $1150$ m 六等分点")

    # (b) 上：O–A0–W 余弦三角形。作图用 r=1150，OA 边标 r
    O = np.array([0.0, 0.0])
    A0 = np.array([RING_R, 0.0])
    Wpt = ARENA_R * np.array([math.cos(ang), math.sin(ang)])
    bx.plot([O[0], A0[0]], [O[1], A0[1]], color=fs.INK, lw=1.35, zorder=3)
    bx.plot([O[0], Wpt[0]], [O[1], Wpt[1]], color=fs.INK, lw=1.35, zorder=3)
    bx.plot([A0[0], Wpt[0]], [A0[1], Wpt[1]], color=fs.ROSE, lw=1.5, zorder=4)
    bx.plot(*O, "o", color=fs.BLUE, ms=6, zorder=6)
    bx.plot(*A0, "s", color=fs.SAND, ms=6, zorder=6)
    bx.plot(*Wpt, "*", color=fs.ROSE, ms=10, zorder=6)
    bx.annotate(r"$O$", O, textcoords="offset points", xytext=(-12, -13), fontsize=8, color=fs.BLUE)
    bx.annotate(r"$A_0$", A0, textcoords="offset points", xytext=(-4, -14), fontsize=8, color=fs.SAND)
    bx.annotate(r"$W$", Wpt, textcoords="offset points", xytext=(4, 4), fontsize=8, color=fs.ROSE)
    bx.annotate(r"$r$", (0.52 * RING_R, -95), fontsize=8, ha="center", color=fs.INK)
    mid_ow = 0.5 * (O + Wpt) + np.array([-70, 40])
    bx.annotate(r"$1800$", mid_ow, fontsize=8, color=fs.INK, ha="center")
    mid_aw = 0.5 * (A0 + Wpt) + np.array([55, -8])
    bx.annotate(r"$d(r)$", mid_aw, fontsize=8, color=fs.ROSE, ha="left")
    bx.add_patch(Arc(O, 380, 380, angle=0, theta1=0, theta2=30, color=fs.ROSE, lw=1.15, zorder=5))
    bx.annotate(r"$30^\circ$", (230, 48), fontsize=8, color=fs.ROSE)
    bx.set_xlim(-220, 1780)
    bx.set_ylim(-280, 1180)
    fs.style_map(bx, title=r"(b) 最坏点余弦三角形")

    rs = np.linspace(500.0, 1400.0, 240)
    ds = cover_d(rs)
    cx.plot(rs, ds, color=fs.BLUE, lw=1.55, zorder=3)
    cx.axhline(COVER_R, color=fs.MUTED, ls="--", lw=1.05, zorder=2)
    cx.annotate(r"$d=1000$", (520, COVER_R), textcoords="offset points",
                xytext=(4, 5), fontsize=7, color=fs.MUTED)
    cx.axvline(r_star, color=fs.ROSE, ls=":", lw=0.9, zorder=2)
    cx.plot(r_star, COVER_R, "o", color=fs.ROSE, ms=5.5, zorder=5)
    cx.annotate(
        r"$r^*\approx 1123$", (r_star, COVER_R),
        textcoords="offset points", xytext=(-52, 10), fontsize=7.5, color=fs.ROSE,
    )
    cx.plot(RING_R, d_ring, "s", color=fs.SAND, ms=5.5, zorder=5)
    cx.annotate(
        r"$r=1150$，$d=988.5$",
        (RING_R, d_ring),
        textcoords="offset points", xytext=(10, -16), fontsize=7.5, color=fs.SAND,
    )
    cx.set_xlim(500, 1480)
    cx.set_ylim(860, 1450)
    cx.set_xlabel(r"$r$ / m")
    cx.set_ylabel(r"$d(r)$ / m")
    fs.style_xy(cx, title=r"$d(r)$–$r$（$n=6$）")
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
    for i, (x, y, ch) in enumerate(clears):
        ax.plot(x, y, "*", color=fs.GREEN, ms=11, zorder=6)
        dx, dy = (6, 5) if (i % 2 == 0) else (6, -12)
        if x > 900:
            dx = -16
        ax.annotate(str(ch), (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=7.5, color=fs.GREEN)
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
