# -*- coding: utf-8 -*-
"""问题3 论文图：覆盖网、官方演练轨迹、时间账。"""
from __future__ import annotations

import json
import math
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
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
    events = []  # (x,y,kind,ch,t)
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
        "parts": {
            "移动": move_s,
            "检测": meas_s,
            "切频道": sw_s,
            "清除": clr_s,
        },
        "T": T,
        "n_meas": n_meas,
        "n_switch": n_switch,
    }


def fig_coverage(ax):
    th = np.linspace(0, 2 * np.pi, 360)
    ax.plot(ARENA_R * np.cos(th), ARENA_R * np.sin(th), color="#222", lw=1.4, label="目标圆域 1800 m")
    ax.add_patch(Circle((0, 0), COVER_R, fc="#1f77b4", ec="#1f77b4", alpha=0.08, lw=0))
    ax.plot(COVER_R * np.cos(th), COVER_R * np.sin(th), color="#1f77b4", lw=1.0, ls="--", alpha=0.8)
    rings = ring_pts()
    for i, p in enumerate(rings):
        ax.add_patch(Circle(p, COVER_R, fc="#ff7f0e", ec="#ff7f0e", alpha=0.07, lw=0))
        ax.plot(p[0], p[1], "s", color="#d35400", ms=7, zorder=5)
        ax.annotate(f"$A_{i}$", p, textcoords="offset points", xytext=(6, 6), fontsize=9, color="#d35400")
    ax.plot(0, 0, "o", color="#1f77b4", ms=8, zorder=6)
    ax.annotate("$O$", (0, 0), textcoords="offset points", xytext=(-16, -14), fontsize=11, color="#1f77b4")
    # 最差点：外缘角平分线
    ang = math.pi / 6
    w = ARENA_R * np.array([math.cos(ang), math.sin(ang)])
    a0 = rings[0]
    d = float(np.linalg.norm(w - a0))
    ax.plot(w[0], w[1], "*", color="#c0392b", ms=14, zorder=7)
    ax.plot([w[0], a0[0]], [w[1], a0[1]], color="#c0392b", lw=1.1, ls=":")
    ax.annotate(
        f"最差点\n{d:.1f} m $<$ 1000",
        w,
        textcoords="offset points",
        xytext=(12, 8),
        fontsize=8.5,
        color="#c0392b",
    )
    ax.set_aspect("equal")
    ax.set_xlim(-1950, 1950)
    ax.set_ylim(-1950, 1950)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("(a) 覆盖网：原点 + 1150 m 六等分点")
    ax.grid(True, ls=":", alpha=0.4)


def fig_traj(ax, data):
    th = np.linspace(0, 2 * np.pi, 360)
    ax.plot(ARENA_R * np.cos(th), ARENA_R * np.sin(th), color="#888", lw=1.0)
    for p in ring_pts():
        ax.plot(p[0], p[1], "s", color="#d35400", ms=6, zorder=4)
    ax.plot(0, 0, "o", color="#1f77b4", ms=7, zorder=5)
    xy = np.array([[x, y] for x, y, *_ in data["path_xy"]])
    ax.plot(xy[:, 0], xy[:, 1], color="#555", lw=0.9, alpha=0.75, zorder=2)
    clears = [(x, y, ch) for x, y, k, ch, t in data["events"] if k == "clear_success"]
    heard = [(x, y) for x, y, k, ch, t in data["events"] if k == "direction"]
    if heard:
        h = np.array(heard)
        ax.plot(h[:, 0], h[:, 1], "+", color="#2c7bb6", ms=7, zorder=3, label="测到示向度")
    for x, y, ch in clears:
        ax.plot(x, y, "*", color="#27ae60", ms=12, zorder=6)
        ax.annotate(str(ch), (x, y), textcoords="offset points", xytext=(4, 3), fontsize=8, color="#1e8449")
    ax.set_aspect("equal")
    ax.set_xlim(-1950, 1950)
    ax.set_ylim(-1950, 1950)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("(b) 演练轨迹（星号为清除点，数字为频道）")
    ax.grid(True, ls=":", alpha=0.4)


def fig_time(ax, data):
    times = data["summary"]["clear_virtual_times"]
    items = sorted((float(t), int(ch)) for ch, t in times.items())
    ts = [0.0] + [t for t, _ in items]
    ns = list(range(0, len(items) + 1))
    ax.step(ts, ns, where="post", color="#1f77b4", lw=2.0)
    ax.axvline(119, color="#888", ls="--", lw=1.0)
    ax.text(125, 0.4, "原点扫频结束 119 s", fontsize=8, color="#555")
    ax.set_xlabel("虚拟时间 / s")
    ax.set_ylabel("已清除个数")
    ax.set_title("(c) 清除进度与时间构成")
    ax.set_ylim(0, 16)
    ax.grid(True, ls=":", alpha=0.4)
    parts = data["parts"]
    tot = sum(parts.values())
    text = "\n".join(f"{k}  {v:.0f} s  ({100*v/tot:.1f}%)" for k, v in parts.items())
    text += f"\n合计  {data['T']:.0f} s，平均 {data['T']/data['summary']['n_cleared']:.0f} s/个"
    ax.text(
        0.98,
        0.05,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#aaa", alpha=0.92),
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    data = parse_log(LOG)
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.8), constrained_layout=True)
    fig_coverage(axes[0])
    fig_traj(axes[1], data)
    fig_time(axes[2], data)
    fig.suptitle("问题 3　覆盖网、演练轨迹与时间账（案例 WU4D-52HU-EUR9-Y4QR）", fontsize=13, fontweight="bold")
    png = os.path.join(OUT, "fig_q3_overview.png")
    pdf = os.path.join(OUT, "fig_q3_overview.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print("saved", png)
    print("parts", {k: round(v, 1) for k, v in data["parts"].items()}, "T", data["T"])
    kinds = Counter(k for _, _, k, _, _ in data["events"])
    print("events", dict(kinds))

    mc_path = os.path.join(HERE, "logs", "mc_local.json")
    if os.path.isfile(mc_path):
        mc = json.loads(open(mc_path, encoding="utf-8").read())
        fig2, ax = plt.subplots(figsize=(6.4, 4.0))
        avgs = [r["avg_s"] for r in mc["rows"]]
        ax.hist(avgs, bins=8, color="#4C78A8", edgecolor="white", alpha=0.9)
        ax.axvline(370.7, color="#c0392b", ls="--", lw=1.4, label="官方演练 371 s/个")
        ax.axvline(mc["mean_avg_s"], color="#222", ls="-", lw=1.2, label=f"本地 {mc['n_trials']} 局均值 {mc['mean_avg_s']:.0f} s/个")
        ax.set_xlabel("平均定位清除时间 / s")
        ax.set_ylabel("局数")
        ax.set_title("本地对照：平均定位清除时间（24 局全清）")
        ax.legend(fontsize=8.5, framealpha=0.92)
        ax.grid(True, ls=":", alpha=0.4, axis="y")
        png2 = os.path.join(OUT, "fig_q3_mc.png")
        fig2.savefig(png2, dpi=300, bbox_inches="tight")
        fig2.savefig(os.path.join(OUT, "fig_q3_mc.pdf"), bbox_inches="tight")
        print("saved", png2)


if __name__ == "__main__":
    main()
