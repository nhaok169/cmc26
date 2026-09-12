# -*- coding: utf-8 -*-
"""问题2 设计图：L2' 直径热力图、成功条件下的 E[T]–赌注，分两张输出。"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "paper"))
from geometry import DELTA, k_expr, expected_time  # noqa: E402
import figstyle as fs  # noqa: E402

fs.apply()

K = 40.0 / (2 * DELTA)
D_NEAR, D_FAR = 5.0, 1500.0
EPS5 = np.linspace(-DELTA, DELTA, 5)
c_lever = 0.5 / np.sqrt(1 - 0.25)


def lever_h(t):
    return c_lever * max(D_FAR - t, t - D_NEAR)


ts = np.arange(150, 1451, 15.0)
Ds = np.linspace(5, 1500, 145)
Z = np.zeros((len(Ds), len(ts)))
for i, D in enumerate(Ds):
    for j, t in enumerate(ts):
        h = lever_h(t)
        Z[i, j] = 2 * DELTA * np.sqrt(min(k_expr(t, h, D, e) for e in EPS5))
Zc = np.clip(Z, 0, 150)

SCHOOLS = [
    (550, 498, "minimax", fs.ROSE, "o"),
    (675, 476, "概率最优", fs.PURPLE, "s"),
    (750, 450, "保守", fs.GREEN, "D"),
    (500, 250, "时间最优", fs.SAND, "^"),
]


def interval(t, h):
    m = np.array([max(k_expr(t, h, D, e) for e in EPS5) <= K * K for D in Ds])
    idx = np.where(m)[0]
    if len(idx) == 0:
        return None
    return Ds[idx[0]], Ds[idx[-1]]


OUT = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT, exist_ok=True)


def fig_heatmap():
    fig, ax = fs.new_fig()
    pm = ax.pcolormesh(ts, Ds, Zc, cmap=fs.CMAP_DIAM, shading="auto", vmin=0, vmax=150)
    cs = ax.contour(ts, Ds, Z, levels=[40], colors=fs.ROSE, linewidths=1.6)
    ax.clabel(cs, fmt="40 m", fontsize=9)
    for t, h, name, col, mk in SCHOOLS:
        iv = interval(t, h)
        if iv:
            ax.plot([t, t], [iv[0], iv[1]], color=col, lw=2.2, alpha=0.9, zorder=4)
            ax.plot(t, 0.5 * (iv[0] + iv[1]), mk, color=col, ms=7, zorder=5, label=name)
    ax.set_xlabel("赌注 $t$ / m（侧偏取杠杆下界）")
    ax.set_ylabel("真实距离 $D$ / m")
    ax.legend(loc="upper right", fontsize=8.5)
    fs.style_xy(ax, title="L2′ 解析直径（筛选层）")
    fig.colorbar(pm, ax=ax, fraction=0.046, pad=0.04, label="直径 / m（显示截断于 150）")
    fs.save(fig, os.path.join(OUT, "fig_q2_heatmap.png"))


def fig_et():
    fig, ax = fs.new_fig()
    rho_ad = lambda L: max(50.0, 0.5 * L)
    w = Ds.copy()
    w /= w.sum()
    tt = np.arange(350, 1001, 50.0)
    et_time, et_cons = [], []
    for t in tt:
        et1, _, _ = expected_time(t, 250.0, Ds, w=w, rho_rule=rho_ad)
        et2, _, _ = expected_time(t, lever_h(t), Ds, w=w, rho_rule=rho_ad)
        et_time.append(et1)
        et_cons.append(et2)
    ax.plot(tt, et_time, "o-", color=fs.SAND, lw=1.8, ms=5.5, label="时间流派（$h=250$）")
    ax.plot(tt, et_cons, "s-", color=fs.GREEN, lw=1.8, ms=5.5, label="保守流派（杠杆下界）")
    ax.set_xlabel("赌注 $t$ / m")
    ax.set_ylabel("成功条件下的期望总时间 $E[T]$ / s")
    ax.legend(loc="upper left")
    fs.style_xy(ax, title="序贯仿真：成功清除的期望时间")
    fs.save(fig, os.path.join(OUT, "fig_q2_et.png"))
    print("E[T]|success time school  min at t=", tt[int(np.argmin(et_time))], "value", min(et_time))
    print("E[T]|success conserv school min at t=", tt[int(np.argmin(et_cons))], "value", min(et_cons))
    for t, h, name, *_ in SCHOOLS:
        et, p, en = expected_time(t, h, Ds, w=w, rho_rule=rho_ad)
        print(f"school {name}: E[T]|ok={et:.1f}s  P(ok)={p:.3f}  E[n]|ok={en:.2f}  interval={interval(t,h)}")


if __name__ == "__main__":
    fig_heatmap()
    fig_et()
