# -*- coding: utf-8 -*-
"""问题二推导图：雅可比映射、不等距平行四边形、2δD 下界、杠杆蝶形、四准则网格。"""
from __future__ import annotations

import os
import sys

import matplotlib
import numpy as np
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Polygon, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper"))
from geometry import DELTA, diam_and_axis, expected_time, k_expr, mec, region, wedge_hp  # noqa: E402
import figstyle as sty  # noqa: E402

sty.apply()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

K = 40.0 / (2.0 * np.tan(DELTA))
C_LEVER = 0.5 / np.sqrt(0.75)
D_NEAR, D_FAR = 5.0, 1500.0
EPS3 = np.array([-DELTA, 0.0, DELTA])
SCHOOLS = [
    (550.0, 450.0, "minimax", sty.ROSE, "o"),
    (675.0, 500.0, "概率", sty.PURPLE, "s"),
    (750.0, 450.0, "保守", sty.GREEN, "D"),
    (500.0, 250.0, "时间", sty.SAND, "^"),
]


def save(fig, name, size=None):
    sty.save(fig, os.path.join(OUT, name), size=size)


def unit(ang):
    return np.array([np.cos(ang), np.sin(ang)], float)


def two_station_poly(S1, S2, G, delta=DELTA):
    th1 = np.arctan2(G[1] - S1[1], G[0] - S1[0])
    th2 = np.arctan2(G[1] - S2[1], G[0] - S2[0])
    return region(wedge_hp(S1, th1) + wedge_hp(S2, th2)) if abs(delta - DELTA) < 1e-15 else region(
        [
            (S1, np.array([np.cos(th1 - delta), np.sin(th1 - delta)]), +1),
            (S1, np.array([np.cos(th1 + delta), np.sin(th1 + delta)]), -1),
            (S2, np.array([np.cos(th2 - delta), np.sin(th2 - delta)]), +1),
            (S2, np.array([np.cos(th2 + delta), np.sin(th2 + delta)]), -1),
        ]
    )


def l2_para(S1, S2, G, delta=DELTA):
    d1 = np.linalg.norm(G - S1)
    d2 = np.linalg.norm(G - S2)
    u1 = (G - S1) / d1
    u2 = (G - S2) / d2
    n1 = np.array([-u1[1], u1[0]])
    n2 = np.array([-u2[1], u2[0]])
    J = np.vstack([n1 / d1, n2 / d2])
    Jinv = np.linalg.inv(J)
    a, b = Jinv[:, 0], Jinv[:, 1]
    verts = np.array(
        [
            G + delta * (a + b),
            G + delta * (a - b),
            G + delta * (-a - b),
            G + delta * (-a + b),
        ],
        float,
    )
    c = verts.mean(axis=0)
    ang = np.arctan2(verts[:, 1] - c[1], verts[:, 0] - c[0])
    return verts[np.argsort(ang)], a, b


def fill_poly(ax, poly, fc, ec, alpha=0.35, lw=1.4, z=4, label=None, ls="-"):
    if len(poly) < 3:
        return
    closed = np.vstack([poly, poly[0]])
    ax.fill(closed[:, 0], closed[:, 1], fc=fc, ec="none", alpha=alpha, lw=0, zorder=z, label=None)
    ax.plot(closed[:, 0], closed[:, 1], color=ec, lw=lw, ls=ls, zorder=z + 1, label=label)


def dloc_closed(d1, d2, gamma, delta=DELTA):
    s = np.sin(gamma)
    c = np.abs(np.cos(gamma))
    return 2.0 * delta * np.sqrt((d1 * d1 + d2 * d2 + 2.0 * d1 * d2 * c) / (s * s))


def para_aspect(a, b):
    """平行四边形长宽比：较长边 / 对应高。"""
    la, lb = np.linalg.norm(a), np.linalg.norm(b)
    sinphi = abs(float(a[0] * b[1] - a[1] * b[0])) / (la * lb + 1e-15)
    w, h = la, lb * sinphi
    return max(w, h) / max(min(w, h), 1e-9)


def station_pair(d1, d2, gamma):
    """S1 在原点、G 在 (d1,0)、∠S1GS2=γ，||G-S2||=d2。"""
    S1 = np.array([0.0, 0.0])
    G = np.array([float(d1), 0.0])
    S2 = np.array([d1 - d2 * np.cos(gamma), d2 * np.sin(gamma)])
    return S1, S2, G


def fig_jacobian():
    """(a) 站址；(b) 误差盒；(c) J^{-1} 把方块映成平行四边形。"""
    d1, d2, gdeg = 800.0, 400.0, 90.0
    gamma = np.deg2rad(gdeg)
    S1, S2, G = station_pair(d1, d2, gamma)
    verts, a, b = l2_para(S1, S2, G, DELTA)
    asp = para_aspect(DELTA * a, DELTA * b)
    if asp > 4.0:
        raise RuntimeError("parallelogram aspect {:.2f} > 4; adjust d1,d2,gamma".format(asp))

    fig, axes = sty.new_fig(1, 3, gridspec_kw={"width_ratios": [1.12, 0.88, 1.22]})
    ax, bx, cx = axes

    ax.plot([S1[0], G[0]], [S1[1], G[1]], color=sty.INK, lw=1.35, zorder=3)
    ax.plot([S2[0], G[0]], [S2[1], G[1]], color=sty.INK, lw=1.35, zorder=3)
    ax.plot([S1[0], S2[0]], [S1[1], S2[1]], color=sty.MUTED, lw=0.85, ls="--", zorder=2)
    ax.plot(*S1, "o", color=sty.INK, ms=6, zorder=6)
    ax.plot(*S2, "o", color=sty.BLUE, ms=6, zorder=6)
    ax.plot(*G, "o", color=sty.INK, ms=4, zorder=6)
    ax.annotate(r"$S_1$", S1, textcoords="offset points", xytext=(-16, -14), fontsize=8)
    ax.annotate(r"$S_2$", S2, textcoords="offset points", xytext=(4, 6), fontsize=8, color=sty.BLUE)
    ax.annotate(r"$G$", G, textcoords="offset points", xytext=(6, -13), fontsize=8)
    ax.annotate(r"$d_1$", (0.52 * d1, -48), fontsize=8, ha="center", color=sty.MUTED)
    mid2 = 0.5 * (S2 + G) + np.array([18, 10])
    ax.annotate(r"$d_2$", mid2, fontsize=8, color=sty.MUTED)
    t1 = np.degrees(np.arctan2(S1[1] - G[1], S1[0] - G[0]))
    t2 = np.degrees(np.arctan2(S2[1] - G[1], S2[0] - G[0]))
    dccw = (t2 - t1) % 360.0
    if dccw <= 180.0:
        a1, a2 = t1, t1 + dccw
    else:
        a1, a2 = t2, t2 + ((t1 - t2) % 360.0)
    ax.add_patch(Arc(G, 160, 160, angle=0, theta1=a1, theta2=a2, color=sty.ROSE, lw=1.35, zorder=5))
    mid = np.radians(0.5 * (a1 + a2))
    ax.annotate(r"$\gamma=50^\circ$", G + 118 * unit(mid), fontsize=8, color=sty.ROSE, ha="center")
    ax.set_xlim(-90, 900)
    ax.set_ylim(-160, 420)
    ax.set_aspect("equal", adjustable="box")
    sty.style_map(ax, title=r"(a) 站址几何", xlabel="x / m", ylabel="y / m")

    dlt = 1.0
    bx.add_patch(Rectangle((-dlt, -dlt), 2 * dlt, 2 * dlt, fc=sty.FILL_IO, ec=sty.BLUE, lw=1.5, zorder=3))
    corners = [(-dlt, dlt), (dlt, dlt), (-dlt, -dlt), (dlt, -dlt)]
    labs = [r"$(-\delta,\delta)$", r"$(\delta,\delta)$", r"$(-\delta,-\delta)$", r"$(\delta,-\delta)$"]
    offs = [(-3, 5), (3, 5), (-3, -13), (3, -13)]
    has = ["right", "left", "right", "left"]
    for (x, y), lab, off, ha in zip(corners, labs, offs, has):
        bx.plot(x, y, "o", color=sty.BLUE, ms=4.5, zorder=5)
        bx.annotate(lab, (x, y), textcoords="offset points", xytext=off, fontsize=6.5, ha=ha, color=sty.INK)
    bx.axhline(0, color=sty.GRID, lw=0.7)
    bx.axvline(0, color=sty.GRID, lw=0.7)
    bx.set_xlim(-1.95, 1.95)
    bx.set_ylim(-1.95, 1.95)
    bx.set_xlabel(r"$\delta\theta_1$ / °")
    bx.set_ylabel(r"$\delta\theta_2$ / °")
    bx.set_aspect("equal")
    sty.style_xy(bx, title=r"(b) 误差盒 $[-\delta,\delta]^2$")

    da, db = DELTA * a, DELTA * b
    fill_poly(cx, verts, sty.FILL_IO, sty.ROSE, 0.35, 1.45, 3)
    named = [
        (G + da + db, r"$G+\delta a+\delta b$"),
        (G + da - db, r"$G+\delta a-\delta b$"),
        (G - da + db, r"$G-\delta a+\delta b$"),
        (G - da - db, r"$G-\delta a-\delta b$"),
    ]
    for p, lab in named:
        cx.plot(*p, "o", color=sty.ROSE, ms=3.5, zorder=6)
    # 按最左/最右/最上/最下分开，避免邻角公式挤在一起
    by_x = sorted(named, key=lambda t: t[0][0])
    by_y = sorted(named, key=lambda t: t[0][1])
    place = {
        id(by_x[0][0]): (by_x[0][0], by_x[0][1], (-10, 2), "right", "center"),
        id(by_x[-1][0]): (by_x[-1][0], by_x[-1][1], (10, 2), "left", "center"),
        id(by_y[-1][0]): (by_y[-1][0], by_y[-1][1], (0, 11), "center", "bottom"),
        id(by_y[0][0]): (by_y[0][0], by_y[0][1], (0, -12), "center", "top"),
    }
    for p, lab in named:
        rec = place.get(id(p))
        if rec is None:
            rec = (p, lab, (8, 8), "left", "bottom")
        _, _, off, ha, va = rec
        cx.annotate(lab, p, textcoords="offset points", xytext=off, fontsize=6.4,
                    ha=ha, va=va, color=sty.INK, zorder=8)
    cx.annotate("", xy=G + da, xytext=G, arrowprops=dict(arrowstyle="-|>", color=sty.SAND, lw=1.25, mutation_scale=8))
    cx.annotate("", xy=G + db, xytext=G, arrowprops=dict(arrowstyle="-|>", color=sty.GREEN, lw=1.25, mutation_scale=8))
    cx.text(*(G + 0.42 * da + np.array([-6.0, 4.0])), r"$\delta a$", fontsize=8, color=sty.SAND, ha="center")
    cx.text(*(G + 0.48 * db + np.array([3.2, 3.2])), r"$\delta b$", fontsize=8, color=sty.GREEN, ha="center")
    dsum, ddif = da + db, da - db
    cx.plot(
        [G[0] - dsum[0], G[0] + dsum[0]], [G[1] - dsum[1], G[1] + dsum[1]],
        color=sty.PURPLE, lw=1.45, zorder=4, label=r"$2\delta\|a+b\|$",
    )
    cx.plot(
        [G[0] - ddif[0], G[0] + ddif[0]], [G[1] - ddif[1], G[1] + ddif[1]],
        color=sty.BLUE, lw=1.2, ls="--", zorder=4, label=r"$2\delta\|a-b\|$",
    )
    cx.plot(*G, "o", color=sty.INK, ms=4.5, zorder=7)
    cx.annotate(r"$G$", G, textcoords="offset points", xytext=(-8, 6), fontsize=8, ha="right", va="bottom")
    cx.legend(loc="lower left", fontsize=6.3, framealpha=0.95, borderpad=0.25)
    pad = 2.45 * max(np.linalg.norm(dsum), np.linalg.norm(ddif), np.linalg.norm(da) + np.linalg.norm(db))
    cx.set_xlim(G[0] - pad, G[0] + pad)
    cx.set_ylim(G[1] - pad, G[1] + pad)
    cx.set_aspect("equal")
    sty.style_map(cx, title=r"(c) $J^{-1}$ 的像", xlabel="x / m", ylabel="y / m")
    print("jacobian: d1={} d2={} g={} aspect={:.2f} |a|={:.1f} |b|={:.1f}".format(
        d1, d2, gdeg, asp, np.linalg.norm(a), np.linalg.norm(b)))
    save(fig, "fig_q2_jacobian.png")


def fig_para_vs_poly():
    """(a) 精确多边形 vs 一阶平行四边形；(b) 相对差随 γ。"""
    d1, d2, gdeg = 800.0, 280.0, 38.0
    gamma = np.deg2rad(gdeg)
    S1, S2, G = station_pair(d1, d2, gamma)
    dlt = np.deg2rad(5.0)
    vpoly = two_station_poly(S1, S2, G, dlt)
    vpara, a, b = l2_para(S1, S2, G, dlt)
    dpoly, _ = diam_and_axis(vpoly)
    dpara = 2 * dlt * max(np.linalg.norm(a + b), np.linalg.norm(a - b))
    rel5 = abs(dpara - dpoly) / dpoly * 100

    fig, axes = sty.new_fig(1, 2)
    ax, bx = axes
    fill_poly(ax, vpoly, sty.FILL_IO, sty.BLUE, 0.40, 1.7, 4, label="交会多边形（精确）")
    closed = np.vstack([vpara, vpara[0]])
    ax.plot(closed[:, 0], closed[:, 1], color=sty.ROSE, lw=1.8, ls="--", zorder=6, label="一阶平行四边形")
    ax.plot(*G, "o", color=sty.INK, ms=4, zorder=7)
    ax.annotate(r"$G$", G, textcoords="offset points", xytext=(8, -14), fontsize=9)
    ax.legend(loc="upper right", fontsize=7.5)
    ax.text(
        0.03, 0.04,
        rf"$d_1=800$ m，$d_2=280$ m，$\gamma=38^\circ$" + "\n"
        + rf"$\delta=5^\circ$：多边形 $d={dpoly:.1f}$ m" + "\n"
        + rf"平行四边形 $d={dpara:.1f}$ m",
        transform=ax.transAxes, fontsize=7.5, va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2.0),
    )
    xmin = min(vpoly[:, 0].min(), vpara[:, 0].min()) - 10
    xmax = max(vpoly[:, 0].max(), vpara[:, 0].max()) + 10
    ymin = min(vpoly[:, 1].min(), vpara[:, 1].min()) - 22
    ymax = max(vpoly[:, 1].max(), vpara[:, 1].max()) + 10
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    sty.style_map(ax, title=r"(a) 精确多边形 vs 一阶平行四边形")

    gams = np.linspace(25.0, 155.0, 131)
    # 等距时相对差几乎不随距离变；只画最坏 $d_1=d_2=500$ m，避免三条曲线叠成一条。
    ys500 = []
    for gd in gams:
        g = np.deg2rad(gd)
        P1, P2, GG = station_pair(500.0, 500.0, g)
        poly = two_station_poly(P1, P2, GG, DELTA)
        dp, _ = diam_and_axis(poly)
        dl = dloc_closed(500.0, 500.0, g, DELTA)
        ys500.append(abs(dl - dp) / dp * 100.0)
    ys500 = np.asarray(ys500)
    bx.plot(gams, ys500, color=sty.GREEN, lw=1.7, label=r"$d_1=d_2=500$ m（最坏）")
    i30 = int(np.argmin(np.abs(gams - 30.0)))
    i25 = int(np.argmin(np.abs(gams - 25.0)))
    bx.axhline(0.43, color=sty.MUTED, ls="--", lw=1.05, label=r"工作角域上界 $0.43\%$")
    bx.plot(30.0, ys500[i30], "o", color=sty.INK, ms=5, zorder=5)
    bx.annotate(
        rf"$30^\circ$ 处 ${ys500[i30]:.2f}\%$",
        (30.0, ys500[i30]), textcoords="offset points", xytext=(8, 8), fontsize=7.5,
    )
    bx.plot(25.0, ys500[i25], "s", color=sty.INK, ms=5, zorder=5)
    bx.annotate(
        rf"$25^\circ$ 处 ${ys500[i25]:.2f}\%$",
        (25.0, ys500[i25]), textcoords="offset points", xytext=(10, -2), fontsize=7.5,
    )
    bx.set_xlim(25, 155)
    bx.set_ylim(0.0, 0.72)
    bx.set_xlabel(r"交会角 $\gamma$ / °")
    bx.set_ylabel(r"$|D_{\mathrm{loc}}-d|/d$ / %")
    bx.legend(loc="upper right", fontsize=7)
    sty.style_xy(bx, title=r"(b) 相对差随 $\gamma$（$\delta=1^\circ$，等距）")
    rel_at = {500.0: (float(ys500[i30]), float(ys500[i25]))}
    print(f"para vs poly: d1=800 d2={d2:.1f} g=38 Dpoly={dpoly:.3f} Dloc={dpara:.3f} rel5={rel5:.3f}%")
    print("rel@30/25", rel_at)
    save(fig, "fig_q2_para.png")


def fig_bound():
    """等距只为隔离 γ；绝对下界 inf D_loc=2δD。"""
    d1 = 800.0
    fig, axes = sty.new_fig(1, 2)
    ax, bx = axes

    G = np.array([d1, 0.0])
    S1 = np.array([0.0, 0.0])
    d_draw = np.deg2rad(8.0)
    w = d1 * np.tan(d_draw)
    ax.plot([S1[0], G[0] + 80], [0, 0], color=sty.INK, lw=1.1, zorder=3)
    ax.plot([S1[0], G[0] + 80], [0, (G[0] + 80) * np.tan(d_draw)], color=sty.BLUE, lw=1.0, ls="--")
    ax.plot([S1[0], G[0] + 80], [0, -(G[0] + 80) * np.tan(d_draw)], color=sty.BLUE, lw=1.0, ls="--")
    ax.fill_betweenx([-w, w], G[0] - 22, G[0] + 22, color=sty.FILL_IO, alpha=0.85, zorder=2)
    ax.annotate(
        "", xy=(G[0] + 30, w), xytext=(G[0] + 30, -w),
        arrowprops=dict(arrowstyle="<->", color=sty.BLUE, lw=1.15),
    )
    ax.text(G[0] + 48, 0, r"$2D\tan\delta$", fontsize=9, color=sty.BLUE, va="center")
    S2 = G + np.array([0.0, 95.0])
    ax.plot(*S1, "o", color=sty.INK, ms=7)
    ax.plot(*S2, "o", color=sty.BLUE, ms=7)
    ax.plot(*G, "o", color=sty.INK, ms=4)
    ax.plot([S2[0], G[0]], [S2[1], G[1]], color=sty.BLUE, lw=1.1)
    ax.annotate(r"$S_1$", S1, textcoords="offset points", xytext=(8, 10), fontsize=10)
    ax.annotate(r"$S_2$", S2, textcoords="offset points", xytext=(8, 6), fontsize=10, color=sty.BLUE)
    ax.annotate(r"$G$", G, textcoords="offset points", xytext=(10, -18), fontsize=10)
    ax.text(0.04, 0.96, r"张角放大至 $\pm 8^\circ$；真实半宽仅 $D\tan 1^\circ\approx 14$ m",
            transform=ax.transAxes, fontsize=8, va="top")
    ax.set_xlim(-40, 980)
    ax.set_ylim(-160, 200)
    ax.set_aspect("equal")
    sty.style_xy(ax, title=r"(a) 第一锥在 $G$ 处的横向宽度削不掉")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")

    d2s = np.linspace(40.0, 1200.0, 80)
    Dloc = 2 * DELTA * np.sqrt(d1 ** 2 + d2s ** 2)  # gamma=90
    bx.plot(d2s, Dloc, color=sty.BLUE, lw=1.9, label=r"$\gamma=90^\circ$，$D_{\mathrm{loc}}=2\delta\sqrt{d_1^2+d_2^2}$")
    bx.axhline(2 * DELTA * d1, color=sty.ROSE, ls="--", lw=1.2, label=rf"下界 $2\delta D={2*DELTA*d1:.1f}$ m")
    bx.axhline(2 * np.sqrt(2) * DELTA * d1, color=sty.GREEN, ls=":", lw=1.3, label=rf"等距 $2\sqrt{{2}}\delta D={2*np.sqrt(2)*DELTA*d1:.1f}$ m")
    bx.plot(800.0, 2 * np.sqrt(2) * DELTA * d1, "s", color=sty.GREEN, ms=7, zorder=5)
    bx.annotate(r"$d_2=d_1=800$", (800, 2 * np.sqrt(2) * DELTA * d1), textcoords="offset points",
                xytext=(10, 8), fontsize=8, color=sty.GREEN)
    bx.axhline(40, color=sty.MUTED, lw=0.8, ls="-.")
    bx.text(50, 43, "40 m", fontsize=8, color=sty.MUTED)
    bx.set_xlim(0, 1200)
    bx.set_ylim(20, 70)
    bx.set_xlabel(r"$d_2=\|G-S_2\|$ / m")
    bx.set_ylabel(r"$D_{\mathrm{loc}}$ / m")
    bx.legend(loc="upper left", fontsize=7.5)
    sty.style_xy(bx, title=r"(b) 固定 $d_1=800$ m、$\gamma=90^\circ$")
    save(fig, "fig_q2_bound.png")


def fig_lever():
    """(a) 固定 D 禁止区；(b) 对顶锥族与包络；(c) 蝶形斜带。"""
    ts = np.linspace(0, 1500, 400)
    env = C_LEVER * np.maximum(np.abs(D_FAR - ts), np.abs(ts - D_NEAR))
    fig, axes = sty.new_fig(1, 3, gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]})
    ax, bx, cx = axes

    D = 800.0
    tline = np.linspace(-80, 1600, 260)
    hcut = C_LEVER * np.abs(D - tline)
    ax.fill_between(tline, -hcut, hcut, color=sty.FILL_BAD, alpha=0.72, zorder=1, label=r"禁止区 $|h|<|D-t|/\sqrt{3}$")
    ax.plot(tline, hcut, color=sty.ROSE, lw=1.25)
    ax.plot(tline, -hcut, color=sty.ROSE, lw=1.25)
    ax.add_patch(Arc((D, 0), 420, 420, angle=0, theta1=0, theta2=30, color=sty.INK, lw=1.2, zorder=5))
    ax.annotate(r"$30^\circ$", (D + 250, 58), fontsize=8, color=sty.INK)
    ax.plot(D, 0, "o", color=sty.INK, ms=5, zorder=6)
    ax.annotate(r"$(800,0)$", (D, 0), textcoords="offset points", xytext=(-36, -16), fontsize=7.5)
    ax.set_xlim(-40, 1480)
    ax.set_ylim(-720, 720)
    ax.set_xlabel(r"$t$ / m")
    ax.set_ylabel(r"$h$ / m")
    ax.legend(loc="upper left", fontsize=6.5)
    sty.style_xy(ax, title=r"(a) 固定 $D=800$ m")

    t2 = np.linspace(0, 1500, 240)
    fam = [(300.0, sty.MUTED, 1.0, ":"), (800.0, sty.GREEN, 1.15, "-"), (1300.0, sty.SAND, 1.0, "--")]
    for Dv, col, lw, ls in fam:
        hc = C_LEVER * np.abs(Dv - t2)
        bx.plot(t2, hc, color=col, lw=lw, ls=ls, label=rf"$D={Dv:.0f}$")
        bx.plot(t2, -hc, color=col, lw=lw, ls=ls)
    bx.plot(ts, env, color=sty.BLUE, lw=2.15, zorder=4, label=r"包络")
    bx.plot(ts, -env, color=sty.BLUE, lw=2.15, zorder=4)
    bx.set_xlim(0, 1500)
    bx.set_ylim(-950, 1050)
    bx.set_xlabel(r"$t$ / m")
    bx.set_ylabel(r"$h$ / m")
    bx.legend(loc="upper right", fontsize=6.3)
    sty.style_xy(bx, title=r"(b) 对顶锥族与包络")

    cx.fill_between(ts, env, 1100, color=sty.FILL_IO, alpha=0.78, zorder=1)
    cx.fill_between(ts, -1100, -env, color=sty.FILL_IO, alpha=0.78, zorder=1)
    cx.plot(ts, env, color=sty.BLUE, lw=1.55, zorder=3)
    cx.plot(ts, -env, color=sty.BLUE, lw=1.55, zorder=3)
    cx.plot(500, 250, "^", color=sty.SAND, ms=8, zorder=6)
    cx.annotate(
        r"$(500,250)$ 带外", (500, 250),
        textcoords="offset points", xytext=(10, -16), fontsize=7.5, color=sty.SAND,
    )
    cx.set_xlim(0, 1500)
    cx.set_ylim(-1000, 1100)
    cx.set_xlabel(r"$t$ / m")
    cx.set_ylabel(r"$h$ / m")
    sty.style_xy(cx, title=r"(c) 蝶形斜带 $|h|\geq$ 包络")
    save(fig, "fig_q2_lever.png")


def fig_schools():
    """四准则在 (t,h) 网格上怎么取点。"""
    ts = np.arange(250.0, 1001.0, 25.0)
    hs = np.arange(150.0, 701.0, 25.0)
    Ds = np.linspace(5.0, 1500.0, 50)
    w = Ds.copy()
    w /= w.sum()
    T, H = np.meshgrid(ts, hs)
    P = np.zeros_like(T)
    Pref = np.zeros_like(T)
    for i, h in enumerate(hs):
        for j, t in enumerate(ts):
            mask = np.array([max(k_expr(t, h, D, e) for e in EPS3) <= K * K for D in Ds])
            P[i, j] = float(w[mask].sum()) if mask.any() else 0.0
            if mask[0]:
                k = int(np.argmax(~np.append(mask, False))) if (~mask).any() else len(mask)
                # prefix length
                k = 0
                while k < len(mask) and mask[k]:
                    k += 1
                Pref[i, j] = Ds[k - 1] if k else 0.0

    fig, axes = sty.new_fig(1, 2)
    ax, bx = axes
    pm = ax.pcolormesh(ts, hs, P * 100, cmap=sty.CMAP_DIAM, shading="auto", vmin=0, vmax=30)
    cs = ax.contour(ts, hs, P * 100, levels=[20, 25, 27], colors=sty.INK, linewidths=0.6)
    ax.clabel(cs, fmt="%.0f%%", fontsize=7)
    # lever
    tb = np.linspace(250, 1000, 80)
    hb = C_LEVER * np.maximum(np.abs(D_FAR - tb), np.abs(tb - D_NEAR))
    ax.plot(tb, hb, color=sty.GREEN, lw=1.3, ls="--", label="杠杆下界")
    for t, h, name, col, mk in SCHOOLS:
        ax.plot(t, h, mk, color=col, ms=8, zorder=6, label=name)
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_xlabel(r"$t$ / m")
    ax.set_ylabel(r"$h$ / m")
    fig.colorbar(pm, ax=ax, fraction=0.046, pad=0.03, label="两步成功概率 / %（面积先验）")
    sty.style_xy(ax, title=r"(a) $31\times 23$（713 点）网格上的两步成功概率")

    Ds_et = np.linspace(5.0, 1500.0, 40)
    w_et = Ds_et.copy()
    w_et /= w_et.sum()
    rho = lambda L: max(50.0, 0.5 * L)
    tt = np.arange(350.0, 951.0, 50.0)
    et_time, et_cons = [], []
    for t in tt:
        et1, _, _ = expected_time(t, 250.0, Ds_et, w=w_et, rho_rule=rho)
        et2, _, _ = expected_time(t, C_LEVER * max(abs(D_FAR - t), abs(t - D_NEAR)), Ds_et, w=w_et, rho_rule=rho)
        et_time.append(et1)
        et_cons.append(et2)
    bx.plot(tt, et_time, "o-", color=sty.SAND, lw=1.7, ms=5, label=r"时间流派（$h=250$）")
    bx.plot(tt, et_cons, "s-", color=sty.GREEN, lw=1.7, ms=5, label="保守流派（杠杆下界）")
    bx.plot(500, et_time[list(tt).index(500.0)], "^", color=sty.SAND, ms=10, zorder=5)
    bx.plot(750, et_cons[list(tt).index(750.0)], "D", color=sty.GREEN, ms=8, zorder=5)
    bx.set_xlabel(r"$t$ / m")
    bx.set_ylabel(r"$E[T]$ / s（成功样本）")
    bx.legend(loc="upper left", fontsize=8)
    sty.style_xy(bx, title=r"(b) 计入第三次补测后的期望时间")
    print("school fig ET time@500", et_time[list(tt).index(500.0)], "cons@750", et_cons[list(tt).index(750.0)])
    save(fig, "fig_q2_schools.png")


def fig_rescue_pair():
    """长轴切割拆成两张可读的图。"""
    D = 1500.0
    t, h = 700.0, 462.0
    G = np.array([D, 0.0])
    S1 = np.array([0.0, 0.0])
    S2 = np.array([t, h])
    th2 = np.arctan2(G[1] - S2[1], G[0] - S2[0])
    HPs2 = wedge_hp(S1, 0.0) + wedge_hp(S2, th2)
    v2 = region(HPs2)
    diam2, axv = diam_and_axis(v2)
    r2, C2 = mec(v2)
    rho = max(50.0, 0.5 * diam2)
    nhat = np.array([-axv[1], axv[0]])
    if nhat @ (S2 - C2) < 0:
        nhat = -nhat
    Sn = C2 + rho * nhat
    th3 = np.arctan2(G[1] - Sn[1], G[0] - Sn[0])
    v3 = region(HPs2 + wedge_hp(Sn, th3))
    diam3, _ = diam_and_axis(v3)
    r3, C3 = mec(v3)

    fig, ax = sty.new_fig()
    ax.add_patch(Circle((0, 0), 1800, fill=False, ec=sty.ARENA, lw=0.9, ls=":", zorder=1))
    ax.plot([S1[0], G[0]], [S1[1], G[1]], color=sty.ROSE, lw=1.1, zorder=2)
    ax.plot([S2[0], G[0]], [S2[1], G[1]], color=sty.BLUE, lw=1.1, zorder=2)
    fill_poly(ax, v2, sty.FILL_IO, sty.BLUE, 0.95, 1.3, 4, label=rf"两次后的 $P$，$d={diam2:.0f}$ m")
    ax.plot(*S1, "o", color=sty.INK, ms=7, zorder=7)
    ax.plot(*S2, "o", color=sty.BLUE, ms=7, zorder=7)
    ax.plot(*G, "o", color=sty.INK, ms=4, zorder=7)
    ax.plot(*Sn, "s", color=sty.SAND, ms=7, zorder=7)
    ax.annotate(r"$S_1$", S1, textcoords="offset points", xytext=(8, 10), fontsize=11)
    ax.annotate(r"$S_2$", S2, textcoords="offset points", xytext=(8, 8), fontsize=11, color=sty.BLUE)
    ax.annotate(r"$G$", G, textcoords="offset points", xytext=(-18, -18), fontsize=11)
    ax.annotate(r"$S_{\mathrm{new}}$", Sn, textcoords="offset points", xytext=(8, 12), fontsize=10, color=sty.SAND)
    ax.annotate(r"示向度 $\theta_1$", (620, -90), fontsize=9, color=sty.ROSE)
    ax.annotate("", xy=Sn, xytext=C2, arrowprops=dict(arrowstyle="-|>", color=sty.SAND, lw=1.2, mutation_scale=9))
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(-80, 1720)
    ax.set_ylim(-260, 720)
    sty.style_map(ax, title=rf"两次测向后交会区仍细长（直径 {diam2:.0f} m，$\rho={rho:.0f}$ m）")
    axins = inset_axes(ax, width="34%", height="38%", loc="upper left", borderpad=0.8)
    fill_poly(axins, v2, sty.FILL_IO, sty.BLUE, 0.95, 1.1, 4)
    axins.plot(*G, "o", color=sty.INK, ms=4, zorder=7)
    axins.plot(*Sn, "s", color=sty.SAND, ms=5, zorder=7)
    pad_in = 40.0
    axins.set_xlim(v2[:, 0].min() - pad_in, v2[:, 0].max() + pad_in)
    axins.set_ylim(v2[:, 1].min() - pad_in, v2[:, 1].max() + pad_in)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_aspect("equal")
    for s in axins.spines.values():
        s.set_color(sty.PURPLE)
        s.set_linewidth(1.0)
    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec=sty.PURPLE, lw=0.8, ls="--")
    save(fig, "fig_q2_rescue_a.png")

    fig, bx = sty.new_fig()
    closed2 = np.vstack([v2, v2[0]])
    bx.plot(closed2[:, 0], closed2[:, 1], color=sty.BLUE, lw=1.2, ls="--", zorder=3, label=rf"两次后 $P$（直径 {diam2:.0f} m）")
    fill_poly(bx, v3, sty.FILL_OK, sty.GREEN, 0.50, 1.6, 4, label=rf"三次后 $P$（直径 {diam3:.0f} m）")
    bx.add_patch(Circle(C3, r3, fill=False, ec=sty.PURPLE, lw=1.4, label=rf"MEC，$R={r3:.1f}$ m"))
    bx.add_patch(Circle(C3, 20, fill=False, ec=sty.GREEN, lw=1.3, ls="--", label=r"$20$ m 清除圆"))
    p = C2 + (diam2 / 2) * axv
    q = C2 - (diam2 / 2) * axv
    bx.plot([p[0], q[0]], [p[1], q[1]], color=sty.ROSE, lw=1.2, ls=":", label="两次后的长轴")
    bx.plot(*C3, "+", color=sty.PURPLE, ms=9, zorder=6)
    bx.plot(*G, "o", color=sty.INK, ms=5, zorder=7)
    bx.annotate(r"$G$", G, textcoords="offset points", xytext=(8, -14), fontsize=10)
    bx.annotate(r"$C_{\mathrm{MEC}}$", C3, textcoords="offset points", xytext=(8, 8), fontsize=9, color=sty.PURPLE)
    bx.legend(loc="upper right", fontsize=8)
    pad = 25
    xs = np.vstack([v2, v3])
    bx.set_xlim(xs[:, 0].min() - pad, xs[:, 0].max() + pad)
    bx.set_ylim(xs[:, 1].min() - pad, xs[:, 1].max() + pad)
    sty.style_map(bx, title=r"第三次沿法向切入后，$P$ 落入清除圆")
    print(f"rescue: diam2={diam2:.2f} r2={r2:.2f} rho={rho:.1f} diam3={diam3:.2f} r3={r3:.2f}")
    save(fig, "fig_q2_rescue_b.png")


def fig_geom_sin():
    """单独一张：sinγ 几何，标签离开线段。"""
    S1 = np.array([0.0, 0.0])
    D = 800.0
    t, h = 500.0, 500.0
    G = np.array([D, 0.0])
    S2 = np.array([t, h])
    d1 = float(np.linalg.norm(G - S1))
    d2 = float(np.linalg.norm(G - S2))
    gvec1 = S1 - G
    gvec2 = S2 - G
    gamma = np.degrees(np.arccos(np.clip((gvec1 @ gvec2) / (d1 * d2), -1.0, 1.0)))

    fig, ax = sty.new_fig()
    ax.plot([S1[0], G[0]], [S1[1], G[1]], color=sty.INK, lw=1.4, zorder=3)
    ax.plot([S2[0], G[0]], [S2[1], G[1]], color=sty.INK, lw=1.4, zorder=3)
    ax.plot([S1[0], S2[0]], [S1[1], S2[1]], color=sty.MUTED, lw=1.0, ls="--", zorder=2)
    ax.plot([t, t], [0, h], color=sty.BLUE, lw=0.9, ls=":", zorder=2)
    ax.plot([0, t], [0, 0], color=sty.BLUE, lw=1.1, zorder=2)
    ax.plot(*S1, "o", color=sty.INK, ms=7, zorder=6)
    ax.plot(*S2, "o", color=sty.BLUE, ms=7, zorder=6)
    ax.plot(*G, "o", color=sty.INK, ms=4, zorder=6)
    ax.annotate(r"$S_1$", S1, textcoords="offset points", xytext=(-22, -16), fontsize=11)
    ax.annotate(r"$S_2$", S2, textcoords="offset points", xytext=(10, 8), fontsize=11, color=sty.BLUE)
    ax.annotate(r"$G$", G, textcoords="offset points", xytext=(10, -16), fontsize=11)
    ax.annotate(r"$t$", (t * 0.45, -55), fontsize=11, color=sty.BLUE, ha="center")
    ax.annotate(r"$h$", (t + 40, h * 0.45), fontsize=11, color=sty.BLUE)
    ax.annotate(r"$d_1=D$", (D * 0.55, 48), fontsize=10, color=sty.MUTED, ha="center")
    ax.annotate(r"$d_2$", 0.5 * (S2 + G) + np.array([24, 8]), fontsize=10, color=sty.MUTED)
    t1 = np.degrees(np.arctan2(S1[1] - G[1], S1[0] - G[0]))
    t2 = np.degrees(np.arctan2(S2[1] - G[1], S2[0] - G[0]))
    dccw = (t2 - t1) % 360.0
    if dccw <= 180.0:
        a1, a2 = t1, t1 + dccw
    else:
        a1, a2 = t2, t2 + ((t1 - t2) % 360.0)
    ax.add_patch(Arc(G, 180, 180, angle=0, theta1=a1, theta2=a2, color=sty.ROSE, lw=1.6, zorder=5))
    mid = np.radians(0.5 * (a1 + a2))
    ax.annotate(r"$\gamma$", G + 120 * unit(mid), fontsize=12, color=sty.ROSE, ha="center")
    ax.text(
        0.03, 0.97,
        r"$\varepsilon=0$ 时 $\sin\gamma=|h|/d_2$" + "\n" + rf"本图 $\gamma={gamma:.0f}^\circ$",
        transform=ax.transAxes, fontsize=9, va="top",
    )
    ax.set_xlim(-90, 900)
    ax.set_ylim(-140, 620)
    sty.style_map(ax, title=r"交会角：$\sin\gamma=|h|/d_2$（测得轴与真实方向重合时）", xlabel="t / m", ylabel="h / m")
    print(f"sin gamma fig: gamma={gamma:.1f}")
    save(fig, "fig_q2_geom.png")


def main():
    fig_geom_sin()
    fig_jacobian()
    fig_para_vs_poly()
    fig_bound()
    fig_lever()
    fig_schools()
    fig_rescue_pair()


if __name__ == "__main__":
    main()
