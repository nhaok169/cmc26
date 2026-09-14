# -*- coding: utf-8 -*-
"""问题2 论文图：全部围着「候选区 → 选点」来画。

图序与正文一致：
  fig_q2_wedge      第一次测向后源在楔形里（标 S1、示向度）
  fig_q2_geom       sinγ 几何 + L1 多边形 / L2 平行四边形
  fig_q2_gamma      左：S2 绕 G 的轨迹；右：直径随 γ，90° 最低
  fig_q2_candidate  世界坐标蝶形（同一组 S1、示向度，四点策略）
  fig_q2_rescue     两次不够时的长轴切割（S1、S2 都在图内）
E[T] 曲线仍由 design_chart.py 输出。
"""
import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper"))
from geometry import (  # noqa: E402
    DELTA, wedge_hp, region, diam_and_axis, mec, k_expr, clip,
)
import figstyle as sty  # noqa: E402

sty.apply()

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

K = 40.0 / (2 * np.tan(DELTA))
C_LEVER = 0.5 / np.sqrt(0.75)
D_NEAR, D_FAR = 5.0, 1500.0
R_MOVE = 1200.0

# 楔形与蝶形共用同一组 S1、示向度，读者可以对着看
S1_W = np.array([-520.0, 280.0])
TH1 = np.deg2rad(15.0)
E_TH = np.array([np.cos(TH1), np.sin(TH1)])
N_TH = np.array([-np.sin(TH1), np.cos(TH1)])

SCHOOLS = [
    (550, 450, "minimax", sty.ROSE, "o"),
    (675, 500, "概率", sty.PURPLE, "s"),
    (750, 450, "保守", sty.GREEN, "D"),
    (500, 250, "时间", sty.SAND, "^"),
]


def save(fig, name):
    sty.save(fig, os.path.join(OUT, name))


def unit(ang):
    return np.array([np.cos(ang), np.sin(ang)], float)


def world_s2(t, h, S1=S1_W, e=E_TH, n=N_TH):
    return np.asarray(S1, float) + t * e + h * n


def lever_h(t):
    t = np.asarray(t, float)
    return C_LEVER * np.maximum(np.abs(D_FAR - t), np.abs(t - D_NEAR))


def wedge_hp_delta(S, th, delta):
    return [
        (S, np.array([np.cos(th - delta), np.sin(th - delta)]), +1),
        (S, np.array([np.cos(th + delta), np.sin(th + delta)]), -1),
    ]


def clip_disk(poly, center, radius, n=72):
    p = np.asarray(poly, float)
    c = np.asarray(center, float)
    for k in range(n):
        ang = 2 * np.pi * k / n
        e = unit(ang)
        Q = c + radius * e
        e_dir = np.array([-e[1], e[0]])
        if len(p) < 3:
            return np.zeros((0, 2))
        p = clip(p, Q, e_dir, +1)
    return p


def sector_poly(S, th, delta, radius, n=36):
    angs = np.linspace(th - delta, th + delta, n)
    pts = [np.asarray(S, float)]
    pts += [S + radius * unit(a) for a in angs]
    return np.array(pts)


def fill_poly(ax, poly, fc, ec, alpha=0.35, lw=1.4, z=4, label=None):
    if len(poly) < 3:
        return
    closed = np.vstack([poly, poly[0]])
    ax.fill(closed[:, 0], closed[:, 1], fc=fc, ec=ec, alpha=alpha, lw=lw, zorder=z, label=label)
    ax.plot(closed[:, 0], closed[:, 1], color=ec, lw=lw, zorder=z + 1)


def rays(ax, S, th, delta, length, color, lw=1.15):
    S = np.asarray(S, float)
    ax.plot(*zip(S, S + length * unit(th)), color=color, lw=lw + 0.35, zorder=5)
    for d in (-delta, delta):
        ax.plot(*zip(S, S + length * unit(th + d)), color=color, lw=lw, ls="--", zorder=5)


def two_station_poly(S1, S2, G, delta=DELTA):
    th1 = np.arctan2(G[1] - S1[1], G[0] - S1[0])
    th2 = np.arctan2(G[1] - S2[1], G[0] - S2[0])
    return region(wedge_hp_delta(S1, th1, delta) + wedge_hp_delta(S2, th2, delta), box=5000.0)


def l2_parallelogram(S1, S2, G, delta=DELTA):
    """误差方块 [±δ,±δ]² 经 J^{-1} 映成的平行四边形顶点。"""
    d1 = np.linalg.norm(G - S1)
    d2 = np.linalg.norm(G - S2)
    u1 = (G - S1) / d1
    u2 = (G - S2) / d2
    n1 = np.array([-u1[1], u1[0]])
    n2 = np.array([-u2[1], u2[0]])
    J = np.vstack([n1 / d1, n2 / d2])
    Jinv = np.linalg.inv(J)
    a, b = Jinv[:, 0], Jinv[:, 1]
    verts = np.array([
        G + delta * (a + b),
        G + delta * (a - b),
        G + delta * (-a - b),
        G + delta * (-a + b),
    ], float)
    c = verts.mean(axis=0)
    ang = np.arctan2(verts[:, 1] - c[1], verts[:, 0] - c[0])
    return verts[np.argsort(ang)], a, b


def arc_span(vertex, p1, p2):
    a1 = np.degrees(np.arctan2(p1[1] - vertex[1], p1[0] - vertex[0]))
    a2 = np.degrees(np.arctan2(p2[1] - vertex[1], p2[0] - vertex[0]))
    dccw = (a2 - a1) % 360.0
    if dccw <= 180.0:
        return a1, a1 + dccw
    return a2, a2 + ((a1 - a2) % 360.0)


def fig_wedge():
    """第一次测向：源只能落在 S1 示向度的楔形里，距离未知。"""
    fig, ax = sty.new_fig()
    O = np.array([0.0, 0.0])
    d_draw = np.deg2rad(8.0)
    ax.add_patch(Circle(O, 1800, fc="#F4F4F4", ec="none", alpha=0.9, zorder=0))
    ax.add_patch(Circle(O, 1800, fill=False, ec=sty.ARENA, lw=1.15, zorder=2))
    ax.add_patch(Circle(S1_W, 1500, fill=False, ec=sty.BLUE, lw=1.0, ls="--", zorder=2))

    sec = sector_poly(S1_W, TH1, d_draw, 1500, n=48)
    W = clip_disk(sec, O, 1800, n=80)
    fill_poly(ax, W, sty.FILL_P, sty.ROSE, alpha=0.42, lw=1.2, z=3)
    rays(ax, S1_W, TH1, d_draw, 1650, sty.ROSE, lw=1.05)

    tip = S1_W + 1580 * E_TH
    ax.annotate(
        "", xy=tip, xytext=S1_W,
        arrowprops=dict(arrowstyle="-|>", color=sty.INK, lw=1.35, mutation_scale=11),
    )
    mid = S1_W + 820 * E_TH
    ax.annotate(r"示向度 $\theta_1$", mid, textcoords="offset points",
                xytext=(12, 14), fontsize=10, color=sty.INK)

    ax.plot(*S1_W, "o", color=sty.BLUE, ms=8, zorder=7)
    ax.annotate(r"$S_1$", S1_W, textcoords="offset points", xytext=(-36, 12),
                fontsize=11, color=sty.BLUE)
    ax.plot(0, 0, "+", color=sty.INK, ms=8, zorder=6)
    ax.annotate("场地圆心", (80, -220), fontsize=9, color=sty.MUTED, ha="left")

    for dist, lab in [(450, ""), (950, ""), (1400, "")]:
        g = S1_W + dist * E_TH
        ax.plot(*g, "o", mfc="white", mec=sty.INK, ms=6, zorder=6)
    ax.annotate("距离未知，源可能在楔形内任一点",
                (200, 80),
                fontsize=9, color=sty.MUTED, ha="left")

    ax.set_xlim(-1950, 1950)
    ax.set_ylim(-1950, 1950)
    sty.style_map(ax, title=r"第一次测向后：干扰源落在 $S_1$ 示向度的楔形里")
    save(fig, "fig_q2_wedge.png")


def fig_geom():
    """(a) sinγ 的三角形；(b) 同一组站的 L1 多边形与 L2 平行四边形。"""
    S1 = np.array([0.0, 0.0])
    D = 800.0
    t, h = 500.0, 500.0
    G = np.array([D, 0.0])
    S2 = np.array([t, h])
    d1 = float(np.linalg.norm(G - S1))
    d2 = float(np.linalg.norm(G - S2))
    gvec1 = S1 - G
    gvec2 = S2 - G
    gamma = np.degrees(np.arccos(np.clip(
        (gvec1 @ gvec2) / (d1 * d2), -1.0, 1.0)))

    fig, axes = sty.new_fig(1, 2)
    ax, bx = axes

    ax.plot([S1[0], G[0]], [S1[1], G[1]], color=sty.INK, lw=1.4, zorder=3)
    ax.plot([S2[0], G[0]], [S2[1], G[1]], color=sty.INK, lw=1.4, zorder=3)
    ax.plot([S1[0], S2[0]], [S1[1], S2[1]], color=sty.MUTED, lw=1.0, ls="--", zorder=2)
    ax.plot([t, t], [0, h], color=sty.BLUE, lw=0.9, ls=":", zorder=2)
    ax.plot([0, t], [0, 0], color=sty.BLUE, lw=1.1, zorder=2)
    ax.annotate(
        "", xy=(t + 40, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=sty.BLUE, lw=1.1, mutation_scale=9),
    )

    ax.plot(*S1, "o", color=sty.INK, ms=8, zorder=6)
    ax.plot(*S2, "o", color=sty.BLUE, ms=8, zorder=6)
    ax.plot(*G, "*", color=sty.INK, ms=12, zorder=6)
    ax.annotate(r"$S_1$", S1, textcoords="offset points", xytext=(-22, -16), fontsize=11)
    ax.annotate(r"$S_2$", S2, textcoords="offset points", xytext=(8, 8), fontsize=11, color=sty.BLUE)
    ax.annotate(r"$G$", G, textcoords="offset points", xytext=(8, -16), fontsize=11)
    ax.annotate(r"$t$", (t * 0.45, -48), fontsize=11, color=sty.BLUE, ha="center")
    ax.annotate(r"$h$", (t + 28, h * 0.45), fontsize=11, color=sty.BLUE)
    ax.annotate(r"$d_1$", (D * 0.62, 42), fontsize=10, color=sty.MUTED, ha="center")
    ax.annotate(r"$d_2$", 0.5 * (S2 + G) + np.array([18, 10]), fontsize=10, color=sty.MUTED)

    t1, t2 = arc_span(G, S1, S2)
    rad = 130.0
    ax.add_patch(Arc(G, 2 * rad, 2 * rad, angle=0, theta1=t1, theta2=t2,
                     color=sty.ROSE, lw=1.5, zorder=5))
    mid_ang = np.radians(0.5 * (t1 + t2))
    ax.annotate(r"$\gamma$", G + (rad + 28) * unit(mid_ang), fontsize=12, color=sty.ROSE, ha="center")
    ax.text(0.03, 0.97, r"$\sin\gamma=|h|/d_2$" + f"\n$\\gamma={gamma:.0f}^\\circ$",
            transform=ax.transAxes, fontsize=9, color=sty.INK, va="top")
    ax.set_xlim(-90, 900)
    ax.set_ylim(-160, 620)
    sty.style_map(ax, title=r"(a) $\sin\gamma$ 的几何：$S_1,S_2,G$", xlabel="t / m", ylabel="h / m")

    v1 = two_station_poly(S1, S2, G, DELTA)
    para, a_vec, b_vec = l2_parallelogram(S1, S2, G, DELTA)
    fill_poly(bx, v1, sty.FILL_IO, sty.BLUE, alpha=0.45, lw=1.5, z=4, label="L1 交会多边形")
    closed = np.vstack([para, para[0]])
    bx.plot(closed[:, 0], closed[:, 1], color=sty.ROSE, lw=1.7, ls="--", zorder=6,
            label="L2 平行四边形")
    bx.plot(*G, "*", color=sty.INK, ms=10, zorder=7)
    bx.annotate(r"$G$", G, textcoords="offset points", xytext=(8, -14), fontsize=10)
    # 平行四边形的两条生成边
    bx.annotate(
        "", xy=G + DELTA * a_vec, xytext=G,
        arrowprops=dict(arrowstyle="-|>", color=sty.SAND, lw=1.2, mutation_scale=9),
    )
    bx.annotate(
        "", xy=G + DELTA * b_vec, xytext=G,
        arrowprops=dict(arrowstyle="-|>", color=sty.GREEN, lw=1.2, mutation_scale=9),
    )
    bx.text(*(G + DELTA * a_vec + np.array([-18, 10])), r"$\delta a$", fontsize=9, color=sty.SAND)
    bx.text(*(G + DELTA * b_vec + np.array([6, -12])), r"$\delta b$", fontsize=9, color=sty.GREEN)
    diam, _ = diam_and_axis(v1)
    bx.legend(loc="upper right", fontsize=8)
    bx.text(0.04, 0.04, f"直径 {diam:.1f} m", transform=bx.transAxes, fontsize=9, va="bottom")
    span = 55
    bx.set_xlim(G[0] - span, G[0] + span)
    bx.set_ylim(G[1] - span, G[1] + span)
    sty.style_map(bx, title=r"(b) L1 多边形与 L2 平行四边形")
    print(f"geom: gamma={gamma:.1f} deg, L1 diam={diam:.2f} m")
    save(fig, "fig_q2_geom.png")


def fig_gamma_pair():
    """左：S2 绕 G；右：直径–γ 曲线。"""
    d = 800.0
    G = np.array([0.0, 0.0])
    S1 = np.array([-d, 0.0])

    fig, axes = sty.new_fig(1, 2)
    ax, bx = axes

    circ = plt.Circle(G, d, fill=False, ec=sty.GRID, lw=1.1, ls="--", zorder=1)
    ax.add_patch(circ)
    ax.plot([S1[0], G[0]], [S1[1], G[1]], color=sty.INK, lw=1.2, zorder=3)
    ax.plot(*S1, "o", color=sty.BLUE, ms=8, zorder=6)
    ax.plot(*G, "*", color=sty.INK, ms=11, zorder=6)
    ax.annotate(r"$S_1$", S1, textcoords="offset points", xytext=(10, -22), fontsize=11, color=sty.BLUE)
    ax.annotate(r"$G$", G, textcoords="offset points", xytext=(8, -16), fontsize=11)

    gammas = np.linspace(25, 155, 80)
    traj = np.array([d * unit(np.pi - np.deg2rad(g)) for g in gammas])
    ax.plot(traj[:, 0], traj[:, 1], color=sty.BLUE, lw=1.8, zorder=3)
    ax.annotate(r"$S_2$ 轨迹", traj[12], textcoords="offset points",
                xytext=(-62, 6), fontsize=9, color=sty.BLUE)

    marks = [(30, sty.ROSE, "o"), (90, sty.GREEN, "s"), (150, sty.SAND, "D")]
    for gdeg, col, mk in marks:
        S2 = d * unit(np.pi - np.deg2rad(gdeg))
        ax.plot([S2[0], G[0]], [S2[1], G[1]], color=col, lw=1.15, zorder=4)
        ax.plot(*S2, mk, color=col, ms=8, zorder=7)
        ax.annotate(rf"$S_2$({gdeg:.0f}$^\circ$)", S2, textcoords="offset points",
                    xytext=(8, 8) if gdeg != 90 else (8, 10),
                    fontsize=9, color=col)
        t1, t2 = arc_span(G, S1, S2)
        rad = 160 if gdeg == 90 else 110
        ax.add_patch(Arc(G, 2 * rad, 2 * rad, angle=0, theta1=t1, theta2=t2,
                         color=col, lw=1.2, zorder=5))

    ax.set_xlim(-1050, 950)
    ax.set_ylim(-420, 1050)
    sty.style_map(ax, title=r"(a) $S_2$ 绕 $G$ 改变交会角 $\gamma$")

    gam = np.linspace(12, 168, 320)
    gr = np.deg2rad(gam)
    num = 2 * d * d * (1 + np.abs(np.cos(gr)))
    Dloc = 2 * DELTA * np.sqrt(num / np.sin(gr) ** 2)
    bx.plot(gam, Dloc, color=sty.BLUE, lw=2.0, zorder=3)
    for gdeg, col, mk in marks:
        i = np.argmin(np.abs(gam - gdeg))
        bx.plot(gdeg, Dloc[i], mk, color=col, ms=8, zorder=5,
                label=rf"{gdeg:.0f}°，{Dloc[i]:.1f} m")
        bx.annotate("", xy=(gdeg, Dloc[i]),
                    xytext=(gdeg, Dloc[i] + (18 if gdeg != 90 else -22)),
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.6))
    bx.axhline(40, color=sty.ROSE, ls="--", lw=0.95, alpha=0.85)
    bx.text(14, 46, "40 m 筛选线", fontsize=9, color=sty.ROSE)
    bx.set_xlim(10, 170)
    bx.set_ylim(0, 160)
    bx.set_xlabel(r"交会角 $\gamma$ / °")
    bx.set_ylabel(r"定位直径 $D_{\mathrm{loc}}$ / m")
    bx.legend(loc="upper center", ncol=1, fontsize=8.5)
    sty.style_xy(bx, title=r"(b) 直径随 $\gamma$ 变化，$90^\circ$ 最低")
    print("gamma90 diam", Dloc[np.argmin(np.abs(gam - 90))])
    save(fig, "fig_q2_gamma.png")


def fig_candidate_world():
    """蝶形画在场地里：S1、示向度、两侧翼、四个策略点。"""
    ts = np.linspace(80, 1400, 180)
    hs = np.linspace(-900, 900, 221)
    T, H = np.meshgrid(ts, hs)
    lev = np.abs(H) >= lever_h(T)
    inside = (T ** 2 + H ** 2) <= R_MOVE ** 2
    X = S1_W[0] + T * E_TH[0] + H * N_TH[0]
    Y = S1_W[1] + T * E_TH[1] + H * N_TH[1]
    in_arena = X ** 2 + Y ** 2 <= 1800 ** 2
    k_hit = np.full(T.shape, np.inf)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            t, h = float(T[i, j]), float(H[i, j])
            if abs(h) < 20:
                continue
            k_hit[i, j] = k_expr(t, h, t, 0.0)
    k_ok = k_hit <= (K * K)
    mask_lever = lev & inside & in_arena
    mask_core = mask_lever & k_ok

    fig, ax = sty.new_fig()
    ax.add_patch(Circle((0, 0), 1800, fc="#F7F7F7", ec=sty.ARENA, lw=1.1, zorder=0))
    rays(ax, S1_W, TH1, np.deg2rad(8.0), 1650, sty.ROSE, lw=0.9)
    tip = S1_W + 1700 * E_TH
    ax.annotate(
        "", xy=tip, xytext=S1_W,
        arrowprops=dict(arrowstyle="-|>", color=sty.INK, lw=1.25, mutation_scale=11),
    )
    ax.annotate(r"示向度 $\theta_1$", S1_W + 900 * E_TH,
                textcoords="offset points", xytext=(10, -22), fontsize=10)

    ax.scatter(X[mask_lever], Y[mask_lever], s=1.6, c=sty.FILL_IO, linewidths=0, zorder=2, rasterized=True)
    ax.scatter(X[mask_core], Y[mask_core], s=1.6, c=sty.FILL_OK, linewidths=0, zorder=3, rasterized=True)

    # 杠杆边界映到世界坐标
    tb = np.linspace(120, 1350, 160)
    hb = lever_h(tb)
    for sign in (1.0, -1.0):
        pts = np.array([world_s2(t, sign * h) for t, h in zip(tb, hb)])
        keep = (tb ** 2 + hb ** 2) <= R_MOVE ** 2
        ax.plot(pts[keep, 0], pts[keep, 1], color=sty.BLUE, lw=1.4, zorder=4)

    ax.plot(*S1_W, "o", color=sty.BLUE, ms=8, zorder=8)
    ax.annotate(r"$S_1$", S1_W, textcoords="offset points", xytext=(-32, 10),
                fontsize=11, color=sty.BLUE)

    offsets = {
        "minimax": (10, 12),
        "概率": (12, -6),
        "保守": (12, 10),
        "时间": (-52, -18),
    }
    for t, h, name, col, mk in SCHOOLS:
        p = world_s2(t, h)
        ax.plot(*p, mk, color=col, ms=9, zorder=8, label=rf"{name} $({t:.0f},{h:.0f})$")
        dx, dy = offsets.get(name, (8, 8))
        ax.annotate(name, p, textcoords="offset points", xytext=(dx, dy), fontsize=8, color=col)
        p2 = world_s2(t, -h)
        ax.plot(*p2, mk, color=col, ms=6, mfc="white", zorder=7)

    ax.legend(loc="lower left", fontsize=8, framealpha=0.95)
    ax.set_xlim(-1950, 1950)
    ax.set_ylim(-1950, 1950)
    sty.style_map(ax, title=r"第二点候选区：示向度两侧的蝶形（浅色为杠杆带）")
    save(fig, "fig_q2_candidate.png")


def fig_rescue():
    """两次不够：全局能看见 S1、S2、G；局部看切割。"""
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

    fig, axes = sty.new_fig(1, 2)
    ax, bx = axes

    focus = np.vstack([v2, v3, C2.reshape(1, 2), Sn.reshape(1, 2)])
    xmin, xmax = focus[:, 0].min(), focus[:, 0].max()
    ymin, ymax = focus[:, 1].min(), focus[:, 1].max()
    pad = 0.28 * max(xmax - xmin, ymax - ymin, 80)
    box_x0, box_y0 = xmin - pad, ymin - pad
    box_w, box_h = (xmax + pad) - box_x0, (ymax + pad) - box_y0

    ax.add_patch(Circle((0, 0), 1800, fill=False, ec=sty.ARENA, lw=0.9, ls=":", zorder=1))
    rays(ax, S1, 0.0, DELTA, 1600, sty.ROSE, lw=0.9)
    rays(ax, S2, th2, DELTA, 1100, sty.BLUE, lw=0.9)
    fill_poly(ax, v2, sty.FILL_IO, sty.BLUE, alpha=0.90, lw=1.3, z=4)
    ax.add_patch(Rectangle((box_x0, box_y0), box_w, box_h, fill=False,
                           ec=sty.PURPLE, lw=1.15, ls="--", zorder=8))
    ax.plot(*S1, "o", color=sty.INK, ms=8, zorder=7)
    ax.plot(*S2, "o", color=sty.BLUE, ms=8, zorder=7)
    ax.plot(*G, "*", color=sty.INK, ms=11, zorder=7)
    ax.plot(*Sn, "s", color=sty.SAND, ms=7, zorder=7)
    ax.annotate(r"$S_1$", S1, textcoords="offset points", xytext=(8, 10), fontsize=11)
    ax.annotate(r"$S_2$", S2, textcoords="offset points", xytext=(8, 8), fontsize=11, color=sty.BLUE)
    ax.annotate(r"$G$", G, textcoords="offset points", xytext=(-22, -18), fontsize=11)
    ax.annotate(r"$S_{\mathrm{new}}$", Sn, textcoords="offset points",
                xytext=(8, 12), fontsize=9, color=sty.SAND)
    ax.annotate(r"示向度 $\theta_1$", (620, -80), fontsize=9, color=sty.ROSE)
    ax.annotate(
        "", xy=Sn, xytext=C2,
        arrowprops=dict(arrowstyle="-|>", color=sty.SAND, lw=1.2, mutation_scale=9),
    )
    ax.set_xlim(-120, 1720)
    ax.set_ylim(-280, 720)
    sty.style_map(ax, title=rf"(a) 两次测向后　直径 {diam2:.0f} m")

    closed2 = np.vstack([v2, v2[0]])
    bx.plot(closed2[:, 0], closed2[:, 1], color=sty.BLUE, lw=1.0, ls="--", zorder=3)
    fill_poly(bx, v3, sty.FILL_OK, sty.GREEN, alpha=0.55, lw=1.5, z=4)
    bx.add_patch(Circle(C3, r3, fill=False, ec=sty.PURPLE, lw=1.2, ls="--"))
    bx.add_patch(Circle(C3, 20, fill=False, ec=sty.GREEN, lw=1.1, ls=":"))
    p = C2 + (diam2 / 2) * axv
    q = C2 - (diam2 / 2) * axv
    bx.plot([p[0], q[0]], [p[1], q[1]], color=sty.ROSE, lw=1.3, ls="--")
    bx.plot(*C3, "+", color=sty.PURPLE, ms=9, zorder=6)
    bx.plot(*G, "*", color=sty.INK, ms=10, zorder=6)
    bx.annotate(r"$G$", G, textcoords="offset points", xytext=(8, -14), fontsize=9)
    bx.text(0.03, 0.97, "蓝虚线：两次后的 $P$\n绿圆：$20$ m 清除圆",
            transform=bx.transAxes, fontsize=8, va="top", color=sty.INK)
    bx.set_xlim(box_x0, box_x0 + box_w)
    bx.set_ylim(box_y0, box_y0 + box_h)
    sty.style_map(bx, title=rf"(b) 沿长轴法向再测　直径 {diam3:.0f} m")
    print(f"rescue: diam2={diam2:.2f} r2={r2:.2f} rho={rho:.1f} diam3={diam3:.2f} r3={r3:.2f}")
    print(f"  S1={S1} S2={S2} Sn={Sn}")
    save(fig, "fig_q2_rescue.png")


def main():
    fig_wedge()
    fig_geom()
    fig_gamma_pair()
    fig_candidate_world()
    fig_rescue()


if __name__ == "__main__":
    main()
