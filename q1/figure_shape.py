"""问题1示意：定位区域怎么来、直径怎么取、覆盖怎么判。

顶点、直径、圆覆盖一律调用 solver.py。形状图把 ±1° 放大到 ±8°，便于看清连线。
"""
import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from solver import bearing_deg, locate

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "paper"))
import figstyle as fs  # noqa: E402

fs.apply()


def unit(ang_deg):
    t = np.radians(ang_deg)
    return np.array([np.cos(t), np.sin(t)])


def require_ok(sensors, thetas, delta):
    r = locate(sensors, thetas, delta)
    if r.status != "OK" or r.vertices is None:
        raise RuntimeError("locate 失败：{} {}".format(r.status, r.message))
    return r


G = np.array([80.0, 220.0])
S1 = np.array([-920.0, -480.0])
S2 = np.array([980.0, -420.0])
SENSORS = [S1, S2]
THETAS = [bearing_deg(S, G) for S in SENSORS]
DELTA_DRAW = 8.0
COLORS = [fs.BLUE, fs.GREEN]
RAY_LEN = 2600
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)


def ray_segment(S, ang, length):
    return np.asarray(S, float), np.asarray(S, float) + length * unit(ang)


def draw_cone_rays(ax, S, theta, delta, color, length, lw_main=1.8, lw_edge=1.1):
    p0, p1 = ray_segment(S, theta, length)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw_main, zorder=3)
    for dang in (-delta, delta):
        q0, q1 = ray_segment(S, theta + dang, length)
        ax.plot([q0[0], q1[0]], [q0[1], q1[1]], color=color, lw=lw_edge, ls="--", alpha=0.9, zorder=3)


def fill_cone(ax, S, theta, delta, color, radius, alpha=0.10):
    n = 48
    angs = np.linspace(theta - delta, theta + delta, n)
    pts = np.array([S] + [S + radius * unit(a) for a in angs])
    ax.fill(pts[:, 0], pts[:, 1], color=color, alpha=alpha, lw=0, zorder=1)


def draw_polygon(ax, poly, fc, ec, alpha, lw=2.0, zorder=4):
    closed = np.vstack([poly, poly[0]])
    ax.fill(closed[:, 0], closed[:, 1], fc=fc, ec=ec, alpha=alpha, lw=lw, zorder=zorder)
    ax.plot(closed[:, 0], closed[:, 1], color=ec, lw=lw, zorder=zorder + 1)


def style_ax(ax, title, xlim, ylim, xlabel=False, ylabel=False):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    fs.style_map(ax, title=title, xlabel="x / m" if xlabel else "", ylabel="y / m" if ylabel else "")
    if not xlabel:
        ax.set_xlabel("")
    if not ylabel:
        ax.set_ylabel("")


r_draw = require_ok(SENSORS, THETAS, DELTA_DRAW)
P_draw = r_draw.vertices
r2 = require_ok(SENSORS, THETAS, 1.0)
P2, D2, A2, B2 = r2.vertices, r2.diameter, r2.A, r2.B
O2, R2 = r2.center, r2.radius
print("示意 ±{}°: {} 边形".format(DELTA_DRAW, len(P_draw)))
print("真实 ±1°: {} 边形, D={:.3f} m, 覆盖={}".format(len(P2), D2, r2.covers))


def draw_form_and_diameter():
    """图1：(a) 两锥交出 P；(b) 全部顶点连线，最长者即直径。"""
    fig, axes = fs.new_fig(1, 2)
    ax_a, ax_b = axes

    fill_cone(ax_a, S1, THETAS[0], DELTA_DRAW, COLORS[0], 2400, 0.12)
    fill_cone(ax_a, S2, THETAS[1], DELTA_DRAW, COLORS[1], 2400, 0.12)
    draw_cone_rays(ax_a, S1, THETAS[0], DELTA_DRAW, COLORS[0], RAY_LEN)
    draw_cone_rays(ax_a, S2, THETAS[1], DELTA_DRAW, COLORS[1], RAY_LEN)
    draw_polygon(ax_a, P_draw, fs.FILL_P, fs.ROSE, 0.55, lw=1.8)
    ax_a.plot(S1[0], S1[1], "o", color=COLORS[0], ms=9, zorder=8)
    ax_a.plot(S2[0], S2[1], "o", color=COLORS[1], ms=9, zorder=8)
    ax_a.annotate(r"$S_1$", S1, textcoords="offset points", xytext=(14, 10), fontsize=11, color=COLORS[0])
    ax_a.annotate(r"$S_2$", S2, textcoords="offset points", xytext=(-42, 10), fontsize=11, color=COLORS[1])
    ax_a.plot(G[0], G[1], marker="*", color=fs.INK, ms=11, zorder=8)
    ax_a.annotate("$G$", G, textcoords="offset points", xytext=(10, -16), fontsize=10)
    mid1 = S1 + 0.42 * (G - S1)
    ax_a.annotate(r"$\theta_1$", mid1, textcoords="offset points", xytext=(-18, 10), fontsize=10, color=COLORS[0])
    mid2 = S2 + 0.38 * (G - S2)
    ax_a.annotate(r"$\theta_2$", mid2, textcoords="offset points", xytext=(8, -14), fontsize=10, color=COLORS[1])
    style_ax(ax_a, r"(a) 两误差锥交出定位区域 $P$（张角放大）", (-1650, 1650), (-1400, 1600), ylabel=True)

    m = len(P2)
    pairs = []
    for i in range(m):
        for j in range(i + 1, m):
            pairs.append((float(np.linalg.norm(P2[i] - P2[j])), i, j))
    pairs.sort(reverse=True)
    # 非直径的边和对角线
    for d, i, j in pairs[1:]:
        ax_b.plot(
            [P2[i][0], P2[j][0]], [P2[i][1], P2[j][1]],
            color=fs.MUTED, lw=1.0, ls="-", alpha=0.55, zorder=4,
        )
    ax_b.plot(
        [A2[0], B2[0]], [A2[1], B2[1]],
        color=fs.PURPLE, lw=2.4, zorder=6, label="直径（最长连线）",
    )
    draw_polygon(ax_b, P2, fs.FILL_P, fs.ROSE, 0.28, lw=1.5)
    c = P2.mean(axis=0)
    for k, p in enumerate(P2):
        ax_b.plot(p[0], p[1], "o", color=fs.ROSE, ms=6, zorder=7)
        v = p - c
        n = np.linalg.norm(v) + 1e-9
        ax_b.annotate(
            f"$V_{{{k+1}}}$", p,
            textcoords="offset points",
            xytext=(13 * v[0] / n, 13 * v[1] / n),
            fontsize=10, color=fs.ROSE, ha="center", va="center", zorder=8,
        )
    ax_b.legend(loc="lower right", fontsize=8.5)
    pad = 90
    xmin, ymin = P2.min(axis=0) - pad
    xmax, ymax = P2.max(axis=0) + pad
    style_ax(ax_b, r"(b) 全部顶点连线，最长者为直径", (xmin, xmax), (ymin, ymax))
    fs.save(fig, os.path.join(OUT, "fig_q1_shape_ab.png"))


def draw_cover_and_counter():
    """图2：(a) 直径中点检验覆盖；(b) 锐角三角形反例。"""
    fig, axes = fs.new_fig(1, 2)
    ax_a, ax_b = axes

    draw_polygon(ax_a, P2, fs.FILL_P, fs.ROSE, 0.40, lw=1.6)
    ax_a.plot([A2[0], B2[0]], [A2[1], B2[1]], color=fs.PURPLE, lw=2.2, zorder=5)
    ax_a.add_patch(Circle(O2, R2, fill=False, ec=fs.PURPLE, lw=1.6))
    ax_a.plot(O2[0], O2[1], "+", color=fs.PURPLE, ms=11, mew=1.6, zorder=6)
    ax_a.annotate("$O$", O2, textcoords="offset points", xytext=(8, -14), fontsize=11, color=fs.PURPLE)
    ax_a.annotate("$A$", A2, textcoords="offset points", xytext=(-16, 6), fontsize=11, color=fs.PURPLE)
    ax_a.annotate("$B$", B2, textcoords="offset points", xytext=(8, 6), fontsize=11, color=fs.PURPLE)
    for p in P2:
        ax_a.plot(p[0], p[1], "o", color=fs.ROSE, ms=5.5, zorder=6)
        ax_a.plot([O2[0], p[0]], [O2[1], p[1]], color=fs.PURPLE, lw=0.8, ls=":", zorder=5)
    s = max(R2 * 2.15, 48.0)
    style_ax(
        ax_a,
        r"(a) 圆心取直径中点，再量各顶点到中点",
        (O2[0] - s, O2[0] + s), (O2[1] - s, O2[1] + s), xlabel=True, ylabel=True,
    )

    # 边长 2 的等边三角形：以底边为直径的圆盖不住第三顶点
    A = np.array([0.0, 0.0])
    B = np.array([2.0, 0.0])
    C = np.array([1.0, np.sqrt(3)])
    O = (A + B) / 2
    R = 1.0
    tri = np.vstack([A, B, C])
    draw_polygon(ax_b, tri, fs.FILL_WARN, fs.INK, 0.35, lw=1.6)
    ax_b.plot([A[0], B[0]], [A[1], B[1]], color=fs.PURPLE, lw=2.3, zorder=5)
    ax_b.add_patch(Circle(O, R, fill=False, ec=fs.PURPLE, lw=1.6))
    ax_b.plot(O[0], O[1], "+", color=fs.PURPLE, ms=11, mew=1.6, zorder=6)
    ax_b.plot([O[0], C[0]], [O[1], C[1]], color=fs.ROSE, lw=1.2, ls="--", zorder=5)
    for p, lab, off in [(A, "$A$", (-12, -14)), (B, "$B$", (8, -14)), (C, "$C$", (6, 8)), (O, "$O$", (8, -12))]:
        ax_b.plot(p[0], p[1], "o", color=fs.INK if lab != "$O$" else fs.PURPLE, ms=6, zorder=7)
        ax_b.annotate(lab, p, textcoords="offset points", xytext=off, fontsize=11, color=fs.PURPLE if lab == "$O$" else fs.INK)
    ax_b.annotate(r"$\sqrt{3}>1$", (O + C) / 2, textcoords="offset points", xytext=(10, 0), fontsize=10, color=fs.ROSE)
    ax_b.set_xlim(-0.35, 2.55)
    ax_b.set_ylim(-1.25, 2.15)
    fs.style_map(ax_b, title=r"(b) 反例：第三顶点在直径圆外", xlabel="x", ylabel="y")
    fs.save(fig, os.path.join(OUT, "fig_q1_shape_cd.png"))


if __name__ == "__main__":
    draw_form_and_diameter()
    draw_cover_and_counter()
