"""问题1示意：定位区域怎么来、直径怎么取、覆盖怎么判。

顶点、直径、圆覆盖一律调用 solver.py。形状图把 ±1° 放大到 ±8°，便于看清连线。
"""
import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Wedge

from solver import bearing_deg, cone_halfplanes, locate, minimum_enclosing_circle, polygon_diameter

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


def _bbox_gap(b1, b2):
    dx = max(0.0, b1.x0 - b2.x1, b2.x0 - b1.x1)
    dy = max(0.0, b1.y0 - b2.y1, b2.y0 - b1.y1)
    if dx == 0.0 and dy == 0.0:
        ox = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
        oy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
        return -min(ox, oy)
    if dx == 0.0:
        return dy
    if dy == 0.0:
        return dx
    return float(np.hypot(dx, dy))


def _poly_outside(poly, origin, direction, extra):
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) + 1e-15)
    proj = max(float((np.asarray(v) - origin) @ d) for v in poly)
    return origin + (proj + extra) * d


def _fan_label_pos(S, theta, delta, side, dist):
    """扇形内、主射线与锥边之间、靠外一侧。"""
    return np.asarray(S, float) + dist * unit(theta + side * 0.50 * delta)


def check_ax_a_labels(ax, poly, points, segments, min_px=8.0):
    """自检：标签矩形与其它标签 / 点 / 线段 / 多边形边的间距。"""
    fig = ax.figure
    fig.set_dpi(fs.DPI)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    boxes = []
    for t in ax.texts:
        if not str(t.get_text()).strip():
            continue
        boxes.append((str(t.get_text()), t.get_window_extent(renderer=rend)))
    issues = []
    for i in range(len(boxes)):
        s1, b1 = boxes[i]
        for j in range(i + 1, len(boxes)):
            s2, b2 = boxes[j]
            g = _bbox_gap(b1, b2)
            if g < min_px:
                issues.append("text {} vs {} gap={:.1f}px".format(s1, s2, g))
    for s, b in boxes:
        pad = b.expanded(1.0 + min_px / max(b.width, 1.0), 1.0 + min_px / max(b.height, 1.0))
        pad = matplotlib.transforms.Bbox.from_extents(
            b.x0 - min_px, b.y0 - min_px, b.x1 + min_px, b.y1 + min_px
        )
        for p in points:
            disp = ax.transData.transform(p)
            if pad.contains(disp[0], disp[1]):
                issues.append("text {} near point ({:.0f},{:.0f})".format(s, p[0], p[1]))
        for p0, p1 in segments:
            p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
            hit = False
            for a in np.linspace(0.0, 1.0, 28):
                q = (1.0 - a) * p0 + a * p1
                disp = ax.transData.transform(q)
                if pad.contains(disp[0], disp[1]):
                    issues.append("text {} overlaps ray/edge".format(s))
                    hit = True
                    break
            if hit:
                break
        closed = np.vstack([poly, poly[0]])
        hit = False
        for k in range(len(closed) - 1):
            p0, p1 = closed[k], closed[k + 1]
            for a in np.linspace(0.0, 1.0, 20):
                q = (1.0 - a) * p0 + a * p1
                disp = ax.transData.transform(q)
                if pad.contains(disp[0], disp[1]):
                    issues.append("text {} overlaps polygon".format(s))
                    hit = True
                    break
            if hit:
                break
    if issues:
        print("fig1(a) LABEL CHECK FAIL:")
        for line in issues:
            print(" ", line)
    else:
        print("fig1(a) LABEL CHECK OK (gap>={}px)".format(min_px))
    return issues


def draw_form_and_diameter():
    """图1：(a) 两锥交出 P；(b) 全部顶点连线，最长者即直径。"""
    fig, axes = fs.new_fig(1, 2)
    ax_a, ax_b = axes
    FS = 10.5

    fill_cone(ax_a, S1, THETAS[0], DELTA_DRAW, COLORS[0], 2400, 0.12)
    fill_cone(ax_a, S2, THETAS[1], DELTA_DRAW, COLORS[1], 2400, 0.12)
    draw_cone_rays(ax_a, S1, THETAS[0], DELTA_DRAW, COLORS[0], RAY_LEN)
    draw_cone_rays(ax_a, S2, THETAS[1], DELTA_DRAW, COLORS[1], RAY_LEN)
    draw_polygon(ax_a, P_draw, fs.FILL_P, fs.ROSE, 0.55, lw=1.8)
    ax_a.plot(S1[0], S1[1], "o", color=COLORS[0], ms=7, zorder=8)
    ax_a.plot(S2[0], S2[1], "o", color=COLORS[1], ms=7, zorder=8)
    ax_a.annotate(
        r"$S_1$", S1, textcoords="offset points", xytext=(10, -24),
        fontsize=FS, color=COLORS[0], ha="left", va="top", zorder=9,
    )
    ax_a.annotate(
        r"$S_2$", S2, textcoords="offset points", xytext=(10, -24),
        fontsize=FS, color=COLORS[1], ha="left", va="top", zorder=9,
    )
    ax_a.plot(G[0], G[1], marker="o", color=fs.INK, ms=3.5, zorder=7)
    # G / P：箭头与文字分开，避免 annotate 的 window extent 把引线算进碰撞盒
    p_mean = P_draw.mean(axis=0)
    g_lab = np.array([97.0, -300.0])
    g_hit = _poly_outside(P_draw, G, g_lab - G, extra=22.0)
    ax_a.annotate(
        "", xy=g_hit, xytext=g_lab, textcoords="data",
        arrowprops=dict(arrowstyle="-|>", color=fs.INK, lw=0.8, mutation_scale=7),
        zorder=8,
    )
    ax_a.text(g_lab[0], g_lab[1], r"$G$", fontsize=FS, color=fs.INK, ha="center", va="top", zorder=9)
    p_lab = np.array([69.0, 880.0])
    p_hit = _poly_outside(P_draw, p_mean, p_lab - p_mean, extra=20.0)
    ax_a.annotate(
        "", xy=p_hit, xytext=p_lab, textcoords="data",
        arrowprops=dict(arrowstyle="-|>", color=fs.ROSE, lw=0.8, mutation_scale=7),
        zorder=8,
    )
    ax_a.text(p_lab[0], p_lab[1], r"$P$", fontsize=FS, color=fs.ROSE, ha="center", va="bottom", zorder=9)
    # θ：靠外锥边的远扇
    th1 = S1 + 2150.0 * unit(THETAS[0] + 1.75 * DELTA_DRAW)
    th2 = S2 + 2100.0 * unit(THETAS[1] - 1.75 * DELTA_DRAW)
    ax_a.annotate(r"$\theta_1$", th1, fontsize=FS, color=COLORS[0], ha="center", va="center", zorder=9)
    ax_a.annotate(r"$\theta_2$", th2, fontsize=FS, color=COLORS[1], ha="center", va="center", zorder=9)
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
    segs_a = []
    for S, th in ((S1, THETAS[0]), (S2, THETAS[1])):
        for dang in (0.0, -DELTA_DRAW, DELTA_DRAW):
            segs_a.append((S, S + RAY_LEN * unit(th + dang)))
    check_ax_a_labels(ax_a, P_draw, [S1, S2, G], segs_a)
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


def _farthest_excess(verts, O, R):
    dist = np.linalg.norm(np.asarray(verts) - O, axis=1)
    k = int(np.argmax(dist))
    return verts[k], float(dist[k] - R)


def draw_case_g_mec():
    """图3：直径圆与最小包围圆，2×2。"""
    G0 = np.array([0.0, 0.0])
    Ss = [
        np.array([800.0, 0.0]),
        np.array([-400.0, 400.0 * np.sqrt(3.0)]),
        np.array([-400.0, -400.0 * np.sqrt(3.0)]),
    ]
    ths = [bearing_deg(S, G0) for S in Ss]
    rG = require_ok(Ss, ths, 1.0)
    rG8 = require_ok(Ss, ths, 8.0)
    rB = require_ok(SENSORS, THETAS, 1.0)

    A_e = np.array([0.0, 0.0])
    B_e = np.array([2.0, 0.0])
    C_e = np.array([1.0, np.sqrt(3.0)])
    tri = np.vstack([A_e, B_e, C_e])
    d_e, Ae, Be = polygon_diameter(tri)
    O_e = (Ae + Be) / 2.0
    R_e = d_e / 2.0
    C_mec_e, Rm_e = minimum_enclosing_circle(tri)

    fig, axes = plt.subplots(2, 2, figsize=(fs.W_IN, 4.9), layout="constrained")
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]
    fs_leg = 6.2

    # (a) 等边三角形
    draw_polygon(ax_a, tri, fs.FILL_WARN, fs.INK, 0.32, lw=1.3)
    ax_a.plot([Ae[0], Be[0]], [Ae[1], Be[1]], color=fs.PURPLE, lw=1.8, zorder=5)
    ax_a.add_patch(Circle(O_e, R_e, fill=False, ec=fs.PURPLE, lw=1.3, label="直径圆"))
    ax_a.add_patch(
        Circle(C_mec_e, Rm_e, fill=False, ec=fs.GREEN, lw=1.3, ls="--", label=r"最小包围圆")
    )
    ax_a.plot([O_e[0], C_e[0]], [O_e[1], C_e[1]], color=fs.ROSE, lw=1.0, ls="--", zorder=5)
    ax_a.plot(O_e[0], O_e[1], "+", color=fs.PURPLE, ms=8, mew=1.3, zorder=6)
    ax_a.plot(C_mec_e[0], C_mec_e[1], "+", color=fs.GREEN, ms=8, mew=1.3, zorder=6)
    for p, lab, off in [
        (A_e, r"$A$", (-10, -12)),
        (B_e, r"$B$", (6, -12)),
        (C_e, r"$C$", (5, 6)),
        (O_e, r"$O$", (7, -11)),
    ]:
        ax_a.plot(p[0], p[1], "o", color=fs.INK, ms=4.5, zorder=7)
        ax_a.annotate(lab, p, textcoords="offset points", xytext=off, fontsize=8, color=fs.INK)
    ax_a.annotate(
        r"$\sqrt{3}\,d/2$",
        (O_e + C_e) / 2.0,
        textcoords="offset points",
        xytext=(8, 2),
        fontsize=7.5,
        color=fs.ROSE,
        ha="left",
        va="center",
        zorder=8,
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.92),
    )
    ax_a.legend(loc="lower left", fontsize=fs_leg, handlelength=1.4, borderpad=0.25)
    ax_a.set_xlim(-0.45, 2.65)
    ax_a.set_ylim(-1.15, 2.20)
    fs.style_map(ax_a, title=r"(a) 等边三角形：$R_{\mathrm{MEC}}=d/\sqrt{3}$", xlabel="", ylabel="")
    ax_a.set_xlabel("")
    ax_a.set_ylabel("")
    ax_a.title.set_fontsize(8.0)

    # (b) 算例 G，±8°
    P8, D8, A8, B8 = rG8.vertices, rG8.diameter, rG8.A, rG8.B
    O8, R8 = rG8.center, rG8.radius
    C8, Rm8 = rG8.mec_center, rG8.mec_radius
    far8, gap8 = _farthest_excess(P8, O8, R8)
    draw_polygon(ax_b, P8, fs.FILL_P, fs.ROSE, 0.40, lw=1.2)
    ax_b.plot([A8[0], B8[0]], [A8[1], B8[1]], color=fs.PURPLE, lw=1.5, zorder=5)
    ax_b.add_patch(Circle(O8, R8, fill=False, ec=fs.PURPLE, lw=1.25, label="直径圆"))
    ax_b.add_patch(
        Circle(C8, Rm8, fill=False, ec=fs.GREEN, lw=1.2, ls="--", label=r"最小包围圆")
    )
    ax_b.plot(O8[0], O8[1], "+", color=fs.PURPLE, ms=7, mew=1.2, zorder=6)
    for p in P8:
        ax_b.plot(p[0], p[1], "o", color=fs.ROSE, ms=3.5, zorder=6)
    ax_b.plot(far8[0], far8[1], "o", color=fs.ROSE, ms=6.5, zorder=8)
    ax_b.plot([O8[0], far8[0]], [O8[1], far8[1]], color=fs.ROSE, lw=1.0, ls="--", zorder=5)
    ax_b.annotate(
        "缺口 {:.2f} m".format(gap8),
        xy=far8,
        xytext=(78, -155),
        textcoords="data",
        fontsize=7.0,
        color=fs.ROSE,
        ha="left",
        va="center",
        zorder=8,
        arrowprops=dict(arrowstyle="-|>", color=fs.ROSE, lw=0.8, mutation_scale=7),
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.92),
    )
    ax_b.plot(G0[0], G0[1], "o", color=fs.INK, ms=3.5, zorder=7)
    ax_b.annotate(r"$G$", G0, textcoords="offset points", xytext=(6, 8), fontsize=8)
    ax_b.legend(loc="lower left", fontsize=fs_leg, handlelength=1.4, borderpad=0.25)
    sb = max(float(np.max(np.abs(P8))), Rm8) * 1.22
    ax_b.set_xlim(-sb, sb)
    ax_b.set_ylim(-sb, sb)
    fs.style_map(
        ax_b,
        title=r"(b) 算例 G：$\pm 1^\circ$ 放大为 $\pm 8^\circ$，直径圆盖不住 $P$",
        xlabel="",
        ylabel="",
    )
    ax_b.set_xlabel("")
    ax_b.set_ylabel("")
    ax_b.title.set_fontsize(7.6)

    # (c) 算例 G 真实 ±1°
    P, D, A, B = rG.vertices, rG.diameter, rG.A, rG.B
    O, R = rG.center, rG.radius
    C, Rm = rG.mec_center, rG.mec_radius
    far, gap = _farthest_excess(P, O, R)
    draw_polygon(ax_c, P, fs.FILL_P, fs.ROSE, 0.40, lw=1.2)
    ax_c.plot([A[0], B[0]], [A[1], B[1]], color=fs.PURPLE, lw=1.5, zorder=5)
    ax_c.add_patch(Circle(O, R, fill=False, ec=fs.PURPLE, lw=1.25, label="直径圆"))
    ax_c.add_patch(
        Circle(C, Rm, fill=False, ec=fs.GREEN, lw=1.2, ls="--", label=r"最小包围圆")
    )
    ax_c.add_patch(Circle(C, 20.0, fill=False, ec=fs.SAND, lw=1.15, ls=":", label=r"$20$ m 清除圆"))
    ax_c.plot(O[0], O[1], "+", color=fs.PURPLE, ms=7, mew=1.2, zorder=6)
    ax_c.plot(C[0], C[1], "+", color=fs.GREEN, ms=7, mew=1.2, zorder=6)
    for p in P:
        ax_c.plot(p[0], p[1], "o", color=fs.ROSE, ms=3.5, zorder=6)
    ax_c.plot(far[0], far[1], "o", color=fs.ROSE, ms=6.5, zorder=8)
    ax_c.plot([O[0], far[0]], [O[1], far[1]], color=fs.ROSE, lw=1.0, ls="--", zorder=5)
    ax_c.annotate(
        "最远顶点超出 {:.3f} m".format(gap),
        xy=far,
        xytext=(21.8, 7.5),
        textcoords="data",
        fontsize=7.0,
        color=fs.ROSE,
        ha="left",
        va="bottom",
        zorder=8,
        arrowprops=dict(arrowstyle="-|>", color=fs.ROSE, lw=0.8, mutation_scale=7),
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.92),
    )
    ax_c.plot(G0[0], G0[1], "o", color=fs.INK, ms=3.5, zorder=7)
    ax_c.annotate(r"$G$", G0, textcoords="offset points", xytext=(-16, -12), fontsize=8)
    ax_c.legend(loc="upper left", fontsize=fs_leg, handlelength=1.4, borderpad=0.25)
    sc = 29.0
    ax_c.set_xlim(-sc, sc)
    ax_c.set_ylim(-sc, sc)
    fs.style_map(
        ax_c,
        title=rf"(c) 算例 G 真实 $\pm 1^\circ$：$R_{{\mathrm{{MEC}}}}={Rm:.3f}<20$ m，可清除",
        xlabel="x / m",
        ylabel="y / m",
    )
    ax_c.title.set_fontsize(7.6)

    # (d) 算例 B
    PB, DB, AB, BB = rB.vertices, rB.diameter, rB.A, rB.B
    OB, RB = rB.center, rB.radius
    CB, RmB = rB.mec_center, rB.mec_radius
    draw_polygon(ax_d, PB, fs.FILL_P, fs.ROSE, 0.40, lw=1.2)
    ax_d.plot([AB[0], BB[0]], [AB[1], BB[1]], color=fs.PURPLE, lw=1.5, zorder=5)
    ax_d.add_patch(
        Circle(OB, RB, fill=False, ec=fs.PURPLE, lw=1.25, label=r"直径圆 $R=d/2={:.2f}$ m".format(RB))
    )
    ax_d.add_patch(Circle(CB, 20.0, fill=False, ec=fs.SAND, lw=1.15, ls=":", label=r"$20$ m 清除圆"))
    ax_d.plot(OB[0], OB[1], "+", color=fs.PURPLE, ms=7, mew=1.2, zorder=6)
    for p in PB:
        ax_d.plot(p[0], p[1], "o", color=fs.ROSE, ms=3.5, zorder=6)
    ax_d.plot(G[0], G[1], "o", color=fs.INK, ms=3.5, zorder=7)
    ax_d.annotate(r"$G$", G, textcoords="offset points", xytext=(5, -11), fontsize=8)
    ax_d.legend(loc="lower left", fontsize=fs_leg, handlelength=1.4, borderpad=0.25)
    pad = 18
    xmin, ymin = PB.min(axis=0) - pad
    xmax, ymax = PB.max(axis=0) + pad
    ax_d.set_xlim(xmin, xmax)
    ax_d.set_ylim(ymin, ymax)
    fs.style_map(
        ax_d,
        title=rf"(d) 算例 B：覆盖，$R_{{\mathrm{{MEC}}}}={RmB:.3f}>20$ m，不可清除",
        xlabel="",
        ylabel="",
    )
    ax_d.set_xlabel("")
    ax_d.set_ylabel("")
    ax_d.title.set_fontsize(7.6)

    print(
        "fig3 G±1: d={:.3f} R_diam={:.3f} excess={:.3f} R_MEC={:.3f}".format(D, R, gap, Rm)
    )
    print(
        "fig3 G±8: d={:.2f} R_MEC={:.2f} gap={:.2f}".format(D8, Rm8, gap8)
    )
    print("fig3 B: d={:.3f} R_MEC={:.3f} covers={}".format(DB, RmB, rB.covers))
    print("fig3 E: d={:.3f} R_MEC={:.6f}".format(d_e, Rm_e))
    fs.save(fig, os.path.join(OUT, "fig_q1_mec.png"), size=(fs.W_IN, 4.9))


def half_plane_poly(xlim, ylim, origin, normal):
    """将闭半平面 (X-origin)·normal ≥ 0 裁成坐标框内的多边形。"""
    origin = np.asarray(origin, float)
    n = np.asarray(normal, float)
    n = n / (np.linalg.norm(n) + 1e-15)
    xs = [xlim[0], xlim[1], xlim[1], xlim[0]]
    ys = [ylim[0], ylim[0], ylim[1], ylim[1]]
    corners = [np.array([x, y]) for x, y in zip(xs, ys)]
    edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]

    def side(p):
        return float((p - origin) @ n)

    pts = []
    for a, b in edges:
        sa, sb = side(a), side(b)
        if sa >= -1e-12:
            pts.append(a)
        if sa * sb < 0:
            t = sa / (sa - sb)
            pts.append(a + t * (b - a))
    uniq = []
    for p in pts:
        if not any(np.linalg.norm(p - q) < 1e-8 for q in uniq):
            uniq.append(p)
    if len(uniq) < 3:
        return None
    c = np.mean(uniq, axis=0)
    ang = [np.arctan2(p[1] - c[1], p[0] - c[0]) for p in uniq]
    order = np.argsort(ang)
    return np.array([uniq[i] for i in order])


def _normal_gap_stats(origins, thetas, delta=8.0):
    planes = []
    for S, th in zip(origins, thetas):
        planes.extend(cone_halfplanes(S, th, delta))
    angs = np.sort([float(np.arctan2(h.normal[1], h.normal[0])) for h in planes])
    gaps = [float(x) for x in np.diff(angs)]
    gaps.append(float(angs[0] + 2.0 * np.pi - angs[-1]))
    kstar = int(np.argmax(gaps))
    gmax = gaps[kstar]
    if kstar < len(angs) - 1:
        astar = 0.5 * (angs[kstar] + angs[kstar + 1])
        a0, a1 = angs[kstar], angs[kstar + 1]
    else:
        astar = 0.5 * (angs[-1] + angs[0] + 2.0 * np.pi)
        a0, a1 = angs[-1], angs[0] + 2.0 * np.pi
    u = -np.array([np.cos(astar), np.sin(astar)])
    return planes, angs, gaps, gmax, astar, a0, a1, u, kstar


def _draw_gap_inset(ax, angs, gaps, gmax, a0, a1, u, unbounded, kstar):
    ins = ax.inset_axes([0.615, 0.535, 0.375, 0.44])
    ins.set_xlim(-1.45, 1.45)
    ins.set_ylim(-1.45, 1.45)
    ins.set_aspect("equal")
    ins.set_xticks([])
    ins.set_yticks([])
    for s in ins.spines.values():
        s.set_color(fs.GRID)
        s.set_linewidth(0.6)
    ins.set_facecolor("white")
    if unbounded:
        th_u = np.degrees(np.arctan2(u[1], u[0]))
        ins.add_patch(
            Wedge((0, 0), 1.42, th_u - 90.0, th_u + 90.0, facecolor=fs.PURPLE, alpha=0.12, lw=0, zorder=0)
        )
    ins.add_patch(Circle((0, 0), 1.0, fill=False, ec=fs.INK, lw=0.8, zorder=2))
    seq = list(angs) + [angs[0] + 2.0 * np.pi]
    for i, g in enumerate(gaps):
        t1, t2 = np.degrees(seq[i]), np.degrees(seq[i + 1])
        is_max = i == kstar
        ins.add_patch(
            Arc(
                (0, 0),
                2.0,
                2.0,
                theta1=t1,
                theta2=t2,
                color=fs.ROSE if is_max else fs.MUTED,
                lw=2.0 if is_max else 0.8,
                zorder=3,
            )
        )
    # 圆周上的短矢，避免四支从原点画出叠成两支
    for ang in angs:
        n = np.array([np.cos(ang), np.sin(ang)])
        ins.plot(n[0], n[1], "o", color=fs.PURPLE, ms=3.2, zorder=5)
        ins.annotate(
            "",
            xy=0.92 * n,
            xytext=0.42 * n,
            arrowprops=dict(arrowstyle="-|>", color=fs.PURPLE, lw=1.05, mutation_scale=7),
            zorder=4,
        )
    mid = 0.5 * (a0 + a1)
    lab = 1.30 * np.array([np.cos(mid), np.sin(mid)])
    ins.text(
        lab[0],
        lab[1],
        r"$g_{\max}$" + "\n" + r"${:.0f}^\circ$".format(np.degrees(gmax)),
        fontsize=7.5,
        ha="center",
        va="center",
        color=fs.ROSE,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=fs.GRID, lw=0.5),
        zorder=6,
    )


def draw_normal_gap():
    """图4：两锥主图 + 右上角单位圆插图。"""
    fig, axes = fs.new_fig(1, 2)
    ax, bx = axes
    cases = [
        (
            ax,
            [np.array([-420.0, 0.0]), np.array([420.0, 0.0])],
            [0.0, 180.0],
            False,
            [r"$S_1$", r"$S_2$"],
            [(-22, -20), (8, -20)],
        ),
        (
            bx,
            [np.array([-220.0, -80.0]), np.array([180.0, -80.0])],
            [8.0, 0.0],
            True,
            [r"$S_1$", r"$S_2$"],
            [(-22, -20), (8, -20)],
        ),
    ]
    gdegs = []
    for ax_i, origins, thetas, unbounded, labs, offs in cases:
        planes, angs, gaps, gmax, astar, a0, a1, u, kstar = _normal_gap_stats(origins, thetas, 8.0)
        gdegs.append(np.degrees(gmax))
        cols = [fs.BLUE, fs.GREEN]
        for S, th, col in zip(origins, thetas, cols):
            fill_cone(ax_i, S, th, 8.0, col, 900, 0.10)
            draw_cone_rays(ax_i, S, th, 8.0, col, 1100, lw_main=1.4, lw_edge=1.05)
            ax_i.plot(S[0], S[1], "o", color=col, ms=6.5, zorder=6)
        for S, lab, off, col in zip(origins, labs, offs, cols):
            ax_i.annotate(lab, S, textcoords="offset points", xytext=off, fontsize=9.5, color=col)
        for h in planes:
            uu = unit(h.dir_angle)
            dist = 560.0
            p0 = np.asarray(h.origin, float) + dist * uu
            # 右上角插图约覆盖 x>400、y>50，箭头缩回以免叠进框内
            if p0[0] > 420.0 and p0[1] > 8.0:
                p0 = np.asarray(h.origin, float) + 300.0 * uu
            nn = h.normal / (np.linalg.norm(h.normal) + 1e-15)
            ax_i.annotate(
                "",
                xy=p0 + 90.0 * nn,
                xytext=p0,
                arrowprops=dict(arrowstyle="-|>", color=fs.PURPLE, lw=1.0, mutation_scale=7),
                zorder=7,
            )
        if unbounded:
            poly = half_plane_poly((-700, 1100), (-700, 700), np.zeros(2), u)
            if poly is not None:
                ax_i.fill(poly[:, 0], poly[:, 1], color=fs.PURPLE, alpha=0.10, lw=0, zorder=0)
            ax_i.annotate(
                "",
                xy=960.0 * u,
                xytext=240.0 * u,
                arrowprops=dict(arrowstyle="-|>", color=fs.ROSE, lw=1.45, mutation_scale=10),
                zorder=8,
            )
            perp = np.array([u[1], -u[0]])
            if perp[1] > 0:
                perp = -perp
            tp = 0.5 * (240.0 + 960.0) * u + 150.0 * perp
            # 锥面占 y≈0 一带，文字再下移以免压 S2 与母线
            tp = np.array([tp[0], min(tp[1], -320.0)])
            ax_i.text(
                tp[0],
                tp[1],
                r"沿公共方向 $\mathbf{u}$ 走向无穷",
                fontsize=7.5,
                color=fs.ROSE,
                ha="center",
                va="top",
                zorder=9,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.92),
            )
        _draw_gap_inset(ax_i, angs, gaps, gmax, a0, a1, u, unbounded, kstar)
        handle = Line2D([0], [0], color=fs.PURPLE, lw=1.4, label=r"内法向 $\mathbf{n}_i$")
        ax_i.legend(handles=[handle], loc="lower left", fontsize=7.5, handlelength=1.2, borderpad=0.3)
        ax_i.set_xlim(-700, 1100)
        ax_i.set_ylim(-700, 700)
        gdeg = np.degrees(gmax)
        if unbounded:
            title = r"(b) 近乎同向：$g_{\max}=" + "{:.0f}".format(gdeg) + r"^\circ$≥$180^\circ$，无界"
        else:
            title = r"(a) 对向交会：$g_{\max}=" + "{:.0f}".format(gdeg) + r"^\circ<180^\circ$，有界"
        fs.style_map(ax_i, title=title)
        ax_i.title.set_fontsize(9.5)
        print("fig4 g_max = {:.3f} deg, unbounded={}".format(gdeg, unbounded))
    fs.save(fig, os.path.join(OUT, "fig_q1_unbounded.png"))


if __name__ == "__main__":
    draw_form_and_diameter()
    draw_cover_and_counter()
    draw_case_g_mec()
    draw_normal_gap()
