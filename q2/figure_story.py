# -*- coding: utf-8 -*-
"""问题2 必讲图：流程、候选楔形、局部坐标、交会角对比、候选蝶形、长轴切割。

对应论文阅读顺序，不另画 12 张附图。设计图 (t,D) 热力图与 E[T] 曲线见 design_chart.py。
"""
import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import (
    DELTA, wedge_hp, region, diam_and_axis, mec, k_expr, clip,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

K = 40.0 / (2 * np.tan(DELTA))
C_LEVER = 0.5 / np.sqrt(0.75)
D_NEAR, D_FAR = 5.0, 1500.0


def save(fig, name):
    png = os.path.join(OUT, name)
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(png.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    print("saved", png, os.path.getsize(png), "bytes")
    plt.close(fig)


def unit(ang):
    return np.array([np.cos(ang), np.sin(ang)], float)


def wedge_hp_delta(S, th, delta):
    return [
        (S, np.array([np.cos(th - delta), np.sin(th - delta)]), +1),
        (S, np.array([np.cos(th + delta), np.sin(th + delta)]), -1),
    ]


def clip_disk(poly, center, radius, n=72):
    """用外切正多边形近似圆盘裁剪。"""
    p = np.asarray(poly, float)
    c = np.asarray(center, float)
    for k in range(n):
        ang = 2 * np.pi * k / n
        e = unit(ang)
        # 切点 Q = c + R e，保留 cross(e, P-Q) 与 inward 一致
        # 要 (P-c)·e <= R。用 clip: 以切点为顶点、切向为 e 的旋转。
        Q = c + radius * e
        # geometry.clip: sign * cross(e_dir, P-S) >= 0
        # 取 e_dir = (-e_y, e_x)（逆时针切向），S=Q，sign=+1
        # cross(e_dir, P-Q) = -e_y (P_x-Q_x) wait:
        # e_dir = (-e[1], e[0]); cross = e_dir_x * (Py-Qy) - e_dir_y * (Px-Qx)
        # = -e[1](Py-Qy) - e[0](Px-Qx) = - e·(P-Q) = R - e·(P-c)
        # 我们要 e·(P-c) <= R 即 R - e·(P-c) >= 0，即 cross >= 0。正好。
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


def rays(ax, S, th, delta, length, color, lw=1.3):
    S = np.asarray(S, float)
    ax.plot(*zip(S, S + length * unit(th)), color=color, lw=lw + 0.4, zorder=5)
    for d in (-delta, delta):
        ax.plot(*zip(S, S + length * unit(th + d)), color=color, lw=lw, ls="--", zorder=5)


# ---------------------------------------------------------------------------
# 图 1  六步流程
# ---------------------------------------------------------------------------
def box(ax, xy, w, h, text, kind="proc", fs=8.6):
    x, y = xy
    styles = {
        "start": dict(fc="#f4f4f4", ec="#222222", rad=0.4),
        "proc": dict(fc="#ffffff", ec="#222222", rad=0.04),
        "io": dict(fc="#eef6fb", ec="#1f4e79", rad=0.04),
        "ok": dict(fc="#e8f5e9", ec="#2e7d32", rad=0.10),
        "note": dict(fc="#fff8e1", ec="#8d6e00", rad=0.06),
    }
    s = styles[kind]
    p = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={s['rad']}",
        linewidth=1.15, facecolor=s["fc"], edgecolor=s["ec"],
    )
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color="#111111", linespacing=1.28)
    return (x, y)


def diamond(ax, xy, w, h, text, fs=8.4):
    x, y = xy
    verts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
    ax.add_patch(Polygon(verts, closed=True, facecolor="#fff8e1", edgecolor="#222222", lw=1.15))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color="#111111")
    return (x, y)


def arr(ax, p1, p2, text=None, side="right"):
    ax.annotate(
        "", xy=p2, xytext=p1,
        arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1.05, mutation_scale=9),
    )
    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        dx = 0.16 if side == "right" else -0.16
        ax.text(mx + dx, my, text, fontsize=7.8, color="#333",
                ha="left" if side == "right" else "right", va="center")


def fig_flow():
    fig, ax = plt.subplots(figsize=(8.2, 9.4))
    ax.set_xlim(-2.55, 2.65)
    ax.set_ylim(-0.35, 9.55)
    ax.set_aspect("equal")
    ax.axis("off")

    s = box(ax, (0, 9.20), 1.20, 0.38, "开始", "start")
    p1 = box(ax, (0, 8.52), 2.55, 0.48, "第一步  S1 测向，得示向度 θ1", "io")
    p2 = box(ax, (0, 7.68), 2.85, 0.62,
             "第二步  候选集 W = 楔形(θ1±1°)\n"
             "∩ 圆盘(S1, 1500) ∩ 圆盘(O, 1800)", "proc")
    p3 = box(ax, (0, 6.72), 2.85, 0.58,
             "第三步  局部坐标 (t, h) 选 S2\n"
             "t：沿示向度的赌注　　h：侧偏", "proc")
    p4 = box(ax, (0, 5.78), 2.90, 0.62,
             "第四步  S2 测向，两楔形求交\n"
             "精确：凸多边形（L1）  一阶：平行四边形（L2）", "proc")
    p5 = box(ax, (0, 4.85), 2.70, 0.50, "第五步  算直径 D_loc 与最小覆盖圆 MEC", "proc")
    d = diamond(ax, (0, 3.88), 2.05, 0.78, "MEC 半径 ≤ 20 m？")
    yes = box(ax, (-1.45, 2.55), 1.70, 0.62, "第六步（能）\n走到 MEC 圆心 /clear", "ok", fs=8.3)
    no = box(ax, (1.45, 2.55), 1.85, 0.62, "第六步（不能）\n长轴切割，追加 S_new", "note", fs=8.3)
    loop = box(ax, (1.45, 1.45), 1.85, 0.48, "回到第五步，再判 MEC", "proc", fs=8.2)
    out = box(ax, (0, 0.35), 2.40, 0.42, "输出：定位清除完成", "io")

    arr(ax, (s[0], s[1] - 0.19), (p1[0], p1[1] + 0.24))
    arr(ax, (p1[0], p1[1] - 0.24), (p2[0], p2[1] + 0.31))
    arr(ax, (p2[0], p2[1] - 0.31), (p3[0], p3[1] + 0.29))
    arr(ax, (p3[0], p3[1] - 0.29), (p4[0], p4[1] + 0.31))
    arr(ax, (p4[0], p4[1] - 0.31), (p5[0], p5[1] + 0.25))
    arr(ax, (p5[0], p5[1] - 0.25), (d[0], d[1] + 0.39))
    ax.annotate(
        "", xy=(-1.45, 2.86), xytext=(-0.55, 3.55),
        arrowprops=dict(arrowstyle="-|>", color="#222", lw=1.05, mutation_scale=9,
                        connectionstyle="angle,angleA=-90,angleB=180,rad=0"),
    )
    ax.text(-1.72, 3.35, "是", fontsize=8.0, color="#2e7d32")
    ax.annotate(
        "", xy=(1.45, 2.86), xytext=(0.55, 3.55),
        arrowprops=dict(arrowstyle="-|>", color="#222", lw=1.05, mutation_scale=9,
                        connectionstyle="angle,angleA=-90,angleB=0,rad=0"),
    )
    ax.text(1.55, 3.35, "否", fontsize=8.0, color="#8d6e00")
    arr(ax, (no[0], no[1] - 0.31), (loop[0], loop[1] + 0.24))
    ax.annotate(
        "", xy=(1.05, 4.85), xytext=(1.45, 1.69),
        arrowprops=dict(arrowstyle="-|>", color="#222", lw=1.0, mutation_scale=9,
                        connectionstyle="arc3,rad=0.35"),
    )
    arr(ax, (yes[0], yes[1] - 0.31), (out[0] - 0.55, out[1] + 0.21))

    ax.set_title("问题 2　单目标第二检测点：六步决策链", fontsize=13, pad=8)
    ax.text(
        0, -0.12,
        "终判据用问题 1 的多边形（L1）与 Welzl/枚举 MEC；平行四边形只用于 L2 设计公式。",
        ha="center", va="top", fontsize=8.2, color="#444",
    )
    save(fig, "fig_q2_flow.png")


# ---------------------------------------------------------------------------
# 图 2  几何故事：W、局部坐标、γ 曲线、三交会区
# ---------------------------------------------------------------------------
def panel_W(ax):
    O = np.array([0.0, 0.0])
    S1 = np.array([-620.0, 380.0])
    th = np.deg2rad(18.0)
    # 示意张角放大，否则 2° 在全场上看不见
    d_draw = np.deg2rad(7.0)
    ax.add_patch(Circle(O, 1800, fill=False, ec="#888", lw=1.2, ls="-", zorder=2))
    ax.add_patch(Circle(S1, 1500, fill=False, ec="#1f77b4", lw=1.0, ls="--", zorder=2))
    ax.add_patch(Circle(O, 1800, fc="#eeeeee", ec="none", alpha=0.35, zorder=0))

    sec = sector_poly(S1, th, d_draw, 1500, n=48)
    W = clip_disk(sec, O, 1800, n=80)
    fill_poly(ax, W, "#c62828", "#b71c1c", alpha=0.28, lw=1.2, z=3)
    rays(ax, S1, th, d_draw, 1650, "#b71c1c", lw=1.15)

    ax.plot(*S1, "o", color="#1565c0", ms=7, zorder=6)
    ax.plot(0, 0, "k+", ms=8, zorder=6)
    ax.annotate("S1", S1 + np.array([-90, 70]), fontsize=10, color="#1565c0")
    ax.annotate("O", (40, -80), fontsize=10)
    ax.text(80, -1680, "目标圆 1800 m", fontsize=8.2, color="#555")
    ax.text(S1[0] - 560, S1[1] + 920, "接收 1500 m", fontsize=8.2, color="#1565c0")
    ax.text(
        1100, 280,
        "候选集 W\n方向已知、距离未知",
        fontsize=9.0, color="#7f1d1d", ha="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#c62828", alpha=0.9),
    )
    ax.text(
        0.03, 0.97, "楔形张角示意放大至 ±7°（真实 ±1°）",
        transform=ax.transAxes, fontsize=7.4, color="#666", va="top",
    )
    ax.set_xlim(-1950, 1950)
    ax.set_ylim(-1950, 1950)
    ax.set_aspect("equal")
    ax.set_title("(a) 第一次测向后的候选集 W", fontsize=10)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.tick_params(labelsize=8)


def panel_coords(ax):
    """局部 (t,h)：测得轴、真实轴偏 ε、S2、未知 D。张角放大以便看见楔形。"""
    d_draw = np.deg2rad(8.0)
    eps = np.deg2rad(8.0)  # 示意放大；真实 ε∈[-1°,1°]
    S1 = np.array([0.0, 0.0])
    t, h = 500.0, 250.0
    S2 = np.array([t, h])
    D = 900.0
    G = D * unit(eps)

    ax.axhline(0, color="#bbb", lw=0.6)
    ax.axvline(0, color="#bbb", lw=0.6)
    ax.annotate(
        "", xy=(1300, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.4, mutation_scale=12),
    )
    ax.text(1280, -70, "t（沿测得示向度）", fontsize=8.5, ha="right")
    ax.annotate(
        "", xy=(0, 720), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.4, mutation_scale=12),
    )
    ax.text(30, 700, "h（侧偏）", fontsize=8.5)

    rays(ax, S1, 0.0, d_draw, 1250, "#c62828", lw=1.0)
    ax.plot([0, 1250 * np.cos(eps)], [0, 1250 * np.sin(eps)], color="#2e7d32", lw=1.3, ls="-.")
    ax.plot(*G, "k*", ms=11, zorder=7)
    ax.plot(*S2, "o", color="#1565c0", ms=8, zorder=7)
    ax.plot(*S1, "o", color="#333", ms=7, zorder=7)
    ax.annotate("S1", (-40, -90), fontsize=10)
    ax.text(t + 28, h + 28, "S2 = (t, h)", fontsize=9.5, color="#1565c0", clip_on=True)
    ax.annotate("G", G + np.array([16, -28]), fontsize=10)
    ax.plot([t, t], [0, h], color="#1565c0", lw=0.9, ls=":")
    ax.plot([0, t], [h, h], color="#1565c0", lw=0.9, ls=":")
    ax.text(t / 2, -55, "t", color="#1565c0", fontsize=9, ha="center")
    ax.text(t + 25, h / 2, "h", color="#1565c0", fontsize=9)
    ax.text(380, 40, "测得轴", fontsize=8, color="#c62828")
    ax.text(980, 200, "真实方向（偏 ε）", fontsize=8, color="#2e7d32")
    ax.text(220, 380, "交会角 γ = ∠S1GS2", fontsize=8.2, color="#333")

    ax.set_xlim(-180, 1400)
    ax.set_ylim(-280, 780)
    ax.set_aspect("equal")
    ax.set_title("(b) 局部坐标：赌注 t 与侧偏 h", fontsize=10)
    ax.set_xlabel("t / m")
    ax.set_ylabel("h / m")
    ax.tick_params(labelsize=8)


def panel_gamma_curve(ax):
    d1 = d2 = 800.0
    gam = np.linspace(12, 168, 320)
    gr = np.deg2rad(gam)
    num = d1 * d1 + d2 * d2 + 2 * d1 * d2 * np.abs(np.cos(gr))
    Dloc = 2 * DELTA * np.sqrt(num / np.sin(gr) ** 2)
    ax.plot(gam, Dloc, color="#1f4e79", lw=2.0)
    for g, col, mk in [(30, "#c62828", "o"), (90, "#2e7d32", "s"), (150, "#e65100", "D")]:
        i = np.argmin(np.abs(gam - g))
        ax.plot(g, Dloc[i], mk, color=col, ms=8, zorder=5)
        ax.annotate(f"{g}°\n{Dloc[i]:.1f} m", (g, Dloc[i] + 8), ha="center", fontsize=8, color=col)
    ax.axhline(40, color="#c62828", ls="--", lw=0.9, alpha=0.7)
    ax.text(20, 43, "40 m 两步成败线", fontsize=7.6, color="#c62828")
    ax.set_xlim(10, 170)
    ax.set_ylim(0, 160)
    ax.set_xlabel("交会角 γ / °")
    ax.set_ylabel("D_loc / m")
    ax.set_title("(c) d1 = d2 = 800 m 时直径随 γ", fontsize=10)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)


def two_station_poly(S1, S2, G, delta=DELTA):
    th1 = np.arctan2(G[1] - S1[1], G[0] - S1[0])
    th2 = np.arctan2(G[1] - S2[1], G[0] - S2[0])
    return region(wedge_hp_delta(S1, th1, delta) + wedge_hp_delta(S2, th2, delta), box=5000.0)


def panel_gamma_shape(ax, gamma_deg, tag):
    """G 在原点，两站等距 800 m，夹角 gamma。真实 ±1°，视窗放大到交会区。"""
    d = 800.0
    G = np.array([0.0, 0.0])
    S1 = d * unit(np.pi)                 # (-800, 0)
    S2 = d * unit(np.pi - np.deg2rad(gamma_deg))
    v = two_station_poly(S1, S2, G, DELTA)
    fill_poly(ax, v, "#bbdefb", "#1565c0", alpha=0.55, lw=1.5, z=4)
    rays(ax, S1, np.arctan2(G[1] - S1[1], G[0] - S1[0]), DELTA, 1100, "#c62828", lw=0.8)
    rays(ax, S2, np.arctan2(G[1] - S2[1], G[0] - S2[0]), DELTA, 1100, "#2e7d32", lw=0.8)
    ax.plot(*S1, "o", color="#c62828", ms=6)
    ax.plot(*S2, "o", color="#2e7d32", ms=6)
    ax.plot(*G, "k*", ms=8)
    if len(v) >= 3:
        diam, _axv = diam_and_axis(v)
        _r, C = mec(v)
        ax.plot(*C, "+", color="#6a1b9a", ms=8)
        ax.text(
            0.03, 0.03, f"D_loc = {diam:.1f} m",
            transform=ax.transAxes, fontsize=8.0, color="#333",
        )
    ax.set_aspect("equal")
    # 视窗：交会区尺度约数十米
    span = 55 if gamma_deg == 90 else 95
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_title(f"{tag}  γ = {gamma_deg:.0f}°", fontsize=10)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("x / m", fontsize=8)
    ax.set_ylabel("y / m", fontsize=8)


def fig_geometry():
    fig = plt.figure(figsize=(12.4, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.12, 1.0], hspace=0.32, wspace=0.28)
    axW = fig.add_subplot(gs[0, 0])
    axC = fig.add_subplot(gs[0, 1])
    axE = fig.add_subplot(gs[0, 2])
    ax30 = fig.add_subplot(gs[1, 0])
    ax90 = fig.add_subplot(gs[1, 1])
    ax150 = fig.add_subplot(gs[1, 2])
    panel_W(axW)
    panel_coords(axC)
    panel_gamma_curve(axE)
    panel_gamma_shape(ax30, 30, "(d)")
    panel_gamma_shape(ax90, 90, "(e)")
    panel_gamma_shape(ax150, 150, "(f)")
    fig.suptitle(
        "问题 2  几何故事：W、决策坐标、交会角（(d)(e)(f) 为真实 ±1°，视窗放大到交会区）",
        fontsize=12, y=0.98,
    )
    save(fig, "fig_q2_geometry.png")


# ---------------------------------------------------------------------------
# 图 3  候选区域蝶形
# ---------------------------------------------------------------------------
def lever_h(t):
    t = np.asarray(t, float)
    return C_LEVER * np.maximum(np.abs(D_FAR - t), np.abs(t - D_NEAR))


def fig_candidate():
    ts = np.linspace(50, 1450, 220)
    hs = np.linspace(-900, 900, 241)
    T, H = np.meshgrid(ts, hs)
    lev = np.abs(H) >= lever_h(T)
    # 命中赌注 D=t、ε=0 时的 K-条件（示意“核心带”）
    k_hit = np.full(T.shape, np.inf)
    for i in range(0, T.shape[0], 1):
        for j in range(T.shape[1]):
            t, h = T[i, j], H[i, j]
            if abs(h) < 20:
                continue
            k_hit[i, j] = k_expr(t, h, t, 0.0)
    k_ok = k_hit <= (K * K)

    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    ax.add_patch(plt.Circle((0, 0), 1200, fill=False, ls="--", ec="#888", lw=1.0))

    mask_lever = lev & (T ** 2 + H ** 2 <= 1200 ** 2)
    mask_core = mask_lever & k_ok
    Z = np.zeros(T.shape)
    Z[mask_lever] = 1
    Z[mask_core] = 2
    Zm = np.ma.masked_where(Z == 0, Z)
    ax.pcolormesh(T, H, Zm, cmap=matplotlib.colors.ListedColormap(["#bbdefb", "#81c784"]),
                  vmin=1, vmax=2, shading="auto", alpha=0.85, zorder=1)
    ax.plot(ts, lever_h(ts), color="#1565c0", lw=1.6, zorder=2)
    ax.plot(ts, -lever_h(ts), color="#1565c0", lw=1.6, zorder=2)

    schools = [
        (550, 498, "minimax\n(550, 498)", "#c62828", (-92, 14)),
        (675, 476, "概率最优\n(675, 476)", "#6a1b9a", (-18, 34)),
        (750, 450, "保守\n(750, 450)", "#2e7d32", (28, -38)),
        (500, 250, "时间流派\n(500, 250)", "#e65100", (14, -58)),
    ]
    for t, h, name, col, off in schools:
        ax.plot(t, h, "o", color=col, ms=8, zorder=6)
        ax.plot(t, -h, "o", color=col, ms=6, mfc="white", zorder=6)
        ax.annotate(name, (t, h), xytext=off, textcoords="offset points",
                    fontsize=7.6, color=col)

    ax.axhline(0, color="#999", lw=0.6)
    ax.axvline(0, color="#999", lw=0.6)
    ax.set_xlim(0, 1450)
    ax.set_ylim(-850, 850)
    ax.set_aspect("equal")
    ax.set_xlabel("t / m（沿示向度）")
    ax.set_ylabel("h / m（侧偏）")
    ax.set_title("第二检测点候选区域 C（蝶形）", fontsize=12)
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#bbdefb", ec="#1565c0", label="杠杆条件（探距带）"),
        plt.Rectangle((0, 0), 1, 1, fc="#81c784", ec="#2e7d32", label="叠加：命中赌注时满足 K-条件"),
        Line2D([0], [0], color="#1565c0", lw=1.6, label="|h| = 0.577 max(|1500−t|, |t−5|)"),
        Line2D([0], [0], color="#888", lw=1.0, ls="--", label="移动预算 1200 m"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    ax.annotate(
        "探距带外：远目标不保证两步成功，靠长轴切割兜底",
        xy=(500, 250), xytext=(160, 360),
        textcoords="data", fontsize=8.0, color="#e65100",
        arrowprops=dict(arrowstyle="-|>", color="#e65100", lw=0.9),
        bbox=dict(boxstyle="round,pad=0.25", fc="#fff3e0", ec="#e65100", alpha=0.95),
    )
    save(fig, "fig_q2_candidate.png")


# ---------------------------------------------------------------------------
# 图 4  长轴切割
# ---------------------------------------------------------------------------
def fig_rescue():
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

    # 两图共用视窗，对照收缩；S2 离区域太远，不拉进坐标轴
    focus = np.vstack([v2, C2.reshape(1, 2), G.reshape(1, 2), Sn.reshape(1, 2)])
    xmin, xmax = focus[:, 0].min(), focus[:, 0].max()
    ymin, ymax = focus[:, 1].min(), focus[:, 1].max()
    pad = 0.35 * max(xmax - xmin, ymax - ymin, 80)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2))
    for ax, v, r, C, title, extra in [
        (axes[0], v2, r2, C2, f"(a) 两次测向后　　直径 {diam2:.1f} m，MEC {r2:.1f} m", True),
        (axes[1], v3, r3, C3, f"(b) 长轴切割后再测　　直径 {diam3:.1f} m，MEC {r3:.1f} m", False),
    ]:
        fill_poly(ax, v, "#bbdefb", "#1565c0", alpha=0.55, lw=1.6)
        ax.add_patch(Circle(C, r, fill=False, ec="#6a1b9a", lw=1.3, ls="--", label="MEC"))
        ax.add_patch(Circle(C, 20, fill=False, ec="#2e7d32", lw=1.1, ls=":", label="20 m 清除圆"))
        ax.plot(*C, "+", color="#6a1b9a", ms=10, zorder=6)
        ax.plot(*G, "k*", ms=10, zorder=6)
        ax.annotate("G", G + np.array([4, 6]), fontsize=9)
        if extra:
            p = C2 + (diam2 / 2) * axv
            q = C2 - (diam2 / 2) * axv
            ax.plot([p[0], q[0]], [p[1], q[1]], color="#c62828", lw=1.6, label="长轴")
            ax.annotate(
                "", xy=Sn, xytext=C2,
                arrowprops=dict(arrowstyle="-|>", color="#e65100", lw=1.5, mutation_scale=11),
            )
            ax.plot(*Sn, "s", color="#e65100", ms=8, zorder=7)
            ax.annotate("新测点 S_new", Sn + np.array([6, 8]),
                        fontsize=8.5, color="#e65100")
            ax.annotate("C", C2 + np.array([6, -12]), fontsize=9, color="#6a1b9a")
            ax.legend(loc="upper right", fontsize=8)
            ax.text(
                0.03, 0.04, "S2 = (700, 462) 在图外左上",
                transform=ax.transAxes, fontsize=8.0, color="#1565c0",
            )
        else:
            ax.plot(*Sn, "s", color="#e65100", ms=7)
            ok = "≤20 m，可清除" if r3 <= 20 else ">20 m"
            ax.text(0.04, 0.04, f"MEC {ok}", transform=ax.transAxes,
                    fontsize=9, color="#2e7d32" if r3 <= 20 else "#c62828",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#888"))
        ax.set_aspect("equal")
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.tick_params(labelsize=8)

    fig.suptitle(
        "长轴切割：D = 1500 m，赌注 (700, 462)，ρ = max(50, 0.5 L)　绿点线为 20 m 清除圆",
        fontsize=12,
    )
    print(f"rescue numbers: diam2={diam2:.2f} r2={r2:.2f} rho={rho:.1f} diam3={diam3:.2f} r3={r3:.2f}")
    save(fig, "fig_q2_rescue.png")


def main():
    fig_flow()
    fig_geometry()
    fig_candidate()
    fig_rescue()


if __name__ == "__main__":
    main()
