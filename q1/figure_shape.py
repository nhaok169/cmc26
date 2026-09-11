"""问题1示意：交会定位区域的形状、直径与圆覆盖。

顶点、直径、圆覆盖一律调用 q1/solver.py，避免示意图与求解结果不一致。
本文件只负责绘图。图中把测向误差从 ±1° 放大到 ±8°，便于看清形状。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from solver import bearing_deg, locate

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False


def unit(ang_deg):
    t = np.radians(ang_deg)
    return np.array([np.cos(t), np.sin(t)])


def require_ok(sensors, thetas, delta):
    r = locate(sensors, thetas, delta)
    if r.status != "OK" or r.vertices is None:
        raise RuntimeError("locate 失败：{} {}".format(r.status, r.message))
    return r


# ========================= 场景（示意用，非题面数据） =========================
# 三个检测点围住同一个未知源；示向度按“指向 G”生成，交会区一定含 G。
G = np.array([80.0, 220.0])
S1 = np.array([-920.0, -480.0])
S2 = np.array([980.0, -420.0])
S3 = np.array([-40.0, 1180.0])
SENSORS = [S1, S2, S3]
THETAS = [bearing_deg(S, G) for S in SENSORS]

DELTA_TRUE = 1.0   # 题面真实误差
DELTA_DRAW = 8.0   # 画形状用的放大误差（1° 在全图上几乎是一条线）

COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e"]  # S1蓝 S2绿 S3橙


def ray_segment(S, ang, length):
    e = np.asarray(S, float) + length * unit(ang)
    return np.asarray(S, float), e


def draw_cone_rays(ax, S, theta, delta, color, length, lw_main=1.8, lw_edge=1.1):
    p0, p1 = ray_segment(S, theta, length)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw_main, zorder=3)
    for dang in (-delta, delta):
        q0, q1 = ray_segment(S, theta + dang, length)
        ax.plot(
            [q0[0], q1[0]], [q0[1], q1[1]],
            color=color, lw=lw_edge, ls="--", alpha=0.9, zorder=3,
        )


def fill_cone(ax, S, theta, delta, color, radius, alpha=0.10):
    """用多边形填充误差锥（避免 Wedge 角度跨 0/360 的坑）。"""
    n = 48
    angs = np.linspace(theta - delta, theta + delta, n)
    pts = [S] + [S + radius * unit(a) for a in angs]
    pts = np.array(pts)
    ax.fill(pts[:, 0], pts[:, 1], color=color, alpha=alpha, lw=0, zorder=1)


def draw_polygon(ax, poly, fc, ec, alpha, lw=2.0, zorder=4, label=None):
    if len(poly) < 3:
        return
    closed = np.vstack([poly, poly[0]])
    ax.fill(
        closed[:, 0], closed[:, 1],
        fc=fc, ec=ec, alpha=alpha, lw=lw, zorder=zorder, label=label,
    )
    ax.plot(closed[:, 0], closed[:, 1], color=ec, lw=lw, zorder=zorder + 1)


def mark_vertices(ax, poly, color="crimson", prefix="P"):
    c = poly.mean(axis=0)
    for i, p in enumerate(poly):
        ax.plot(p[0], p[1], "o", color=color, ms=5.5, zorder=6)
        v = p - c
        n = np.linalg.norm(v) + 1e-9
        ax.annotate(
            f"${prefix}_{{{i+1}}}$",
            p,
            textcoords="offset points",
            xytext=(14 * v[0] / n, 14 * v[1] / n),
            fontsize=10,
            color=color,
            zorder=7,
            ha="center",
            va="center",
        )


def mark_sensor(ax, S, name, color, text_offset):
    ax.plot(S[0], S[1], "o", color=color, ms=9, zorder=8)
    ax.annotate(
        name, S, textcoords="offset points", xytext=text_offset,
        fontsize=13, fontweight="bold", color=color, zorder=8,
    )


def style_ax(ax, title, xlim, ylim, xlabel=False, ylabel=False):
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if xlabel:
        ax.set_xlabel("x / m")
    if ylabel:
        ax.set_ylabel("y / m")
    ax.set_title(title, fontsize=11, pad=6)
    ax.grid(True, ls=":", alpha=0.45)
    ax.set_facecolor("#fafafa")


# ========================= 计算各情形的定位区域（与 solver.py 同一套几何） =========================
r2 = require_ok(SENSORS[:2], THETAS[:2], DELTA_DRAW)
r3 = require_ok(SENSORS[:3], THETAS[:3], DELTA_DRAW)
r2_true = require_ok(SENSORS[:2], THETAS[:2], DELTA_TRUE)
P2, P3, P2_true = r2.vertices, r3.vertices, r2_true.vertices
D2, A2, B2 = r2.diameter, r2.A, r2.B
covers2, O2, R2 = r2.covers, r2.center, r2.radius

print("==== 放大误差 ±{}° 下的形状 ====".format(DELTA_DRAW))
print("2 个检测点: {} 边形, 直径 = {:.1f} m, 直径圆覆盖? {}".format(
    len(P2), D2, "是" if covers2 else "否"))
print("3 个检测点: {} 边形, 直径 = {:.1f} m".format(len(P3), r3.diameter))
print("真实 ±1° 两站交会: {} 边形, 直径 = {:.1f} m （更细长）".format(
    len(P2_true), r2_true.diameter))


# ========================= 绘图 =========================
fig, axes = plt.subplots(2, 2, figsize=(11.4, 10.2), constrained_layout=True)
ax_a, ax_b, ax_c, ax_d = axes.ravel()
RAY_LEN = 2600

# ---- (a) 1 个检测点：无限误差锥 ----
fill_cone(ax_a, S1, THETAS[0], DELTA_DRAW, COLORS[0], radius=2400, alpha=0.16)
draw_cone_rays(ax_a, S1, THETAS[0], DELTA_DRAW, COLORS[0], RAY_LEN)
mark_sensor(ax_a, S1, r"$S_1$", COLORS[0], (-70, -36))
ax_a.plot(G[0], G[1], marker="*", color="k", ms=13, zorder=8)
ax_a.annotate("真实源 $G$（未知，仅示意）", G, textcoords="offset points",
              xytext=(10, -18), fontsize=9)
# 锥角标注
mid = S1 + 620 * unit(THETAS[0])
ax_a.annotate(
    "误差锥\n宽度 $2\\delta=16^\\circ$（放大）\n真实仅为 $2^\\circ$",
    xy=mid, fontsize=9, ha="center",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1f77b4", alpha=0.92),
)
style_ax(
    ax_a,
    "(a) $N=1$：无界误差锥，无法定义直径",
    (-1600, 1400), (-1100, 1500),
    ylabel=True,
)

# ---- (b) 2 个检测点：四边形 ----
fill_cone(ax_b, S1, THETAS[0], DELTA_DRAW, COLORS[0], 2400, 0.08)
fill_cone(ax_b, S2, THETAS[1], DELTA_DRAW, COLORS[1], 2400, 0.08)
draw_cone_rays(ax_b, S1, THETAS[0], DELTA_DRAW, COLORS[0], RAY_LEN)
draw_cone_rays(ax_b, S2, THETAS[1], DELTA_DRAW, COLORS[1], RAY_LEN)
draw_polygon(ax_b, P2, fc="#d62728", ec="#9b1b1b", alpha=0.38, lw=2.2)
mark_vertices(ax_b, P2)
mark_sensor(ax_b, S1, r"$S_1$", COLORS[0], (-78, -36))
mark_sensor(ax_b, S2, r"$S_2$", COLORS[1], (12, -36))
ax_b.plot(G[0], G[1], marker="*", color="k", ms=13, zorder=8)
ax_b.annotate("两锥交会区域 $P$\n（对应题面图2的红四边形）",
              xy=P2.mean(axis=0), fontsize=9, ha="center",
              xytext=(0, 70), textcoords="offset points",
              arrowprops=dict(arrowstyle="->", color="#9b1b1b"),
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#9b1b1b", alpha=0.92))
style_ax(
    ax_b,
    "(b) $N=2$：交会成凸四边形（可退化为三角形/线段/点）",
    (-1600, 1600), (-1100, 1500),
)

# ---- (c) 3 个检测点：本示例切成六边形 ----
fill_cone(ax_c, S1, THETAS[0], DELTA_DRAW, COLORS[0], 2400, 0.06)
fill_cone(ax_c, S2, THETAS[1], DELTA_DRAW, COLORS[1], 2400, 0.06)
fill_cone(ax_c, S3, THETAS[2], DELTA_DRAW, COLORS[2], 2400, 0.10)
draw_cone_rays(ax_c, S1, THETAS[0], DELTA_DRAW, COLORS[0], RAY_LEN, lw_main=1.2, lw_edge=0.8)
draw_cone_rays(ax_c, S2, THETAS[1], DELTA_DRAW, COLORS[1], RAY_LEN, lw_main=1.2, lw_edge=0.8)
draw_cone_rays(ax_c, S3, THETAS[2], DELTA_DRAW, COLORS[2], RAY_LEN)
# 先画旧四边形虚线轮廓，再画新六边形
closed2 = np.vstack([P2, P2[0]])
ax_c.plot(closed2[:, 0], closed2[:, 1], color="#9b1b1b", lw=1.4, ls=":", alpha=0.8,
          label="2 站时的四边形", zorder=4)
draw_polygon(ax_c, P3, fc="#d62728", ec="#9b1b1b", alpha=0.45, lw=2.2)
mark_vertices(ax_c, P3)
mark_sensor(ax_c, S3, r"$S_3$", COLORS[2], (10, 12))
ax_c.plot(G[0], G[1], marker="*", color="k", ms=13, zorder=8)
ax_c.legend(loc="lower right", fontsize=9, framealpha=0.92)
ax_c.annotate(
    "本示例中橙色锥切掉四边形两角\n得到凸六边形（一般至多 2n 边）",
    xy=P3[np.argmax(P3[:, 1])],
    fontsize=9, ha="center",
    xytext=(0.70, 0.88), textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", color="#9b1b1b"),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ff7f0e", alpha=0.92),
)
# 放大到四边形附近，六条边才分得清；橙色虚线即 S3 的误差边界
pc = np.vstack([P2, P3])
style_ax(
    ax_c,
    "(c) $N=3$：本示例中切成六边形（一般至多 $2N$ 边）",
    (pc[:, 0].min() - 140, pc[:, 0].max() + 180),
    (pc[:, 1].min() - 100, pc[:, 1].max() + 160),
    xlabel=True, ylabel=True,
)

# ---- (d) 直径 + 圆覆盖（局部放大四边形） ----
pad_x, pad_y = 150, 110
xmin, ymin = P2.min(axis=0) - np.array([pad_x, pad_y])
xmax, ymax = P2.max(axis=0) + np.array([90, pad_y])
# 局部也把三条边画一点，帮助认方向
draw_polygon(ax_d, P2, fc="#d62728", ec="#9b1b1b", alpha=0.28, lw=2.4, label="定位区域 $P$")
ax_d.plot([A2[0], B2[0]], [A2[1], B2[1]], color="#6a3d9a", lw=2.8, zorder=5, label="直径 $AB$（最远两顶点）")
ax_d.plot(A2[0], A2[1], "o", color="#6a3d9a", ms=8, zorder=6)
ax_d.plot(B2[0], B2[1], "o", color="#6a3d9a", ms=8, zorder=6)
ax_d.annotate("$A$", A2, textcoords="offset points", xytext=(-16, 6), fontsize=14, color="#6a3d9a")
ax_d.annotate("$B$", B2, textcoords="offset points", xytext=(8, 6), fontsize=14, color="#6a3d9a")
others = [p for p in P2 if np.linalg.norm(p - A2) > 1e-6 and np.linalg.norm(p - B2) > 1e-6]
for lab, p, off in zip(["C", "D"], others, [(-14, 8), (8, -16)]):
    ax_d.plot(p[0], p[1], "o", color="crimson", ms=6, zorder=6)
    ax_d.annotate(f"${lab}$", p, textcoords="offset points", xytext=off, fontsize=13, color="crimson")
    ax_d.plot([O2[0], p[0]], [O2[1], p[1]], color="#6a3d9a", lw=0.9, ls="--", alpha=0.7)
circ = Circle(O2, R2, fill=False, ec="#6a3d9a", lw=2.0, ls="-", label="以 $AB$ 为直径的圆")
ax_d.add_patch(circ)
ax_d.plot(O2[0], O2[1], "+", color="#6a3d9a", ms=12, mew=2, zorder=6)
ax_d.annotate("$O$", O2, textcoords="offset points", xytext=(8, -14), fontsize=12, color="#6a3d9a")
ax_d.plot(G[0], G[1], marker="*", color="k", ms=13, zorder=8)
cover_txt = "能覆盖 P" if covers2 else "不能覆盖 P"
ax_d.annotate(
    "只需检查 C、D 是否都在圆内\n"
    "P 凸、圆也凸：顶点在内则整块都在内\n"
    f"本例：{cover_txt}",
    xy=O2, fontsize=9, ha="left",
    xytext=(-195, 78), textcoords="offset points",
    bbox=dict(boxstyle="round,pad=0.35", fc="#f3e8ff", ec="#6a3d9a", alpha=0.95),
)
ax_d.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
style_ax(
    ax_d,
    "(d) 直径取最远顶点对；再检验直径圆是否覆盖 $P$",
    (xmin, xmax), (ymin, ymax),
    xlabel=True,
)

fig.suptitle(
    "交会定位区域的形状、直径与圆覆盖（示意图）",
    fontsize=14, fontweight="bold",
)
fig.text(
    0.5, -0.01,
    "注：为显示形状，将测向误差由题面 $\\pm 1^\\circ$ 放大为 $\\pm 8^\\circ$；真实区域更细长。"
    "本图只画角度交会区域，不叠加半径 1800 m 目标圆域。"
    "$N=1$ 为无界锥，$N=2$ 多为四边形；$N=3$ 在本示例中为六边形，一般边数不超过 $2N$。",
    ha="center", va="top", fontsize=9.0, color="#333333",
)

out_dir = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(out_dir, exist_ok=True)
paper_png = os.path.join(out_dir, "fig_geometry.png")
paper_pdf = os.path.join(out_dir, "fig_geometry.pdf")
fig.savefig(paper_png, dpi=300, bbox_inches="tight")
fig.savefig(paper_pdf, bbox_inches="tight")
print("saved", paper_png)
