"""问题1 算法流程图（论文用）。"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")


def box(ax, xy, w, h, text, kind="proc"):
    x, y = xy
    styles = {
        "start": dict(fc="#f4f4f4", ec="#222222", rad=0.45),
        "proc": dict(fc="#ffffff", ec="#222222", rad=0.04),
        "io": dict(fc="#eef6fb", ec="#222222", rad=0.04),
        "ok": dict(fc="#e8f5e9", ec="#2e7d32", rad=0.08),
        "fail": dict(fc="#fdecea", ec="#c62828", rad=0.08),
    }
    s = styles[kind]
    if kind == "start":
        p = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle=f"round,pad=0.02,rounding_size={s['rad']}",
            linewidth=1.2, facecolor=s["fc"], edgecolor=s["ec"],
        )
    else:
        p = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="square,pad=0.0",
            linewidth=1.2, facecolor=s["fc"], edgecolor=s["ec"],
        )
        if kind in ("ok", "fail"):
            p.set_boxstyle("round,pad=0.02,rounding_size=0.12")
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=9.2, color="#111111", linespacing=1.25)
    return (x, y)


def diamond(ax, xy, w, h, text):
    x, y = xy
    verts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
    p = Polygon(verts, closed=True, facecolor="#fff8e1", edgecolor="#222222", linewidth=1.2)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=9.0, color="#111111")
    return (x, y)


def arrow(ax, p1, p2, text=None, text_side="right"):
    ax.annotate(
        "", xy=p2, xytext=p1,
        arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1.05, mutation_scale=10),
    )
    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        dx = 0.18 if text_side == "right" else -0.18
        ax.text(mx + dx, my, text, fontsize=8.2, color="#333333", ha="left" if text_side == "right" else "right", va="center")


def main():
    fig, ax = plt.subplots(figsize=(7.6, 10.2))
    ax.set_xlim(-2.35, 2.55)
    ax.set_ylim(-0.45, 10.15)
    ax.set_aspect("equal")
    ax.axis("off")

    # 主链 x=0，失败分支在右侧
    s = box(ax, (0, 9.7), 1.35, 0.42, "开始", "start")
    inn = box(ax, (0, 9.05), 2.55, 0.50, "输入：检测点 $S_i$、示向度 $\\theta_i$、误差限 $\\delta$", "io")
    dN = diamond(ax, (0, 8.22), 1.70, 0.72, "$N<2$？")
    u1 = box(ax, (1.85, 8.22), 1.20, 0.48, "UNBOUNDED\n（须加检测点）", "fail")

    p2 = box(ax, (0, 7.28), 2.70, 0.55, "由 $\\theta_i\\pm\\delta$ 构造 $2N$ 条边界直线\n及对应半平面", "proc")
    dU = diamond(ax, (0, 6.42), 1.95, 0.78, "公共区域无界？\n（法向最大间隙$\\geq 180^\\circ$）")
    u2 = box(ax, (1.90, 6.42), 1.15, 0.48, "UNBOUNDED", "fail")

    p3 = box(ax, (0, 5.52), 2.70, 0.55, "直线两两求交，保留满足全部\n半平面约束的交点，得顶点集 $V$", "proc")
    dE = diamond(ax, (0, 4.58), 1.95, 0.78, "顶点集为空？")
    e1 = box(ax, (1.90, 4.58), 1.15, 0.48, "EMPTY\n（定位失败）", "fail")

    p7 = box(ax, (0, 3.58), 2.55, 0.48, "按质心极角排序，得到凸多边形 $P$", "proc")
    p8 = box(ax, (0, 2.82), 2.70, 0.50, "枚举顶点对，直径 $D=\\max\\|u-v\\|$，端点 $A,B$", "proc")
    p9 = box(ax, (0, 2.08), 2.70, 0.50, "圆心 $O=(A+B)/2$，$R=D/2$\n并求最小包围圆", "proc")
    dC = diamond(ax, (0, 1.22), 2.05, 0.78, "全部顶点到 $O$\n的距离 $\\leq R$？")
    yes = box(ax, (-1.15, 0.38), 1.35, 0.42, "直径圆覆盖 $P$", "ok")
    no = box(ax, (1.15, 0.38), 1.45, 0.42, "直径圆不覆盖 $P$", "fail")
    out = box(ax, (0, 0.0), 3.05, 0.42, "输出：OK，$V$，$D$，覆盖判定，MEC", "io")

    arrow(ax, (s[0], s[1] - 0.21), (inn[0], inn[1] + 0.25))
    arrow(ax, (inn[0], inn[1] - 0.25), (dN[0], dN[1] + 0.36))
    arrow(ax, (dN[0] + 0.85, dN[1]), (u1[0] - 0.60, u1[1]), "是", "right")
    arrow(ax, (dN[0], dN[1] - 0.36), (p2[0], p2[1] + 0.275), "否", "right")
    arrow(ax, (p2[0], p2[1] - 0.275), (dU[0], dU[1] + 0.39))
    arrow(ax, (dU[0] + 0.975, dU[1]), (u2[0] - 0.575, u2[1]), "是", "right")
    arrow(ax, (dU[0], dU[1] - 0.39), (p3[0], p3[1] + 0.275), "否", "right")
    arrow(ax, (p3[0], p3[1] - 0.275), (dE[0], dE[1] + 0.39))
    arrow(ax, (dE[0] + 0.975, dE[1]), (e1[0] - 0.575, e1[1]), "是", "right")
    arrow(ax, (dE[0], dE[1] - 0.39), (p7[0], p7[1] + 0.24), "否", "right")
    arrow(ax, (p7[0], p7[1] - 0.24), (p8[0], p8[1] + 0.25))
    arrow(ax, (p8[0], p8[1] - 0.25), (p9[0], p9[1] + 0.225))
    arrow(ax, (p9[0], p9[1] - 0.25), (dC[0], dC[1] + 0.39))
    ax.annotate(
        "", xy=(-1.15, 0.59), xytext=(-0.55, 0.95),
        arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1.05, mutation_scale=10,
                        connectionstyle="angle,angleA=-90,angleB=180,rad=0"),
    )
    ax.text(-1.55, 0.92, "是", fontsize=8.2, color="#333333")
    ax.annotate(
        "", xy=(1.15, 0.59), xytext=(0.55, 0.95),
        arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1.05, mutation_scale=10,
                        connectionstyle="angle,angleA=-90,angleB=0,rad=0"),
    )
    ax.text(1.42, 0.92, "否", fontsize=8.2, color="#333333")
    arrow(ax, (-1.15, 0.17), (out[0] - 0.70, out[1] + 0.12))
    arrow(ax, (1.15, 0.17), (out[0] + 0.70, out[1] + 0.12))

    ax.set_title("算法 1  交会定位区域直径与圆覆盖", fontsize=13, pad=8)
    ax.text(
        0.0, -0.22,
        "本问仅计算示向度误差锥的角度交会区域。无界由法向间隙判定，有界后再枚举交点。",
        ha="center", va="top", fontsize=8.5, color="#333333",
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, "fig_algorithm_flow.png")
    pdf = os.path.join(OUT_DIR, "fig_algorithm_flow.pdf")
    fig.tight_layout()
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print("saved", png, pdf)


if __name__ == "__main__":
    main()
