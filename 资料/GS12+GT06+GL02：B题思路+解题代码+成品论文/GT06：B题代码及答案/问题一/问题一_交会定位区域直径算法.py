# -*- coding: utf-8 -*-
"""
问题一：交会定位区域直径算法与直径圆覆盖性
------------------------------------------------------------------
输入：若干检测点坐标 S_i 及各点测得的示向度 theta_i
输出：交会定位区域（凸多边形）顶点、直径、以及"以直径为直径的圆能否覆盖定位区域"的判定

算法组成：
  1) 楔形 -> 半平面组等价转化（避免角度跨 0 度不连续）
  2) Sutherland-Hodgman 逐次半平面裁剪求交
  3) 凸多边形直径：暴力枚举 + 旋转卡壳 双实现互验
  4) 直径圆覆盖判据：Thales 逆定理（凸四边形充要条件）
  5) 大规模随机构型的蒙特卡洛统计（两种口径：纯交会 / 加目标圆域截断）
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import font_manager
from matplotlib.patches import Circle, Polygon as MplPolygon
import warnings
warnings.filterwarnings('ignore')

# ========== 全局样式（必须先设置：style 会重置 font.sans-serif） ==========
plt.style.use('seaborn-v0_8-whitegrid')

# ========== 中文字体配置（强制，确保图例中文正常显示；置于 style 之后） ==========
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti SC', 'Heiti TC', 'PingFang SC',
    'PingFang HK', 'Hiragino Sans GB', 'Songti SC', 'STHeiti', 'Arial Unicode MS',
    'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.edgecolor'] = '#CCCCCC'
for fname in font_manager.findSystemFonts():
    try:
        f = font_manager.FontProperties(fname=fname)
        if any(kw in f.get_name() for kw in ['Hei', 'Song', 'Ming', 'Fang', 'Kai',
                                             'SimSun', 'SimHei', 'PingFang', 'STHeiti']):
            font_manager.fontManager.addfont(fname)
    except Exception:
        pass

CMAP_SEQUENTIAL = plt.cm.viridis
CMAP_CATEGORICAL = plt.cm.tab10
CMAP_DIVERGING = plt.cm.RdBu_r

COLORS_10 = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
             '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD']

BASE = '/Users/rzn/Desktop/IP计划/CUMCM2026Problems(4)/B-1/求解'
OUT_DIR = BASE + '/问题一/'
FIG_DIR = OUT_DIR + 'figures/'
os.makedirs(FIG_DIR, exist_ok=True)


def save_fig(fig, name_cn):
    fig.savefig(FIG_DIR + name_cn)
    plt.close(fig)


def save_csv(df, name_cn):
    df.to_csv(OUT_DIR + name_cn, index=False, encoding='utf-8-sig')


def apply_mpl_style(fig, despine=True):
    ax = fig.gca()
    if despine:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.4, color='#AAAAAA')
    ax.tick_params(labelsize=10)
    fig.tight_layout()


def set_legend_cn(ax, **kwargs):
    return ax.legend(prop={'family': plt.rcParams['font.sans-serif']}, **kwargs)


# ==================================================================
# 全局常量（题目给定）
# ==================================================================
R_TARGET = 1800.0            # 目标区域半径 (m)
DELTA_DEG = 1.0              # 示向度误差上界 (度)
DELTA = np.deg2rad(DELTA_DEG)
BOX_HALF = 20000.0           # 初始裁剪盒半宽 (m)，远大于目标区域，用于识别"无界"


# ==================================================================
# 1. 楔形 -> 半平面组
# ==================================================================
def wedge_halfplanes(S, theta_deg, delta_deg=DELTA_DEG):
    """把位于 S、示向度 theta 的楔形表示为两个半平面约束。

    楔形 = {S + rho*(cos a, sin a): rho >= 0, |a - theta| <= delta}
         = 两条边界射线所张的凸锥。

    边界射线方位角 a 的左法向为 n = (-sin a, cos a)，对点 X 令 d = X - S，
    则 d 处于该射线逆时针一侧 <=> d.n >= 0。于是
        锥 = {d : d.n(theta-delta) >= 0 且 d.n(theta+delta) <= 0}
    该写法对任意 theta 都无需处理 0/360 度的角度回绕。

    返回 [(S, n_low, +1), (S, n_high, -1)]，符号 +1 表示 d.n >= 0，-1 表示 d.n <= 0。
    """
    a_low = np.deg2rad(theta_deg - delta_deg)
    a_high = np.deg2rad(theta_deg + delta_deg)
    n_low = np.array([-np.sin(a_low), np.cos(a_low)])
    n_high = np.array([-np.sin(a_high), np.cos(a_high)])
    return [(np.asarray(S, float), n_low, +1), (np.asarray(S, float), n_high, -1)]


def region_halfplanes(points, bearings_deg, delta_deg=DELTA_DEG, cap_circle=True):
    """组装定位区域的全部半平面约束。

    cap_circle=True 时附加目标圆域 D 的内接 720 边形约束（保证区域有界）。
    """
    H = []
    for S, th in zip(points, bearings_deg):
        H.extend(wedge_halfplanes(S, th, delta_deg))

    m = BOX_HALF
    ex, ey = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    H.extend([(np.array([-m, 0.0]), ex, +1),   # x >= -m
              (np.array([m, 0.0]), ex, -1),    # x <=  m
              (np.array([0.0, -m]), ey, +1),   # y >= -m
              (np.array([0.0, m]), ey, -1)])   # y <=  m
    if cap_circle:
        # 目标圆域 D 的内接 720 边形（顶点落在圆上，相对误差 1-cos(pi/720) ~ 1e-5）
        for a in np.linspace(0, 2 * np.pi, 720, endpoint=False):
            e = np.array([np.cos(a), np.sin(a)])
            H.append((R_TARGET * e, e, -1))    # (X - R e).e <= 0  <=>  X.e <= R
    return H


# ==================================================================
# 2. Sutherland-Hodgman 逐次半平面裁剪
# ==================================================================
def _clip_polygon(poly, S, n, sign):
    """用半平面 {X: sign*(X-S).n >= 0} 裁剪凸多边形 poly（顶点数组 Nx2）。"""
    if len(poly) == 0:
        return np.zeros((0, 2))

    def signed(P):
        return sign * float(np.dot(P - S, n))

    out = []
    h = len(poly)
    for k in range(h):
        A = poly[k]
        B = poly[(k + 1) % h]
        sA, sB = signed(A), signed(B)
        inA, inB = sA >= -1e-9, sB >= -1e-9
        if inA:
            out.append(A)
            if not inB:
                denom = sA - sB
                t = sA / denom if abs(denom) > 1e-18 else 0.0
                out.append(A + np.clip(t, 0.0, 1.0) * (B - A))
        elif inB:
            denom = sA - sB
            t = sA / denom if abs(denom) > 1e-18 else 0.0
            out.append(A + np.clip(t, 0.0, 1.0) * (B - A))
    if len(out) == 0:
        return np.zeros((0, 2))
    return np.array(out)


def intersection_region(points, bearings_deg, delta_deg=DELTA_DEG, cap_circle=True):
    """求交会定位区域：2m 个楔形半平面（及目标圆域）的交多边形。"""
    poly = np.array([[-BOX_HALF, -BOX_HALF], [BOX_HALF, -BOX_HALF],
                     [BOX_HALF, BOX_HALF], [-BOX_HALF, BOX_HALF]], float)
    for (S, n, sgn) in region_halfplanes(points, bearings_deg, delta_deg, cap_circle):
        poly = _clip_polygon(poly, S, n, sgn)
        if len(poly) == 0:
            return np.zeros((0, 2))
    return poly


# ==================================================================
# 3. 直径：暴力枚举 与 旋转卡壳
# ==================================================================
def convex_hull(P):
    """Andrew 单调链，去除共线冗余点。"""
    P = np.asarray(P, float)
    if len(P) <= 3:
        return P
    P = np.unique(np.round(P, 9), axis=0)
    if len(P) <= 3:
        return P

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    P = P[np.lexsort((P[:, 1], P[:, 0]))]
    lower = []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)
    upper = []
    for p in P[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def diameter_bruteforce(poly):
    """顶点对暴力枚举求直径。返回 (直径, 端点对)。O(h^2)。"""
    if len(poly) == 0:
        return np.nan, (None, None)
    if len(poly) == 1:
        return 0.0, (poly[0], poly[0])
    P = np.asarray(poly, float)
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    return float(D[i, j]), (P[i], P[j])


def diameter_rotating_calipers(poly):
    """旋转卡壳求凸多边形直径。返回 (直径, 端点对)。O(h)。"""
    H = convex_hull(poly)
    h = len(H)
    if h == 0:
        return np.nan, (None, None)
    if h == 1:
        return 0.0, (H[0], H[0])
    if h == 2:
        return float(np.linalg.norm(H[1] - H[0])), (H[0], H[1])

    def area2(i, j, k):
        return abs((H[j][0] - H[i][0]) * (H[k][1] - H[i][1])
                   - (H[j][1] - H[i][1]) * (H[k][0] - H[i][0]))

    best, pair = -1.0, (None, None)
    j = 1
    for i in range(h):
        ni = (i + 1) % h
        while True:
            nj = (j + 1) % h
            if area2(i, ni, nj) > area2(i, ni, j):
                j = nj
            else:
                break
        for a, b in [(i, j), (ni, j)]:
            d = float(np.linalg.norm(H[a] - H[b]))
            if d > best:
                best, pair = d, (H[a], H[b])
    return best, pair


def min_enclosing_circle(poly):
    """凸多边形最小包围圆（由 2 点或 3 点确定），返回 (半径, 圆心)。"""
    P = convex_hull(poly)
    if len(P) == 0:
        return np.nan, None
    if len(P) == 1:
        return 0.0, P[0]

    def in_circle(c, r, pts):
        return np.all(np.linalg.norm(pts - c, axis=1) <= r + 1e-9)

    best_r, best_c = np.inf, None
    n = len(P)
    for i in range(n):
        for j in range(i + 1, n):
            c = (P[i] + P[j]) / 2.0
            r = float(np.linalg.norm(P[i] - P[j]) / 2.0)
            if r < best_r and in_circle(c, r, P):
                best_r, best_c = r, c
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                A, B, C = P[i], P[j], P[k]
                d = 2.0 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
                if abs(d) < 1e-12:
                    continue
                ux = ((A[0]**2 + A[1]**2) * (B[1] - C[1]) + (B[0]**2 + B[1]**2) * (C[1] - A[1])
                      + (C[0]**2 + C[1]**2) * (A[1] - B[1])) / d
                uy = ((A[0]**2 + A[1]**2) * (C[0] - B[0]) + (B[0]**2 + B[1]**2) * (A[0] - C[0])
                      + (C[0]**2 + C[1]**2) * (B[0] - A[0])) / d
                c = np.array([ux, uy])
                r = float(np.linalg.norm(A - c))
                if r < best_r and in_circle(c, r, P):
                    best_r, best_c = r, c
    return best_r, best_c


# ==================================================================
# 4. 直径圆覆盖判据
# ==================================================================
def diameter_circle_covers(poly):
    """判定以定位区域直径为直径的圆能否覆盖该区域。

    对凸多边形 P：设最长顶点距离端点为 A、C，直径圆为以 AC 为直径的圆，
    则 P 被覆盖 <=> 全部顶点落在该圆内。
    对凸四边形可等价地用 Thales 逆定理表述：
        点 X 在以 AC 为直径的圆内 <=> (X-A).(X-C) <= 0 <=> angle AXC >= 90 度。
    """
    H = convex_hull(poly)
    if len(H) == 0:
        return dict(diameter=np.nan, center=None, radius=np.nan, covered=False,
                    n_vertices=0, worst_vertex_dist=np.nan, worst_excess=np.nan,
                    criterion=None)
    D, (A, C) = diameter_bruteforce(H)
    center = (A + C) / 2.0
    radius = D / 2.0
    dists = np.linalg.norm(H - center, axis=1)
    worst = float(dists.max())
    crit = None
    if len(H) == 4:
        iA = int(np.argmin(np.linalg.norm(H - A, axis=1)))
        iC = int(np.argmin(np.linalg.norm(H - C, axis=1)))
        crit = {int(k): float(np.dot(H[k] - A, H[k] - C))
                for k in range(4) if k not in (iA, iC)}
    return dict(diameter=D, center=center, radius=radius,
                covered=bool(worst <= radius + 1e-9),
                n_vertices=len(H), worst_vertex_dist=worst,
                worst_excess=float(worst - radius), criterion=crit)


def is_degenerate(poly):
    """判断定位区域是否触及初始裁剪盒（即未加目标圆域约束时无界）。"""
    if len(poly) == 0:
        return False
    return bool(np.any(np.abs(poly) >= BOX_HALF - 1e-6))


# ==================================================================
# 5. 单构型完整求解
# ==================================================================
def solve_configuration(points, bearings_deg, delta_deg=DELTA_DEG, cap_circle=True):
    poly = intersection_region(points, bearings_deg, delta_deg, cap_circle=cap_circle)
    if len(poly) == 0:
        return dict(empty=True, poly=poly)
    res = diameter_circle_covers(poly)
    d_bf, _ = diameter_bruteforce(convex_hull(poly))
    d_rc, _ = diameter_rotating_calipers(poly)
    r_mec, c_mec = min_enclosing_circle(poly)
    res.update(empty=False, poly=poly, d_bruteforce=d_bf, d_rotating=d_rc,
               min_enclosing_radius=r_mec, mec_center=c_mec,
               ratio_mec_over_diameter=(r_mec / d_bf if d_bf and d_bf > 1e-12 else np.nan))
    return res


# ==================================================================
# 6. 蒙特卡洛统计
# ==================================================================
def monte_carlo(n_cases=5000, seed=20260913, delta_deg=DELTA_DEG, cap_circle=True):
    """随机生成 2 检测点 + 1 干扰源构型，统计定位区域与直径圆覆盖情况。

    示向度 = 真实方位角 + 服从 U[-1,1] 度的固定环境偏差（同一检测点误差固定）。
    因真实源必落在两楔形交内，故定位区域总是包含真值，构型均可解释。
    """
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_cases):
        def sample_disk(n=1):
            ang = 2 * np.pi * rng.random(n)
            rad = R_TARGET * np.sqrt(rng.random(n))
            return np.c_[rad * np.cos(ang), rad * np.sin(ang)]

        S = sample_disk(2)
        G = sample_disk(1)[0]
        true_b = np.rad2deg(np.arctan2(G[1] - S[:, 1], G[0] - S[:, 0])) % 360.0
        eps = rng.uniform(-delta_deg, delta_deg, size=2)
        meas_b = (true_b + eps) % 360.0

        base = dict(测点1_x=float(S[0, 0]), 测点1_y=float(S[0, 1]),
                    测点2_x=float(S[1, 0]), 测点2_y=float(S[1, 1]),
                    源_x=float(G[0]), 源_y=float(G[1]),
                    示向度1=float(meas_b[0]), 示向度2=float(meas_b[1]))

        res = solve_configuration(S, meas_b, delta_deg=delta_deg, cap_circle=cap_circle)
        if res.get('empty', False):
            rows.append(dict(序号=t + 1, 是否为空集=True, 是否退化=None, 直径=np.nan,
                             是否被覆盖=None, 最小包围圆半径=np.nan,
                             包围圆半径与直径之比=np.nan, 超出量=np.nan, 交会角=np.nan,
                             源到测点1距离=np.nan, **base))
            continue

        degenerate = is_degenerate(intersection_region(S, meas_b, delta_deg, cap_circle=False))
        v1, v2 = G - S[0], G - S[1]
        cosang = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
        alpha = float(np.rad2deg(np.arccos(np.clip(cosang, -1, 1))))
        rows.append(dict(
            序号=t + 1, 是否为空集=False, 是否退化=bool(degenerate),
            直径=res['diameter'], 是否被覆盖=bool(res['covered']),
            最小包围圆半径=res['min_enclosing_radius'],
            包围圆半径与直径之比=res['ratio_mec_over_diameter'],
            超出量=max(0.0, res['worst_excess']),   # 最远顶点超出直径圆的距离 (m)
            交会角=alpha, 源到测点1距离=float(np.linalg.norm(v1)), **base))
    return pd.DataFrame(rows)


def sensitivity_delta(deltas=(0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0), n_cases=1500, seed=7):
    """测向精度灵敏度：覆盖率与超出量随示向度误差上界 delta 的变化。"""
    rows = []
    for d in deltas:
        mc = monte_carlo(n_cases=n_cases, seed=seed, delta_deg=d, cap_circle=False)
        v = mc[(mc['是否为空集'] == False) & (mc['是否退化'] == False)]
        rows.append(dict(误差上界=float(d),
                         有效构型数=len(v),
                         覆盖率=float(v['是否被覆盖'].mean() * 100),
                         直径均值=float(v['直径'].mean()),
                         超出量均值=float(v['超出量'].mean()),
                         超出量最大值=float(v['超出量'].max())))
    return pd.DataFrame(rows)


# ==================================================================
# 主流程
# ==================================================================
def main():
    print('=' * 72)
    print('问题一：交会定位区域直径算法与直径圆覆盖性')
    print('=' * 72)

    # ---------------- 6.1 示范构型 ----------------
    S_demo = np.array([[0.0, 0.0], [1200.0, 0.0]])
    G_demo = np.array([900.0, 700.0])
    true_b = np.rad2deg(np.arctan2(G_demo[1] - S_demo[:, 1],
                                   G_demo[0] - S_demo[:, 0])) % 360.0
    meas_b = (true_b + np.array([0.6, -0.8])) % 360.0
    demo = solve_configuration(S_demo, meas_b, cap_circle=False)
    print(f"\n[示范构型] S1={S_demo[0]}, S2={S_demo[1]}, 真值 G={G_demo}")
    print(f"  真实方位角 = {np.round(true_b, 3)} 度, 示向度 = {np.round(meas_b, 3)} 度")
    print(f"  定位区域顶点数 = {demo['n_vertices']}")
    print(f"  直径 = {demo['diameter']:.4f} m  (旋转卡壳 {demo['d_rotating']:.4f} m, "
          f"双算法偏差 {abs(demo['diameter'] - demo['d_rotating']):.2e} m)")
    print(f"  最小包围圆半径 = {demo['min_enclosing_radius']:.4f} m, "
          f"半径/直径 = {demo['ratio_mec_over_diameter']:.4f}  (Jung 上界 0.5774)")
    print(f"  直径圆是否覆盖: {'是' if demo['covered'] else '否'}")
    print(f"  Thales 判据符号量 (X-A).(X-C) = "
          f"{ {k: round(v, 3) for k, v in demo['criterion'].items()} }  (均<=0 即覆盖)")

    # ---------------- 6.2 蒙特卡洛：两种口径 ----------------
    mc_pure = monte_carlo(n_cases=5000, cap_circle=False)     # 纯交会四边形
    mc_cap = monte_carlo(n_cases=5000, cap_circle=True)       # 加目标圆域截断

    rows_desc = []
    for tag, mc in [('纯交会四边形', mc_pure), ('加目标圆域截断', mc_cap)]:
        nondeg = mc[(mc['是否为空集'] == False) & (mc['是否退化'] == False)]
        rows_desc.append(dict(
            口径=tag, 总构型数=len(mc),
            非空构型占比=len(mc[mc['是否为空集'] == False]) / len(mc) * 100,
            无界退化占比=mc['是否退化'].mean() * 100,
            有效构型数=len(nondeg),
            直径圆覆盖率=nondeg['是否被覆盖'].mean() * 100,
            直径均值=nondeg['直径'].mean(), 直径中位数=nondeg['直径'].median(),
            直径标准差=nondeg['直径'].std(),
            包围圆半径与直径之比均值=nondeg['包围圆半径与直径之比'].mean(),
            包围圆半径与直径之比最大值=nondeg['包围圆半径与直径之比'].max(),
        ))
    desc2 = pd.DataFrame(rows_desc)
    print('\n[两种口径下的蒙特卡洛统计]')
    print(desc2.round(4).to_string(index=False))

    # 以纯交会口径为主口径，输出分组表与描述统计
    valid = mc_pure[(mc_pure['是否为空集'] == False) & (mc_pure['是否退化'] == False)].copy()
    print(f"\n[主口径：纯交会四边形] 有效(有界非空)构型 {len(valid)}/{len(mc_pure)} 组")
    print(f"  直径圆覆盖率   = {valid['是否被覆盖'].mean()*100:.3f}%")
    print(f"  覆盖失败构型数 = {int((~valid['是否被覆盖'].astype(bool)).sum())}")

    bins = [0, 30, 45, 60, 75, 90, 105, 120, 150, 180]
    valid['交会角分组'] = pd.cut(valid['交会角'], bins=bins)
    g = valid.groupby('交会角分组', observed=True).agg(
        构型数=('序号', 'count'),
        直径均值=('直径', 'mean'),
        直径中位数=('直径', 'median'),
        覆盖率=('是否被覆盖', 'mean'),
        包围圆半径与直径之比均值=('包围圆半径与直径之比', 'mean'),
    ).reset_index()
    g['覆盖率'] = (g['覆盖率'] * 100).round(4)
    g = g.rename(columns={'交会角分组': '交会角区间(度)'})
    g['交会角区间(度)'] = g['交会角区间(度)'].astype(str)
    save_csv(g, '覆盖率统计表.csv')
    print('\n[覆盖率按交会角分组 · 纯交会口径]')
    print(g.round(4).to_string(index=False))

    desc = pd.DataFrame({
        '统计量': ['有效构型数', '非空构型占比(%)', '无界退化构型占比(%)', '直径圆覆盖率(%)',
                 '直径均值(m)', '直径中位数(m)', '直径标准差(m)',
                 '直径最大值(m)', '直径最小值(m)',
                 '包围圆半径/直径 均值', '包围圆半径/直径 最大值', 'Jung 定理上界 1/sqrt(3)'],
        '数值': [len(valid),
               len(mc_pure[mc_pure['是否为空集'] == False]) / len(mc_pure) * 100,
               mc_pure['是否退化'].mean() * 100,
               valid['是否被覆盖'].mean() * 100,
               valid['直径'].mean(), valid['直径'].median(), valid['直径'].std(),
               valid['直径'].max(), valid['直径'].min(),
               valid['包围圆半径与直径之比'].mean(),
               valid['包围圆半径与直径之比'].max(), 1 / np.sqrt(3)],
    })
    save_csv(desc, '定位区域直径描述统计.csv')
    print('\n[描述统计 · 纯交会口径]')
    print(desc.round(6).to_string(index=False))

    # ---------------- 6.3 挑覆盖失败反例 ----------------
    fails = valid[~valid['是否被覆盖'].astype(bool)]
    if len(fails):
        worst = fails.loc[fails['超出量'].idxmax()]
        print(f"\n[覆盖失败反例] 取自纯交会口径超出量最大的构型：序号 {int(worst['序号'])}, "
              f"交会角 {worst['交会角']:.2f} 度, 超出量 {worst['超出量']:.4f} m")
    else:
        worst = valid.loc[valid['超出量'].idxmax()]
    print(f"  失败构型超出量：均值 {fails['超出量'].mean() if len(fails) else 0:.4f} m, "
          f"最大 {fails['超出量'].max() if len(fails) else 0:.4f} m, "
          f"占直径比例最大 "
          f"{(fails['超出量'] / fails['直径']).max() if len(fails) else 0:.4%}")
    S_fail = np.array([[worst['测点1_x'], worst['测点1_y']],
                       [worst['测点2_x'], worst['测点2_y']]])
    b_fail = np.array([worst['示向度1'], worst['示向度2']])
    G_fail = np.array([worst['源_x'], worst['源_y']])
    res_fail = solve_configuration(S_fail, b_fail, cap_circle=False)
    print(f"  直径 = {res_fail['diameter']:.4f} m, 最小包围圆半径 = "
          f"{res_fail['min_enclosing_radius']:.4f} m, "
          f"覆盖 = {'是' if res_fail['covered'] else '否'}")
    print(f"  Thales 符号量 = "
          f"{ {k: round(v, 3) for k, v in (res_fail['criterion'] or {}).items()} }")

    # ---------------- 6.4 图1：交会定位区域示意图（1×2：全局 + 局部放大） ----------------
    poly = demo['poly']
    D, (A, C) = diameter_bruteforce(convex_hull(poly))
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2))

    for pi, ax in enumerate(axes):
        for i, (Sp, th) in enumerate(zip(S_demo, meas_b)):
            for k, lab in [(-1, '示向度 $\\pm1^\\circ$ 边界射线'), (+1, None)]:
                a = np.deg2rad(th + k * DELTA_DEG)
                far = Sp + 4200 * np.array([np.cos(a), np.sin(a)])
                ax.plot([Sp[0], far[0]], [Sp[1], far[1]], '--', color=COLORS_10[i],
                        linewidth=1.1, alpha=0.9, label=lab if k == -1 else None)
            a = np.deg2rad(th)
            far = Sp + 4200 * np.array([np.cos(a), np.sin(a)])
            ax.plot([Sp[0], far[0]], [Sp[1], far[1]], '-', color=COLORS_10[i], linewidth=1.6,
                    label=f'检测点 $S_{i+1}$ 示向度 {th:.2f}$^\\circ$')
            ax.scatter(*Sp, s=110, marker='^', color=COLORS_10[i], edgecolor='k', zorder=5,
                       label=f'检测点 $S_{i+1}$')
        ax.scatter(*G_demo, s=170, marker='*', color='k', zorder=6,
                   label='干扰源真实位置 $G$')
        ax.add_patch(MplPolygon(poly, closed=True, facecolor=COLORS_10[3],
                                alpha=0.70 if pi else 0.45,
                                edgecolor=COLORS_10[3], linewidth=1.6,
                                label='交会定位区域'))
        ax.add_patch(Circle((A + C) / 2, D / 2, fill=False, edgecolor=COLORS_10[2],
                            linewidth=2.0, zorder=4,
                            label=f'以直径为直径的圆 ($D_P$={D:.1f} m)'))
        ax.plot([A[0], C[0]], [A[1], C[1]], '-', color=COLORS_10[2], linewidth=2.0,
                zorder=5, label='定位区域直径 $D_P$')
        ax.set_aspect('equal')
        ax.set_xlabel('X 方向坐标 (m)', fontsize=11)
        ax.set_ylabel('Y 方向坐标 (m)', fontsize=11)
        if pi == 0:
            ax.set_xlim(-350, 2700)
            ax.set_ylim(-800, 2000)
            ax.set_title('(a) 全局视图：两检测点的示向度射线束', fontsize=12)
        else:
            cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
            pad = max(45.0, 1.35 * D / 2)
            ax.set_xlim(cx - pad, cx + pad)
            ax.set_ylim(cy - pad, cy + pad)
            ax.set_title('(b) 局部放大：定位区域、直径与直径圆', fontsize=12)
        apply_mpl_style(fig)
    set_legend_cn(axes[0], loc='upper left', fontsize=8.5)
    set_legend_cn(axes[1], loc='upper left', fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.suptitle(f'交会定位区域几何（$D_P$={D:.2f} m，被直径圆覆盖）', fontsize=13)
    save_fig(fig, '交会定位区域示意图.png')

    # ---------------- 6.5 图2：覆盖与不覆盖对照（1×2） ----------------
    def draw_case(ax, S, meas_b_, G, title, cap=False):
        r = solve_configuration(S, meas_b_, cap_circle=cap)
        p = convex_hull(r['poly'])
        ax.add_patch(MplPolygon(p, closed=True, facecolor=COLORS_10[3], alpha=0.45,
                                edgecolor=COLORS_10[3], linewidth=1.5))
        for i in range(2):
            Sp = S[i]
            for k in (-1, 1):
                a = np.deg2rad(meas_b_[i] + k * DELTA_DEG)
                far = Sp + 1600 * np.array([np.cos(a), np.sin(a)])
                ax.plot([Sp[0], far[0]], [Sp[1], far[1]], '--', color=COLORS_10[i],
                        linewidth=0.9, alpha=0.85)
            a = np.deg2rad(meas_b_[i])
            far = Sp + 1600 * np.array([np.cos(a), np.sin(a)])
            ax.plot([Sp[0], far[0]], [Sp[1], far[1]], '-', color=COLORS_10[i], linewidth=1.2)
            ax.scatter(*Sp, s=80, marker='^', color=COLORS_10[i], edgecolor='k', zorder=5)
        ax.scatter(*G, s=120, marker='*', color='k', zorder=6)
        D, (A, C) = diameter_bruteforce(p)
        ax.add_patch(Circle((A + C) / 2, D / 2, fill=False, edgecolor=COLORS_10[2],
                            linewidth=1.8))
        ax.plot([A[0], C[0]], [A[1], C[1]], '-', color=COLORS_10[2], linewidth=1.8)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11.5)
        ax.set_xlabel('X 方向坐标 (m)', fontsize=10)
        ax.set_ylabel('Y 方向坐标 (m)', fontsize=10)
        apply_mpl_style(fig)
        if r['n_vertices'] > 0:
            cx, cy = r['poly'][:, 0], r['poly'][:, 1]
            span = max(cx.max() - cx.min(), cy.max() - cy.min(), D)
            pad = max(12.0, 0.62 * span)
            ax.set_xlim(cx.mean() - pad, cx.mean() + pad)
            ax.set_ylim(cy.mean() - pad, cy.mean() + pad)
        return r

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8))
    draw_case(axes[0], S_demo, meas_b, G_demo,
              f"覆盖成立：包围圆半径/直径 = {demo['ratio_mec_over_diameter']:.4f}")
    if res_fail['n_vertices'] > 0:
        draw_case(axes[1], S_fail, b_fail, G_fail,
                  f"覆盖失败：包围圆半径/直径 = {res_fail['ratio_mec_over_diameter']:.4f}"
                  if not res_fail['covered'] else
                  f"覆盖成立：包围圆半径/直径 = {res_fail['ratio_mec_over_diameter']:.4f}")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.suptitle('直径圆覆盖交会定位区域的两种情形', fontsize=13)
    save_fig(fig, '直径圆覆盖与不覆盖对比图.png')

    # ---------------- 6.6 灵敏度：覆盖率随测向误差上界变化 ----------------
    sens = sensitivity_delta()
    save_csv(sens, '覆盖率对测向精度的灵敏度.csv')
    print('\n[灵敏度 · 覆盖率随示向度误差上界的变化]')
    print(sens.round(4).to_string(index=False))

    # ---------------- 6.7 图3：覆盖率、直径分布与灵敏度（1×3） ----------------
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.0))

    ax = axes[0]
    hi = valid['直径'].quantile(0.995)
    ax.hist(valid['直径'].clip(upper=hi), bins=60, color=COLORS_10[0], alpha=0.85,
            edgecolor='white', linewidth=0.4)
    med = valid['直径'].median()
    ax.axvline(med, color=COLORS_10[3], linewidth=1.8, linestyle='--',
               label=f'中位数 {med:.1f} m')
    ax.set_xlabel('交会定位区域直径 $D_P$ (m)', fontsize=11)
    ax.set_ylabel('构型数量', fontsize=11)
    ax.set_title('定位区域直径分布', fontsize=12)
    apply_mpl_style(fig)
    set_legend_cn(ax)

    ax = axes[1]
    gg = valid.groupby(pd.cut(valid['交会角'], bins=bins), observed=True)
    keys = list(gg.groups.keys())
    xs = [iv.mid for iv in keys]
    ys = [gg.get_group(k)['是否被覆盖'].mean() * 100 for k in keys]
    ns = [len(gg.get_group(k)) for k in keys]
    ax.plot(xs, ys, 'o-', color=COLORS_10[2], linewidth=2.0, markersize=7)
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(f'n={n}', (x, y), textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=8, color='#444444')
    ax.set_ylim(80, 105)
    ax.set_xlabel('交会角 $\\alpha$ (度)', fontsize=11)
    ax.set_ylabel('直径圆覆盖率 (%)', fontsize=11)
    ax.set_title('覆盖率随交会角的变化', fontsize=12)
    apply_mpl_style(fig)

    ax = axes[2]
    ax.plot(sens['误差上界'], sens['覆盖率'], 's-', color=COLORS_10[4],
            linewidth=2.0, markersize=7, label='直径圆覆盖率')
    ax.set_ylim(80, 105)
    ax.set_xlabel('示向度误差上界 $\\delta$ (度)', fontsize=11)
    ax.set_ylabel('直径圆覆盖率 (%)', fontsize=11)
    ax2 = ax.twinx()
    ax2.plot(sens['误差上界'], sens['超出量最大值'], '^--', color=COLORS_10[1],
             linewidth=1.8, markersize=7, label='最大超出量')
    ax2.set_ylabel('失败构型最大超出量 (m)', fontsize=11, color=COLORS_10[1])
    ax2.tick_params(axis='y', colors=COLORS_10[1])
    ax2.spines['top'].set_visible(False)
    ax.set_title('覆盖率对测向精度的灵敏度', fontsize=12)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    set_legend_cn(ax, handles=h1 + h2, labels=l1 + l2, loc='center right', fontsize=9)
    apply_mpl_style(fig)
    save_fig(fig, '直径圆覆盖率与直径分布图.png')

    # ---------------- 6.7 结果 CSV ----------------
    out_rows = []
    for tag, S_, b_, cap in [('示范构型', S_demo, meas_b, False),
                             ('覆盖失败反例', S_fail, b_fail, False)]:
        r = solve_configuration(S_, b_, cap_circle=cap)
        if r['n_vertices'] == 0:
            continue
        H = convex_hull(r['poly'])
        row = dict(构型=tag, 检测点数=2, 顶点数=len(H),
                   直径=round(r['diameter'], 6),
                   旋转卡壳直径=round(r['d_rotating'], 6),
                   双算法偏差=abs(r['diameter'] - r['d_rotating']),
                   最小包围圆半径=round(r['min_enclosing_radius'], 6),
                   包围圆半径与直径之比=round(r['ratio_mec_over_diameter'], 6),
                   是否被覆盖=bool(r['covered']),
                   Thales符号量1=None, Thales符号量2=None)
        if r['criterion']:
            vals = sorted(r['criterion'].values())
            row['Thales符号量1'] = round(vals[0], 6)
            row['Thales符号量2'] = round(vals[-1], 6)
        for k in range(min(4, len(H))):
            row[f'顶点{k+1}_x'] = round(float(H[k][0]), 6)
            row[f'顶点{k+1}_y'] = round(float(H[k][1]), 6)
        out_rows.append(row)
    save_csv(pd.DataFrame(out_rows), '定位区域直径结果.csv')
    save_csv(desc2, '两种口径对比统计表.csv')
    print('\n[定位区域直径结果]')
    print(pd.DataFrame(out_rows).to_string(index=False))

    # 清理临时文件
    try:
        os.remove(OUT_DIR + '_mc_tmp.csv')
    except OSError:
        pass

    # 保存一份蒙特卡洛明细供核对
    valid.head(200).to_csv(OUT_DIR + '蒙特卡洛构型抽样明细.csv',
                           index=False, encoding='utf-8-sig')
    print('\n问题一求解完成，结果已保存至', OUT_DIR)


if __name__ == '__main__':
    main()
