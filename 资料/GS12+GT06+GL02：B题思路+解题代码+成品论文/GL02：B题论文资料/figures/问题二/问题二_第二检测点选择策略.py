# -*- coding: utf-8 -*-
"""
问题二：第二个检测点的选择策略与候选区域
------------------------------------------------------------------
输入：第一个检测点 S1 及其测得的示向度 theta1（全向干扰源）
输出：第二个检测点 S2 的最优位置、选择策略、以及满足定位精度要求的候选区域

模型组成：
  1) 可行域建模：单次示向度把干扰源约束在张角 2° 的楔形 Omega 内
  2) 误差传播：两条方位线交点的扰动解析式 E(S2; G)
  3) 解析特例：垂直布站下最优基线 b* = sqrt(2) r，最优交会角 54.74°，E_min = 3*delta*r
  4) 数值求解：在 (b, phi) 上做期望准则与极小化极大准则的最优布站
  5) 候选区域：目标函数次水平集
  6) 灵敏度：计入机动时间成本后的最优基线漂移
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import font_manager
from matplotlib.patches import Circle, Wedge as MplWedge, Polygon as MplPolygon
from scipy.optimize import minimize, minimize_scalar
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
OUT_DIR = BASE + '/问题二/'
FIG_DIR = OUT_DIR + 'figures/'
os.makedirs(FIG_DIR, exist_ok=True)


def save_fig(fig, name_cn):
    fig.savefig(FIG_DIR + name_cn)
    plt.close(fig)


def save_csv(df, name_cn):
    df.to_csv(OUT_DIR + name_cn, index=False, encoding='utf-8-sig')


def apply_mpl_style(fig, despine=True, ax=None):
    ax = ax if ax is not None else fig.gca()
    if despine:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.4, color='#AAAAAA')
    ax.tick_params(labelsize=10)
    fig.tight_layout()


def set_legend_cn(ax, **kwargs):
    return ax.legend(prop={'family': plt.rcParams['font.sans-serif']}, **kwargs)


# ==================================================================
# 全局常量
# ==================================================================
R_TARGET = 1800.0        # 目标区域半径 (m)
DELTA_DEG = 1.0          # 示向度误差上界 (度)
DELTA = np.deg2rad(DELTA_DEG)
R_HI = 1500.0            # 有效接收半径上界，亦为径向可行域上界 (m)
V_DOG = 5.0              # 机器狗速度 (m/s)


# ==================================================================
# 1. 可行域 Omega：单次示向度给出的楔形
# ==================================================================
def sample_omega(n=6000, seed=20260913, s1=(0.0, 0.0), theta1_deg=0.0,
                 r_hi=R_HI, delta_deg=DELTA_DEG):
    """在楔形 Omega 内按面积均匀采样。

    坐标已作刚性变换：S1 平移到原点，示向度 theta1 旋转到 +x 轴。
    于是 Omega = {(r cos a, r sin a): 0 <= r <= r_hi, |a| <= delta_deg}。
    """
    rng = np.random.default_rng(seed)
    r = r_hi * np.sqrt(rng.random(n))
    a = np.deg2rad(rng.uniform(-delta_deg, delta_deg, n))
    G = np.c_[r * np.cos(a), r * np.sin(a)]
    return G + np.asarray(s1, float)


def omega_boundary(s1=(0.0, 0.0), r_hi=R_HI, delta_deg=DELTA_DEG, n_arc=200):
    """返回楔形边界的闭合点列（用于绘图/判据）。"""
    s1 = np.asarray(s1, float)
    a = np.deg2rad(np.linspace(-delta_deg, delta_deg, n_arc))
    top = s1 + r_hi * np.c_[np.cos(a), np.sin(a)]
    return np.vstack([[s1], top, [s1]])


# ==================================================================
# 2. 误差传播：两条方位线交点的扰动
# ==================================================================
def localization_error(S2, G, delta=DELTA):
    """给定第二检测点 S2 与真实源位置 G，方位线扰动导致的定位偏差最坏上界。

    Delta P = r1*dphi1/sin(alpha_x) * u2 + r2*dphi2/sin(alpha_x) * u1
    取 |dphi1| = |dphi2| = delta 且同号，得
        E = delta/sin(alpha_x) * sqrt(r1^2 + r2^2 + 2 r1 r2 cos(alpha_x))
    其中 alpha_x 为 G 处两视线的交会角，r1 = |S1 G|，r2 = |S2 G|。

    S2 与 G 数组广播：G 为 (N,2)，S2 为 (2,) 或 (M,2)。
    """
    G = np.atleast_2d(G)
    S2arr = np.atleast_2d(S2)
    r1 = np.linalg.norm(G, axis=1)                       # S1 在原点
    d = G[None, :, :] - S2arr[:, None, :]
    r2 = np.linalg.norm(d, axis=2)                        # (M, N)
    b = np.linalg.norm(S2arr, axis=1)[:, None]            # (M, 1)
    r1b = r1[None, :]
    cosx = np.clip((r1b**2 + r2**2 - b**2) / (2 * r1b * r2 + 1e-18), -1.0, 1.0)
    sinx = np.sqrt(np.maximum(1e-18, 1.0 - cosx**2))
    E = delta / sinx * np.sqrt(r1b**2 + r2**2 + 2 * r1b * r2 * cosx)
    return E.squeeze()


def localization_error_perp(b, r, delta=DELTA):
    """垂直布站（基线方向 ⊥ 视线方向）时 E 的解析式：
        E(b; r) = delta / b * sqrt((r^2 + b^2)(4 r^2 + b^2))
    """
    return delta / b * np.sqrt((r**2 + b**2) * (4 * r**2 + b**2))


# ==================================================================
# 3. 准则函数
# ==================================================================
def objective_expected(S2, G, delta=DELTA, lam=0.0):
    """期望准则：Omega 内平均定位误差 (+ 可选机动时间成本 lam * |S2| / v)。"""
    E = localization_error(S2, G, delta)
    return float(np.mean(E) + lam * np.linalg.norm(S2) / V_DOG)


def objective_minimax(S2, G, delta=DELTA, lam=0.0):
    """极小化极大准则：Omega 内最坏定位误差 (+ 可选机动时间成本)。"""
    E = localization_error(S2, G, delta)
    return float(np.max(E) + lam * np.linalg.norm(S2) / V_DOG)


def search_optimal(G, kind='expected', b_grid=None, phi_grid=None,
                   delta=DELTA, lam=0.0):
    """在极坐标 (b, phi) 网格粗搜 + 局部精搜，返回最优布站。

    phi in [0, 180)，因为 +/- 对称由 phi 与 -phi 自动覆盖。
    """
    if b_grid is None:
        b_grid = np.arange(50.0, 3200.0, 25.0)
    if phi_grid is None:
        phi_grid = np.deg2rad(np.arange(4.0, 180.0, 1.5))
    f = objective_expected if kind == 'expected' else objective_minimax

    best = (np.inf, None, None)
    for b in b_grid:
        for p in phi_grid:
            v = f(np.array([b * np.cos(p), b * np.sin(p)]), G, delta, lam)
            if v < best[0]:
                best = (v, b, p)
    # 局部精搜
    x0 = np.array([best[1] * np.cos(best[2]), best[1] * np.sin(best[2])])
    res = minimize(lambda x: f(x, G, delta, lam), x0, method='Nelder-Mead',
                   options=dict(xatol=1e-3, fatol=1e-6, maxiter=4000))
    S2 = res.x
    return dict(S2=S2, b=float(np.linalg.norm(S2)),
                phi_deg=float(np.rad2deg(np.arctan2(S2[1], S2[0])) % 180.0),
                value=float(res.fun))


# ==================================================================
# 4. 候选区域：目标函数次水平集
# ==================================================================
def candidate_region(G, S2_opt, tol=0.05, kind='expected', delta=DELTA):
    """候选择区域 C = {S2 : J(S2) <= (1+tol) * J(S2*)}（在极坐标网格上求得）。"""
    f = objective_expected if kind == 'expected' else objective_minimax
    j_star = f(S2_opt, G, delta)
    bs = np.arange(50.0, 3200.0, 20.0)
    phis = np.deg2rad(np.arange(2.0, 180.0, 1.0))
    B, P = np.meshgrid(bs, phis, indexing='ij')
    XY = np.stack([B * np.cos(P), B * np.sin(P)], axis=-1).reshape(-1, 2)
    J = np.array([f(xy, G, delta) for xy in XY]).reshape(B.shape)
    mask = J <= (1 + tol) * j_star
    pts = XY[mask.ravel()]
    return dict(mask=mask, B=B, P=P, J=J, bs=bs, phis=phis,
                j_star=j_star, points=pts,
                b_range=(float(pts[:, 0].var() and np.linalg.norm(pts, axis=1).min()),
                         float(np.linalg.norm(pts, axis=1).max())),
                phi_range_deg=(float(np.rad2deg(np.arctan2(pts[:, 1], pts[:, 0])).min() % 180),
                               float(np.rad2deg(np.arctan2(pts[:, 1], pts[:, 0])).max() % 180)))


def describe_region(pts):
    """把候选点集概括为极坐标描述（相对 S1，角度已含示向度方向）。"""
    r = np.linalg.norm(pts, axis=1)
    a = np.rad2deg(np.arctan2(pts[:, 1], pts[:, 0])) % 360.0
    # 只保留上半平面（phi in [0,180)），再对 theta1 做对称
    return dict(b_min=float(r.min()), b_max=float(r.max()),
                phi_min=float(a.min()), phi_max=float(a.max()))


# ==================================================================
# 主流程
# ==================================================================
def main():
    print('=' * 72)
    print('问题二：第二个检测点的选择策略与候选区域')
    print('=' * 72)

    G = sample_omega(n=8000)

    # ---------------- 2.1 解析特例：垂直布站 ----------------
    print('\n[解析结论 · 垂直布站] E(b; r) = delta/b * sqrt((r^2+b^2)(4r^2+b^2))')
    print('  令 t = b^2/r^2，E ∝ sqrt(t + 5 + 4/t)，在 t = 2 处取极小')
    print(f'  => b* = sqrt(2) r, 最优交会角 arccos(1/sqrt(3)) = '
          f'{np.rad2deg(np.arccos(1/np.sqrt(3))):.2f} 度, E_min = 3*delta*r = '
          f'{3*DELTA:.6f} r')
    rows_analytic = []
    for r in [200, 400, 600, 800, 1000, 1200, 1400, 1500]:
        bb = np.linspace(1.0, 4000.0, 400000)
        ee = localization_error_perp(bb, r)
        i = int(np.argmin(ee))
        rows_analytic.append(dict(距离r=float(r), 最优基线b=float(bb[i]),
                                  b与r之比=float(bb[i] / r),
                                  最优交会角=float(np.rad2deg(np.arccos(r / np.hypot(r, bb[i])))),
                                  最小误差E=float(ee[i]),
                                  E与r之比=float(ee[i] / r),
                                  解析系数3delta=float(3 * DELTA)))
    df_ana = pd.DataFrame(rows_analytic)
    print('\n[垂直布站解析最优核对]')
    print(df_ana.round(6).to_string(index=False))

    # ---------------- 2.2 数值最优布站 ----------------
    opt_exp = search_optimal(G, kind='expected')
    opt_mm = search_optimal(G, kind='minimax')
    opt_perp = search_optimal(G, kind='expected',
                              phi_grid=np.deg2rad(np.array([90.0])))
    print('\n[数值最优布站 · 极坐标网格 + Nelder-Mead 精搜]')
    for name, o in [('期望准则', opt_exp), ('极小化极大准则', opt_mm),
                    ('限定垂直布站', opt_perp)]:
        E = localization_error(o['S2'], G)
        print(f'  {name:<8s}: b* = {o["b"]:7.1f} m, phi* = {o["phi_deg"]:6.2f} 度, '
              f'J = {o["value"]:6.2f} m | E均值 {E.mean():6.2f} m, '
              f'E95 {np.percentile(E, 95):6.2f} m, E最大 {E.max():6.2f} m')

    # ---------------- 2.3 候选区域 ----------------
    creg = candidate_region(G, opt_exp['S2'], tol=0.05, kind='expected')
    desc = describe_region(creg['points'])
    print(f'\n[候选区域 · 期望准则次水平集 (容差 5%)]')
    print(f'  最优准则值 J* = {creg["j_star"]:.2f} m, 阈值 = {1.05*creg["j_star"]:.2f} m')
    print(f'  径向范围 b ∈ [{desc["b_min"]:.1f}, {desc["b_max"]:.1f}] m')
    print(f'  方位角范围 phi ∈ [{desc["phi_min"]:.2f}, {desc["phi_max"]:.2f}] 度 (相对示向度方向)')

    # ---------------- 2.4 灵敏度：机动时间成本权重 ----------------
    print('\n[灵敏度 · 计入机动时间成本 lam * |S2|/v 后的最优基线]')
    rows_lam = []
    for lam in [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        o = search_optimal(G, kind='expected',
                           b_grid=np.arange(50.0, 3200.0, 50.0),
                           phi_grid=np.deg2rad(np.arange(4.0, 180.0, 2.0)), lam=lam)
        E = localization_error(o['S2'], G)
        rows_lam.append(dict(成本权重lam=lam, 最优基线b=o['b'],
                             最优方位角phi=o['phi_deg'],
                             定位误差均值=float(E.mean()),
                             机动时间成本秒=o['b'] / V_DOG,
                             综合目标值=o['value']))
    df_lam = pd.DataFrame(rows_lam)
    print(df_lam.round(3).to_string(index=False))
    save_csv(df_lam, '机动成本灵敏度.csv')

    # ---------------- 2.5 图1：可行域与误差热力图（1×3） ----------------
    wb = omega_boundary(r_hi=R_HI)
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.6))

    ax = axes[0]
    ax.add_patch(MplPolygon(wb, closed=True, facecolor=COLORS_10[3], alpha=0.30,
                            edgecolor=COLORS_10[3], linewidth=1.6,
                            label='可行域 $\\Omega$（张角 $2^\\circ$）'))
    ax.scatter(0, 0, s=140, marker='^', color=COLORS_10[0], edgecolor='k', zorder=6,
               label='检测点 $S_1$')
    ax.annotate('', xy=(R_HI, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', color=COLORS_10[0], lw=1.8))
    ax.text(R_HI * 0.45, 30, '示向度方向 $\\theta_1$', fontsize=10, color=COLORS_10[0])
    ax.set_xlim(-100, 1650)
    ax.set_ylim(-32, 32)
    ax.set_xticks(np.arange(0, 1601, 300))
    ax.set_xlabel('沿示向度方向距离 (m)', fontsize=11)
    ax.set_ylabel('侧向偏移 (m，已放大)', fontsize=11)
    ax.set_title('(a) 可行域 $\\Omega$ 的楔形张角（纵轴放大）', fontsize=12)
    apply_mpl_style(fig, ax=ax)
    set_legend_cn(ax, loc='upper left', fontsize=9)

    ax = axes[1]
    ax.add_patch(MplPolygon(wb, closed=True, facecolor=COLORS_10[3], alpha=0.30,
                            edgecolor=COLORS_10[3], linewidth=1.6))
    ax.scatter(G[::25, 0], G[::25, 1], s=3, color=COLORS_10[3], alpha=0.55,
               label='可行域采样点')
    ax.add_patch(Circle((0, 0), R_HI, fill=False, linestyle=':', linewidth=1.4,
                        edgecolor='#666666', label='有效接收半径上界 1500 m'))
    ax.plot([0, opt_exp['S2'][0]], [0, opt_exp['S2'][1]], '-', color=COLORS_10[2],
            linewidth=1.8, label='最优基线 $S_1S_2^*$')
    ax.plot([0, opt_exp['S2'][0]], [0, -opt_exp['S2'][1]], '-', color=COLORS_10[2],
            linewidth=1.2, alpha=0.55)
    ax.scatter(0, 0, s=140, marker='^', color=COLORS_10[0], edgecolor='k', zorder=6,
               label='检测点 $S_1$')
    ax.scatter(*opt_exp['S2'], s=190, marker='*', color=COLORS_10[2], edgecolor='k',
               zorder=7, label=f'最优 $S_2^*$（{opt_exp["b"]:.0f} m, '
                               f'{opt_exp["phi_deg"]:.0f}$^\\circ$）')
    ax.scatter(opt_exp['S2'][0], -opt_exp['S2'][1], s=190, marker='*',
               color=COLORS_10[2], edgecolor='k', zorder=7)
    ax.set_aspect('equal')
    ax.set_xlim(-250, 1750)
    ax.set_ylim(-900, 900)
    ax.set_xlabel('沿示向度方向距离 (m)', fontsize=11)
    ax.set_ylabel('侧向偏移 (m)', fontsize=11)
    ax.set_title('(b) 可行域与最优第二检测点', fontsize=12)
    apply_mpl_style(fig, ax=ax)
    set_legend_cn(ax, loc='lower left', fontsize=8.5)

    ax = axes[2]
    bs_ = creg['bs']
    phs_ = np.rad2deg(creg['phis'])
    pc = ax.pcolormesh(phs_, bs_, creg['J'], cmap=CMAP_SEQUENTIAL,
                       shading='auto', vmin=np.percentile(creg['J'], 0),
                       vmax=np.percentile(creg['J'], 40))
    cb = fig.colorbar(pc, ax=ax)
    cb.set_label('期望定位误差 $J(S_2)$ (m)', fontsize=10)
    cs = ax.contour(phs_, bs_, creg['J'], levels=[1.05 * creg['j_star']],
                    colors=[COLORS_10[3]], linewidths=2.2)
    ax.clabel(cs, fmt='+5%%', fontsize=9)
    ax.contourf(phs_, bs_, creg['J'], levels=[-1, 1.05 * creg['j_star']],
                colors=[COLORS_10[3]], alpha=0.25)
    ax.scatter(opt_exp['phi_deg'], opt_exp['b'], s=180, marker='*', color=COLORS_10[2],
               edgecolor='k', zorder=7, label='期望准则最优 $S_2^*$')
    ax.scatter(opt_mm['phi_deg'], opt_mm['b'], s=130, marker='D', color=COLORS_10[4],
               edgecolor='k', zorder=7, label='极小化极大准则最优')
    ax.scatter(90.0, opt_perp['b'], s=130, marker='s', color=COLORS_10[1],
               edgecolor='k', zorder=7, label='垂直布站最优（次优）')
    ax.set_xlabel('基线方位角 $\\varphi$ (度，相对示向度方向)', fontsize=11)
    ax.set_ylabel('基线长度 $b$ (m)', fontsize=11)
    ax.set_title('(b) 期望定位误差与最优布站、候选区域', fontsize=12)
    apply_mpl_style(fig, ax=ax)
    set_legend_cn(ax, loc='upper right', fontsize=9)
    fig.suptitle('第二个检测点的可行域、误差分布与最优布站', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_fig(fig, '可行域与候选区域示意图.png')

    # ---------------- 2.6 图2：解析关系（1×2） ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    ax = axes[0]
    rr = np.linspace(50, R_HI, 200)
    ax.plot(rr, np.sqrt(2) * rr, '-', color=COLORS_10[0], linewidth=2.0,
            label='解析最优基线 $b^*=\\sqrt{2}r$')
    ax.plot(df_ana['距离r'], df_ana['最优基线b'], 'o', color=COLORS_10[3],
            markersize=7, label='数值搜索最优基线')
    ax.set_xlabel('干扰源距离 $r$ (m)', fontsize=11)
    ax.set_ylabel('最优基线长度 $b^*$ (m)', fontsize=11)
    ax.set_title('(a) 最优基线随源距离的变化', fontsize=12)
    apply_mpl_style(fig, ax=ax)
    set_legend_cn(ax)

    ax = axes[1]
    for idx, r in enumerate([300, 700, 1100, 1500]):
        bb = np.linspace(20, 3200, 1200)
        ax.plot(bb, localization_error_perp(bb, r), '-', color=COLORS_10[idx],
                linewidth=1.8, label=f'$r$ = {r} m')
        bstar = np.sqrt(2) * r
        ax.scatter(bstar, localization_error_perp(bstar, r), s=70,
                   color=COLORS_10[idx], edgecolor='k', zorder=5)
    ax.axvline(2121.3, color='#888888', linestyle='--', linewidth=1.4,
               label='$r=1500$ 时的 $b^*$=2121 m')
    ax.set_xlabel('基线长度 $b$ (m)', fontsize=11)
    ax.set_ylabel('定位误差 $E$ (m)', fontsize=11)
    ax.set_yscale('log')
    ax.set_title('(b) 定位误差随基线的变化（垂直布站）', fontsize=12)
    apply_mpl_style(fig, ax=ax)
    set_legend_cn(ax, fontsize=9)
    fig.suptitle('垂直布站下的解析最优基线关系', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_fig(fig, '最优基线与距离关系图.png')

    # ---------------- 2.7 图3：平面热力图与成本灵敏度（1×2） ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    ax = axes[0]
    gx = np.linspace(-500, 1800, 260)
    gy = np.linspace(-1300, 1300, 260)
    GX, GY = np.meshgrid(gx, gy)
    XY = np.stack([GX.ravel(), GY.ravel()], axis=1)
    Jmap = np.full(XY.shape[0], np.nan)
    for k in range(0, XY.shape[0], 1):
        pass
    # 向量化计算：分块以避免内存峰值
    chunk = 4000
    vals = []
    for s in range(0, XY.shape[0], chunk):
        vals.append(localization_error(XY[s:s + chunk], G).mean(axis=1))
    Jmap = np.concatenate(vals).reshape(GX.shape)
    Jlog = np.log10(Jmap)
    fin = Jlog[np.isfinite(Jlog)]
    pc = ax.pcolormesh(gx, gy, Jlog, cmap=CMAP_SEQUENTIAL, shading='auto',
                       vmin=np.percentile(fin, 1), vmax=np.percentile(fin, 99))
    cb = fig.colorbar(pc, ax=ax)
    ticks = [1, 3, 10, 30, 100, 300, 1000, 10000]
    cb.set_ticks(np.log10(ticks))
    cb.set_ticklabels([str(t) for t in ticks])
    cb.set_label('期望定位误差 $J(S_2)$ (m，对数刻度)', fontsize=10)
    ax.contour(gx, gy, Jmap, levels=[1.05 * creg['j_star']],
               colors=[COLORS_10[3]], linewidths=2.2)
    ax.add_patch(MplPolygon(wb, closed=True, facecolor='none', edgecolor='#333333',
                            linewidth=1.4, linestyle='--', label='可行域 $\\Omega$ 边界'))
    ax.scatter(0, 0, s=150, marker='^', color=COLORS_10[0], edgecolor='k', zorder=6,
               label='$S_1$')
    ax.scatter(*opt_exp['S2'], s=180, marker='*', color=COLORS_10[2], edgecolor='k',
               zorder=7, label='最优 $S_2^*$')
    ax.scatter(*[opt_exp['S2'][0], -opt_exp['S2'][1]], s=180, marker='*',
               color=COLORS_10[2], edgecolor='k', zorder=7)
    ax.set_aspect('equal')
    ax.set_xlabel('X 方向坐标 (m)', fontsize=11)
    ax.set_ylabel('Y 方向坐标 (m)', fontsize=11)
    ax.set_title('(a) 平面上的期望定位误差与候选区域边界', fontsize=12)
    apply_mpl_style(fig, ax=ax)
    set_legend_cn(ax, loc='lower left', fontsize=9)

    ax = axes[1]
    ax.plot(df_lam['成本权重lam'], df_lam['最优基线b'], 'o-', color=COLORS_10[0],
            linewidth=2.0, markersize=7, label='最优基线 $b^*$')
    ax.set_xlabel('机动时间成本权重 $\\lambda$', fontsize=11)
    ax.set_ylabel('最优基线 $b^*$ (m)', fontsize=11)
    ax2 = ax.twinx()
    ax2.plot(df_lam['成本权重lam'], df_lam['定位误差均值'], 's--', color=COLORS_10[1],
             linewidth=1.8, markersize=7, label='期望定位误差')
    ax2.set_ylabel('期望定位误差 (m)', fontsize=11, color=COLORS_10[1])
    ax2.tick_params(axis='y', colors=COLORS_10[1])
    ax2.spines['top'].set_visible(False)
    ax.set_title('(b) 计入机动成本后的最优基线漂移', fontsize=12)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    set_legend_cn(ax, handles=h1 + h2, labels=l1 + l2, loc='center right', fontsize=9)
    apply_mpl_style(fig, ax=ax)
    fig.suptitle('最优布站的平面结构与成本灵敏度', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_fig(fig, '最优布站平面结构与成本灵敏度图.png')

    # ---------------- 2.8 输出 CSV ----------------
    summary = pd.DataFrame([
        dict(准则='期望准则(主)', 最优基线b=opt_exp['b'], 最优方位角phi=opt_exp['phi_deg'],
             期望误差=float(localization_error(opt_exp['S2'], G).mean()),
             分位95误差=float(np.percentile(localization_error(opt_exp['S2'], G), 95)),
             最坏误差=float(localization_error(opt_exp['S2'], G).max())),
        dict(准则='极小化极大准则', 最优基线b=opt_mm['b'], 最优方位角phi=opt_mm['phi_deg'],
             期望误差=float(localization_error(opt_mm['S2'], G).mean()),
             分位95误差=float(np.percentile(localization_error(opt_mm['S2'], G), 95)),
             最坏误差=float(localization_error(opt_mm['S2'], G).max())),
        dict(准则='限定垂直布站(次优)', 最优基线b=opt_perp['b'],
             最优方位角phi=opt_perp['phi_deg'],
             期望误差=float(localization_error(opt_perp['S2'], G).mean()),
             分位95误差=float(np.percentile(localization_error(opt_perp['S2'], G), 95)),
             最坏误差=float(localization_error(opt_perp['S2'], G).max())),
    ])
    save_csv(summary, '最优基线与误差结果.csv')
    save_csv(df_ana, '垂直布站解析最优核对表.csv')

    cr = creg['points']
    save_csv(pd.DataFrame(dict(
        相对S1距离=np.linalg.norm(cr, axis=1),
        相对示向度方位角=np.rad2deg(np.arctan2(cr[:, 1], cr[:, 0])) % 180,
        x=cr[:, 0], y=cr[:, 1])), '候选区域点集.csv')

    print('\n[最优布站汇总]')
    print(summary.round(4).to_string(index=False))
    print('\n问题二求解完成，结果已保存至', OUT_DIR)


if __name__ == '__main__':
    main()
