# -*- coding: utf-8 -*-
"""
问题三：全向干扰源的搜索、定位与清除策略（理论建模）
------------------------------------------------------------------
模型组成：
  1) 可检测域模型与"保证可测"覆盖判据
  2) 保证覆盖网：由 Kershner 定理给出最少观测点下界 7，并求解最小行程的环半径
  3) 分层策略：全域粗探 -> 交会定距 -> 逼近 -> 就地清除
  4) 定位精度阈值：由问题二结论反解"进入清除精度所需的逼近距离"
  5) 虚拟时间模型与理论下界（含 BHH 旅行商下界）
  6) 灵敏度：覆盖网规模、侧移系数 beta 对总时间的影响

说明：本题按纯理论建模口径求解，不接入模拟器；正式测试结果表（表1）保留空表头待回填。
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
from scipy.optimize import minimize_scalar, brentq
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
OUT_DIR = BASE + '/问题三/'
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
# 全局常量（题目给定）
# ==================================================================
R_TARGET = 1800.0        # 目标区域半径 (m)
DELTA = np.deg2rad(1.0)  # 示向度误差上界 (rad)
R_C_MIN = 1000.0         # 有效接收半径下界（保证可测半径）
R_C_MAX = 1500.0         # 有效接收半径上界
CLEAR_R = 20.0           # 清除半径 (m)
NEAR_R = 5.0             # 近距离阈值 (m)
V_DOG = 5.0              # 机器狗速度 (m/s)
T_MEASURE = 5.0          # 单次检测耗时 (s)
T_SWITCH = 1.0           # 频道切换耗时 (s)
T_CLEAR_OK = 5.0         # 成功清除耗时 (s)
T_CLEAR_NONE = 3.0       # 未发现耗时 (s)
N_CHANNEL = 20           # 频道数
N_SRC_RANGE = (10, 16)   # 干扰源个数范围


# ==================================================================
# 1. 保证覆盖网
# ==================================================================
def covering_radius(d, n_boundary=40000, n_radial=400, n_theta=900):
    """'1 个中心点 + 6 个六边形环点(环半径 d)'构型对目标圆域 D 的覆盖半径。

    覆盖半径 = max_{X in D} min_{Q in Qset} |X - Q|
    """
    a = np.linspace(0, 2 * np.pi, n_boundary, endpoint=False)
    boundary = np.c_[R_TARGET * np.cos(a), R_TARGET * np.sin(a)]
    rr = R_TARGET * np.linspace(0, 1, n_radial)
    tt = np.linspace(0, 2 * np.pi, n_theta)
    RR, TT = np.meshgrid(rr, tt, indexing='ij')
    interior = np.c_[(RR * np.cos(TT)).ravel(), (RR * np.sin(TT)).ravel()]
    X = np.vstack([boundary, interior])
    Q = covering_network(1.0)
    Q[:, 0] *= d
    Q[:, 1] *= d
    Dist = np.sqrt(((X[:, None, :] - Q[None, :, :]) ** 2).sum(-1)).min(axis=1)
    return float(Dist.max()), X


def covering_network(d):
    """返回 '1 中心 + 6 环点' 的单位构型（环半径 d）。"""
    pts = [np.array([0.0, 0.0])]
    for k in range(6):
        pts.append(np.array([d * np.cos(k * np.pi / 3), d * np.sin(k * np.pi / 3)]))
    return np.array(pts)


def covering_radius_6pt(d, n_boundary=40000, n_radial=200, n_theta=400):
    """6 个环点（无中心点）构型的覆盖半径，用于验证 6 点不可行。

    注意必须对圆域整体采样：若只采边界会漏掉最坏点（圆心）。
    """
    a = np.linspace(0, 2 * np.pi, n_boundary, endpoint=False)
    bnd = np.c_[R_TARGET * np.cos(a), R_TARGET * np.sin(a)]
    rr = R_TARGET * np.linspace(0, 1, n_radial)
    tt = np.linspace(0, 2 * np.pi, n_theta)
    RR, TT = np.meshgrid(rr, tt, indexing='ij')
    inn = np.c_[(RR * np.cos(TT)).ravel(), (RR * np.sin(TT)).ravel()]
    X = np.vstack([bnd, inn])
    Q = covering_network(d)[1:]
    dmin = np.empty(len(X))
    step = 200000
    for s0 in range(0, len(X), step):
        c = X[s0:s0 + step]
        dmin[s0:s0 + step] = np.sqrt(((c[:, None, :] - Q[None, :, :]) ** 2).sum(-1)).min(axis=1)
    return float(dmin.max())


def solve_covering_network(r_guarantee=R_C_MIN, verbose=True):
    """求解保证覆盖网：最小观测点数、环半径区间与最小行程方案。"""
    f = lambda d: covering_radius(d)[0]
    res = minimize_scalar(f, bounds=(1.0, 3000.0), method='bounded',
                          options=dict(xatol=1e-4))
    d_star, rho_star = float(res.x), float(res.fun)
    d1 = float(brentq(lambda d: f(d) - r_guarantee, 1.0, d_star, xtol=1e-4))
    d2 = float(brentq(lambda d: f(d) - r_guarantee, d_star, 3000.0, xtol=1e-4))
    res6 = minimize_scalar(covering_radius_6pt, bounds=(1.0, 3000.0), method='bounded',
                           options=dict(xatol=1e-4))
    if verbose:
        print(f'  7 点构型 (1 中心 + 6 环点):')
        print(f'    最优环半径 d* = {d_star:.2f} m -> 最小覆盖半径 rho* = {rho_star:.4f} m')
        print(f'    覆盖半径恰为 {r_guarantee:.0f} m 时: d1 = {d1:.2f} m (最小行程), '
              f'd2 = {d2:.2f} m')
        print(f'  6 点构型 (仅 6 环点): 最优覆盖半径 = {res6.fun:.2f} m '
              f'(d = {res6.x:.2f} m) > {r_guarantee:.0f} m -> 不可行')
        print(f'  Kershner 定理: 6 个半径 {r_guarantee:.0f} m 的圆盘最多覆盖半径 '
              f'{r_guarantee * np.sqrt(3):.2f} m 的圆域 < {R_TARGET:.0f} m，'
              f'故最少观测点数为 7')
    return dict(n_points=7, d_star=d_star, rho_star=rho_star, d1=d1, d2=d2,
                rho_6pt=float(res6.fun), d_6pt=float(res6.x),
                points=covering_network(d1), guarantee=r_guarantee)


def verify_coverage(points, r_guarantee=R_C_MIN, n_samples=2000000, seed=2026):
    """稠密采样验证：目标域内任意点至少被一个半径 r_guarantee 的圆盘覆盖。"""
    rng = np.random.default_rng(seed)
    # 边界密集采样 + 内部均匀采样
    a = np.linspace(0, 2 * np.pi, n_samples // 4, endpoint=False)
    bnd = np.c_[R_TARGET * np.cos(a), R_TARGET * np.sin(a)]
    t = 2 * np.pi * rng.random(n_samples // 2)
    r = R_TARGET * np.sqrt(rng.random(n_samples // 2))
    inn = np.c_[r * np.cos(t), r * np.sin(t)]
    X = np.vstack([bnd, inn])
    dmin = np.empty(len(X))
    step = 200000
    for s in range(0, len(X), step):
        chunk = X[s:s + step]
        dmin[s:s + step] = np.sqrt(
            ((chunk[:, None, :] - points[None, :, :]) ** 2).sum(-1)).min(axis=1)
    return float(dmin.max()), int((dmin > r_guarantee + 1e-6).sum()), len(X)


# ==================================================================
# 2. 定位精度阈值（承接问题二结论）
# ==================================================================
def approach_threshold(clear_r=CLEAR_R, delta=DELTA, side=1.0):
    """由 E = 3*delta*r <= clear_r 反解所需逼近距离阈值 (m)。

    side=1.0 表示按问题二垂直布站最优基线 b = sqrt(2) r 取定距精度。
    """
    return clear_r / (3 * delta * side)


def triangulate_error(r, b, phi_deg=90.0, delta=DELTA):
    """给定源距离 r、基线 b、基线方位 phi，返回交会定位误差 E 与不确定半径。"""
    phi = np.deg2rad(phi_deg)
    S2 = np.array([b * np.cos(phi), b * np.sin(phi)])
    r1 = r
    r2 = float(np.linalg.norm(np.array([r, 0.0]) - S2))
    cosx = np.clip((r1**2 + r2**2 - b**2) / (2 * r1 * r2), -1, 1)
    sinx = np.sqrt(max(1e-18, 1 - cosx**2))
    return float(delta / sinx * np.sqrt(r1**2 + r2**2 + 2 * r1 * r2 * cosx))


# ==================================================================
# 3. 虚拟时间模型
# ==================================================================
def tsp_lower_bound(n_src, R=R_TARGET, beta_bhh=0.7124):
    """Beardwood-Halton-Hammersley 旅行商下界 L ~ beta * sqrt(n * A)。"""
    A = np.pi * R**2
    return beta_bhh * np.sqrt(n_src * A)


def time_model(n_src, net, beta_lateral=0.25, r_det=None, adaptive_scan=True):
    """问题三虚拟时间分解模型。

    阶段一（全域粗探）：从原点出发沿六边形环巡回一周，每点测 20 个频道。
    阶段二（定位清除）：旅行商位移 + 每源侧移定距 + 固定动作开销。
    """
    Q = net['points']
    d1 = net['d1']
    n_q = net['n_points']

    # ---- 阶段一 ----
    L_scan = 6.0 * d1
    # 粗探按"每点测全部 20 个频道"计，给出保守（上界）估计；
    # 实际执行时已清除的频道不再复测，故为保守值。
    n_meas_scan = N_CHANNEL * n_q
    n_switch_scan = n_q * (N_CHANNEL - 1)
    T_scan = L_scan / V_DOG + T_MEASURE * n_meas_scan + T_SWITCH * n_switch_scan

    # ---- 阶段二 ----
    L_tsp = tsp_lower_bound(n_src)
    if r_det is None:
        r_det = L_tsp / n_src              # 巡回的平均相邻间距，作为探测距离尺度
    L_lateral = n_src * beta_lateral * r_det
    T_travel = (L_tsp + L_lateral) / V_DOG
    # 每个源：2 次补充检测 + 2 次频道切换 + 1 次成功清除
    n_meas_term = 2 * n_src
    n_switch_term = 2 * n_src
    T_action = T_MEASURE * n_meas_term + T_SWITCH * n_switch_term + T_CLEAR_OK * n_src

    T_total = T_scan + T_travel + T_action
    return dict(
        n_src=n_src,
        扫描位移=L_scan, 扫描检测次数=n_meas_scan, 扫描切换次数=n_switch_scan,
        阶段一时间=T_scan,
        巡回位移=L_tsp, 侧移位移=L_lateral,
        阶段二移动时间=T_travel, 阶段二动作时间=T_action,
        总时间=T_total, 平均定位清除时间=T_total / n_src,
        理想下界=L_tsp / V_DOG + (2 * T_MEASURE + T_SWITCH + T_CLEAR_OK) * n_src,
    )


# ==================================================================
# 主流程
# ==================================================================
def main():
    print('=' * 74)
    print('问题三：全向干扰源的搜索、定位与清除策略（理论建模）')
    print('=' * 74)

    # ---------------- 3.1 保证覆盖网 ----------------
    print('\n[保证覆盖网求解]')
    net = solve_covering_network()
    Q = net['points']
    print('  七点覆盖网坐标 (m):')
    for k, p in enumerate(Q):
        print(f'    Q{k}: ({p[0]:9.2f}, {p[1]:9.2f})')

    dmax, nbad, ntot = verify_coverage(Q)
    print(f'\n[覆盖性验证] 稠密采样 {ntot} 点，最大最近邻距离 = {dmax:.6f} m '
          f'(保证半径 {R_C_MIN:.0f} m + 数值容差 1e-6)，超出容差点 = {nbad} 个 '
          f'-> 覆盖率 = {100*(1-nbad/ntot):.6f}%')

    # 覆盖半径曲线
    ds = np.linspace(600, 2200, 260)
    rhos = np.array([covering_radius(d, n_boundary=8000, n_radial=120, n_theta=300)[0]
                     for d in ds])
    df_net = pd.DataFrame(dict(环半径d=ds, 覆盖半径rho=rhos,
                               是否可行=rhos <= R_C_MIN))
    save_csv(df_net, '覆盖半径随环半径变化.csv')
    save_csv(pd.DataFrame(dict(点编号=[f'Q{k}' for k in range(len(Q))],
                               x=Q[:, 0], y=Q[:, 1])), '覆盖网点位表.csv')

    # ---------------- 3.2 定位精度阈值 ----------------
    r_thr = approach_threshold()
    print(f'\n[定位精度阈值] 由 E = 3*delta*r <= {CLEAR_R:.0f} m 反解：')
    print(f'  需逼近到 r <= {r_thr:.2f} m 才能保证一次交会即达清除精度')
    rows_thr = []
    for r in [1500, 1200, 900, 600, 400, 381, 300, 200, 100]:
        b_star = np.sqrt(2) * r
        rows_thr.append(dict(源距离r=r, 最优基线b=np.sqrt(2) * r,
                             交会误差E=3 * DELTA * r,
                             是否满足清除精度=bool(3 * DELTA * r <= CLEAR_R)))
    df_thr = pd.DataFrame(rows_thr)
    save_csv(df_thr, '定位精度阈值表.csv')
    print(df_thr.round(3).to_string(index=False))

    # ---------------- 3.3 时间模型与下界 ----------------
    print('\n[虚拟时间模型与理论下界]')
    rows = []
    for n in range(N_SRC_RANGE[0], N_SRC_RANGE[1] + 1):
        rows.append(time_model(n, net))
    df_time = pd.DataFrame(rows)
    cols = ['n_src', '阶段一时间', '阶段二移动时间', '阶段二动作时间',
            '总时间', '平均定位清除时间', '理想下界']
    print(df_time[cols].round(1).to_string(index=False))
    save_csv(df_time, '虚拟时间下界与期望表.csv')

    # ---------------- 3.4 灵敏度 ----------------
    rows_b = []
    for beta in [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]:
        r = time_model(13, net, beta_lateral=beta)
        rows_b.append(dict(侧移系数beta=beta, 总时间=r['总时间'],
                           平均定位清除时间=r['平均定位清除时间'],
                           侧移位移=r['侧移位移']))
    df_beta = pd.DataFrame(rows_b)
    save_csv(df_beta, '侧移系数灵敏度.csv')

    rows_c = []
    for nq in [7, 8, 9, 10, 12]:
        n2 = dict(net)
        # 更多观测点：环点数增加，环半径按同等覆盖半径缩放
        k = nq - 1
        n2['n_points'] = nq
        # 用同覆盖半径近似估计行程
        L = 2 * k * net['d1'] * np.sin(np.pi / k) if k >= 3 else 2 * net['d1']
        n2['d1'] = net['d1']
        r = time_model(13, n2)
        r['观测点数'] = nq
        r['扫描位移'] = L
        rows_c.append(dict(观测点数=nq, 扫描检测次数=N_CHANNEL * nq,
                           扫描位移=L, 总时间=r['总时间']))
    df_nq = pd.DataFrame(rows_c)
    save_csv(df_nq, '观测点数灵敏度.csv')

    # ---------------- 3.5 图1：保证覆盖网（1×2） ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2))
    ax = axes[0]
    th = np.linspace(0, 2 * np.pi, 500)
    ax.plot(R_TARGET * np.cos(th), R_TARGET * np.sin(th), '-', color='#333333',
            linewidth=1.8, label='目标区域边界 (R=1800 m)')
    for k, p in enumerate(Q):
        ax.add_patch(Circle(p, R_C_MIN, fill=(k == 0), alpha=0.10,
                            facecolor=COLORS_10[k % 10], edgecolor=COLORS_10[k % 10],
                            linewidth=1.1))
    ax.add_patch(Circle(Q[0], R_C_MIN, fill=False, edgecolor=COLORS_10[0],
                        linewidth=1.6, label='保证可测圆盘 (半径 1000 m)'))
    # 巡回路径
    tour = np.vstack([Q[0], Q[1:], Q[1]])
    ax.plot(tour[:, 0], tour[:, 1], '--', color=COLORS_10[3], linewidth=1.6,
            label=f'粗探巡回路径（周长 {6*net["d1"]:.0f} m）')
    ax.scatter(Q[1:, 0], Q[1:, 1], s=110, marker='o', color=COLORS_10[2],
               edgecolor='k', zorder=6, label='环上观测点 ($d_1$=1123 m)')
    ax.scatter(Q[0, 0], Q[0, 1], s=170, marker='^', color=COLORS_10[0],
               edgecolor='k', zorder=7, label='机器狗出发点 $Q_0$（原点）')
    ax.set_aspect('equal')
    ax.set_xlim(-2200, 2200)
    ax.set_ylim(-2200, 2200)
    ax.set_xlabel('X 方向坐标 (m)', fontsize=11)
    ax.set_ylabel('Y 方向坐标 (m)', fontsize=11)
    ax.set_title('(a) 七点保证覆盖网与粗探巡回', fontsize=12)
    apply_mpl_style(fig, ax=ax)
    set_legend_cn(ax, loc='upper right', fontsize=8.5)

    ax = axes[1]
    ax.plot(ds, rhos, '-', color=COLORS_10[0], linewidth=2.2, label='覆盖半径 $\\rho(d)$')
    ax.axhline(R_C_MIN, color=COLORS_10[3], linestyle='--', linewidth=1.8,
               label='保证可测半径 1000 m')
    ax.scatter(net['d_star'], net['rho_star'], s=120, marker='*', color=COLORS_10[2],
               edgecolor='k', zorder=6,
               label=f'覆盖半径最小点 $d^*$={net["d_star"]:.0f} m, $\\rho^*$=900 m')
    ax.scatter(net['d1'], R_C_MIN, s=120, marker='D', color=COLORS_10[1],
               edgecolor='k', zorder=6,
               label=f'最小行程可行解 $d_1$={net["d1"]:.0f} m')
    ax.scatter(net['d2'], R_C_MIN, s=120, marker='s', color=COLORS_10[4],
               edgecolor='k', zorder=6, label=f'另一可行解 $d_2$={net["d2"]:.0f} m')
    ax.axvspan(net['d1'], net['d2'], color=COLORS_10[2], alpha=0.10,
               label='覆盖可行的环半径区间')
    ax.set_xlabel('环半径 $d$ (m)', fontsize=11)
    ax.set_ylabel('对目标域的覆盖半径 $\\rho$ (m)', fontsize=11)
    ax.set_title('(b) 覆盖半径随环半径的变化', fontsize=12)
    apply_mpl_style(fig, ax=ax)
    set_legend_cn(ax, loc='upper center', fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.suptitle('保证覆盖网：最少 7 个观测点可覆盖整个目标区域', fontsize=13)
    save_fig(fig, '七点保证覆盖网示意图.png')

    # ---------------- 3.6 图2：分层搜索几何（1×3） ----------------
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.8))
    src = np.array([[620.0, 480.0], [-900.0, 350.0]])
    g = src[0]
    P0 = np.array([0.0, 0.0])
    uv = (g - P0) / np.linalg.norm(g - P0)
    nvec = np.array([-uv[1], uv[0]])
    r_est = float(np.linalg.norm(g - P0))
    b = 0.25 * r_est
    P1 = P0 + b * nvec
    Ghat = g + np.array([35.0, -28.0])
    P2 = Ghat - 0.35 * r_est * uv
    unc = 3 * DELTA * r_est

    # (a) 粗探阶段
    ax = axes[0]
    ax.plot(R_TARGET * np.cos(th), R_TARGET * np.sin(th), '-', color='#333333',
            linewidth=1.6)
    for p_ in Q:
        ax.add_patch(Circle(p_, R_C_MIN, fill=False, edgecolor=COLORS_10[0],
                            alpha=0.22, linewidth=0.9))
    for gi, (gg, cc) in enumerate(zip(src, [COLORS_10[3], COLORS_10[4]])):
        for p_ in Q:
            if np.linalg.norm(p_ - gg) <= R_C_MIN:
                ax.plot([p_[0], gg[0]], [p_[1], gg[1]], ':', color=cc,
                        linewidth=1.0, alpha=0.85)
        ax.scatter(gg[0], gg[1], s=190, marker='*', color=cc, edgecolor='k', zorder=8)
    ax.scatter(Q[:, 0], Q[:, 1], s=90, marker='^', color=COLORS_10[0],
               edgecolor='k', zorder=6)
    ax.scatter(Q[0, 0], Q[0, 1], s=150, marker='^', color=COLORS_10[0],
               edgecolor='k', zorder=7)
    ax.set_aspect('equal')
    ax.set_xlim(-2200, 2200)
    ax.set_ylim(-2200, 2200)
    ax.set_xlabel('X 方向坐标 (m)', fontsize=11)
    ax.set_ylabel('Y 方向坐标 (m)', fontsize=11)
    ax.set_title('(a) 粗探阶段：覆盖网点给出示向度射线', fontsize=12)
    apply_mpl_style(fig, ax=ax)
    set_legend_cn(ax, loc='upper right', fontsize=8.5, handles=[
        plt.Line2D([], [], marker='^', color='w', markerfacecolor=COLORS_10[0],
                   markeredgecolor='k', markersize=10, label='观测点（含出发点）'),
        plt.Line2D([], [], marker='*', color='w', markerfacecolor=COLORS_10[3],
                   markeredgecolor='k', markersize=12, label='干扰源'),
        plt.Line2D([], [], linestyle=':', color=COLORS_10[3], label='示向度射线'),
        plt.Line2D([], [], color=COLORS_10[0], alpha=0.4, linewidth=1.6,
                   label='保证可测圆盘')])

    # (b) 定位阶段整体几何
    ax = axes[1]
    ax.plot([P0[0], P0[0] + 900 * uv[0]], [P0[1], P0[1] + 900 * uv[1]], '-',
            color=COLORS_10[0], linewidth=1.6, label='$P_0$ 处示向度')
    for sgn in (1, -1):
        vv = uv + sgn * np.tan(DELTA) * nvec
        vv = vv / np.linalg.norm(vv)
        ax.plot([P0[0], P0[0] + 900 * vv[0]], [P0[1], P0[1] + 900 * vv[1]], '--',
                color=COLORS_10[0], linewidth=1.0, alpha=0.85,
                label='$\\pm1^\\circ$ 示向度边界' if sgn > 0 else None)
    ax.plot([P0[0], P1[0]], [P0[1], P1[1]], '-', color=COLORS_10[1], linewidth=2.0,
            label=f'侧移定距 $b=0.25r$={b:.0f} m')
    ax.plot([P1[0], Ghat[0]], [P1[1], Ghat[1]], ':', color=COLORS_10[1], linewidth=1.2)
    ax.plot([P1[0], P2[0]], [P1[1], P2[1]], '-', color=COLORS_10[2], linewidth=2.0,
            label='逼近路径')
    ax.add_patch(Circle(Ghat, unc, fill=True, alpha=0.30, facecolor=COLORS_10[3],
                        edgecolor=COLORS_10[3], label=f'定距后不确定区 $3\\delta r$={unc:.0f} m'))
    ax.scatter([P0[0], P1[0], P2[0]], [P0[1], P1[1], P2[1]], s=[130, 110, 110],
               marker='o', color=[COLORS_10[0], COLORS_10[1], COLORS_10[2]],
               edgecolor='k', zorder=7)
    ax.annotate('$P_0$', P0, textcoords='offset points', xytext=(-6, -16), fontsize=10)
    ax.annotate('$P_1$', P1, textcoords='offset points', xytext=(-4, 12), fontsize=10)
    ax.annotate('$P_2$', P2, textcoords='offset points', xytext=(-4, 12), fontsize=10)
    ax.scatter(g[0], g[1], s=200, marker='*', color=COLORS_10[3], edgecolor='k',
               zorder=9, label='干扰源真实位置 $G$')
    ax.plot([P2[0], g[0]], [P2[1], g[1]], '->', color=COLORS_10[2], linewidth=2.0)
    ax.set_aspect('equal')
    lim = 1150
    ax.set_xlim(g[0] - 0.75 * lim, g[0] + lim)
    ax.set_ylim(g[1] - lim, g[1] + lim)
    ax.set_xlabel('X 方向坐标 (m)', fontsize=11)
    ax.set_ylabel('Y 方向坐标 (m)', fontsize=11)
    ax.set_title('(b) 定位阶段：侧移定距与逼近路径', fontsize=12)
    apply_mpl_style(fig, ax=ax)
    set_legend_cn(ax, loc='upper left', fontsize=7.8)

    # (c) 逼近末端放大
    ax = axes[2]
    ax.add_patch(Circle(g, CLEAR_R, fill=True, alpha=0.22, facecolor=COLORS_10[2],
                        edgecolor=COLORS_10[2], linewidth=1.8,
                        label=f'激光清除半径 {CLEAR_R:.0f} m'))
    ax.add_patch(Circle(g, NEAR_R, fill=True, alpha=0.45, facecolor=COLORS_10[3],
                        edgecolor=COLORS_10[3], linewidth=1.6,
                        label=f'近距离阈值 {NEAR_R:.0f} m（直接清除）'))
    t = np.linspace(-55, 12, 200)
    track = g + np.outer(t, uv) + np.outer(np.full_like(t, 3.0), nvec)
    ax.plot(track[:, 0], track[:, 1], '-', color=COLORS_10[2], linewidth=2.0,
            label='逼近航迹（含 $\\pm3$ m 横向残差）')
    ax.scatter(g[0], g[1], s=200, marker='*', color=COLORS_10[3], edgecolor='k',
               zorder=9, label='干扰源 $G$')
    ax.annotate('', xy=g, xytext=g + 60 * uv,
                arrowprops=dict(arrowstyle='-|>', color=COLORS_10[2], lw=2.0))
    ax.set_aspect('equal')
    ax.set_xlim(g[0] - 70, g[0] + 70)
    ax.set_ylim(g[1] - 70, g[1] + 70)
    ax.set_xlabel('X 方向坐标 (m)', fontsize=11)
    ax.set_ylabel('Y 方向坐标 (m)', fontsize=11)
    ax.set_title('(c) 逼近末端：触发 near 后直接清除', fontsize=12)
    apply_mpl_style(fig, ax=ax)
    set_legend_cn(ax, loc='upper left', fontsize=8.5)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.suptitle('分层搜索策略的几何结构（粗探 → 定距 → 逼近 → 清除）', fontsize=13)
    save_fig(fig, '分层搜索几何结构图.png')

    # ---------------- 3.7 图3：时间模型（1×3） ----------------
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))

    ax = axes[0]
    ns = df_time['n_src'].values
    ax.bar(ns, df_time['阶段一时间'], color=COLORS_10[0], label='全域粗探', width=0.62)
    ax.bar(ns, df_time['阶段二移动时间'], bottom=df_time['阶段一时间'],
           color=COLORS_10[2], label='清除阶段移动', width=0.62)
    ax.bar(ns, df_time['阶段二动作时间'],
           bottom=df_time['阶段一时间'] + df_time['阶段二移动时间'],
           color=COLORS_10[1], label='清除阶段检测/切换/清除', width=0.62)
    ax.set_xlabel('干扰源个数 $N$', fontsize=11)
    ax.set_ylabel('虚拟总时间 (s)', fontsize=11)
    ax.set_title('(a) 虚拟总时间构成', fontsize=12)
    apply_mpl_style(fig, ax=ax)
    set_legend_cn(ax, fontsize=9)

    ax = axes[1]
    ax.plot(ns, df_time['总时间'] / 60, 'o-', color=COLORS_10[0], linewidth=2.0,
            markersize=6, label='模型预测总时间')
    ax.plot(ns, df_time['理想下界'] / 60, 's--', color=COLORS_10[3], linewidth=1.8,
            markersize=6, label='理想下界（忽略探测开销）')
    ax.set_xlabel('干扰源个数 $N$', fontsize=11)
    ax.set_ylabel('时间 (min)', fontsize=11)
    ax.set_title('(b) 总时间与理想下界的对比', fontsize=12)
    apply_mpl_style(fig, ax=ax)
    set_legend_cn(ax, fontsize=9)

    ax = axes[2]
    ax.plot(df_beta['侧移系数beta'], df_beta['平均定位清除时间'], 'o-',
            color=COLORS_10[2], linewidth=2.0, markersize=6,
            label='平均定位清除时间')
    ax.axvline(0.25, color='#888888', linestyle='--', linewidth=1.5,
               label='取值 $\\beta=0.25$')
    ax.set_xlabel('侧移系数 $\\beta$', fontsize=11)
    ax.set_ylabel('平均定位清除时间 (s)', fontsize=11)
    ax.set_title('(c) 侧移系数对平均时间的影响', fontsize=12)
    apply_mpl_style(fig, ax=ax)
    set_legend_cn(ax, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.suptitle('问题三虚拟时间模型与灵敏度分析', fontsize=13)
    save_fig(fig, '虚拟时间构成与灵敏度图.png')

    # ---------------- 3.8 输出汇总 ----------------
    print('\n[关键结论]')
    print(f'  最少保证观测点数 N_q = 7（Kershner 定理下界）')
    print(f'  最小行程覆盖网环半径 d1 = {net["d1"]:.2f} m，覆盖半径 = {R_C_MIN:.0f} m')
    print(f'  粗探巡回位移 = {6*net["d1"]:.1f} m，检测 {N_CHANNEL*7} 次，切换 {7*19} 次，'
          f'耗时 {df_time["阶段一时间"].iloc[0]:.1f} s')
    print(f'  清除精度要求的逼近阈值 r <= {r_thr:.1f} m')
    print(f'  N=13 时模型总时间 = {df_time[df_time.n_src==13]["总时间"].iloc[0]:.1f} s, '
          f'平均定位清除时间 = '
          f'{df_time[df_time.n_src==13]["平均定位清除时间"].iloc[0]:.1f} s')
    print('\n问题三求解完成，结果已保存至', OUT_DIR)


if __name__ == '__main__':
    main()
