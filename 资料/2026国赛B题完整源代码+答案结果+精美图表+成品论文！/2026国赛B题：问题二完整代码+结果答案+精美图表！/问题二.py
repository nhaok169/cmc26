# -*- coding: utf-8 -*-
"""
问题二：已知某全向干扰源在一个检测点处测得的示向度，给出第二个检测点的选择策略与候选区域。

坐标系：以 S1 为原点、示向度 θ1 方向为 +x 轴（旋转后）。源 G=(r,0)，r∈[r_lo,r_hi]=[5,1500]。
双点交会定位区域近似平行四边形：面积 A ≈ 4ε²·r·r2²/|y2|，γ 为交会角，sinγ=|y2|/r2。
约束：
  接收（全 r 可达，取保证半径 R0）：max_r r2(r) ≤ R0  <=>  m²+y2² ≤ R0²，m=max(|x2-r_lo|,|x2-r_hi|)
  交会角（全 r ≥ γ_min）：|y2| ≥ tan(γ_min)·m
候选区域 C = {(x2,y2): tan(γ_min)·m ≤ |y2| ≤ sqrt(R0²−m²)}，最优取 x2=r_mid, |y2|=sqrt(R0²−h²)。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import *
from scipy.spatial import HalfspaceIntersection
import itertools

FIG = FIG_DIR
OUT = OUT_DIR

EPS_DEG = 1.0
EPS = np.deg2rad(EPS_DEG)
R_LO, R_HI = 5.0, 1500.0      # 源距 S1 的可达范围（收到示向度⇒>5 且 ≤有效半径 1500）
R0 = 1000.0                    # 保证接收半径（有效半径下限）
GAMMA_MIN = np.deg2rad(30.0)   # 交会角下限

# ---------- 几何工具 ----------
def unit(deg):
    a = np.deg2rad(deg)
    return np.array([np.cos(a), np.sin(a)])

def left_normal(deg):
    a = np.deg2rad(deg)
    return np.array([-np.sin(a), np.cos(a)])

def ang_of(v):
    return (np.degrees(np.arctan2(v[1], v[0]))) % 360.0

def build_halfspaces(S, theta, eps=1.0):
    H = []
    for Si, ti in zip(S, theta):
        n1 = left_normal(ti - eps); n2 = left_normal(ti + eps)
        H.append([-n1[0], -n1[1], n1[0]*Si[0] + n1[1]*Si[1]])
        H.append([ n2[0],  n2[1], -n2[0]*Si[0] - n2[1]*Si[1]])
    return np.array(H)

def chebyshev_center(H):
    from scipy.optimize import linprog
    A = H[:, :2]; b = H[:, 2]
    c = [0.0, 0.0, -1.0]
    A_ub = np.hstack([A, np.linalg.norm(A, axis=1, keepdims=True)])
    b_ub = -b
    bounds = [(None, None), (None, None), (0.0, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    return None if res.x is None else res.x[:2]

def polygon_vertices(S, theta, eps=1.0, true_G=None):
    H = build_halfspaces(S, theta, eps)
    for ip in [chebyshev_center(H), true_G]:
        if ip is None:
            continue
        ip = np.asarray(ip, float).ravel()
        if ip.shape[0] != 2:
            continue
        try:
            verts = HalfspaceIntersection(H, ip).intersections
            c = verts.mean(axis=0)
            verts = verts[np.argsort([ang_of(v - c) for v in verts])]
            return verts
        except Exception:
            continue
    return None

def polygon_area(verts):
    n = len(verts); s = 0.0
    for i in range(n):
        x1, y1 = verts[i]; x2, y2 = verts[(i+1) % n]
        s += x1*y2 - x2*y1
    return abs(s) / 2

def area_approx(r, x2, y2):
    """A ≈ 4ε²·r·r2²/|y2|"""
    if abs(y2) < 1e-9:
        return np.inf
    r2 = np.hypot(x2 - r, y2)
    return 4 * EPS**2 * r * r2**2 / abs(y2)

def worst_area(x2, y2, rs):
    return max(area_approx(r, x2, y2) for r in rs)

# ---------- 候选区域（闭式） ----------
def m_of(x2):
    return max(abs(x2 - R_LO), abs(x2 - R_HI))

def candidate_region(R0v=R0, gamma_min=GAMMA_MIN, n=600):
    """返回候选区域边界点 (x2, y2_lo, y2_hi)"""
    xlim = [R_LO - R0v, R_HI + R0v]
    xs = np.linspace(xlim[0], xlim[1], n)
    xin, lo, hi = [], [], []
    for x in xs:
        m = m_of(x)
        if m > R0v * np.cos(gamma_min):
            continue
        ylo = np.tan(gamma_min) * m
        yhi = np.sqrt(R0v**2 - m**2)
        if yhi >= ylo:
            xin.append(x); lo.append(ylo); hi.append(yhi)
    return np.array(xin), np.array(lo), np.array(hi)

def optimal_S2(R0v=R0):
    r_mid = (R_LO + R_HI) / 2.0
    h = (R_HI - R_LO) / 2.0
    y2 = np.sqrt(R0v**2 - h**2)
    return r_mid, y2, h

# ---------- 图 2-1 单点示向度与可行源弧段 ----------
def fig1_feasible_segment():
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_aspect('equal', adjustable='box')
    # ±1° 扇区
    L = 2100.0
    for dd in (-1.0, 1.0):
        uu = unit(dd)
        ax.plot([0, L*uu[0]], [0, L*uu[1]], color='#d53f8c', lw=1.1, ls=':', alpha=0.9)
    u = unit(0)
    ax.plot([0, L], [0, 0], color='#6b46c1', lw=1.5, ls='--')
    t = np.linspace(-1, 1, 60)
    arcx = L*np.cos(np.deg2rad(t)); arcy = L*np.sin(np.deg2rad(t))
    poly = np.vstack([[0, 0], np.column_stack([arcx, arcy])])
    ax.fill(poly[:, 0], poly[:, 1], color='#d53f8c', alpha=0.18, lw=0)
    # 可行源弧段 r∈[5,1500]
    ax.plot([R_LO, R_HI], [0, 0], color='#0d9488', lw=5, alpha=0.85, solid_capstyle='butt')
    ax.plot([R_LO], [0], 'o', ms=7, color='#0d9488', mec='k', mew=0.9)
    ax.plot([R_HI], [0], 's', ms=8, color='#0d9488', mec='k', mew=0.9)
    ax.annotate(f'r$_{{lo}}$={R_LO:.0f} m', (R_LO-140, 90), fontsize=10, color='#0f766e')
    ax.annotate(f'r$_{{hi}}$={R_HI:.0f} m', (R_HI-30, 90), fontsize=10, color='#0f766e')
    # 接收半径示意：两个端点处的 1000m 圆
    ax.add_patch(Circle((R_LO, 0), R0, fill=False, ec='#d97706', lw=1.2, ls='-', alpha=0.7))
    ax.add_patch(Circle((R_HI, 0), R0, fill=False, ec='#d97706', lw=1.2, ls='-', alpha=0.7))
    ax.annotate('有效接收半径 R=1000~1500 m', (R_LO-100, 1080), fontsize=9, color='#92400e')
    ax.plot([0], [0], 'o', ms=9, color='#6b46c1', mec='k', mew=1.0, zorder=5)
    ax.annotate('S$_1$', (-10, -120), fontsize=12, color='#5b21b6')
    ax.set_xlabel('沿示向度方向距离 (m)'); ax.set_ylabel('垂直示向度方向 (m)')
    ax.set_xlim(-350, 2350); ax.set_ylim(-1150, 1250)
    despine(ax)
    save_fig(fig, '图2-1_单点示向度与可行源弧段')

# ---------- 图 2-2 候选区域（闭式） ----------
def fig2_candidate_region():
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.set_aspect('equal', adjustable='box')
    xs, lo, hi = candidate_region(R0, GAMMA_MIN)
    # 上下对称
    ax.fill_between(xs, -hi, -lo, color='#0d9488', alpha=0.28, lw=0)
    ax.fill_between(xs, lo, hi, color='#0d9488', alpha=0.28, lw=0)
    ax.plot(xs, lo, color='#6b46c1', lw=1.6, label=f'交会角下限 γ≥{np.degrees(GAMMA_MIN):.0f}°')
    ax.plot(xs, hi, color='#d53f8c', lw=1.6, label=f'接收上限 r2≤{R0:.0f} m')
    ax.plot(xs, -lo, color='#6b46c1', lw=1.6)
    ax.plot(xs, -hi, color='#d53f8c', lw=1.6)
    # 可行源弧段
    ax.plot([R_LO, R_HI], [0, 0], color='#0d9488', lw=4, alpha=0.8, solid_capstyle='butt')
    ax.plot([0], [0], 'o', ms=8, color='#6b46c1', mec='k', mew=1.0, zorder=5)
    ax.annotate('S$_1$', (-10, -120), fontsize=11, color='#5b21b6')
    # 推荐点
    xo, yo, _ = optimal_S2(R0)
    ax.plot([xo, xo], [yo, -yo], 'o', ms=10, color='#dc2626', mec='k', mew=1.1, zorder=6)
    ax.annotate(f'S$_2$* = ({xo:.0f}, ±{yo:.0f})', (xo-30, yo+130), fontsize=11, color='#991b1b')
    ax.set_xlabel('沿示向度方向 x$_2$ (m)'); ax.set_ylabel('垂直示向度方向 y$_2$ (m)')
    ax.set_xlim(-300, 2100); ax.set_ylim(-1150, 1150)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    despine(ax)
    save_fig(fig, '图2-2_第二检测点候选区域')

# ---------- 图 2-3 最坏定位面积热力 ----------
def fig3_area_heatmap():
    rs = np.linspace(R_LO, R_HI, 200)
    X = np.linspace(-200, 1700, 45)
    Y = np.linspace(50, 1200, 42)
    XX, YY = np.meshgrid(X, Y)
    A = np.zeros_like(XX)
    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            A[i, j] = worst_area(XX[i, j], YY[i, j], rs)
    A_log = np.log10(A)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    mesh = ax.pcolormesh(XX, YY, A_log, cmap='YlGnBu', shading='auto',
                         edgecolors='black', linewidth=0.25)
    cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('lg(最坏定位区域面积 A$_{max}$ / m$^2$)')
    # 候选区域边界叠加
    xs, lo, hi = candidate_region(R0, GAMMA_MIN)
    ax.plot(xs, lo, color='#dc2626', lw=1.8)
    ax.plot(xs, hi, color='#dc2626', lw=1.8)
    ax.plot([R_LO, R_HI], [0, 0], color='#ffffff', lw=3, alpha=0.9)
    xo, yo, _ = optimal_S2(R0)
    ax.plot([xo], [yo], 'o', ms=11, color='#fbbf24', mec='k', mew=1.2, zorder=6)
    ax.annotate('S$_2$* (最优)', (xo-20, yo+90), fontsize=10, color='#92400e')
    ax.set_xlabel('x$_2$ (m)'); ax.set_ylabel('y$_2$ (m)')
    ax.set_xlim(-200, 1700); ax.set_ylim(50, 1200)
    despine(ax)
    save_fig(fig, '图2-3_最坏定位面积热力与候选区域')

# ---------- 图 2-4 垂直基线策略分析 ----------
def fig4_perpendicular():
    Ls = np.linspace(100, 1800, 300)
    rs = np.linspace(R_LO, R_HI, 120)
    gamma_worst = []; A_worst = []
    for L in Ls:
        # 垂直基线 S2=(0,L)：γ(r)=atan(L/r)，最坏在 r=1500；接收 r2=sqrt(r²+L²)
        g = np.arctan2(L, R_HI)
        gamma_worst.append(np.degrees(g))
        A_worst.append(worst_area(0.0, L, rs))
    A_worst = np.array(A_worst)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
    ax1.plot(Ls, gamma_worst, color='#6b46c1', lw=2.0, label='最坏交会角 γ (r=1500m)')
    ax1.axhline(np.degrees(GAMMA_MIN), color='#0d9488', lw=1.2, ls='--')
    ax1.annotate('γ$_min$=30°', xy=(1500, 31), xytext=(1560, 35), fontsize=9, color='#0f766e')
    ax1.set_xlabel('垂直基线长度 L (m)'); ax1.set_ylabel('最坏交会角 γ (deg)')
    ax1.set_ylim(0, 100); ax1.grid(axis='y', lw=0.5, alpha=0.3, color='#94a3b8')
    ax2.plot(Ls, A_worst, color='#d53f8c', lw=2.0, label='最坏定位面积 A$_{max}$')
    ax2.set_xlabel('垂直基线长度 L (m)'); ax2.set_ylabel('最坏定位面积 (m$^2$)')
    ax2.grid(axis='y', lw=0.5, alpha=0.3, color='#94a3b8')
    ax2.set_yscale('log')
    ax1.legend(fontsize=9, loc='upper left'); ax2.legend(fontsize=9, loc='upper left')
    for a, lab in ((ax1, '(a) 最坏交会角 γ 随 L'), (ax2, '(b) 最坏定位面积 A$_{max}$ 随 L')):
        despine(a); a.tick_params(direction='in')
        a.text(0.5, -0.22, lab, transform=a.transAxes, ha='center', fontsize=11, fontweight='bold')
    save_fig(fig, '图2-4_垂直基线长度对交会角与面积的影响')

# ---------- 图 2-5 推荐点 vs 朴素点 定位区域验证 ----------
def fig5_verify():
    S1 = np.array([0.0, 0.0])
    xo, yo, _ = optimal_S2(R0)
    S2_opt = np.array([xo, yo])
    S2_naive = np.array([0.0, 1500.0])   # 垂直远处（接收差）
    S2_colin = np.array([1500.0, 0.0])   # 共线（交会角≈0）
    r_test = R_HI
    G = np.array([r_test, 0.0])
    cases = [('推荐 S$_2$', S2_opt, '#0d9488'),
             ('垂直远处 S$_2$', S2_naive, '#d97706'),
             ('共线 S$_2$', S2_colin, '#dc2626')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, S2, col) in zip(axes, cases):
        S = np.array([S1, S2])
        th = [ang_of(G - Si) for Si in S]
        verts = polygon_vertices(S, np.array(th), EPS_DEG, G)
        if verts is not None:
            p = np.vstack([verts, verts[0]])
            ax.fill(p[:, 0], p[:, 1], color=col, alpha=0.45, edgecolor='k', lw=1.0)
            a = polygon_area(verts)
            ax.annotate(f'面积={a:.0f} m$^2$', xy=(0.03, 0.93), xycoords='axes fraction',
                        fontsize=11, color='#1f2937')
        ax.plot(S[:, 0], S[:, 1], 'o', ms=8, color='#6b46c1', mec='k', mew=1.0)
        ax.plot([G[0]], [G[1]], '*', ms=16, color='#dc2626', mec='k', mew=1.0)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        ax.set_title('')  # 标题由 caption 承担
        ax.annotate(name, xy=(0.03, 0.03), xycoords='axes fraction', fontsize=11, color='#374151',
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#9ca3af', lw=0.7))
        despine(ax)
    for ax, lab in zip(axes, ['(a) 推荐 S$_2$（x$_2$=752.5, y$_2$=664）',
                              '(b) 垂直远处 S$_2$（L=1500）',
                              '(c) 共线 S$_2$（交会角≈0）']):
        ax.text(0.5, -0.2, lab, transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold')
    save_fig(fig, '图2-5_不同S2方案的定位区域对比')

# ---------- 图 2-6 候选区域随参数变化 ----------
def fig6_param():
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.set_aspect('equal', adjustable='box')
    colors = ['#6b46c1', '#0d9488', '#d97706']
    for R0v, c in zip([1000.0, 1250.0, 1500.0], colors):
        xs, lo, hi = candidate_region(R0v, GAMMA_MIN)
        ax.fill_between(xs, lo, hi, color=c, alpha=0.12, lw=0)
        ax.fill_between(xs, -hi, -lo, color=c, alpha=0.12, lw=0)
        ax.plot(xs, hi, color=c, lw=1.7, label=f'R$_0$={R0v:.0f} m')
        ax.plot(xs, -hi, color=c, lw=1.7)
        ax.plot(xs, lo, color=c, lw=1.2, ls=':')
        ax.plot(xs, -lo, color=c, lw=1.2, ls=':')
    ax.plot([R_LO, R_HI], [0, 0], color='#374151', lw=4, alpha=0.8)
    ax.plot([0], [0], 'o', ms=8, color='#6b46c1', mec='k', mew=1.0, zorder=5)
    ax.annotate('S$_1$', (-10, -120), fontsize=11, color='#5b21b6')
    for R0v, c in zip([1000.0, 1250.0, 1500.0], colors):
        xo, yo, _ = optimal_S2(R0v)
        ax.plot([xo], [yo], 'o', ms=9, color=c, mec='k', mew=1.0, zorder=6)
    ax.set_xlabel('x$_2$ (m)'); ax.set_ylabel('y$_2$ (m)')
    ax.set_xlim(-300, 2100); ax.set_ylim(-1450, 1450)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    despine(ax)
    save_fig(fig, '图2-6_候选区域随保证接收半径的变化')

# ---------- 主流程 ----------
if __name__ == '__main__':
    print('=' * 64)
    print('问题二：第二检测点选择策略与候选区域')
    print(f'源距范围 r∈[{R_LO:.0f},{R_HI:.0f}] m，误差 ε=±{EPS_DEG}°，交会角下限 γ_min={np.degrees(GAMMA_MIN):.0f}°，保证接收半径 R0={R0:.0f} m')
    xo, yo, h = optimal_S2(R0)
    print(f'推荐 S2：沿示向度方向 x2 = {(R_LO+R_HI)/2:.1f} m，垂直偏移 y2 = ±{yo:.1f} m')
    print(f'  -> 最坏交会角 γ = {np.degrees(np.arctan2(yo, h)):.2f}°（r=r_hi 处）')
    A_star = worst_area(xo, yo, np.linspace(R_LO, R_HI, 400))
    print(f'  -> 最坏定位面积 A_max ≈ {A_star:.1f} m^2')
    # 交叉验证：用问题一的半平面交计算真实多边形面积
    S1 = np.array([0.0, 0.0]); S2 = np.array([xo, yo]); G = np.array([R_HI, 0.0])
    th = [ang_of(G - S1), ang_of(G - S2)]
    verts = polygon_vertices(np.array([S1, S2]), np.array(th), EPS_DEG, G)
    a_real = polygon_area(verts)
    print(f'[交叉验证] 半平面交真实面积 = {a_real:.1f} m^2，近似 A = 4ε²·r·r2²/|y2| = {A_star:.1f} m^2，相对差 {abs(a_real-A_star)/a_real*100:.1f}%')
    print('=' * 64)

    fig1_feasible_segment()
    fig2_candidate_region()
    fig3_area_heatmap()
    fig4_perpendicular()
    fig5_verify()
    fig6_param()

    # CSV：候选区域边界 + 推荐点 + 策略对比
    xs, lo, hi = candidate_region(R0, GAMMA_MIN)
    df = pd.DataFrame({'x2(m)': np.round(xs, 1), 'y2_lo(m)': np.round(lo, 1), 'y2_hi(m)': np.round(hi, 1)})
    df.to_csv(os.path.join(OUT, '问题二_候选区域边界.csv'), index=False, encoding='utf-8-sig')
    rows = [{'策略': '推荐(垂直+中点, R0=1000)', 'S2_x(m)': xo, 'S2_y(m)': yo, '最坏交会角(deg)': round(np.degrees(np.arctan2(yo, h)), 2),
             '最坏定位面积(m2)': round(worst_area(xo, yo, np.linspace(R_LO, R_HI, 400)), 1)},
            {'策略': '垂直远处(L=1500)', 'S2_x(m)': 0, 'S2_y(m)': 1500, '最坏交会角(deg)': 45.0,
             '最坏定位面积(m2)': round(worst_area(0, 1500, np.linspace(R_LO, R_HI, 400)), 1)},
            {'策略': '共线(S2在源向远侧)', 'S2_x(m)': 1500, 'S2_y(m)': 0, '最坏交会角(deg)': 0.0,
             '最坏定位面积(m2)': np.inf}]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, '问题二_策略对比.csv'), index=False, encoding='utf-8-sig')
    print('已输出 CSV：候选区域边界 / 策略对比')
    print('全部图片与结果输出完毕。')
