# -*- coding: utf-8 -*-
"""
问题一：交会定位区域直径算法与覆盖判定
给定若干检测点坐标及其关于某干扰源的示向度（含 ±1° 误差），
1) 用半平面交构造交会定位区域（凸多边形）
2) 计算定位区域直径（区域任意两点距离最大值）
3) 计算最小覆盖圆，判定"以定位区域直径为直径的圆能否覆盖该区域"（Jung 定理）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import *
from scipy.spatial import HalfspaceIntersection
import itertools

BASE = BASE_DIR
FIG = FIG_DIR
OUT = OUT_DIR

# ---------------- 几何基础 ----------------
def unit(deg):
    a = np.deg2rad(deg)
    return np.array([np.cos(a), np.sin(a)])

def left_normal(deg):
    """方向角 deg 的左法向量 n_L = (-sin, cos)"""
    a = np.deg2rad(deg)
    return np.array([-np.sin(a), np.cos(a)])

def ang_of(v):
    """向量方向角（度，[0,360)）"""
    return (np.degrees(np.arctan2(v[1], v[0]))) % 360.0

def build_halfspaces(S, theta, eps=1.0):
    """
    每个检测点 S_i 处示向度 theta_i，误差 eps，构造半平面 A*x+B*y+C<=0。
    定位区域 = {G : 对所有 i, nL(theta_i-eps)·(G-S_i)>=0 且 nL(theta_i+eps)·(G-S_i)<=0}
    """
    H = []
    for Si, ti in zip(S, theta):
        n1 = left_normal(ti - eps)   # 下界：n1·(G-S)>=0  ->  -n1·G + n1·S <=0
        n2 = left_normal(ti + eps)   # 上界：n2·(G-S)<=0  ->   n2·G - n2·S <=0
        H.append([-n1[0], -n1[1],  n1[0]*Si[0] + n1[1]*Si[1]])
        H.append([ n2[0],  n2[1], -n2[0]*Si[0] - n2[1]*Si[1]])
    return np.array(H)

def ls_interior_point(S, theta):
    """用各示向线（中心线）的最小二乘交点作为半平面交的内点"""
    A = np.zeros((2, 2)); b = np.zeros(2)
    for Si, ti in zip(S, theta):
        n = left_normal(ti)
        A += np.outer(n, n)
        b += (n @ Si) * n
    return np.linalg.solve(A, b)

def chebyshev_center(H):
    """半平面 a·x+b<=0 的切比雪夫中心（最大化内切半径的内点，线性规划）"""
    from scipy.optimize import linprog
    A = H[:, :2]; b = H[:, 2]
    n = A.shape[0]
    c = [0.0, 0.0, -1.0]
    A_ub = np.hstack([A, np.linalg.norm(A, axis=1, keepdims=True)])
    b_ub = -b
    bounds = [(None, None), (None, None), (0.0, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.x is None:
        return None
    return res.x[:2]

def polygon_vertices(S, theta, eps=1.0, true_G=None):
    """返回定位区域顶点（凸多边形，逆时针）"""
    H = build_halfspaces(S, theta, eps)
    for ip in [chebyshev_center(H), ls_interior_point(S, theta), true_G]:
        if ip is None:
            continue
        ip = np.asarray(ip, dtype=float).ravel()
        if ip.shape[0] != 2:
            continue
        try:
            hs = HalfspaceIntersection(H, ip)
            verts = hs.intersections
            break
        except Exception:
            continue
    # 逆时针排序（按相对质心的方位角）
    c = verts.mean(axis=0)
    order = np.argsort([ang_of(v - c) for v in verts])
    verts = verts[order]
    return verts

def convex_polygon_area(verts):
    n = len(verts)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = verts[i]; x2, y2 = verts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0

def polygon_diameter(verts):
    """凸多边形直径（O(n^2) 精确；n 小）返回 (d, i, j)"""
    n = len(verts)
    d2 = 0.0; pi = pj = 0
    for i in range(n):
        for j in range(i + 1, n):
            dd = np.sum((verts[i] - verts[j]) ** 2)
            if dd > d2:
                d2 = dd; pi, pj = i, j
    return np.sqrt(d2), pi, pj

def _circumcircle(a, b, c):
    ax, ay = a; bx, by = b; cx, cy = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    r = np.hypot(ux - ax, uy - ay)
    return np.array([ux, uy]), r

def minimal_enclosing_circle(verts):
    """最小覆盖圆：枚举 2 点直径圆与 3 点外接圆，取能覆盖全部的最小者"""
    pts = [np.asarray(p, float) for p in verts]
    n = len(pts)
    if n == 1:
        return pts[0], 0.0
    if n == 2:
        return (pts[0] + pts[1]) / 2, np.linalg.norm(pts[0] - pts[1]) / 2
    best_c, best_r = None, np.inf
    for i, j in itertools.combinations(range(n), 2):
        c = (pts[i] + pts[j]) / 2
        r = np.linalg.norm(pts[i] - pts[j]) / 2
        if all(np.linalg.norm(p - c) <= r + 1e-9 for p in pts):
            if r < best_r:
                best_r, best_c = r, c
    for i, j, k in itertools.combinations(range(n), 3):
        cc = _circumcircle(pts[i], pts[j], pts[k])
        if cc is None:
            continue
        c, r = cc
        if all(np.linalg.norm(p - c) <= r + 1e-9 for p in pts):
            if r < best_r:
                best_r, best_c = r, c
    return best_c, best_r

def analyze(S, theta, true_G, eps=1.0):
    verts = polygon_vertices(S, theta, eps, true_G)
    d, i, j = polygon_diameter(verts)
    c_mec, r_mec = minimal_enclosing_circle(verts)
    cover = (r_mec <= d / 2 + 1e-9)
    return verts, d, (i, j), c_mec, r_mec, cover

# ---------------- 演示案例 ----------------
def demo_case():
    # 两个检测点 + 一个真实源（生成含误差示向度）
    S = np.array([[0.0, 0.0], [1200.0, 200.0]])
    G = np.array([900.0, 1100.0])
    rng = np.random.default_rng(7)
    theta = []
    for Si in S:
        tb = ang_of(G - Si)
        err = rng.uniform(-1, 1)
        theta.append((tb + err) % 360.0)
    return S, G, np.array(theta)

def three_point_case():
    S = np.array([[-400.0, -300.0], [1400.0, -500.0], [300.0, 1500.0]])
    G = np.array([400.0, 300.0])
    rng = np.random.default_rng(11)
    theta = []
    for Si in S:
        tb = ang_of(G - Si)
        theta.append((tb + rng.uniform(-1, 1)) % 360.0)
    return S, G, np.array(theta)

# ---------------- 图 1-1 交会定位原理 ----------------
def fig1_principle(S, G, theta, verts):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect('equal', adjustable='box')
    eps = 1.0
    # 每个检测点的 ±1° 扇区
    for Si, ti in zip(S, theta):
        # 中心示向线
        u = unit(ti)
        L = 2600.0
        ax.plot([Si[0], Si[0] + L * u[0]], [Si[1], Si[1] + L * u[1]],
                color='#2b6cb0', lw=1.4, ls='--', alpha=0.85)
        # 误差边界射线
        for dd in (-eps, eps):
            uu = unit(ti + dd)
            ax.plot([Si[0], Si[0] + L * uu[0]], [Si[1], Si[1] + L * uu[1]],
                    color='#d97706', lw=1.0, ls=':', alpha=0.9)
        # 扇区填充
        a0 = ti - eps; a1 = ti + eps
        t = np.linspace(a0, a1, 60)
        arcx = Si[0] + L * np.cos(np.deg2rad(t))
        arcy = Si[1] + L * np.sin(np.deg2rad(t))
        poly = np.vstack([[Si], np.column_stack([arcx, arcy])])
        ax.fill(poly[:, 0], poly[:, 1], color='#f6ad55', alpha=0.22, lw=0)
    # 定位区域
    p = np.vstack([verts, verts[0]])
    ax.fill(p[:, 0], p[:, 1], color='#e53e3e', alpha=0.5, edgecolor='#9b2c2c', lw=1.2)
    ax.plot(S[:, 0], S[:, 1], 'o', ms=8, color='#2b6cb0', zorder=5, mec='k', mew=1.0)
    ax.plot([G[0]], [G[1]], '*', ms=18, color='#2f855a', zorder=6, mec='k', mew=1.0)
    for i, Si in enumerate(S):
        ax.annotate(f'S$_{i+1}$', (Si[0] + 40, Si[1] - 80), fontsize=11)
    ax.annotate('G', (G[0] + 40, G[1] + 40), fontsize=12, color='#276749')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_xlim(-500, 2900); ax.set_ylim(-900, 2200)
    despine(ax)
    save_fig(fig, '图1-1_交会定位原理与定位区域')

# ---------------- 图 1-2 定位区域与直径 ----------------
def fig2_diameter(verts, d, idx):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect('equal', adjustable='box')
    p = np.vstack([verts, verts[0]])
    ax.fill(p[:, 0], p[:, 1], color='#4299e1', alpha=0.45, edgecolor='#2b6cb0', lw=1.3)
    ax.plot(verts[:, 0], verts[:, 1], 'o', ms=6, color='#2b6cb0', mec='k', mew=0.8)
    # 所有顶点对距离（灰），直径（红粗）
    for a, b in itertools.combinations(range(len(verts)), 2):
        ax.plot([verts[a][0], verts[b][0]], [verts[a][1], verts[b][1]],
                color='#a0aec0', lw=0.6, alpha=0.7, zorder=1)
    i, j = idx
    ax.plot([verts[i][0], verts[j][0]], [verts[i][1], verts[j][1]],
            color='#e53e3e', lw=2.6, zorder=3)
    ax.plot([verts[i][0], verts[j][0]], [verts[i][1], verts[j][1]],
            'o', ms=9, color='#e53e3e', mec='k', mew=1.0, zorder=4)
    mid = (verts[i] + verts[j]) / 2
    ax.annotate(f'd = {d:.1f} m', (mid[0] + 15, mid[1] - 25), fontsize=12, color='#9b2c2c')
    # 顶点坐标标注
    for k, v in enumerate(verts):
        ax.annotate(f'V$_{k+1}$({v[0]:.0f},{v[1]:.0f})', (v[0] + 12, v[1] + 12), fontsize=8)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    despine(ax)
    save_fig(fig, '图1-2_定位区域与直径')

# ---------------- 图 1-3 直径圆 vs 最小覆盖圆 ----------------
def fig3_cover(verts, d, c_mec, r_mec, cover):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect('equal', adjustable='box')
    p = np.vstack([verts, verts[0]])
    ax.fill(p[:, 0], p[:, 1], color='#f6e05e', alpha=0.5, edgecolor='#975a16', lw=1.2)
    # 直径圆（以直径中点为圆心，d/2 为半径）
    i, j = idx_global
    a, b = verts[i], verts[j]
    cd = (a + b) / 2
    ax.add_patch(Circle(cd, d / 2, fill=False, ec='#e53e3e', lw=2.0, ls='--'))
    ax.plot([a[0], b[0]], [a[1], b[1]], color='#e53e3e', lw=2.0)
    # 最小覆盖圆
    ax.add_patch(Circle(c_mec, r_mec, fill=False, ec='#2b6cb0', lw=2.0))
    ax.plot([c_mec[0]], [c_mec[1]], '+', ms=14, color='#2b6cb0', mew=2.0)
    # 顶点
    ax.plot(verts[:, 0], verts[:, 1], 'o', ms=7, color='#975a16', mec='k', mew=0.9)
    ax.annotate(f'直径圆 r={d/2:.1f}', (cd[0] + 8, cd[1] + d / 2 - 18), color='#9b2c2c', fontsize=10)
    ax.annotate(f'最小覆盖圆 R={r_mec:.1f}', (c_mec[0] + 8, c_mec[1] + r_mec - 18), color='#2b6cb0', fontsize=10)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    despine(ax)
    save_fig(fig, '图1-3_直径圆与最小覆盖圆对比')

# ---------------- 图 1-4 三检测点 ----------------
def fig4_three(S, G, theta, verts):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect('equal', adjustable='box')
    eps = 1.0; L = 2600.0
    for Si, ti in zip(S, theta):
        u = unit(ti)
        ax.plot([Si[0], Si[0] + L * u[0]], [Si[1], Si[1] + L * u[1]],
                color='#2b6cb0', lw=1.2, ls='--', alpha=0.8)
        for dd in (-eps, eps):
            uu = unit(ti + dd)
            ax.plot([Si[0], Si[0] + L * uu[0]], [Si[1], Si[1] + L * uu[1]],
                    color='#d97706', lw=0.9, ls=':', alpha=0.85)
    p = np.vstack([verts, verts[0]])
    ax.fill(p[:, 0], p[:, 1], color='#48bb78', alpha=0.55, edgecolor='#276749', lw=1.3)
    d, i, j = polygon_diameter(verts)
    ax.plot([verts[i][0], verts[j][0]], [verts[i][1], verts[j][1]],
            color='#e53e3e', lw=2.4)
    ax.plot(S[:, 0], S[:, 1], 'o', ms=8, color='#2b6cb0', zorder=5, mec='k', mew=1.0)
    ax.plot([G[0]], [G[1]], '*', ms=18, color='#9b2c2c', zorder=6, mec='k', mew=1.0)
    for i_, Si in enumerate(S):
        ax.annotate(f'S$_{i_+1}$', (Si[0] + 40, Si[1] - 80), fontsize=11)
    ax.annotate('G', (G[0] + 40, G[1] + 40), fontsize=12, color='#9b2c2c')
    mid = (verts[i] + verts[j]) / 2
    ax.annotate(f'd={d:.1f} m', (mid[0] + 15, mid[1] + 15), fontsize=11, color='#9b2c2c')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_xlim(-900, 2400); ax.set_ylim(-1200, 2300)
    despine(ax)
    save_fig(fig, '图1-4_三检测点交会定位区域')

# ---------------- 图 1-5 覆盖比分布 ----------------
def fig5_ratio_distribution():
    rng = np.random.default_rng(42)
    ratios = []
    d_vals = []; r_vals = []
    for t in range(120):
        # 随机 2~4 个检测点
        m = int(rng.integers(2, 5))
        S = rng.uniform(-1500, 1500, size=(m, 2))
        G = rng.uniform(-800, 800, size=2)
        theta = []
        for Si in S:
            tb = ang_of(G - Si)
            theta.append((tb + rng.uniform(-1, 1)) % 360.0)
        try:
            verts = polygon_vertices(S, np.array(theta), 1.0, G)
            if len(verts) < 3:
                continue
            d, _, _ = polygon_diameter(verts)
            c_mec, r_mec = minimal_enclosing_circle(verts)
            ratios.append(r_mec / (d / 2))
            d_vals.append(d); r_vals.append(r_mec)
        except Exception:
            continue
    ratios = np.array(ratios)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(ratios, bins=24, color='#4299e1', edgecolor='k', lw=0.8, alpha=0.85)
    ax.axvline(1.0, color='#e53e3e', lw=1.6, ls='--')
    ax.axvline(2 / np.sqrt(3), color='#2f855a', lw=1.6, ls='--')
    ax.annotate('R/(d/2)=1（直径圆恰覆盖）', xy=(1.0, 2), xytext=(1.02, 12),
                fontsize=9, color='#9b2c2c')
    ax.annotate(f'Jung 上界 {2/np.sqrt(3):.4f}', xy=(2/np.sqrt(3), 2),
                xytext=(1.07, 6), fontsize=9, color='#276749')
    ax.set_xlabel('R$_{mec}$ / (d/2)')
    ax.set_ylabel('频数')
    ax.set_xlim(0.98, 1.17)
    despine(ax)
    save_fig(fig, '图1-5_覆盖半径比分布')

# ---------------- 图 1-6 等边三角形 Jung 临界 ----------------
def fig6_jung():
    # 边长 s 的等边三角形：直径=边长，最小覆盖圆半径=s/sqrt(3)
    s = 1.0
    tri = np.array([[0.0, 0.0], [s, 0.0], [0.5, np.sqrt(3) / 2 * s]])
    d, i, j = polygon_diameter(tri)
    c_mec, r_mec = minimal_enclosing_circle(tri)
    a, b = tri[i], tri[j]
    cd = (a + b) / 2
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.set_aspect('equal', adjustable='box')
    ax.add_patch(Polygon(tri, closed=True, fill=True, facecolor='#f6ad55', alpha=0.6, edgecolor='#975a16', lw=1.4))
    ax.add_patch(Circle(cd, d / 2, fill=False, ec='#e53e3e', lw=2.0, ls='--'))
    ax.add_patch(Circle(c_mec, r_mec, fill=False, ec='#2b6cb0', lw=2.0))
    ax.plot(tri[:, 0], tri[:, 1], 'o', ms=8, color='#975a16', mec='k', mew=1.0)
    ax.plot([a[0], b[0]], [a[1], b[1]], color='#e53e3e', lw=2.0)
    ax.plot([c_mec[0]], [c_mec[1]], '+', ms=14, color='#2b6cb0', mew=2.0)
    ax.annotate(f'直径 d={d:.3f}', (0.18, -0.12), fontsize=10, color='#9b2c2c')
    ax.annotate(f'直径圆半径 d/2={d/2:.3f}', (0.18, -0.26), fontsize=10, color='#9b2c2c')
    ax.annotate(f'最小覆盖圆 R={r_mec:.3f} = d/√3', (c_mec[0] - 0.3, c_mec[1] + 0.06),
                fontsize=10, color='#2b6cb0')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_xlim(-0.35, 1.35); ax.set_ylim(-0.5, 1.1)
    despine(ax)
    save_fig(fig, '图1-6_等边三角形Jung临界情形')

# ---------------- 主流程 ----------------
if __name__ == '__main__':
    S, G, theta = demo_case()
    verts, d, idx, c_mec, r_mec, cover = analyze(S, theta, G)
    idx_global = idx
    area = convex_polygon_area(verts)
    print('=' * 60)
    print('问题一：交会定位区域直径算法与覆盖判定')
    print('检测点坐标：', S.tolist())
    print('真实源位置：', G.tolist())
    print('示向度(含误差)：', np.round(theta, 2).tolist())
    print('定位区域顶点：')
    for k, v in enumerate(verts):
        print(f'  V{k+1} = ({v[0]:.2f}, {v[1]:.2f})')
    print(f'定位区域面积：{area:.2f} m^2')
    print(f'定位区域直径 d = {d:.4f} m（顶点 V{idx[0]+1}-V{idx[1]+1}）')
    print(f'最小覆盖圆半径 R_mec = {r_mec:.4f} m，圆心 ({c_mec[0]:.2f}, {c_mec[1]:.2f})')
    print(f'直径圆半径 d/2 = {d/2:.4f} m')
    print(f'判定：以直径 d 为直径的圆能否覆盖定位区域？ -> {"能" if cover else "不能"}'
          f'（R_mec/(d/2) = {r_mec/(d/2):.4f}）')
    print('=' * 60)

    # 图
    fig1_principle(S, G, theta, verts)
    fig2_diameter(verts, d, idx)
    fig3_cover(verts, d, c_mec, r_mec, cover)
    S3, G3, t3 = three_point_case()
    v3, d3, idx3, c3, r3, cov3 = analyze(S3, t3, G3)
    print(f'\n[三检测点] 面积={convex_polygon_area(v3):.2f} m^2, d={d3:.2f} m, R={r3:.2f} m, 覆盖={cov3}')
    fig4_three(S3, G3, t3, v3)
    fig5_ratio_distribution()
    fig6_jung()

    # 结果 CSV（样例 + 随机统计）
    rows = []
    for name, SS, GG, tt in [('案例A(2点)', S, G, theta), ('案例B(3点)', S3, G3, t3)]:
        vv, dd, ii, cc, rr, cv = analyze(SS, tt, GG)
        rows.append({'案例': name, '检测点数': len(SS), '定位区域面积(m^2)': round(convex_polygon_area(vv), 2),
                     '直径d(m)': round(dd, 2), '最小覆盖圆半径R(m)': round(rr, 2),
                     'R/(d/2)': round(rr / (dd / 2), 4), '直径圆能否覆盖': '能' if cv else '不能'})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, '问题一_定位区域直径与覆盖.csv'), index=False, encoding='utf-8-sig')
    # 供后续问题引用的预处理样例
    df.to_csv(shared_path('预处理数据.csv'), index=False, encoding='utf-8-sig')
    print('\n已输出结果 CSV：问题一_定位区域直径与覆盖.csv / 预处理数据.csv')
    print('全部图片与结果输出完毕。')
