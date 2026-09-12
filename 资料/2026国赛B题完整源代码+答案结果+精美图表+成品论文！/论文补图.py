# -*- coding: utf-8 -*-
"""
论文排版辅助：重新生成 2 张「竖向/细高」违规图（图1-2、图2-6），改为宽≥高的横构图。
其余求解脚本保持为求解阶段最终版本，本脚本仅针对论文所需的横构图重新出图。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *
from scipy.spatial import HalfspaceIntersection
import itertools

P1_IMG = os.path.join(BASE_DIR, '问题一', '图片')
P2_IMG = os.path.join(BASE_DIR, '问题二', '图片')

def unit(deg):
    a = np.deg2rad(deg); return np.array([np.cos(a), np.sin(a)])

def left_normal(deg):
    a = np.deg2rad(deg); return np.array([-np.sin(a), np.cos(a)])

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
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0.0, None)], method='highs')
    return None if res.x is None else res.x[:2]

def polygon_vertices(S, theta, eps=1.0, true_G=None):
    H = build_halfspaces(S, theta, eps)
    for ip in [chebyshev_center(H), true_G]:
        if ip is None: continue
        ip = np.asarray(ip, float).ravel()
        if ip.shape[0] != 2: continue
        try:
            verts = HalfspaceIntersection(H, ip).intersections
            c = verts.mean(axis=0)
            return verts[np.argsort([ang_of(v - c) for v in verts])]
        except Exception:
            continue
    return None

def polygon_diameter(verts):
    n = len(verts); d2 = 0.0; pi = pj = 0
    for i in range(n):
        for j in range(i + 1, n):
            dd = np.sum((verts[i] - verts[j]) ** 2)
            if dd > d2: d2 = dd; pi, pj = i, j
    return np.sqrt(d2), pi, pj

# ---------- 图 1-2 横构图 ----------
def fig1_2():
    S = np.array([[0.0, 0.0], [1200.0, 200.0]])
    G = np.array([900.0, 1100.0])
    rng = np.random.default_rng(7)
    theta = np.array([(ang_of(G - Si) + rng.uniform(-1, 1)) % 360.0 for Si in S])
    verts = polygon_vertices(S, theta, 1.0, G)
    d, i, j = polygon_diameter(verts)
    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.set_aspect('equal', adjustable='box')
    p = np.vstack([verts, verts[0]])
    ax.fill(p[:, 0], p[:, 1], color='#4299e1', alpha=0.45, edgecolor='#2b6cb0', lw=1.3)
    ax.plot(verts[:, 0], verts[:, 1], 'o', ms=6, color='#2b6cb0', mec='k', mew=0.8)
    for a, b in itertools.combinations(range(len(verts)), 2):
        ax.plot([verts[a][0], verts[b][0]], [verts[a][1], verts[b][1]],
                color='#a0aec0', lw=0.6, alpha=0.7, zorder=1)
    ax.plot([verts[i][0], verts[j][0]], [verts[i][1], verts[j][1]], color='#e53e3e', lw=2.6, zorder=3)
    ax.plot([verts[i][0], verts[j][0]], [verts[i][1], verts[j][1]], 'o', ms=9, color='#e53e3e', mec='k', mew=1.0, zorder=4)
    mid = (verts[i] + verts[j]) / 2
    ax.annotate(f'd = {d:.1f} m', (mid[0] + 14, mid[1] - 20), fontsize=12, color='#9b2c2c')
    for k, v in enumerate(verts):
        ax.annotate(f'V$_{k+1}$', (v[0] + 10, v[1] + 10), fontsize=10)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_xlim(810, 975); ax.set_ylim(1030, 1170)
    despine(ax)
    path = os.path.join(P1_IMG, '图1-2_定位区域与直径.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white'); plt.close(fig)
    print('已重新生成 图1-2（横构）')

# ---------- 图 2-6 横构图 ----------
def m_of(x2):
    return max(abs(x2 - 5.0), abs(x2 - 1500.0))

def candidate_region(R0v=1000.0, gamma_min=np.deg2rad(30.0), n=800):
    xs = np.linspace(5.0 - R0v, 1500.0 + R0v, n)
    xin, lo, hi = [], [], []
    for x in xs:
        m = m_of(x)
        if m > R0v * np.cos(gamma_min): continue
        ylo = np.tan(gamma_min) * m
        yhi = np.sqrt(R0v**2 - m**2)
        if yhi >= ylo:
            xin.append(x); lo.append(ylo); hi.append(yhi)
    return np.array(xin), np.array(lo), np.array(hi)

def fig2_6():
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_aspect('equal', adjustable='box')
    colors = ['#6b46c1', '#0d9488', '#d97706']
    for R0v, c in zip([1000.0, 1250.0, 1500.0], colors):
        xs, lo, hi = candidate_region(R0v)
        ax.fill_between(xs, lo, hi, color=c, alpha=0.12, lw=0)
        ax.fill_between(xs, -hi, -lo, color=c, alpha=0.12, lw=0)
        ax.plot(xs, hi, color=c, lw=1.7, label=f'R$_0$={R0v:.0f} m')
        ax.plot(xs, -hi, color=c, lw=1.7)
        ax.plot(xs, lo, color=c, lw=1.2, ls=':')
        ax.plot(xs, -lo, color=c, lw=1.2, ls=':')
    ax.plot([5.0, 1500.0], [0, 0], color='#374151', lw=4, alpha=0.8)
    ax.plot([0], [0], 'o', ms=8, color='#6b46c1', mec='k', mew=1.0, zorder=5)
    ax.annotate('S$_1$', (-10, -120), fontsize=11, color='#5b21b6')
    for R0v, c in zip([1000.0, 1250.0, 1500.0], colors):
        h = (1500.0 - 5.0) / 2.0
        yo = np.sqrt(R0v**2 - h**2)
        ax.plot([(5.0 + 1500.0) / 2.0], [yo], 'o', ms=9, color=c, mec='k', mew=1.0, zorder=6)
    ax.set_xlabel('x$_2$ (m)'); ax.set_ylabel('y$_2$ (m)')
    ax.set_xlim(-850, 3050); ax.set_ylim(-1550, 1550)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    despine(ax)
    path = os.path.join(P2_IMG, '图2-6_候选区域随保证接收半径的变化.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white'); plt.close(fig)
    print('已重新生成 图2-6（横构）')

if __name__ == '__main__':
    fig1_2()
    fig2_6()
