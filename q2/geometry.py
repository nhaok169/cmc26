# -*- coding: utf-8 -*-
"""问题2 共用几何工具: 角度带楔形 / 半平面裁剪 / 凸多边形直径与长轴 / 最小覆盖圆(MEC)"""
import numpy as np
from itertools import combinations

DELTA = np.deg2rad(1.0)      # 测向误差半宽
V = 5.0                      # 机器狗速度 m/s
MEAS_T = 5.0                 # 单次测量耗时 s
CLEAR_T = 5.0                # 光学定位+激光清除耗时 s


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def wedge_hp(S, th):
    """以 S 为顶点、示向度 th 为中心、±δ 张角的楔形 -> 两个半平面 (S, 法向e, 符号)"""
    return [(S, np.array([np.cos(th - DELTA), np.sin(th - DELTA)]), +1),
            (S, np.array([np.cos(th + DELTA), np.sin(th + DELTA)]), -1)]


def clip(poly, S, e, sign):
    """Sutherland-Hodgman: 保留 sign*cross(e, P-S) >= 0"""
    side = lambda p: sign * cross2(e, p - S)
    res = []
    n = len(poly)
    for i in range(n):
        c, nx = poly[i], poly[(i + 1) % n]
        dc, dn = side(c), side(nx)
        if dc >= 0:
            res.append(c)
        if (dc > 0 and dn < 0) or (dc < 0 and dn > 0):
            res.append(c + dc / (dc - dn) * (nx - c))
    return np.array(res) if res else np.zeros((0, 2))


def region(HPs, box=6000.0):
    """全部半平面与大框的交集 -> 凸多边形顶点"""
    poly = np.array([[-box, -box], [box, -box], [box, box], [-box, box]], float)
    for S, e, s in HPs:
        if len(poly) < 3:
            return np.zeros((0, 2))
        poly = clip(poly, S, e, s)
    return poly


def diam_and_axis(v):
    """凸多边形直径及长轴方向单位向量"""
    best = (0.0, None, None)
    for a, b in combinations(range(len(v)), 2):
        d = np.linalg.norm(v[a] - v[b])
        if d > best[0]:
            best = (d, v[a], v[b])
    ax = best[2] - best[1]
    return best[0], ax / np.linalg.norm(ax)


def mec(v):
    """最小覆盖圆 (半径, 圆心), 暴力: 直径对 或 三点外接圆"""
    n = len(v)
    cand = []
    for a, b in combinations(range(n), 2):
        c = (v[a] + v[b]) / 2
        r = np.linalg.norm(v[a] - c)
        if all(np.linalg.norm(x - c) <= r + 1e-9 for x in v):
            cand.append((r, c))
    for tri in combinations(range(n), 3):
        p, q, r_ = v[tri[0]], v[tri[1]], v[tri[2]]
        d = 2 * (p[0] * (q[1] - r_[1]) + q[0] * (r_[1] - p[1]) + r_[0] * (p[1] - q[1]))
        if abs(d) < 1e-12:
            continue
        ux = ((p @ p) * (q[1] - r_[1]) + (q @ q) * (r_[1] - p[1]) + (r_ @ r_) * (p[1] - q[1])) / d
        uy = ((p @ p) * (r_[0] - q[0]) + (q @ q) * (p[0] - r_[0]) + (r_ @ r_) * (q[0] - p[0])) / d
        c = np.array([ux, uy])
        r = np.linalg.norm(p - c)
        if all(np.linalg.norm(x - c) <= r + 1e-9 for x in v):
            cand.append((r, c))
    return min(cand, key=lambda t: t[0]) if cand else (np.inf, None)


def k_expr(t, h, D, eps=0.0):
    """K-闭式: (max||a±b||)^2 = (d1^2+d2^2+2d1d2|cosγ|)/sin^2γ, 含数值保护"""
    G = np.array([D * np.cos(eps), D * np.sin(eps)])
    S1 = np.array([0., 0.])
    S2 = np.array([t, h])
    d1 = np.linalg.norm(G - S1)
    d2 = np.linalg.norm(G - S2)
    v1, v2 = S1 - G, S2 - G
    sg = abs(cross2(v1, v2)) / (d1 * d2)
    if sg < 1e-4:
        return np.inf
    cg = (v1 @ v2) / (d1 * d2)
    return (d1 * d1 + d2 * d2 + 2 * d1 * d2 * abs(cg)) / (sg * sg)


def episode(t, h, D, rng=None, rho_rule=lambda L: 300.0, max_meas=7):
    """单目标一次完整定位清除流程.
    测量坐标系: S1处测得示向度=0; 真实G=D*(cos e1, sin e1), e_i~U[-δ,δ](rng=None时e=0).
    返回 (总时间s, 总测量次数). 失败救援: 长轴切割规则, rho=L的函数."""
    e1 = rng.uniform(-DELTA, DELTA) if rng is not None else 0.0
    G = np.array([D * np.cos(e1), D * np.sin(e1)])
    S1 = np.array([0., 0.])
    HPs = wedge_hp(S1, 0.0)                    # 第一次测量(给定), 楔形以测得示向度0为中心
    cur = S1
    T = 0.0
    n = 1

    def meas_at(S):
        """在 S 测量: 返回楔形(中心=真实方位+误差)"""
        th = np.arctan2(G[1] - S[1], G[0] - S[0])
        ei = rng.uniform(-DELTA, DELTA) if rng is not None else 0.0
        return wedge_hp(S, th + ei)

    S2 = np.array([t, h])
    T += np.linalg.norm(S2 - cur) / V + MEAS_T
    cur = S2
    n += 1
    HPs += meas_at(S2)

    while n < max_meas:
        v = region(HPs)
        if len(v) < 3:
            return np.inf, n
        d2, ax = diam_and_axis(v)
        r, C = mec(v)
        if r <= 20.0:
            T += np.linalg.norm(C - cur) / V + CLEAR_T
            return T, n
        rho = rho_rule(d2)
        nhat = np.array([-ax[1], ax[0]])
        if nhat @ (cur - C) < 0:
            nhat = -nhat
        Sn = C + rho * nhat
        T += np.linalg.norm(Sn - cur) / V + MEAS_T
        cur = Sn
        n += 1
        HPs += meas_at(Sn)
    v = region(HPs)
    r, C = mec(v)
    T += np.linalg.norm(C - cur) / V + CLEAR_T
    return T, n
