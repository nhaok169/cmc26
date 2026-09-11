"""问题1求解：交会定位区域的顶点、直径、直径圆覆盖。

算法与《第一问_论文参考稿.md》伪代码一致：
  边界直线两两求交 → 半平面筛选 → 凸多边形顶点
  → 顶点对最大距离即直径 → 检查其余顶点是否在直径圆内。

N 为个位数时整体复杂度 O(N^3)。定位区域不含 1800 m 目标圆域；
诊断方框只用来区分空集与无界，不进入 P。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

DELTA_DEG = 1.0  # 题面测向误差限
EPS = 1e-9  # 半平面归属
PARALLEL_EPS = 1e-12  # 直线求交：|det| 过小视为平行
DEDUP_TOL = 1e-7
COVER_TOL = 1e-8
# 仅用于区分“无界 nonempty”与“空集”的诊断方框半边长，不是题面 1800 m 目标圆域
DIAG_HALF = 1.0e6


def _unit(ang_deg: float) -> np.ndarray:
    t = np.radians(ang_deg)
    return np.array([np.cos(t), np.sin(t)], dtype=float)


def _wrap_deg(a: float) -> float:
    return float(a % 360.0)


@dataclass
class Halfplane:
    """(X - origin) · normal >= 0。normal 指向可行域内部。"""
    origin: np.ndarray
    normal: np.ndarray
    dir_angle: float  # 边界直线的方向角（度），仅用于求交


def cone_halfplanes(S: Sequence[float], theta_deg: float, delta_deg: float = DELTA_DEG) -> List[Halfplane]:
    """检测点 S、示向度 theta 的误差锥：两条边界直线对应的半平面。"""
    S = np.asarray(S, dtype=float)
    lo = _wrap_deg(theta_deg - delta_deg)
    hi = _wrap_deg(theta_deg + delta_deg)
    u_lo, u_hi = _unit(lo), _unit(hi)
    # 可行域在 lo 的逆时针侧、hi 的顺时针侧
    n_lo = np.array([-u_lo[1], u_lo[0]])
    n_hi = np.array([u_hi[1], -u_hi[0]])
    return [
        Halfplane(S.copy(), n_lo, lo),
        Halfplane(S.copy(), n_hi, hi),
    ]


def line_intersection(h1: Halfplane, h2: Halfplane) -> Optional[np.ndarray]:
    """两条边界直线求交。直线 i： (X - o_i) · n_i = 0。"""
    A = np.vstack([h1.normal, h2.normal])
    det = float(np.linalg.det(A))
    if abs(det) < PARALLEL_EPS:
        return None
    b = np.array([np.dot(h1.normal, h1.origin), np.dot(h2.normal, h2.origin)])
    return np.linalg.solve(A, b)


def satisfies_all(p: np.ndarray, planes: Sequence[Halfplane], eps: float = EPS) -> bool:
    return all(float(np.dot(p - h.origin, h.normal)) >= -eps for h in planes)


def dedup_points(pts: Sequence[np.ndarray], tol: float = DEDUP_TOL) -> List[np.ndarray]:
    uniq: List[np.ndarray] = []
    for p in pts:
        if not any(np.linalg.norm(p - q) < tol for q in uniq):
            uniq.append(p)
    return uniq


def angular_sort(pts: Sequence[np.ndarray]) -> np.ndarray:
    arr = np.asarray(pts, dtype=float)
    c = arr.mean(axis=0)
    ang = np.arctan2(arr[:, 1] - c[1], arr[:, 0] - c[0])
    return arr[np.argsort(ang)]


def recession_unbounded(planes: Sequence[Halfplane]) -> bool:
    """若所有内法向落在某个闭半平面内（最大角间隙 >= 180°），则公共区域无界。"""
    angs = np.sort([float(np.arctan2(h.normal[1], h.normal[0])) for h in planes])
    if len(angs) < 2:
        return True
    gaps = np.diff(angs)
    wrap = float(angs[0] + 2 * np.pi - angs[-1])
    max_gap = max(float(gaps.max()) if len(gaps) else 0.0, wrap)
    return max_gap >= np.pi - 1e-10


def _box_planes(center: Sequence[float], half: float = DIAG_HALF) -> List[Halfplane]:
    """诊断用大方框，只判断空/无界，不进入定位区域 P。"""
    c = np.asarray(center, dtype=float)
    return [
        Halfplane(c + np.array([-half, 0.0]), np.array([1.0, 0.0]), 90.0),
        Halfplane(c + np.array([half, 0.0]), np.array([-1.0, 0.0]), 90.0),
        Halfplane(c + np.array([0.0, -half]), np.array([0.0, 1.0]), 0.0),
        Halfplane(c + np.array([0.0, half]), np.array([0.0, -1.0]), 0.0),
    ]


def halfplane_vertices(planes: Sequence[Halfplane]) -> List[np.ndarray]:
    cand: List[np.ndarray] = []
    m = len(planes)
    for i in range(m):
        for j in range(i + 1, m):
            p = line_intersection(planes[i], planes[j])
            if p is not None and satisfies_all(p, planes):
                cand.append(p)
    return dedup_points(cand)


def has_feasible_point(planes: Sequence[Halfplane], center: Sequence[float]) -> bool:
    """半平面是否有公共点。无有限顶点时，用诊断方框探测无界可行域。"""
    if halfplane_vertices(planes):
        return True
    boxed = list(planes) + _box_planes(center)
    return len(halfplane_vertices(boxed)) > 0


def polygon_diameter(vertices: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """有界凸集直径：单点为 0，线段/共线顶点取两端，一般情形枚举顶点对。"""
    arr = np.asarray(vertices, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    m = len(arr)
    if m == 0:
        raise ValueError("空顶点集没有直径")
    if m == 1:
        p = arr[0].copy()
        return 0.0, p, p.copy()
    dmax, a, b = -1.0, arr[0], arr[0]
    for i in range(m):
        for j in range(i + 1, m):
            d = float(np.linalg.norm(arr[i] - arr[j]))
            if d > dmax:
                dmax, a, b = d, arr[i], arr[j]
    if dmax < 0.0:
        p = arr[0].copy()
        return 0.0, p, p.copy()
    return dmax, a, b


def circle_covers(
    vertices: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    tol: float = COVER_TOL,
):
    """以 AB 为直径的圆是否覆盖全部顶点。含单点（R=0）与共线退化。"""
    verts = np.asarray(vertices, dtype=float)
    if verts.ndim == 1:
        verts = verts.reshape(1, -1)
    O = (np.asarray(A, dtype=float) + np.asarray(B, dtype=float)) / 2.0
    R = float(np.linalg.norm(np.asarray(A, dtype=float) - np.asarray(B, dtype=float)) / 2.0)
    if len(verts) == 0:
        return True, O, R, np.zeros(0)
    dist = np.linalg.norm(verts - O, axis=1)
    return bool(np.all(dist <= R + tol)), O, R, dist


@dataclass
class LocateResult:
    status: str  # OK / UNBOUNDED / EMPTY
    message: str
    vertices: Optional[np.ndarray]
    diameter: Optional[float]
    A: Optional[np.ndarray]
    B: Optional[np.ndarray]
    center: Optional[np.ndarray]
    radius: Optional[float]
    covers: Optional[bool]
    n_planes: int

    def pretty(self) -> str:
        lines = [f"status = {self.status}", self.message]
        if self.vertices is not None:
            lines.append(f"顶点数 m = {len(self.vertices)}")
            for i, v in enumerate(self.vertices, 1):
                lines.append(f"  V{i} = ({v[0]:.6f}, {v[1]:.6f})")
        if self.diameter is not None:
            lines.append(f"直径 D = {self.diameter:.6f} m")
            lines.append(f"端点 A = ({self.A[0]:.6f}, {self.A[1]:.6f})")
            lines.append(f"端点 B = ({self.B[0]:.6f}, {self.B[1]:.6f})")
            lines.append(f"圆心 O = ({self.center[0]:.6f}, {self.center[1]:.6f}), R = {self.radius:.6f}")
            lines.append(f"直径圆覆盖 P？ {'是' if self.covers else '否'}")
        return "\n".join(lines)


def locate(
    sensors: Sequence[Sequence[float]],
    thetas_deg: Sequence[float],
    delta_deg: float = DELTA_DEG,
) -> LocateResult:
    """交会定位主程序。sensors[i]、thetas_deg[i] 一一对应。

    定位区域 P 只取示向度误差锥的公共交集，不叠加半径 1800 m 目标圆域。
    判定顺序：先看半平面交是否为空，再看是否无界，避免把“无有限顶点的无界锥”
    误判为 EMPTY。
    """
    N = len(sensors)
    if N != len(thetas_deg):
        raise ValueError("检测点与示向度数量不一致")

    if N < 2:
        return LocateResult(
            "UNBOUNDED",
            "N<2：单个误差锥无界，直径无定义，需要增加检测点。",
            None, None, None, None, None, None, None, 2 * max(N, 0),
        )

    planes: List[Halfplane] = []
    for S, th in zip(sensors, thetas_deg):
        planes.extend(cone_halfplanes(S, th, delta_deg))
    m = len(planes)
    verts = halfplane_vertices(planes)
    center = np.mean(np.asarray(sensors, dtype=float), axis=0)

    # 空集优先：无界 recession 也可能对应矛盾约束（空）。无有限顶点 ≠ 空。
    if not has_feasible_point(planes, center):
        return LocateResult(
            "EMPTY",
            "交会区域为空：半平面无公共点，定位失败，应更换或增加检测点。",
            None, None, None, None, None, None, None, m,
        )

    if recession_unbounded(planes):
        return LocateResult(
            "UNBOUNDED",
            "半平面公共区域无界（法向未包围原点），直径无定义，需要增加或调整检测点。",
            np.array(verts) if verts else None,
            None, None, None, None, None, None, m,
        )

    if len(verts) == 0:
        return LocateResult(
            "EMPTY",
            "交会区域为空：半平面无公共点，定位失败，应更换或增加检测点。",
            None, None, None, None, None, None, None, m,
        )

    P = angular_sort(verts)
    D, A, B = polygon_diameter(P)
    covers, O, R, _ = circle_covers(P, A, B)
    n = len(P)
    if n == 1:
        msg = "交会区域退化为单点。半平面数 2N = {}。".format(m)
    elif n == 2:
        msg = "交会区域退化为线段。半平面数 2N = {}。".format(m)
    else:
        msg = "有界凸 {} 边形。半平面数 2N = {}。".format(n, m)
    return LocateResult("OK", msg, P, D, A, B, O, R, covers, m)


def bearing_deg(src, dst) -> float:
    v = np.asarray(dst, float) - np.asarray(src, float)
    return _wrap_deg(np.degrees(np.arctan2(v[1], v[0])))


def _demo():
    # 与示意图同一组站址；示向度指向示意源 G
    G = np.array([80.0, 220.0])
    S1 = np.array([-920.0, -480.0])
    S2 = np.array([980.0, -420.0])
    S3 = np.array([-40.0, 1180.0])
    th1, th2, th3 = bearing_deg(S1, G), bearing_deg(S2, G), bearing_deg(S3, G)

    print("=" * 60)
    print("算例 A  N=1（应 UNBOUNDED）")
    print(locate([S1], [th1]).pretty())

    print("\n" + "=" * 60)
    print("算例 B  N=2，δ=1°（题面真实误差）")
    r2 = locate([S1, S2], [th1, th2], 1.0)
    print(r2.pretty())

    print("\n" + "=" * 60)
    print("算例 C  N=3，δ=1°")
    print(locate([S1, S2, S3], [th1, th2, th3], 1.0).pretty())

    print("\n" + "=" * 60)
    print("算例 D  两站相背，交会为空（应 EMPTY）")
    print(locate([[0.0, 0.0], [200.0, 0.0]], [90.0, 270.0], 1.0).pretty())

    print("\n" + "=" * 60)
    print("算例 E  等边三角形：直径圆不能覆盖（圆覆盖反例）")
    eq = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, np.sqrt(3.0)]])
    D, A, B = polygon_diameter(eq)
    covers, O, R, dist = circle_covers(eq, A, B)
    print(f"边长=2, D={D:.6f}, R={R:.6f}, 顶点到圆心 = {dist}, 覆盖? {covers}")

    print("\n" + "=" * 60)
    print("算例 F  两站同向（无有限顶点的无界区域，应 UNBOUNDED 而非 EMPTY）")
    print(locate([[0.0, 0.0], [200.0, 0.0]], [90.0, 90.0], 1.0).pretty())

    print("\n" + "=" * 60)
    print("退化：单点 / 共线三点")
    p1 = np.array([[1.0, 2.0]])
    d1, a1, b1 = polygon_diameter(p1)
    c1, _, r1, _ = circle_covers(p1, a1, b1)
    print(f"单点 D={d1}, 覆盖? {c1}, R={r1}")
    col = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    dc, ac, bc = polygon_diameter(col)
    cc, _, rc, distc = circle_covers(col, ac, bc)
    print(f"共线 D={dc:.6f}, 覆盖? {cc}, 顶点到圆心 = {distc}")


if __name__ == "__main__":
    _demo()
