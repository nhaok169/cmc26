# -*- coding: utf-8 -*-
"""Q3 双账状态表示 (§5.1).

硬账: 集员凸多边形 W_k (复用 q2/geometry.py 工具) — 保证正确性, 管终止/清除.
软账: 格点后验 w_k(x) — Stone 式更新, 仅管排序.

频道状态: unknown -> detected -> clearable -> cleared  (或 -> empty)
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "q2"))
from geometry import DELTA, wedge_hp, clip, region, diam_and_axis, mec, cross2

# ===================== 常量 =====================
ARENA_R = 1800.0
RECV_R_MIN = 1000.0
RECV_R_MAX = 1500.0
CLEAR_R = 20.0
NEAR_R = 5.0
SPEED = 5.0
SWITCH_COST = 1.0
MEASURE_COST = 5.0
CLEAR_COST = 5.0          # 光学定位 3s + 激光清除 2s
N_CHANNELS = 20

# 7 点覆盖证书构型 (§5.6): 原点 + 1150m 正六边形
# 1150m 环 → 最坏间隙 988.5m < 1000m (覆盖仍有效)
SKELETON = np.array([[0., 0.]] + [
    [1150.0 * np.cos(i * np.pi / 3), 1150.0 * np.sin(i * np.pi / 3)]
    for i in range(6)
])


# ===================== 多边形工具 =====================
def disk_poly(S, R, n=16):
    """正 n 边形逼近圆盘, CCW 顶点."""
    S = np.asarray(S, float)
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return S + R * np.column_stack([np.cos(a), np.sin(a)])


def poly_to_halfplanes(poly):
    """CCW 凸多边形 -> 内向半平面列表 [(S, e, +1), ...].

    e = 归一化边方向; clip 的 cross2(e, p-S)>=0 保留左侧(CCW 内侧).
    """
    hps = []
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        edge = b - a
        norm = np.linalg.norm(edge)
        if norm > 1e-12:
            edge = edge / norm
        hps.append((a, edge, +1))
    return hps


def intersect_disk(poly, S, R, n=64):
    """凸多边形 ∩ 圆盘(正 n 边形逼近)."""
    if len(poly) < 3:
        return np.zeros((0, 2))
    dp = disk_poly(S, R, n)
    for hp_S, hp_e, hp_s in poly_to_halfplanes(dp):
        if len(poly) < 3:
            return np.zeros((0, 2))
        poly = clip(poly, hp_S, hp_e, hp_s)
    return poly


def poly_area(poly):
    """凸多边形面积."""
    if len(poly) < 3:
        return 0.0
    return 0.5 * abs(cross2(poly[0], poly[-1]) +
                     sum(cross2(poly[i], poly[(i + 1) % len(poly)])
                         for i in range(len(poly) - 1)))


def poly_centroid(poly):
    """凸多边形重心."""
    if len(poly) == 0:
        return np.array([0., 0.])
    if len(poly) < 3:
        return poly.mean(axis=0)
    return poly.mean(axis=0)


# ===================== 频道状态 =====================
class ChannelState:
    """单频道双账状态."""

    def __init__(self, ch, n_grid):
        self.ch = ch
        self.state = "unknown"       # unknown/detected/clearable/cleared/empty
        self.bearings = []           # [(pos_xy, bearing_deg), ...]
        self.W = None                # 凸多边形 (候选集, bearing 精化)
        self.hard_mask = np.ones(n_grid, dtype=bool)
        self.soft_w = np.ones(n_grid) / n_grid
        self.skeleton_tested = np.zeros(7, dtype=bool)
        self._mec = None
        self._diam = None
        self._axis = None

    @property
    def is_active(self):
        return self.state not in ("cleared", "empty")

    @property
    def hard_area(self):
        return float(self.hard_mask.sum())

    @property
    def mec(self):
        if self._mec is None and self.W is not None and len(self.W) >= 3:
            r, c = mec(self.W)
            self._mec = (r, c)
        return self._mec

    @property
    def diameter(self):
        if self._diam is None and self.W is not None and len(self.W) >= 3:
            d, ax = diam_and_axis(self.W)
            self._diam = d
            self._axis = ax
        return self._diam if self._diam is not None else 0.0

    @property
    def axis(self):
        if self._axis is None and self.W is not None and len(self.W) >= 3:
            d, ax = diam_and_axis(self.W)
            self._diam = d
            self._axis = ax
        return self._axis

    @property
    def is_clearable(self):
        m = self.mec
        return m is not None and m[0] <= CLEAR_R

    @property
    def centroid(self):
        if self.W is not None and len(self.W) >= 1:
            return poly_centroid(self.W)
        return None

    def invalidate(self):
        self._mec = None
        self._diam = None
        self._axis = None


# ===================== 双账总管 =====================
class Ledger:
    """管理 20 频道 + 共享格点."""

    def __init__(self, eta=50.0):
        self.eta = eta
        self.grid_coords = self._make_grid(eta)
        self.n_grid = len(self.grid_coords)
        self.channels = {ch: ChannelState(ch, self.n_grid) for ch in range(1, N_CHANNELS + 1)}
        self.cur_pos = np.array([0., 0.])
        self.cur_channel = 1
        self.vt = 0.0
        self.cleared = set()
        # 预计算候选格点 (探索选点用)
        self.candidate_grid = self._make_candidate_grid(300.0)

    def _make_grid(self, eta):
        xs = np.arange(-ARENA_R, ARENA_R + eta, eta)
        ys = np.arange(-ARENA_R, ARENA_R + eta, eta)
        coords = []
        for x in xs:
            for y in ys:
                if x * x + y * y <= ARENA_R ** 2:
                    coords.append([x, y])
        return np.array(coords, float)

    def _make_candidate_grid(self, spacing):
        pts = []
        for x in np.arange(-ARENA_R, ARENA_R + spacing, spacing):
            for y in np.arange(-ARENA_R, ARENA_R + spacing, spacing):
                if x * x + y * y <= ARENA_R ** 2 * 0.95:
                    pts.append([x, y])
        return np.array(pts, float)

    # ---- 更新操作 ----
    def update_no_signal(self, ch, S):
        """硬账: F_k \ B(S, 1000); 软账: Stone 式 (1-p_d) 更新."""
        cs = self.channels[ch]
        S = np.asarray(S, float)
        d = np.linalg.norm(self.grid_coords - S, axis=1)

        # 硬: 排除 1000m 内
        cs.hard_mask &= d > RECV_R_MIN

        # 软: Stone 更新
        p_d = np.where(d <= RECV_R_MIN, 1.0,
                        np.where(d >= RECV_R_MAX, 0.0,
                                 (RECV_R_MAX - d) / (RECV_R_MAX - RECV_R_MIN)))
        factor = 1.0 - p_d
        cs.soft_w *= factor
        s = cs.soft_w.sum()
        cs.soft_w = np.divide(cs.soft_w, s, out=np.full_like(cs.soft_w, 1.0 / self.n_grid), where=s > 1e-15)

        # 骨架点追踪
        diffs = SKELETON - S
        for i in range(7):
            if np.linalg.norm(diffs[i]) < self.eta * 0.5 + 1:
                cs.skeleton_tested[i] = True
        cs.invalidate()

    def update_direction(self, ch, S, bearing_deg):
        """硬账: W_k ∩ ±1° 带 ∩ B(S,1500); 软账: 带内保权、带外归零."""
        cs = self.channels[ch]
        S = np.asarray(S, float)
        th = np.radians(bearing_deg)
        cs.bearings.append((S.copy(), bearing_deg))

        # ---- 硬账: 多边形 ----
        if cs.W is None:
            cs.W = disk_poly([0., 0.], ARENA_R, 32)
        hps = wedge_hp(S, th)
        for hp_S, hp_e, hp_s in hps:
            if len(cs.W) >= 3:
                cs.W = clip(cs.W, hp_S, hp_e, hp_s)
        if len(cs.W) >= 3:
            cs.W = intersect_disk(cs.W, S, RECV_R_MAX, 64)
        if cs.state == "unknown":
            cs.state = "detected"

        # ---- 软账: 格点 ----
        d = np.linalg.norm(self.grid_coords - S, axis=1)
        delta = self.grid_coords - S
        angles = np.arctan2(delta[:, 1], delta[:, 0])
        diff = np.abs(((angles - th + np.pi) % (2 * np.pi)) - np.pi)
        in_band = (diff <= DELTA) & (d <= RECV_R_MAX)
        cs.soft_w *= in_band.astype(float)
        cs.hard_mask &= in_band
        s = cs.soft_w.sum()
        cs.soft_w = np.divide(cs.soft_w, s, out=np.full_like(cs.soft_w, 1.0 / self.n_grid), where=s > 1e-15)

        # 骨架点追踪
        diffs = SKELETON - S
        for i in range(7):
            if np.linalg.norm(diffs[i]) < self.eta * 0.5 + 1:
                cs.skeleton_tested[i] = True
        cs.invalidate()

    def update_near(self, ch, S):
        """≤5m 直接可清除."""
        cs = self.channels[ch]
        S = np.asarray(S, float)
        cs.state = "clearable"
        cs._mec = (0.0, S)
        cs._diam = 0.0

    def update_cleared(self, ch):
        cs = self.channels[ch]
        cs.state = "cleared"
        self.cleared.add(ch)

    def update_clear_fail(self, ch):
        """清除失败: 降级回 detected, 需要补测."""
        cs = self.channels[ch]
        cs.state = "detected"
        cs._mec = None
        cs._diam = None

    # ---- 查询 ----
    def check_clearable(self, ch):
        cs = self.channels[ch]
        if cs.is_clearable and cs.state == "detected":
            cs.state = "clearable"
        return cs.state == "clearable"

    def is_channel_empty(self, ch):
        cs = self.channels[ch]
        return bool(cs.skeleton_tested.all()) or cs.hard_area < 1.0

    def active_channels(self):
        return [ch for ch in range(1, N_CHANNELS + 1) if self.channels[ch].is_active]

    def clearable_channels(self):
        return [ch for ch in range(1, N_CHANNELS + 1) if self.channels[ch].state == "clearable"]

    def detected_channels(self):
        return [ch for ch in range(1, N_CHANNELS + 1)
                if self.channels[ch].state in ("detected", "clearable")]

    def unknown_channels(self):
        return [ch for ch in range(1, N_CHANNELS + 1) if self.channels[ch].state == "unknown"]

    def is_terminated(self):
        """全部频道 ∈ 已清除 ∪ {覆盖证书生效 / F_k=∅}."""
        for ch in range(1, N_CHANNELS + 1):
            cs = self.channels[ch]
            if cs.state in ("cleared", "empty"):
                continue
            if cs.state == "unknown" and self.is_channel_empty(ch):
                cs.state = "empty"
                continue
            return False
        return True

    def skeleton_unvisited(self):
        """未访问的骨架点索引."""
        visited = np.zeros(7, dtype=bool)
        for ch in range(1, N_CHANNELS + 1):
            visited |= self.channels[ch].skeleton_tested
        return [i for i in range(7) if not visited[i]]

    def exploration_area(self, ch, p):
        """频道 ch 在 p 点 no_signal 可排除的格点数 (探索分用)."""
        cs = self.channels[ch]
        if not cs.is_active:
            return 0.0
        d = np.linalg.norm(self.grid_coords - np.asarray(p, float), axis=1)
        return float(cs.hard_mask[d <= RECV_R_MIN].sum())
