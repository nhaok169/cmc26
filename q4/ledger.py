# -*- coding: utf-8 -*-
"""Q4 双账状态表示.

与 Q3 的区别:
  - SKELETON: 7 点 → 31 点六边形格点 (s=950m, 间隙 139.1° ≤ 180°)
  - skeleton_tested: 数组大小跟随 SKELETON
  - no_signal 更新: v1 保持 Q3 逻辑 (排除 B(S,1000) 整圆, 对定向源过度乐观)
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
CLEAR_COST = 5.0
N_CHANNELS = 20

# Q4 格点证书: 21 点 (19 点 s=1200 + 2 补点覆盖定向源盲区)
def _make_grid_skeleton(spacing=1200.0, clip_r=2800.0):
    pts = []
    s = spacing
    for i in range(-8, 9):
        for j in range(-8, 9):
            p = np.array([s * (i + 0.5 * (j % 2)), s * np.sqrt(3) / 2 * j])
            if np.linalg.norm(p) <= clip_r:
                pts.append(p)
    # 补 2 点: 覆盖 s=1200 格点的最大角间隙 (src_angle=120/240, heading=30/150)
    pts.append(np.array([0.0, 1645.0]))      # 覆盖 (120°, 30°) 盲区
    pts.append(np.array([-1425.0, -823.0]))   # 覆盖 (240°, 150°) 盲区
    return np.array(pts)

SKELETON = _make_grid_skeleton()
SKELETON_N = len(SKELETON)
SKELETON_INNER_N = SKELETON_N  # 不分内外圈, 全部一起贪心


# ===================== 多边形工具 =====================
def disk_poly(S, R, n=16):
    S = np.asarray(S, float)
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return S + R * np.column_stack([np.cos(a), np.sin(a)])


def poly_to_halfplanes(poly):
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
    if len(poly) < 3:
        return np.zeros((0, 2))
    dp = disk_poly(S, R, n)
    for hp_S, hp_e, hp_s in poly_to_halfplanes(dp):
        if len(poly) < 3:
            return np.zeros((0, 2))
        poly = clip(poly, hp_S, hp_e, hp_s)
    return poly


def poly_area(poly):
    if len(poly) < 3:
        return 0.0
    return 0.5 * abs(cross2(poly[0], poly[-1]) +
                     sum(cross2(poly[i], poly[(i + 1) % len(poly)])
                         for i in range(len(poly) - 1)))


def poly_centroid(poly):
    if len(poly) == 0:
        return np.array([0., 0.])
    if len(poly) < 3:
        return poly.mean(axis=0)
    return poly.mean(axis=0)


# ===================== 频道状态 =====================
class ChannelState:
    def __init__(self, ch, n_grid):
        self.ch = ch
        self.state = "unknown"
        self.bearings = []
        self.W = None
        self.hard_mask = np.ones(n_grid, dtype=bool)
        self.soft_w = np.ones(n_grid) / n_grid
        self.skeleton_tested = np.zeros(SKELETON_N, dtype=bool)
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
    def __init__(self, eta=50.0):
        self.eta = eta
        self.grid_coords = self._make_grid(eta)
        self.n_grid = len(self.grid_coords)
        self.channels = {ch: ChannelState(ch, self.n_grid) for ch in range(1, N_CHANNELS + 1)}
        self.cur_pos = np.array([0., 0.])
        self.cur_channel = 1
        self.vt = 0.0
        self.cleared = set()
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

    def update_no_signal(self, ch, S):
        cs = self.channels[ch]
        S = np.asarray(S, float)
        d = np.linalg.norm(self.grid_coords - S, axis=1)
        cs.hard_mask &= d > RECV_R_MIN
        p_d = np.where(d <= RECV_R_MIN, 1.0,
                        np.where(d >= RECV_R_MAX, 0.0,
                                 (RECV_R_MAX - d) / (RECV_R_MAX - RECV_R_MIN)))
        factor = 1.0 - p_d
        cs.soft_w *= factor
        s = cs.soft_w.sum()
        cs.soft_w = np.divide(cs.soft_w, s, out=np.full_like(cs.soft_w, 1.0 / self.n_grid), where=s > 1e-15)
        diffs = SKELETON - S
        for i in range(SKELETON_N):
            if np.linalg.norm(diffs[i]) < self.eta * 0.5 + 1:
                cs.skeleton_tested[i] = True
        cs.invalidate()

    def update_direction(self, ch, S, bearing_deg):
        cs = self.channels[ch]
        S = np.asarray(S, float)
        th = np.radians(bearing_deg)
        cs.bearings.append((S.copy(), bearing_deg))
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
        d = np.linalg.norm(self.grid_coords - S, axis=1)
        delta = self.grid_coords - S
        angles = np.arctan2(delta[:, 1], delta[:, 0])
        diff = np.abs(((angles - th + np.pi) % (2 * np.pi)) - np.pi)
        in_band = (diff <= DELTA) & (d <= RECV_R_MAX)
        cs.soft_w *= in_band.astype(float)
        cs.hard_mask &= in_band
        s = cs.soft_w.sum()
        cs.soft_w = np.divide(cs.soft_w, s, out=np.full_like(cs.soft_w, 1.0 / self.n_grid), where=s > 1e-15)
        diffs = SKELETON - S
        for i in range(SKELETON_N):
            if np.linalg.norm(diffs[i]) < self.eta * 0.5 + 1:
                cs.skeleton_tested[i] = True
        cs.invalidate()

    def update_near(self, ch, S):
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
        cs = self.channels[ch]
        cs.state = "detected"
        cs._mec = None
        cs._diam = None

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
        visited = np.zeros(SKELETON_N, dtype=bool)
        for ch in range(1, N_CHANNELS + 1):
            visited |= self.channels[ch].skeleton_tested
        return [i for i in range(SKELETON_N) if not visited[i]]

    def exploration_area(self, ch, p):
        cs = self.channels[ch]
        if not cs.is_active:
            return 0.0
        d = np.linalg.norm(self.grid_coords - np.asarray(p, float), axis=1)
        return float(cs.hard_mask[d <= RECV_R_MIN].sum())
