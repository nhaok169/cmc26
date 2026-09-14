# -*- coding: utf-8 -*-
"""Q4 v2 双账状态层: 探针集合 (probe set) + 覆盖证书判空.

核心升级 (相对 q4/ledger.py):
  1. 每个频道记录**实际测量过的位置** (probes), 而不是离散骨架布尔数组.
  2. 判空判据 = "探针集合对全圆域 G 满足环抱条件":
        max_{G in disk(1800)} maxgap(probes ∩ B(G,1000)) <= 180°
     这是定向源 (±90°) + 全向源的**统一、充要**覆盖证书:
       - 若 G 落在 conv({p in probes : |p-G|<=1000}) 内部, 则无论朝向 φ 如何,
         必有探针落在 G 的 180° 覆盖半平面内 -> 必被检出.
       - 反之存在朝向使全圆域内探针都背对 -> 可能漏检.
  3. 证书只依赖"探针几何", 因此追踪途中的停点也自动计入证书 (信息复用).
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

CERT_MARGIN = -1e-6   # 证书阈值 = 180 - CERT_MARGIN = 180 + 1e-6
                      # 定向源覆盖角为**闭**半平面 (±90° 含), 故 maxgap == 180° 仍成立
                      # (此时任意闭 180° 扇区必含一个探针方向); 仅留浮点容差.

# ===================== 覆盖证书点集 =====================
def _make_skeleton():
    d = os.path.dirname(os.path.abspath(__file__))
    # 只加载**已严格校验**的点集; skeleton_best.npy 是扫描候选 (可能未通过超密校验), 不加载.
    for nm in ("skeleton_final.npy", "skeleton_opt2.npy", "skeleton_opt.npy"):
        p = os.path.join(d, nm)
        if os.path.exists(p):
            return np.load(p)
    def ring(r, k, ph=0.0):
        a = ph + np.arange(k) * 2 * np.pi / k
        return np.column_stack([r * np.cos(a), r * np.sin(a)])
    return np.vstack([ring(400, 6), ring(1300, 8), ring(1900, 12)])

SKELETON = _make_skeleton()
SKELETON_N = len(SKELETON)


# ===================== 证书采样点 =====================
def _make_G_samples():
    """判空证书的检验点 (必须足够密, 否则可能误判为空 -> 漏源)."""
    pts = []
    rr = 50.0
    while rr < ARENA_R:
        n = max(60, int(2 * np.pi * rr / 50))
        a = np.arange(n) * 2 * np.pi / n
        pts.append(np.column_stack([rr * np.cos(a), rr * np.sin(a)]))
        rr += 50.0
    for rr in (1750.0, 1780.0, 1800.0):
        n = 720
        a = np.arange(n) * 2 * np.pi / n
        pts.append(np.column_stack([rr * np.cos(a), rr * np.sin(a)]))
    return np.vstack(pts)

G_SAMPLES = _make_G_samples()


def max_gap_all(probes, Gs=None):
    """返回全部 G 的最大角度间隙的最大值 (度). 向量化."""
    Gs = G_SAMPLES if Gs is None else Gs
    probes = np.asarray(probes, float)
    if len(probes) < 3:
        return 999.0
    diff = Gs[:, None, :] - probes[None, :, :]
    d2 = (diff ** 2).sum(-1)
    mask = d2 <= RECV_R_MIN ** 2 + 1e-9
    if not mask.any():
        return 999.0
    ang = np.arctan2(diff[:, :, 1], diff[:, :, 0])
    ang = np.where(mask, ang, np.nan)
    ang.sort(axis=1)
    k = np.sum(~np.isnan(ang), axis=1)
    m = len(Gs)
    gaps = np.diff(ang, axis=1)
    gaps = np.where(np.isnan(gaps), -1.0, gaps)
    gmax = gaps.max(axis=1) if gaps.shape[1] else np.full(m, -1.0)
    last = ang[np.arange(m), np.clip(k - 1, 0, ang.shape[1] - 1)]
    wrap = ang[:, 0] + 2 * np.pi - last
    wrap = np.where(k >= 1, wrap, 2 * np.pi)
    gap = np.maximum(gmax, wrap)
    gap = np.where(k == 0, np.inf, gap)
    return float(np.degrees(gap.max()))


# ===================== 多边形工具 =====================
def disk_poly(S, R, n=16):
    S = np.asarray(S, float)
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return S + R * np.column_stack([np.cos(a), np.sin(a)])


def poly_to_halfplanes(poly):
    hps = []
    n = len(poly)
    for i in range(n):
        a = poly[i]; b = poly[(i + 1) % n]
        edge = b - a
        nm = np.linalg.norm(edge)
        if nm > 1e-12:
            edge = edge / nm
        hps.append((a, edge, +1))
    return hps


def intersect_disk(poly, S, R, n=64):
    if len(poly) < 3:
        return np.zeros((0, 2))
    for hp_S, hp_e, hp_s in poly_to_halfplanes(disk_poly(S, R, n)):
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
    return poly.mean(axis=0)


# ===================== 频道状态 =====================
class ChannelState:
    def __init__(self, ch):
        self.ch = ch
        self.state = "unknown"
        self.bearings = []          # [(pos, bearing_deg), ...]
        self.W = None               # 候选凸多边形
        self.probes = []            # 测量位置 (仅 no_signal / direction 都会记录)
        self._mec = None
        self._diam = None
        self._axis = None
        self._certified = None      # 缓存: (n_probes, bool)

    @property
    def is_active(self):
        return self.state not in ("cleared", "empty")

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
            self._diam = d; self._axis = ax
        return self._diam if self._diam is not None else 0.0

    @property
    def axis(self):
        if self._axis is None and self.W is not None and len(self.W) >= 3:
            d, ax = diam_and_axis(self.W)
            self._diam = d; self._axis = ax
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
        self._mec = None; self._diam = None; self._axis = None


# ===================== 总管 =====================
class Ledger:
    def __init__(self):
        self.channels = {ch: ChannelState(ch) for ch in range(1, N_CHANNELS + 1)}
        self.cur_pos = np.array([0., 0.])
        self.cur_channel = 1
        self.vt = 0.0
        self.cleared = set()
        self._cert_cache = {}

    # ---- 探针 ----
    def add_probe(self, ch, S):
        cs = self.channels[ch]
        cs.probes.append(np.asarray(S, float).copy())

    def check_empty(self, ch):
        """环抱证书: 探针集合对全圆域环抱 -> 频道必空."""
        cs = self.channels[ch]
        n = len(cs.probes)
        if n < 3:
            return False
        c = self._cert_cache.get(ch)
        if c is not None and c[0] == n:
            return c[1]
        if n > 40:
            # 只保留最外圈探针以节省计算 (内点对环抱贡献小)
            P = np.asarray(cs.probes, float)
            r = np.linalg.norm(P, axis=1)
            idx = np.argsort(-r)[:40]
            P = P[idx]
        else:
            P = np.asarray(cs.probes, float)
        ok = max_gap_all(P) <= 180.0 - CERT_MARGIN
        self._cert_cache[ch] = (n, ok)
        return ok

    # ---- 更新 ----
    def update_no_signal(self, ch, S):
        self.add_probe(ch, S)

    def update_direction(self, ch, S, bearing_deg):
        cs = self.channels[ch]
        S = np.asarray(S, float)
        th = np.radians(bearing_deg)
        cs.bearings.append((S.copy(), bearing_deg))
        self.add_probe(ch, S)
        if cs.W is None:
            cs.W = disk_poly([0., 0.], ARENA_R, 32)
        for hp_S, hp_e, hp_s in wedge_hp(S, th):
            if len(cs.W) >= 3:
                cs.W = clip(cs.W, hp_S, hp_e, hp_s)
        if len(cs.W) >= 3:
            cs.W = intersect_disk(cs.W, S, RECV_R_MAX, 64)
        if cs.state == "unknown":
            cs.state = "detected"
        cs.invalidate()

    def update_near(self, ch, S):
        cs = self.channels[ch]
        cs.state = "clearable"
        cs._mec = (0.0, np.asarray(S, float))
        cs._diam = 0.0

    def update_cleared(self, ch):
        cs = self.channels[ch]
        cs.state = "cleared"
        self.cleared.add(ch)

    def update_clear_fail(self, ch):
        cs = self.channels[ch]
        cs.state = "detected"
        cs._mec = None; cs._diam = None

    def check_clearable(self, ch):
        cs = self.channels[ch]
        if cs.is_clearable and cs.state == "detected":
            cs.state = "clearable"
        return cs.state == "clearable"

    # ---- 查询 ----
    def is_channel_empty(self, ch):
        cs = self.channels[ch]
        if cs.state == "empty":
            return True
        return self.check_empty(ch)

    def mark_empty(self):
        for ch in range(1, N_CHANNELS + 1):
            cs = self.channels[ch]
            if cs.state == "unknown" and self.check_empty(ch):
                cs.state = "empty"

    def active_channels(self):
        return [ch for ch in range(1, N_CHANNELS + 1) if self.channels[ch].is_active]

    def unknown_channels(self):
        return [ch for ch in range(1, N_CHANNELS + 1) if self.channels[ch].state == "unknown"]

    def detected_channels(self):
        return [ch for ch in range(1, N_CHANNELS + 1)
                if self.channels[ch].state in ("detected", "clearable")]

    def clearable_channels(self):
        return [ch for ch in range(1, N_CHANNELS + 1) if self.channels[ch].state == "clearable"]

    def resolved(self):
        return len(self.unknown_channels()) == 0 and len(self.detected_channels()) == 0

    def is_terminated(self):
        for ch in range(1, N_CHANNELS + 1):
            cs = self.channels[ch]
            if cs.state in ("cleared", "empty"):
                continue
            if cs.state == "unknown" and self.check_empty(ch):
                cs.state = "empty"
                continue
            return False
        return True
