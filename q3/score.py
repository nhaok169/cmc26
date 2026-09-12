# -*- coding: utf-8 -*-
"""Q3 评分函数 (§5.2 + §5.3).

探索分: 成本版 AdaptiveGreedy — Δ排除质量(p) / 路费(p)  [G&K Thm 5.10]
精化分: E/D 集员分数 — Σ_k diam(W_k)·Δlog det S_{I∪p,k}  [Calafiore D-score]
消融开关: sinγ 版 — Σ_k diam(W_k)·max_i |sin(θ_p - θ_i)|  (v1 对照)

score(p) = α · 探索分(p) + 精化分(p)
"""
import numpy as np
from ledger import Ledger, SPEED, SWITCH_COST, MEASURE_COST, RECV_R_MIN, RECV_R_MAX


def travel_cost(p, cur_pos, n_channels=1):
    """移动 + 测量虚拟时间."""
    dist = np.linalg.norm(np.asarray(p, float) - np.asarray(cur_pos, float))
    return dist / SPEED + n_channels * (MEASURE_COST + SWITCH_COST)


# ===================== 探索分 =====================
def exploration_score(ledger, p):
    """Σ_{k unknown} 可排除格点面积 / 路费.  硬账最坏情形 (R=1000)."""
    unknown = ledger.unknown_channels()
    if not unknown:
        return 0.0
    total_area = 0.0
    for ch in unknown:
        total_area += ledger.exploration_area(ch, p)
    cost = travel_cost(p, ledger.cur_pos, n_channels=max(1, len(unknown) // 4))
    return total_area / max(cost, 1e-6)


# ===================== E/D 精化分 =====================
def bearing_scatter_matrix(bearings, source_est):
    """S_{I,k} = Σ_i u_i^⊥ u_i^⊥^T / d_i^2  (2×2).

    bearings: [(pos_xy, bearing_deg), ...]
    source_est: 估计源位置 (W_k 重心)
    """
    S = np.zeros((2, 2))
    se = np.asarray(source_est, float)
    for pos, _ in bearings:
        p = np.asarray(pos, float)
        diff = se - p
        d2 = diff @ diff
        if d2 < 1.0:
            d2 = 1.0
        theta = np.arctan2(diff[1], diff[0])
        # u^⊥ = (-sin θ, cos θ)
        u_perp = np.array([-np.sin(theta), np.cos(theta)])
        S += np.outer(u_perp, u_perp) / d2
    return S


def det2(M):
    return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]


def ed_refinement_score(ledger, p):
    """Σ_k diam(W_k) · Δlog det S_{I∪p,k}  (D 分数)."""
    p = np.asarray(p, float)
    total = 0.0
    for ch in ledger.detected_channels():
        cs = ledger.channels[ch]
        if cs.W is None or len(cs.W) < 3 or len(cs.bearings) == 0:
            continue
        if not cs.is_active:
            continue
        centroid = cs.centroid
        if centroid is None:
            continue
        # 当前散布矩阵
        S_cur = bearing_scatter_matrix(cs.bearings, centroid)
        det_cur = det2(S_cur)
        if det_cur < 1e-15:
            det_cur = 1e-15
        # 加入候选点 p 后的增量
        diff = centroid - p
        d2 = diff @ diff
        if d2 < 1.0:
            d2 = 1.0
        theta = np.arctan2(diff[1], diff[0])
        u_perp = np.array([-np.sin(theta), np.cos(theta)])
        dS = np.outer(u_perp, u_perp) / d2
        S_new = S_cur + dS
        det_new = det2(S_new)
        if det_new < 1e-15:
            continue
        delta_logdet = np.log(det_new) - np.log(det_cur)
        total += cs.diameter * delta_logdet
    return total


# ===================== sinγ 消融分 =====================
def singamma_refinement_score(ledger, p):
    """Σ_k diam(W_k) · max_i |sin(θ_p - θ_i)|  (v1 消融对照)."""
    p = np.asarray(p, float)
    total = 0.0
    for ch in ledger.detected_channels():
        cs = ledger.channels[ch]
        if cs.W is None or len(cs.W) < 3 or len(cs.bearings) == 0:
            continue
        if not cs.is_active:
            continue
        centroid = cs.centroid
        if centroid is None:
            continue
        # 候选点到源的方位
        diff = centroid - p
        theta_p = np.arctan2(diff[1], diff[0])
        # 找最大交角 sin
        best_sin = 0.0
        for pos, deg in cs.bearings:
            theta_i = np.radians(deg)
            s = abs(np.sin(theta_p - theta_i))
            if s > best_sin:
                best_sin = s
        total += cs.diameter * best_sin
    return total


# ===================== 组合评分 =====================
class Scorer:
    """组合评分器, 含 α 旋钮和 sinγ 消融开关."""

    def __init__(self, alpha=2.0, use_singamma=False):
        self.alpha = alpha
        self.use_singamma = use_singamma

    def score(self, ledger, p):
        """score(p) = α · 探索分 + 精化分."""
        exp_score = exploration_score(ledger, p)
        if self.use_singamma:
            ref_score = singamma_refinement_score(ledger, p)
        else:
            ref_score = ed_refinement_score(ledger, p)
        return self.alpha * exp_score + ref_score

    def score_batch(self, ledger, candidates):
        """批量评分, 返回 (分数数组, 候选点数组)."""
        scores = np.array([self.score(ledger, p) for p in candidates])
        return scores

    def best_candidate(self, ledger, candidates, exclude=None):
        """选最优候选点. exclude: set of indices to skip."""
        if len(candidates) == 0:
            return None
        best_score = -np.inf
        best_idx = -1
        for i, p in enumerate(candidates):
            if exclude and i in exclude:
                continue
            s = self.score(ledger, p)
            if s > best_score:
                best_score = s
                best_idx = i
        if best_idx < 0:
            return None
        return candidates[best_idx]
