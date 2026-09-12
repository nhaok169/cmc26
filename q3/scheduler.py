# -*- coding: utf-8 -*-
"""Q3 调度器 (§5.4 + §5.5).

SmartStart: L ≤ (Θ-1)·t_已耗 => 收割, 否则探索  [Birx-Disser]
2-opt 收割巡回: 最近邻初解 + 2-opt 改进, 环形序约束  [Kamiński Thm 10]
CELF: 停点内频道选择, 惰性优先队列  [Krause §1.4 / Leskovec KDD'07]
"""
import numpy as np
import heapq
from ledger import Ledger, SPEED, SWITCH_COST, MEASURE_COST, RECV_R_MIN, RECV_R_MAX


# ===================== SmartStart =====================
def harvest_tour_length(ledger):
    """可清除队列各 MEC 中心的 2-opt 巡回长度 (虚拟秒)."""
    clearable = ledger.clearable_channels()
    if not clearable:
        return 0.0
    pts = []
    for ch in clearable:
        m = ledger.channels[ch].mec
        if m is not None and m[1] is not None:
            pts.append(m[1])
        else:
            c = ledger.channels[ch].centroid
            if c is not None:
                pts.append(c)
    if not pts:
        return 0.0
    tour, dist = two_opt_tour(pts, ledger.cur_pos)
    return dist / SPEED


def smartstart_check(ledger, theta=2.05):
    """L ≤ (Θ-1)·t_已耗 => True (出发收割)."""
    L = harvest_tour_length(ledger)
    t_elapsed = ledger.vt
    return L <= (theta - 1.0) * max(t_elapsed, 1.0)


# ===================== 2-opt TSP =====================
def two_opt_tour(points, start_pos):
    """2-opt 巡回: 从 start_pos 出发, 访问全部 points, 不需返程.

    返回 (order: [idx], total_dist: float).
    """
    n = len(points)
    if n == 0:
        return [], 0.0
    pts = [np.asarray(p, float) for p in points]
    start = np.asarray(start_pos, float)

    # 初解: 极角序 (环形序约束, Kamiński Thm 10)
    if n >= 2:
        angles = [np.arctan2(p[1] - start[1], p[0] - start[0]) for p in pts]
        order = np.argsort(angles).tolist()
    else:
        order = [0]

    # 最近邻改进初解
    order = _nearest_neighbor(pts, start)

    # 计算总距离 (不返程)
    def tour_dist(ord_):
        d = np.linalg.norm(pts[ord_[0]] - start)
        for i in range(len(ord_) - 1):
            d += np.linalg.norm(pts[ord_[i + 1]] - pts[ord_[i]])
        return d

    best_order = list(order)
    best_dist = tour_dist(best_order)

    # 2-opt
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_order = best_order[:i] + best_order[i:j + 1][::-1] + best_order[j + 1:]
                new_dist = tour_dist(new_order)
                if new_dist < best_dist - 1e-6:
                    best_dist = new_dist
                    best_order = new_order
                    improved = True
        # 只做一轮 (n 小, 足够)
        break

    return best_order, best_dist


def _nearest_neighbor(pts, start):
    """最近邻初解."""
    n = len(pts)
    visited = [False] * n
    order = []
    cur = start
    for _ in range(n):
        best_i = -1
        best_d = np.inf
        for i in range(n):
            if not visited[i]:
                d = np.linalg.norm(pts[i] - cur)
                if d < best_d:
                    best_d = d
                    best_i = i
        order.append(best_i)
        visited[best_i] = True
        cur = pts[best_i]
    return order


# ===================== 收割批次 =====================
def harvest_batch_plan(ledger):
    """生成收割批次: (频道, MEC中心) 列表, 2-opt+环形序排序."""
    clearable = ledger.clearable_channels()
    if not clearable:
        return []

    pts = []
    chs = []
    for ch in clearable:
        cs = ledger.channels[ch]
        m = cs.mec
        center = m[1] if (m and m[1] is not None) else cs.centroid
        if center is not None:
            pts.append(center)
            chs.append(ch)

    if not pts:
        return []

    order, dist = two_opt_tour(pts, ledger.cur_pos)
    return [(chs[order[i]], pts[order[i]]) for i in range(len(order))]


# ===================== CELF 停点内频道调度 =====================
def celf_select(ledger, p, budget_channels=8):
    """CELF lazy greedy: 在停点 p 选择要测量的频道组合.

    收益 = 探索排除面积 (unknown) + E/D 精化增量 (detected)
    成本 = 5s (measure) + 1s (switch if ch != cur)
    """
    candidates = []
    cur_ch = ledger.cur_channel

    # unknown 频道: 探索收益
    for ch in ledger.unknown_channels():
        area = ledger.exploration_area(ch, p)
        cost = MEASURE_COST + (SWITCH_COST if ch != cur_ch else 0)
        if area > 0:
            candidates.append((area / cost, ch, area, cost))

    # detected 频道: 精化收益 (简化: 距候选域近才测)
    from score import bearing_scatter_matrix, det2
    for ch in ledger.detected_channels():
        cs = ledger.channels[ch]
        if cs.W is None or len(cs.W) < 3 or len(cs.bearings) == 0:
            continue
        centroid = cs.centroid
        if centroid is None:
            continue
        d = np.linalg.norm(np.asarray(p, float) - centroid)
        if d > RECV_R_MAX:
            continue
        # E/D 增量
        S_cur = bearing_scatter_matrix(cs.bearings, centroid)
        det_cur = max(det2(S_cur), 1e-15)
        diff = centroid - np.asarray(p, float)
        d2 = max(float(diff @ diff), 1.0)
        theta = np.arctan2(diff[1], diff[0])
        u_perp = np.array([-np.sin(theta), np.cos(theta)])
        dS = np.outer(u_perp, u_perp) / d2
        det_new = max(det2(S_cur + dS), 1e-15)
        benefit = cs.diameter * (np.log(det_new) - np.log(det_cur))
        cost = MEASURE_COST + (SWITCH_COST if ch != cur_ch else 0)
        if benefit > 0:
            candidates.append((benefit / cost, ch, benefit, cost))

    # 按效益/成本比降序
    candidates.sort(key=lambda x: -x[0])

    # 选择直到预算耗尽
    selected = []
    total_cost = 0.0
    for ratio, ch, benefit, cost in candidates:
        if total_cost + cost > budget_channels * (MEASURE_COST + SWITCH_COST):
            break
        if benefit <= 0:
            break
        selected.append(ch)
        total_cost += cost

    return selected


def en_route_clear(ledger, dest, threshold=50.0):
    """顺路清除: 移动到 dest 途中, 距可清除源 MEC ≤ threshold 返回该频道."""
    p = np.asarray(dest, float)
    for ch in ledger.clearable_channels():
        cs = ledger.channels[ch]
        m = cs.mec
        if m and m[1] is not None:
            d = np.linalg.norm(m[1] - p)
            if d <= threshold:
                return ch, m[1]
    return None
