# -*- coding: utf-8 -*-
"""Q3 主循环 (§5.9): 四模块组装 — 双账 + 探索 + 精化 + 收割 + 终止.

v6 自适应追踪版 (Adaptive Pursuit):
  1. 原点全扫 20 频道
  2. 追踪阶段: 步进走向源, 每步 300m + 递减偏角, 边走边测
     - 理论: 顺序方位定位 (SBL), Fisher 信息 ∝ 1/D² 累积
     - 自适应采样 (Krause 2008): 每步基于已有 LSQ 重定向
     - Calafiore D-score: 追踪顺序按 W 直径降序
  3. 补网阶段: 骨架环点扫描 + 免费多频道扫描
  4. 终止: 全频道 ∈ 已清除 ∪ {覆盖证书 / F_k=∅}
"""
import numpy as np
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "q2"))
from geometry import diam_and_axis, mec as compute_mec

from ledger import (Ledger, SKELETON, ARENA_R, RECV_R_MIN, RECV_R_MAX,
                     CLEAR_R, NEAR_R, SPEED, SWITCH_COST, MEASURE_COST,
                     N_CHANNELS, poly_centroid, poly_area)
from score import Scorer, exploration_score, ed_refinement_score
from scheduler import (smartstart_check, harvest_batch_plan, celf_select,
                        en_route_clear, two_opt_tour, harvest_tour_length)

REAL_TIME_LIMIT = 18 * 60

BET_T = 500.0
BET_H = 250.0
MAX_PURSUIT_STEPS = 6
EARLY_CLEAR_R = 40.0

PURSUIT_STEP = 350.0
PURSUIT_OFFSET_INIT = 15.0


def _unit(deg):
    t = math.radians(deg)
    return np.array([math.cos(t), math.sin(t)], dtype=float)


def _clamp_xy(p):
    p = np.asarray(p, float)
    n = float(np.linalg.norm(p))
    if n > 1.8e6:
        p = p * (1.8e6 / n)
    return p


def second_point(S, theta_deg, t=BET_T, h=BET_H):
    e = _unit(theta_deg)
    n = np.array([-e[1], e[0]])
    return _clamp_xy(np.asarray(S, float) + t * e + h * n)


def long_axis_point(vertices, robot_pos):
    v = np.asarray(vertices, float)
    if len(v) < 2:
        return _clamp_xy(v.mean(axis=0) + np.array([80.0, 0.0]))
    r, C = compute_mec(v) if len(v) >= 3 else (0.5 * float(np.linalg.norm(v[0] - v[1])), (v[0] + v[1]) / 2)
    if C is None or not np.isfinite(r):
        C = v.mean(axis=0)
    L, ax = diam_and_axis(v)
    if ax is None or L <= 1e-9:
        ax = np.array([1.0, 0.0])
        L = 40.0
    rho = max(50.0, 0.5 * float(L))
    nhat = np.array([-ax[1], ax[0]], dtype=float)
    if float(nhat @ (np.asarray(robot_pos, float) - C)) < 0:
        nhat = -nhat
    return _clamp_xy(C + rho * nhat)


class Policy:
    """Q3 v2 主策略 (v6 自适应追踪版)."""

    def __init__(self, sim, alpha=2.0, theta=2.05,
                 use_singamma=False, rollout=False, eta=50.0):
        self.sim = sim
        self.ledger = Ledger(eta=eta)
        self.scorer = Scorer(alpha=alpha, use_singamma=use_singamma)
        self.theta = theta
        self.rollout = rollout
        self._visited = []
        self._abort = False

    def _sync(self):
        self.ledger.cur_pos = np.array(self.sim.cur_pos, float)
        self.ledger.cur_channel = self.sim.cur_channel
        self.ledger.vt = self.sim.vt
        self.ledger.cleared = self.sim.cleared

    @staticmethod
    def _lsq(bearings):
        Sxx = Sxy = Syy = sx = sy = 0.0
        for pos, deg in bearings:
            a = math.radians(deg)
            n0, n1 = -math.sin(a), math.cos(a)
            p = np.asarray(pos, float)
            b = n0 * p[0] + n1 * p[1]
            Sxx += n0 * n0; Sxy += n0 * n1; Syy += n1 * n1
            sx += n0 * b; sy += n1 * b
        det = Sxx * Syy - Sxy * Sxy
        if abs(det) < 1e-9:
            return None
        return np.array([(sx * Syy - sy * Sxy) / det,
                         (Sxx * sy - Sxy * sx) / det])

    def _handle_measure(self, ch, pos, resp):
        mr = resp.get("measure_result")
        if mr == "direction":
            self.ledger.update_direction(ch, pos, resp["svd_deg"])
        elif mr == "near":
            if self.sim.clear(self.sim.cur_pos, ch):
                self.ledger.update_cleared(ch)
            else:
                self.ledger.update_near(ch, pos)
        elif mr == "no_signal":
            self.ledger.update_no_signal(ch, pos)

    def initial_scan(self):
        sim = self.sim
        for ch in range(1, N_CHANNELS + 1):
            resp = sim.measure((0.0, 0.0), ch)
            mr = resp.get("measure_result")
            if mr == "direction":
                self.ledger.update_direction(ch, (0., 0.), resp["svd_deg"])
            elif mr == "near":
                if sim.clear(sim.cur_pos, ch):
                    self.ledger.update_cleared(ch)
                else:
                    self.ledger.update_near(ch, (0., 0.))
            elif mr == "no_signal":
                self.ledger.update_no_signal(ch, (0., 0.))
        self._sync()
        self._visited.append(np.array([0., 0.]))
        for ch in range(1, N_CHANNELS + 1):
            self.ledger.check_clearable(ch)

    def _is_visited(self, p, tol=80.0):
        p = np.asarray(p, float)
        for v in self._visited:
            if np.linalg.norm(p - v) < tol:
                return True
        return False

    def hunt_channel(self, ch):
        """自适应追踪: 沿方位走向源, 每步 300m + 递减偏角, 边走边测."""
        if ch in self.sim.cleared:
            return True
        cs = self.ledger.channels[ch]
        if not cs.is_active:
            return True

        for step in range(MAX_PURSUIT_STEPS):
            cs = self.ledger.channels[ch]
            if not cs.is_active or ch in self.sim.cleared:
                return True

            bearings = cs.bearings
            if not bearings:
                return False

            if len(bearings) >= 2:
                est = self._lsq(bearings)
                if est is not None:
                    cur = self.ledger.cur_pos
                    d_est = float(np.linalg.norm(est - cur))
                    step_dist = min(PURSUIT_STEP, max(120.0, 0.35 * d_est))
                    direction = math.degrees(math.atan2(est[1] - cur[1], est[0] - cur[0]))
                    offset = max(5.0, PURSUIT_OFFSET_INIT - step * 2.5)
                    offset = offset if step % 2 == 0 else -offset
                    walk_angle = direction + offset
                else:
                    walk_angle = bearings[-1][1] + (PURSUIT_OFFSET_INIT if step % 2 == 0 else -PURSUIT_OFFSET_INIT)
                    step_dist = PURSUIT_STEP
            else:
                walk_angle = bearings[0][1] + (PURSUIT_OFFSET_INIT if step % 2 == 0 else -PURSUIT_OFFSET_INIT)
                step_dist = PURSUIT_STEP

            if cs.W is not None and len(cs.W) >= 3:
                m = cs.mec
                if m is not None and m[1] is not None:
                    r_mec, center = m
                    if r_mec <= EARLY_CLEAR_R:
                        if self._clear_source(ch, center):
                            return True
                    if r_mec <= CLEAR_R:
                        if cs.W is not None:
                            for v in cs.W:
                                if self._clear_source(ch, v):
                                    return True
                        if self._clear_source(ch, center):
                            return True
                        return False

            e = _unit(walk_angle)
            target = _clamp_xy(self.ledger.cur_pos + step_dist * e)

            if self._is_visited(target, tol=50):
                target = _clamp_xy(self.ledger.cur_pos + step_dist * _unit(walk_angle + 30))

            self._visited.append(np.asarray(target, float))
            resp = self.sim.measure(target, ch)
            self._handle_measure(ch, target, resp)
            self._sync()
            self.ledger.check_clearable(ch)

            if step < 2:
                self._free_scan_at(target, exclude_ch=ch)

            if resp.get("measure_result") == "near":
                if self.sim.clear(self.sim.cur_pos, ch):
                    self.ledger.update_cleared(ch)
                    return True

            if ch in self.sim.cleared:
                return True

        cs = self.ledger.channels[ch]
        if cs.W is not None and len(cs.W) >= 3:
            target = long_axis_point(cs.W, self.ledger.cur_pos)
            self._visited.append(np.asarray(target, float))
            resp = self.sim.measure(target, ch)
            self._handle_measure(ch, target, resp)
            self._sync()
            self.ledger.check_clearable(ch)

            m = cs.mec
            if m is not None and m[1] is not None and m[0] <= CLEAR_R:
                return self._clear_source(ch, m[1])

        if cs.bearings and len(cs.bearings) >= 2:
            est = self._lsq(cs.bearings)
            if est is not None:
                if self._clear_source(ch, est):
                    return True
        return False

    def _free_scan_at(self, pos, exclude_ch=None):
        """在停点免费扫描所有 detected <2 bearings 频道 (CELF 简化).

        越近源的停点信息价值越高 (Fisher ∝ 1/D²).
        """
        for ch in self.ledger.detected_channels():
            if ch == exclude_ch:
                continue
            if ch in self.sim.cleared:
                continue
            cs = self.ledger.channels[ch]
            if len(cs.bearings) >= 2:
                continue
            resp = self.sim.measure(pos, ch)
            self._handle_measure(ch, pos, resp)
            self._sync()
            self.ledger.check_clearable(ch)

    def hunt_all(self):
        """E/D 分数排序 (Calafiore D-score: W 直径降序) + 最近邻路径."""
        pending = [ch for ch in self.ledger.detected_channels()
                   if ch not in self.sim.cleared
                   and self.ledger.channels[ch].is_active]
        if not pending:
            return

        def ed_priority(ch):
            return -self.ledger.channels[ch].diameter

        pending.sort(key=ed_priority)
        remaining = list(pending)
        while remaining:
            candidates = remaining[:min(3, len(remaining))]
            cur = self.ledger.cur_pos
            best_ch = min(candidates, key=lambda ch: np.linalg.norm(
                (self.ledger.channels[ch].centroid if self.ledger.channels[ch].centroid is not None
                 else np.array([0., 0.])) - cur))
            remaining.remove(best_ch)
            self.hunt_channel(best_ch)
            self._sync()

    def ring_cover(self):
        """补网阶段: 2-opt 优化环点顺序 + unknown 扫描.
        优化: unknown≤3时只走最近2点, 已无unknown提前退出.
        """
        ring_pts = [SKELETON[i] for i in range(1, 7)]
        cur = self.ledger.cur_pos

        n_unknown = len(self.ledger.unknown_channels())
        if n_unknown == 0:
            return
        if n_unknown <= 3:
            n_ring = 2
        elif n_unknown <= 6:
            n_ring = 4
        else:
            n_ring = 6

        # 2-opt 优化环点访问顺序 (从当前位置出发)
        indices = list(range(6))
        def tour_len(order):
            pts = [cur] + [ring_pts[i] for i in order]
            return sum(np.linalg.norm(pts[j+1] - pts[j]) for j in range(len(pts)-1))
        best_order = indices[:]
        best_len = tour_len(best_order)
        improved = True
        while improved:
            improved = False
            for i in range(6):
                for j in range(i+1, 6):
                    new_order = best_order[:i] + best_order[i:j+1][::-1] + best_order[j+1:]
                    new_len = tour_len(new_order)
                    if new_len < best_len - 1:
                        best_order = new_order
                        best_len = new_len
                        improved = True

        order = [ring_pts[i] for i in best_order[:n_ring]]
        cover = list(SKELETON)

        visited_ring = set()
        for i in range(n_ring):
            if len(self.sim.cleared) >= N_CHANNELS:
                break
            unknown = self.ledger.unknown_channels()
            if not unknown:
                break
            if self.sim.elapsed_real() > REAL_TIME_LIMIT:
                break

            # 每次从当前位置选最近未访问环点
            cur = self.ledger.cur_pos
            best_target = None
            best_d = float('inf')
            for j, pt in enumerate(ring_pts):
                if j in visited_ring:
                    continue
                d = np.linalg.norm(pt - cur)
                if d < best_d:
                    best_d = d
                    best_target = pt
                    best_j = j
            if best_target is None:
                break
            visited_ring.add(best_j)
            target = best_target

            if not self._is_visited(target, tol=100):
                self._visited.append(np.asarray(target, float))

            unknown_sorted = sorted(unknown,
                key=lambda c: abs(c - self.ledger.cur_channel))
            for ch in unknown_sorted:
                if ch in self.sim.cleared:
                    continue
                if not self.ledger.channels[ch].is_active:
                    continue
                resp = self.sim.measure(target, ch)
                self._handle_measure(ch, target, resp)
                self._sync()

            self._free_scan_at(target)

            for ch in list(unknown):
                self.ledger.check_clearable(ch)

            self._try_enroute(target)
            self.hunt_all()
            self._mark_empty(cover)

    def _mark_empty(self, cover_pts):
        for ch in self.ledger.unknown_channels():
            cs = self.ledger.channels[ch]
            if cs.skeleton_tested.all():
                cs.state = "empty"

    def _clear_source(self, ch, center):
        if self.sim.clear(center, ch):
            self.ledger.update_cleared(ch)
            return True
        # 有限径向扫描: 只试 3 次 (25m, 50m, 75m), 避免大量失败代价
        cs = self.ledger.channels[ch]
        est = self._lsq(cs.bearings) if cs.bearings else None
        if est is not None:
            for rad in [25, 50, 75]:
                for a in range(0, 360, 90):
                    q = (est[0] + rad * math.cos(math.radians(a)),
                         est[1] + rad * math.sin(math.radians(a)))
                    if self.sim.clear(q, ch):
                        self.ledger.update_cleared(ch)
                        return True
        self.ledger.update_clear_fail(ch)
        return False

    def _try_enroute(self, dest):
        for ch in list(self.ledger.clearable_channels()):
            if ch in self.sim.cleared:
                continue
            cs = self.ledger.channels[ch]
            m = cs.mec
            if m and m[1] is not None:
                d = np.linalg.norm(m[1] - np.asarray(dest, float))
                if d <= CLEAR_R * 2.5:
                    self._clear_source(ch, m[1])
                    self._sync()

    def run(self):
        sim = self.sim
        try:
            self.initial_scan()
            self.hunt_all()
            self.ring_cover()
            self._fallback_clear()
            remaining_clearable = self.ledger.clearable_channels()
            if remaining_clearable:
                self.harvest_step()
        except Exception as e:
            print(f"[WARN] 策略异常: {e}")
            import traceback; traceback.print_exc()
            self._abort = True

        self._sync()
        n = len(sim.cleared)
        vt = sim.vt
        print(f"\n========== Q3 v2 统计 ==========")
        print(f"清除干扰源数 : {n}")
        print(f"虚拟世界总时间: {vt:.1f} s ({vt / 3600:.2f} h)")
        if n > 0:
            print(f"平均定位清除时间: {vt / n:.1f} s/个")
        print(f"检测次数 {sim.measure_cnt}, 换频 {sim.switch_cnt} 次, 清除失败 {sim.clear_fail} 次")
        print(f"程序现实运行时间: {sim.elapsed_real():.1f} s")
        return n, vt

    def harvest_step(self):
        batch = harvest_batch_plan(self.ledger)
        if not batch:
            return False
        for ch, center in batch:
            if ch in self.sim.cleared:
                continue
            self._clear_source(ch, center)
            self._sync()
        return True

    def _fallback_clear(self):
        for ch in self.ledger.detected_channels():
            if ch in self.sim.cleared:
                continue
            cs = self.ledger.channels[ch]
            if not cs.bearings or len(cs.bearings) < 2:
                continue
            est = self._lsq(cs.bearings)
            if est is None:
                continue
            if self._is_visited(est, tol=30):
                continue
            self._visited.append(np.asarray(est, float))
            if self._clear_source(ch, est):
                self._sync()
                return True
        return False
