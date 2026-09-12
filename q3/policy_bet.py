# -*- coding: utf-8 -*-
"""Q3 路线A: Q2 赌注点策略 (Bet-Point).

与 policy.py (路线B: 自适应追踪) 的唯一区别在 hunt_channel:
  路线A: 走 Q2 赌注点 (500,250) → 交会 → MEC → 长轴救援 → 清除
  路线B: 6步×350m 逐步逼近, 每步更新 LSQ

其余模块 (initial_scan, hunt_all, ring_cover, 终止) 完全相同.
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
MAX_RESCUE_STEPS = 3
EARLY_CLEAR_R = 40.0


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


class PolicyBet:
    """Q3 路线A: Q2 赌注点策略."""

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
        """Q2 赌注点策略: 走赌注点 → 交会 → MEC → 长轴救援 → 清除."""
        if ch in self.sim.cleared:
            return True
        cs = self.ledger.channels[ch]
        if not cs.is_active:
            return True

        bearings = cs.bearings
        if not bearings:
            return False

        # Step 1: 走 Q2 赌注点 (500, 250)
        first_pos = bearings[0][0]
        first_deg = bearings[0][1]
        target = second_point(first_pos, first_deg)

        if self._is_visited(target, tol=50):
            target = second_point(first_pos, first_deg, t=BET_T * 0.7, h=BET_H * 1.5)

        self._visited.append(np.asarray(target, float))
        resp = self.sim.measure(target, ch)
        self._handle_measure(ch, target, resp)
        self._sync()
        self.ledger.check_clearable(ch)

        self._free_scan_at(target, exclude_ch=ch)

        if resp.get("measure_result") == "near":
            if self.sim.clear(self.sim.cur_pos, ch):
                self.ledger.update_cleared(ch)
                return True

        if ch in self.sim.cleared:
            return True

        # Step 2: 检查 MEC, 如果 ≤20m 直接清除
        cs = self.ledger.channels[ch]
        if cs.W is not None and len(cs.W) >= 3:
            m = cs.mec
            if m is not None and m[1] is not None:
                r_mec, center = m
                if r_mec <= EARLY_CLEAR_R:
                    if self._clear_source(ch, center):
                        return True
                if r_mec <= CLEAR_R:
                    if self._clear_source(ch, center):
                        return True
                    return False

        # Step 3: 长轴救援 (最多 3 次)
        for rescue in range(MAX_RESCUE_STEPS):
            cs = self.ledger.channels[ch]
            if not cs.is_active or ch in self.sim.cleared:
                return True
            if cs.W is None or len(cs.W) < 3:
                break

            target = long_axis_point(cs.W, self.ledger.cur_pos)
            self._visited.append(np.asarray(target, float))
            resp = self.sim.measure(target, ch)
            self._handle_measure(ch, target, resp)
            self._sync()
            self.ledger.check_clearable(ch)

            if resp.get("measure_result") == "near":
                if self.sim.clear(self.sim.cur_pos, ch):
                    self.ledger.update_cleared(ch)
                    return True

            m = cs.mec
            if m is not None and m[1] is not None:
                r_mec, center = m
                if r_mec <= EARLY_CLEAR_R:
                    if self._clear_source(ch, center):
                        return True
                if r_mec <= CLEAR_R:
                    if self._clear_source(ch, center):
                        return True
                    return False

        # Step 4: 用 LSQ 估计兜底清除
        cs = self.ledger.channels[ch]
        if cs.bearings and len(cs.bearings) >= 2:
            est = self._lsq(cs.bearings)
            if est is not None:
                if self._clear_source(ch, est):
                    return True
        return False

    def _free_scan_at(self, pos, exclude_ch=None):
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
        print(f"\n========== Q3 路线A (赌注点) 统计 ==========")
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
