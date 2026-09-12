# -*- coding: utf-8 -*-
"""Q4 主循环: v1 最小可行版.

与 Q3 的区别:
  - ring_cover → grid_cover (31 点格点证书, 替代 7 点环)
  - _mark_empty → _mark_empty_grid (31 点全覆盖才判空)
  - 其余 (hunt_channel, hunt_all, initial_scan 等) 完全同 Q3
"""
import numpy as np
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "q2"))
from geometry import diam_and_axis, mec as compute_mec

from ledger import (Ledger, SKELETON, SKELETON_N, SKELETON_INNER_N,
                     ARENA_R, RECV_R_MIN, RECV_R_MAX,
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
    """Q4 v1 主策略 (Q3 + 31 点格点证书)."""

    def __init__(self, sim, alpha=2.0, theta=2.05,
                 use_singamma=False, rollout=False, eta=50.0):
        self.sim = sim
        self.ledger = Ledger(eta=eta)
        self.scorer = Scorer(alpha=alpha, use_singamma=use_singamma)
        self.theta = theta
        self.rollout = rollout
        self._visited = []
        self._abort = False
        self._phase_times = {}
        self._clear_fail_count = {}  # 方向2: 按频道计数清除失败

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

    def _bearing_span(self, bearings):
        """计算方位线最大夹角跨度 (度). 夹角太小则 LSQ 不稳定."""
        if len(bearings) < 2:
            return 0.0
        degs = [b[1] for b in bearings]
        max_span = 0.0
        for i in range(len(degs)):
            for j in range(i + 1, len(degs)):
                diff = abs((degs[i] - degs[j] + 180) % 360 - 180)
                if diff > max_span:
                    max_span = diff
        return max_span

    def hunt_channel(self, ch):
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

            # 方向2: 清除失败上限 — 超过 5 次直接放弃, 等 grid_cover 重新检出
            if self._clear_fail_count.get(ch, 0) >= 5:
                return False

            # 方向1: 方位线夹角检查 — 小夹角时不清除, 改走垂直方向
            span = self._bearing_span(bearings)
            can_clear = (span >= 15.0) and (len(bearings) >= 2)

            if can_clear and cs.W is not None and len(cs.W) >= 3:
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

            # 方向1: 小夹角时走垂直方向以获取角度多样性
            if not can_clear and len(bearings) >= 2:
                last_deg = bearings[-1][1]
                walk_angle = last_deg + 90 if step % 2 == 0 else last_deg - 90
                step_dist = PURSUIT_STEP

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
            span = self._bearing_span(cs.bearings)
            if span >= 15.0:
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

    def _free_scan_unknown_at(self, pos, max_ch=None):
        """在追踪停点顺便扫描 unknown 频道.

        好处: (1) 可能检出定向源 (2) no_signal 收缩 hard_mask → 格点可跳过
        限制: 仅扫 unknown 频道, 按频道接近度排序省换频
        """
        unknown = self.ledger.unknown_channels()
        if not unknown:
            return
        unknown_sorted = sorted(unknown, key=lambda c: abs(c - self.ledger.cur_channel))
        for ch in unknown_sorted[:max_ch] if max_ch else unknown_sorted:
            if ch in self.sim.cleared:
                continue
            if not self.ledger.channels[ch].is_active:
                continue
            resp = self.sim.measure(pos, ch)
            self._handle_measure(ch, pos, resp)
            self._sync()
            if resp.get("measure_result") == "direction":
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

    def _is_grid_point_worth_visiting(self, pt, idx):
        """检查格点是否值得访问.

        安全判据: 只跳过"已在 idx 点实际测量过所有 unknown 频道"的格点.
        不用 hard_mask (对定向源 no_signal 排除整圆是过度乐观的).
        """
        unknown = self.ledger.unknown_channels()
        if not unknown:
            return False
        for ch in unknown:
            cs = self.ledger.channels[ch]
            if not cs.is_active:
                continue
            if not cs.skeleton_tested[idx]:
                return True
        return False

    def grid_cover(self):
        """31 点格点证书扫描 (替代 Q3 的 ring_cover).

        v3 优化:
        1. 自适应跳过: no_signal 已排除的格点不访问
        2. 2-opt 路径: 预计算最优巡回顺序
        3. 每个格点只扫 unknown 频道
        4. 检出新源 → 立即 hunt_all
        5. 全部 31 点扫完 → 空频道 (覆盖保证)
        """
        grid_pts = [SKELETON[i] for i in range(SKELETON_N)]
        visited = set()
        skipped = set()

        while len(visited) + len(skipped) < len(grid_pts):
            unknown = self.ledger.unknown_channels()
            if not unknown:
                break
            if len(self.sim.cleared) >= N_CHANNELS:
                break
            if self.sim.elapsed_real() > REAL_TIME_LIMIT:
                break

            cur = self.ledger.cur_pos
            best_d = float('inf')
            best_idx = -1
            for j in range(SKELETON_N):
                if j in visited or j in skipped:
                    continue
                d = np.linalg.norm(grid_pts[j] - cur)
                if d < best_d:
                    best_d = d
                    best_idx = j
            if best_idx < 0:
                break
            visited.add(best_idx)
            target = grid_pts[best_idx]

            if not self._is_grid_point_worth_visiting(target, best_idx):
                for ch in unknown:
                    cs = self.ledger.channels[ch]
                    diffs = SKELETON - target
                    for i in range(SKELETON_N):
                        if np.linalg.norm(diffs[i]) < self.ledger.eta * 0.5 + 1:
                            cs.skeleton_tested[i] = True
                    cs.invalidate()
                skipped.add(best_idx)
                continue

            if not self._is_visited(target, tol=100):
                self._visited.append(np.asarray(target, float))

            unknown_sorted = sorted(unknown,
                key=lambda c: abs(c - self.ledger.cur_channel))
            for ch in unknown_sorted:
                if ch in self.sim.cleared:
                    continue
                if not self.ledger.channels[ch].is_active:
                    continue
                if self.ledger.channels[ch].skeleton_tested[best_idx]:
                    continue
                resp = self.sim.measure(target, ch)
                self._handle_measure(ch, target, resp)
                self._sync()

            # 方向C: 只在有未清除检出源时才 free_scan + hunt_all
            has_uncleared_detected = any(
                ch not in self.sim.cleared and self.ledger.channels[ch].is_active
                for ch in self.ledger.detected_channels()
            )
            if has_uncleared_detected:
                self._free_scan_at(target)
                self._try_enroute(target)
                self.hunt_all()
            self._mark_empty_grid(visited | skipped)

    def _mark_empty_grid(self, visited):
        """格点覆盖证书: 全部 31 点扫完且无信号 → 空频道."""
        if len(visited) < SKELETON_N:
            return
        for ch in self.ledger.unknown_channels():
            cs = self.ledger.channels[ch]
            if cs.skeleton_tested.all():
                cs.state = "empty"

    def _clear_source(self, ch, center):
        """清除干扰源: 密螺旋搜索.

        策略:
        1. 尝试给定中心点
        2. 尝试 LSQ 估计点
        3. 8m/16m/24m 螺旋搜索 (8/8/4 方向) 覆盖 50m 半径
        4. 尝试 MEC 质心
        5. 尝试 W 顶点 (最近 3 个)

        关键改进: 原 25/50/75m 搜索全部超出 20m 清除半径, 必然失败.
        新方案 8m 起步, 8 方向, 覆盖源在 LSQ 27m 范围内.
        """
        center = np.asarray(center, float)
        tried = set()

        def _try(p):
            p = np.asarray(p, float)
            key = (round(p[0] / 10), round(p[1] / 10))
            if key in tried:
                return False
            tried.add(key)
            if self.sim.clear(tuple(p), ch):
                self.ledger.update_cleared(ch)
                self._clear_fail_count.pop(ch, None)
                return True
            return False

        # Phase 1: 给定中心
        if _try(center):
            return True

        # Phase 2: LSQ 估计 + 密螺旋
        cs = self.ledger.channels[ch]
        est = self._lsq(cs.bearings) if cs.bearings else None
        if est is not None:
            est = np.asarray(est, float)
            if _try(est):
                return True
            for rad in [8, 16, 24, 32]:
                n_dirs = 8 if rad <= 16 else 4
                for i in range(n_dirs):
                    a = 2 * math.pi * i / n_dirs
                    q = est + rad * np.array([math.cos(a), math.sin(a)])
                    if _try(q):
                        return True

        # Phase 3: MEC 质心 + W 顶点
        m = cs.mec
        if m and m[1] is not None:
            if _try(m[1]):
                return True
        if cs.W is not None and len(cs.W) >= 3:
            cw = np.asarray(cs.W, float)
            cur = self.ledger.cur_pos
            order = np.argsort([np.linalg.norm(v - cur) for v in cw])
            for idx in order[:3]:
                if _try(cw[idx]):
                    return True

        self._clear_fail_count[ch] = self._clear_fail_count.get(ch, 0) + 1
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
            self.grid_cover()
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
        print(f"\n========== Q4 v1 统计 ==========")
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
