# -*- coding: utf-8 -*-
"""Q4 v3 主策略: **合并巡回** (证书 + 定位 + 清除) + 长轴切割 + 末段 TSP 清尾.

对症 v2 (policy2) 的残余浪费:
  - v2 先单独 hunt 一轮 (seed1: 6499m/1375s), 再走证书巡回; 两者路径重复.
  - v3 取消独立 hunt: 证书巡回途中用"顺路自由交会"(_refine) 收集方位,
    凡 MEC<=20m 且绕行代价小的频道就地清除, 其余统一交给末段 TSP 清尾.
  - 末段 TSP: 从当前点出发对剩余待清源做 2-opt 巡回, 而不是逐个螺旋重试.

巡回节点集合 = 证书骨架点 ∪ 已检测源的 MEC 中心 (动态重规划).
"""
import numpy as np
import math
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "q2"))
from geometry import diam_and_axis, mec as compute_mec

from ledger2 import (Ledger, SKELETON, SKELETON_N, ARENA_R, RECV_R_MIN, RECV_R_MAX,
                     CLEAR_R, NEAR_R, SPEED, SWITCH_COST, MEASURE_COST, N_CHANNELS,
                     poly_centroid, max_gap_all, G_SAMPLES)

REAL_TIME_LIMIT = 16.5 * 60

BET_T = 500.0
BET_H = 250.0
MAX_PURSUIT_STEPS = 6
CLEAR_TOL = CLEAR_R
CUT_RHO_MIN = 60.0

ENROUTE_MAX = 420.0       # 巡回途中顺路清除的最大单程绕行
REFINE_D_MAX = 1250.0     # 顺路交会的作用距离
REFINE_BUDGET = 2
REFINE_MIN_ANG = 45.0     # 新方位与已有方位的最小夹角 (度)
SRC_NODE_TOL = 60.0       # 源作为巡回节点的位置不确定度上限
SRC_COOLDOWN = 6          # 一个源节点服务后冷却的节点数

LIT_BONUS = 260.0         # 候选点落在"已点亮"半平面的距离奖励 (m)
LIT_COS = math.cos(math.radians(50.0))
BLIND_PEN = 600.0         # 候选点落在已知"黑侧"的惩罚 (m)
BLIND_COS = math.cos(math.radians(55.0))
MAX_BLIND = 6


def _unit(deg):
    t = math.radians(deg)
    return np.array([math.cos(t), math.sin(t)], dtype=float)


def _clamp(p, R=2.0e6):
    p = np.asarray(p, float)
    n = float(np.linalg.norm(p))
    if n > R:
        p = p * (R / n)
    return p


def _perp(e):
    return np.array([-e[1], e[0]])


def _tsp_open(pts, start, passes=5):
    """开放路径 TSP (起点固定 start), 返回 (长度, 访问次序索引)."""
    pts = np.asarray(pts, float)
    n = len(pts)
    if n == 0:
        return 0.0, []
    A = np.vstack([np.asarray(start, float), pts])
    D = np.linalg.norm(A[:, None, :] - A[None, :, :], axis=-1)
    N = n + 1
    un = list(range(1, N))
    order = [0]; cur = 0
    while un:
        nxt = min(un, key=lambda j: D[cur, j])
        order.append(nxt); un.remove(nxt); cur = nxt
    best = order[:]

    def L(o):
        o = np.asarray(o)
        return float(D[o[:-1], o[1:]].sum())

    bl = L(best)
    for _ in range(passes):
        imp = False
        for i in range(1, N - 1):
            for j in range(i + 1, N):
                no = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                l = L(no)
                if l < bl - 1e-9:
                    best, bl, imp = no, l, True
        if not imp:
            break
    return bl, [o - 1 for o in best[1:]]


class Policy3:
    def __init__(self, sim, verbose=False, refine_budget=REFINE_BUDGET):
        self.sim = sim
        self.ledger = Ledger()
        self.verbose = verbose
        self.refine_budget = refine_budget
        self._visited = []
        self._tried_pts = {}
        self._src_cool = {}
        self._blind = {}
        self.fin_log = []
        self.phase_stats = {}
        # 诊断
        self._last_pos = np.zeros(2)
        self.mv = 0.0
        self.n_meas = 0
        self.n_clear = 0
        self.n_fail = 0
        self.ch_meas = {}
        self._stats_acc = {}
        self._cur_phase = None
        self._ph = (0.0, 0.0, 0, 0)
        self._tour_len = 0.0
        self._order_len = 0.0

    # ---------------- 诊断 ----------------
    def _bump(self, pos):
        p = np.asarray(pos, float)
        self.mv += float(np.linalg.norm(p - self._last_pos))
        self._last_pos = p

    def _phase(self, name):
        if self._cur_phase is not None:
            self._stats_acc[self._cur_phase] = (
                self.sim.vt - self._ph[0], self.mv - self._ph[1],
                self.n_meas - self._ph[2], self.n_clear - self._ph[3])
        self._cur_phase = name
        self._ph = (self.sim.vt, self.mv, self.n_meas, self.n_clear)

    # ---------------- 基础 ----------------
    def _sync(self):
        self.ledger.cur_pos = np.array(self.sim.cur_pos, float)
        self.ledger.cur_channel = self.sim.cur_channel
        self.ledger.vt = self.sim.vt
        self.ledger.cleared = self.sim.cleared

    def _measure(self, pos, ch):
        pos = _clamp(pos)
        self.n_meas += 1
        self.ch_meas[ch] = self.ch_meas.get(ch, 0) + 1
        resp = self.sim.measure(tuple(pos), ch)
        self._bump(pos)
        mr = resp.get("measure_result")
        if mr == "direction":
            self.ledger.update_direction(ch, pos, resp["svd_deg"])
        elif mr == "near":
            self.ledger.update_near(ch, pos)
            if self.sim.clear(tuple(np.asarray(self.sim.cur_pos, float)), ch):
                self.ledger.update_cleared(ch)
        else:
            self.ledger.update_no_signal(ch, pos)
        self._visited.append(np.asarray(pos, float))
        self._sync()
        return mr

    def _clear_at(self, ch, pos):
        pos = _clamp(pos)
        self.n_clear += 1
        ok = self.sim.clear(tuple(pos), ch)
        self._bump(pos)
        if ok:
            self.ledger.update_cleared(ch)
        else:
            self.n_fail += 1
            self.ledger.update_clear_fail(ch)
        self._sync()
        return ok

    def _mec_center(self, ch):
        cs = self.ledger.channels[ch]
        m = cs.mec
        if m is None or m[1] is None:
            return None
        return np.asarray(m[1], float)

    def _try_clear(self, ch, force=False):
        """尝试清除 ch.

        默认: 仅当 MEC 半径 <= CLEAR_TOL 时尝试.
        force=True: 若 MEC 不收敛但已有方位 (centroid 已知), 按候选多边形几何
        中心兜底尝试一次. 失败无害 (5 s 虚拟时间); 成功可救回接近共线/单方位源.
        """
        cs = self.ledger.channels[ch]
        m = cs.mec
        if m is not None and m[1] is not None and m[0] <= CLEAR_TOL:
            return self._clear_at(ch, m[1])
        if force and cs.bearings:
            cen = cs.centroid
            if cen is not None:
                return self._clear_at(ch, np.asarray(cen, float))
        return False

    # ---------------- 追踪 (定向源: 从"已点亮"侧逼近) ----------------
    def _lit_dir(self, ch):
        """由已有 bearing 的测量点方向给出"已点亮"平均方向 (从 MEC 中心看出去)."""
        cs = self.ledger.channels[ch]
        m = cs.mec
        if m is None or m[1] is None:
            return None
        C = np.asarray(m[1], float)
        u = np.zeros(2)
        for S, _ in cs.bearings:
            v = np.asarray(S, float) - C
            n = float(np.linalg.norm(v))
            if n > 1.0:
                u += v / n
        n = float(np.linalg.norm(u))
        if n < 1e-6:
            return None
        return u / n

    def _cand_points(self, ch):
        """候选测量点池.

        (a) Q2 单方位推进点: S0 + T·e ± k·H·n  (k=1,2,3) —— 单 bearing 时最有效,
            且两个镜像点中通常恰有一个落在定向源的**点亮**半平面内.
        (b) **长条端采样** (关键修复): 两条近乎平行的示向度会把 MEC 拉成长条
            (半径可达 ~700m), 而源必落在长条内部. 旧版只在 C 周围 rho<=380m 布点,
            永远够不到长条两端 -> 反复 no_signal, 单源烧掉数千秒.
            现沿 MEC 主轴 ±t·R 与"最后一条示向度"方向 t·R 采样, 直达两端:
            两个镜像端中恰有一端落在定向源的点亮半平面内.
        (c) MEC 中心 C 周围的 12 向 + 已点亮方向, 半径自适应 (~0.9·R, 上界 420m).
        """
        cs = self.ledger.channels[ch]
        out = []
        m = cs.mec
        R = float(m[0]) if (m is not None and m[1] is not None) else 0.0
        if cs.bearings:
            S0, th0 = cs.bearings[-1]
            e = _unit(th0); n = _perp(e)
            S0 = np.asarray(S0, float)
            for mult in (1.0, 2.0, 3.0):
                for sgn in (1.0, -1.0):
                    out.append(S0 + BET_T * e + sgn * mult * BET_H * n)
        if m is not None and m[1] is not None:
            C = np.asarray(m[1], float)
            L = cs.diameter
            if L is None or L <= 1e-9:
                L = max(2.0 * R, 120.0)
            # (b) 长条端采样: 沿主轴两镜像端 + 沿最后示向度方向的远端
            ax = cs.axis
            if R > 40.0 and ax is not None:
                u = np.asarray(ax, float)
                nu = float(np.linalg.norm(u))
                if nu > 1e-9:
                    u = u / nu
                    for t in (0.5, 0.85, 1.2, 1.55):
                        for sgn in (1.0, -1.0):
                            out.append(C + sgn * t * R * u)
            if R > 40.0 and cs.bearings:
                e = _unit(cs.bearings[-1][1])
                for t in (0.6, 1.0, 1.4):
                    out.append(C + t * R * e)
            # (c) 各向候选环
            rho = max(90.0, min(0.9 * R + 30.0, 420.0))
            lu = self._lit_dir(ch)
            if lu is not None:
                out.append(C + rho * lu)
            for k in range(12):
                a = 2 * math.pi * k / 12.0
                out.append(C + rho * np.array([math.cos(a), math.sin(a)]))
        return out

    def _tried(self, ch):
        return self._tried_pts.setdefault(ch, [])

    def _mec_center_raw(self, ch):
        cs = self.ledger.channels[ch]
        m = cs.mec
        return None if (m is None or m[1] is None) else np.asarray(m[1], float)

    def _next_cand(self, ch):
        """选下一个未试过的候选点: 距离为主, 点亮方向轻微加权, 黑侧轻微惩罚.

        长条 MEC (R>150, 近共线双方位) 时, 额外给**沿主轴远端**的候选加分,
        使追踪优先直达长条两端 (源就在其中一端), 而不是在中间反复试探.
        """
        cur = np.asarray(self.sim.cur_pos, float)
        C = self._mec_center_raw(ch)
        cands = self._cand_points(ch)
        tried = self._tried(ch)
        u = self._lit_dir(ch)
        blind = self._blind.get(ch, [])
        cs = self.ledger.channels[ch]
        R = float(cs.mec[0]) if (cs.mec is not None and cs.mec[1] is not None) else 0.0
        axv = None
        if R > 150.0 and cs.axis is not None:
            a = np.asarray(cs.axis, float)
            na = float(np.linalg.norm(a))
            if na > 1e-9:
                axv = a / na
        best = None; bs = 1e18
        for c in cands:
            if not np.all(np.isfinite(c)):
                continue
            d0 = float(np.linalg.norm(c - cur))
            if d0 < 15.0:
                continue
            if any(np.linalg.norm(c - t) < 26.0 for t in tried):
                continue
            if self._is_visited(c, tol=18.0):
                continue
            sc = d0
            if C is not None:
                v = c - C
                nv = float(np.linalg.norm(v))
                if nv > 1e-6:
                    v = v / nv
                    if u is not None and float(v @ u) > LIT_COS:
                        sc -= LIT_BONUS
                    for b in blind:
                        if float(v @ b) > BLIND_COS:
                            sc += BLIND_PEN
                            break
            if axv is not None:
                sc -= 0.45 * abs(float((c - C) @ axv))   # 长条端优先
            if sc < bs:
                bs = sc; best = c
        return best

    def pursue(self, ch, max_steps=MAX_PURSUIT_STEPS):
        """有界追踪: 每步选一个未试过的候选点; no_signal 则记黑侧方向并换点."""
        meas = 0
        while meas < max_steps:
            if ch in self.sim.cleared:
                return True
            cs = self.ledger.channels[ch]
            if not cs.is_active:
                return True
            if self._try_clear(ch):
                return True
            if not cs.bearings:
                return False
            C0 = self._mec_center_raw(ch)
            cand = self._next_cand(ch)
            if cand is None:
                break
            cand = _clamp(cand)
            self._tried(ch).append(np.asarray(cand, float).copy())
            mr = self._measure(cand, ch)
            if mr == "no_signal" and C0 is not None:
                v = np.asarray(cand, float) - C0
                nv = float(np.linalg.norm(v))
                if nv > 1e-6:
                    bl = self._blind.setdefault(ch, [])
                    bl.append(v / nv)
                    if len(bl) > MAX_BLIND:
                        bl.pop(0)
            meas += 1
        return self._try_clear(ch)

    def _is_visited(self, p, tol=25.0):
        p = np.asarray(p, float)
        for v in self._visited[-24:]:
            if np.linalg.norm(p - v) < tol:
                return True
        return False

    # ---------------- 顺路交会 ----------------
    def _refine_at(self, pos, budget=None):
        """在停点对已检出未清除频道补测; **按不确定度 (MEC 半径) 从大到小**优先.

        v2 用"方位夹角多样性"筛选, 但方位夹角依赖非常粗糙的楔形重心估计,
        会把真正困难 (单方位、楔形 MEC~750m) 的频道长期排除在外. 改为不确定性优先.
        """
        budget = self.refine_budget if budget is None else budget
        pos = np.asarray(pos, float)
        cand = []
        for ch in self.ledger.detected_channels():
            if ch in self.sim.cleared:
                continue
            cs = self.ledger.channels[ch]
            cen = cs.centroid
            if cen is None:
                continue
            d = float(np.linalg.norm(np.asarray(cen, float) - pos))
            m = cs.mec
            if m is not None and m[0] <= CLEAR_TOL:
                if d <= ENROUTE_MAX:
                    self._try_clear(ch)
                continue
            if d > REFINE_D_MAX:
                continue
            unc = float(m[0]) if m is not None else 5000.0
            cand.append((-unc, d, ch))
        cand.sort()
        for _, d, ch in cand[:budget]:
            if ch in self.sim.cleared:
                continue
            self._measure(pos, ch)
            self._try_clear(ch)

    # ---------------- 阶段 ----------------
    def initial_scan(self):
        for ch in range(1, N_CHANNELS + 1):
            self._measure((0.0, 0.0), ch)
        self._visited.append(np.zeros(2))

    def _nodes(self):
        out = [("s", i) for i in sorted(self._pend)]
        for ch in self.ledger.detected_channels():
            if ch in self.sim.cleared:
                continue
            if self._src_cool.get(ch, 0) > 0:
                continue
            cs = self.ledger.channels[ch]
            m = cs.mec
            if m is not None and m[1] is not None and m[0] <= SRC_NODE_TOL:
                out.append(("c", ch))
        return out

    def _tick_cool(self):
        for k in list(self._src_cool):
            self._src_cool[k] -= 1
            if self._src_cool[k] <= 0:
                self._src_cool.pop(k, None)

    def _sig(self, nodes):
        s = []
        for t, v in nodes:
            if t == "s":
                s.append(("s", v))
            else:
                c = self._mec_center(v)
                s.append(("c", v, round(float(c[0]), 0), round(float(c[1]), 0)))
        return tuple(s)

    def merged_walk(self, max_passes=3):
        """证书巡回 + 途中清源; 节点动态重规划."""
        for _pass in range(max_passes):
            self._pend = set(range(SKELETON_N))
            self._order = []
            self._osig = None
            guard = 0
            while True:
                guard += 1
                if guard > 400:
                    break
                if self.sim.elapsed_real() > REAL_TIME_LIMIT:
                    return
                if self.ledger.resolved():
                    return
                nodes = self._nodes()
                if not nodes:
                    break
                sig = self._sig(nodes)
                if sig != self._osig:
                    pts = [self._node_pos(t, v) for t, v in nodes]
                    tl, order = _tsp_open(pts, self.sim.cur_pos)
                    self._order = [nodes[k] for k in order]
                    self._osig = sig
                    self._order_len = max(self._order_len, tl)
                if not self._order:
                    break
                nd = self._order.pop(0)
                self._tick_cool()
                if nd[0] == "s":
                    self._serve_skeleton(nd[1])
                else:
                    self._serve_source(nd[1])
                self._osig = None
            # 若仍有未知频道, 再来一遍 (证书未达成)
            if not self.ledger.unknown_channels():
                return

    def _node_pos(self, t, v):
        if t == "s":
            return SKELETON[v]
        c = self._mec_center(v)
        return np.zeros(2) if c is None else c

    def _serve_skeleton(self, idx):
        target = np.asarray(SKELETON[idx], float)
        self._pend.discard(idx)
        unknown = self.ledger.unknown_channels()
        for ch in sorted(unknown, key=lambda c: abs(c - self.ledger.cur_channel)):
            if self.sim.elapsed_real() > REAL_TIME_LIMIT:
                return
            cs = self.ledger.channels[ch]
            if not cs.is_active:
                continue
            if self.ledger.check_empty(ch):
                cs.state = "empty"
                continue
            self._measure(target, ch)
        self.ledger.mark_empty()
        self._refine_at(target)

    def _serve_source(self, ch):
        if ch in self.sim.cleared:
            return
        if not self.ledger.channels[ch].is_active:
            return
        c = self._mec_center(ch)
        if c is None:
            return
        # 走到 MEC 中心并就地测量 (既清除又补探针)
        self._measure(c, ch)
        if not self._try_clear(ch):
            # 兜底: MEC 未收敛时按几何中心强清一次 (救回接近共线/单方位源)
            self._try_clear(ch, force=True)
            # 未清除 -> 顺路再追一小步, 然后冷却, 让 TSP 先走别处
            self.pursue(ch, max_steps=3)
            if ch not in self.sim.cleared:
                self._src_cool[ch] = SRC_COOLDOWN

    def finalize(self, rounds=3):
        """末段: 对剩余待清源做 TSP 巡回清除; 追踪预算逐步放大."""
        for r in range(rounds):
            if self.sim.elapsed_real() > REAL_TIME_LIMIT:
                return
            pend = []
            for ch in self.ledger.detected_channels():
                if ch in self.sim.cleared:
                    continue
                c = self._mec_center(ch)
                if c is not None:
                    pend.append((ch, c))
            if not pend:
                return
            self.fin_log.append(("round", r, "n_pend", len(pend),
                                 "mecs", [round(float(self.ledger.channels[ch].mec[0]), 1)
                                          for ch, _ in pend],
                                 "pos", [round(float(v), 0) for v in self.sim.cur_pos]))
            pts = [c for _, c in pend]
            tl, order = _tsp_open(pts, self.sim.cur_pos)
            self.fin_log.append(("tsp", r, round(tl, 0)))
            progressed = False
            for k in order:
                if self.sim.elapsed_real() > REAL_TIME_LIMIT:
                    return
                ch, c = pend[k]
                if ch in self.sim.cleared:
                    continue
                cs = self.ledger.channels[ch]
                if not cs.is_active:
                    continue
                b0 = len(cs.bearings)
                mv0 = self.mv; n0 = self.n_meas
                self._measure(c, ch)
                if self._try_clear(ch):
                    self.fin_log.append(("clear", ch, "mv", round(self.mv - mv0),
                                         "meas", self.n_meas - n0))
                    progressed = True
                    continue
                # finalize 是最后机会, MEC 未收敛也按 centroid 兜底强清一次
                if self._try_clear(ch, force=True):
                    self.fin_log.append(("clear_fb", ch, "mv", round(self.mv - mv0),
                                         "meas", self.n_meas - n0))
                    progressed = True
                    continue
                self.pursue(ch, max_steps=5 + 3 * r)
                self.fin_log.append(("pursue", ch, "mv", round(self.mv - mv0),
                                     "meas", self.n_meas - n0,
                                     "mec", round(float(cs.mec[0]), 1) if cs.mec else -1,
                                     "bear", len(cs.bearings)))
                if ch in self.sim.cleared or len(self.ledger.channels[ch].bearings) > b0:
                    progressed = True
                elif self._mec_center(ch) is not None:
                    progressed = True
            if not progressed:
                return

    # ---------------- 主循环 ----------------
    def run(self):
        sim = self.sim
        try:
            self._phase("initial")
            self.initial_scan()
            self._phase("merged")
            self.merged_walk()
            self._phase("final")
            self.finalize()
            self._phase("done")
            self.phase_stats = dict(self._stats_acc)
        except Exception as e:
            print(f"[WARN] Policy3 异常: {e}")
            import traceback; traceback.print_exc()
        self._sync()
        n = len(sim.cleared)
        vt = sim.vt
        if self.verbose:
            print(f"[Policy3] cleared={n} vt={vt:.0f} order={self._order_len:.0f} "
                  f"phases={self.phase_stats}")
        return n, vt


# 兼容旧命名: 正式交付统一使用 Policy
Policy = Policy3
