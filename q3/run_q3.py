# -*- coding: utf-8 -*-
"""问题3：全向干扰源搜索、交会定位与清除。

策略：
1. 原点扫 1–20 频道（不移动，约 119 s 虚拟时间）；
2. 已听频道：能交会且 MEC≤20 m 的先就近清除；其余按估计位置追击。
   第二点取 (t,h)=(500,±250) 中离狗更近的一侧；若当前位置已有足够侧向间隔则就地再测；
   MEC>20 m 再按长轴切。每个停点顺便测仍只有一站、且相对第一站接近 (500,250) 的已听频道；
   交会已进 20 m 且顺路的源就地收割，避免专程再跑；
3. 半径 1150 m 上 6 个监听点按最短巡线补网（6! 枚举，含从当前位置到第一环点）；
   环上新听到的源若去第二点 (500,±250) 再进圈的额外路程 > 300 m 则挂起，整圈走完再就近追。
   原点扫完后同样按该额外路程判挂起（对照预规划环的第一点）；墙钟不够则跳出剩余环点并立即 drain。
   `--no-hang` 关闭环上挂起；`--no-origin-hang` 关闭原点挂起；二者独立。
   追击第二点纵向距离默认 500 m；若该频道在别处无信号，则沿射线剔除 |G−M|≤1005 m 的 ρ，
   把赌注夹到允许区间（`--no-miss-constraint` 关闭）。
   `--replan` 打开滚动重规划：任务池 = 未访环点 + 非挂起已听（入口 closer_offset）+ 可清除（顺路 q）；
   每次只执行开路巡线的第一项再重解。默认仍是原点追完再走环（`--no-replan`）。
   `--no-onroute-clear` 关闭顺路清点（默认在 C_MEC 朝下一节点平移）。
   `--adaptive-step` 在已有 MEC 中心时把第二点纵向赌注改成 0.85|C−S0|（默认关）。

用法：
  python q3/run_q3.py
  python q3/run_q3.py --watch
  python q3/run_q3.py --local [--episodes N]
  python q3/run_q3.py --local --replan
  python q3/run_q3.py --local --no-origin-hang --no-miss-constraint
  python q3/run_q3.py --no-hang --no-origin-hang
  python tools/practice_loop.py --problem 3 --n 10
环境变量 ROBOT_ID 可覆盖默认参赛队号。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "q1"))
sys.path.insert(0, str(ROOT / "q2"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from api import Simulator, ApiError  # noqa: E402
from geometry import diam_and_axis, mec  # noqa: E402
from solver import locate  # noqa: E402
from tactics import closer_offset, estimate_pos, lateral_offset, ray_coords  # noqa: E402

ROBOT_ID = os.environ.get("ROBOT_ID", "YOUR_TEAM_ID")
ARENA_R = 1800.0
RING_R = 1150.0
N_RING = 6
BET_T = 500.0
BET_H = 250.0
CLEAR_R = 20.0
MAX_HUNT_MEAS = 7
HANG_M = 300.0
CLEAR_EST = 480.0
REPLAN_DEFAULT = False
ONROUTE_CLEAR_DEFAULT = True

LOG_DIR = Path(__file__).resolve().parent / "logs"


def unit(deg: float) -> np.ndarray:
    t = math.radians(deg)
    return np.array([math.cos(t), math.sin(t)], dtype=float)


def clamp_xy(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = float(np.linalg.norm(p))
    if n > ARENA_R - 10.0 and n > 1e-9:
        p = p * ((ARENA_R - 10.0) / n)
    return p


def ring_points(radius: float = RING_R, n: int = N_RING) -> list:
    return [
        radius * np.array([math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n)])
        for k in range(n)
    ]


def covering_worst_gap(radius: float = RING_R, n: int = N_RING, arena: float = ARENA_R) -> float:
    """圆盘内到最近监听点（原点+环）的最大距离，应 < 1000。"""
    pts = [np.zeros(2)] + ring_points(radius, n)
    worst = 0.0
    for r in np.linspace(0.0, arena, 73):
        n_ang = 240 if r > 1 else 1
        for k in range(n_ang):
            a = 2 * math.pi * k / n_ang
            x = r * np.array([math.cos(a), math.sin(a)])
            d = min(float(np.linalg.norm(x - p)) for p in pts)
            if d > worst:
                worst = d
    return worst


def ring_tour_bruteforce(start, pts):
    """6! 枚举环点顺序，含从当前位置到第一环点的路程，取最短。"""
    start = np.asarray(start, float)
    pts = [np.asarray(p, float) for p in pts]
    n = len(pts)
    best_perm = None
    best_L = float("inf")
    for perm in permutations(range(n)):
        L = 0.0
        cur = start
        for i in perm:
            L += float(np.linalg.norm(pts[i] - cur))
            cur = pts[i]
        if L < best_L:
            best_L = L
            best_perm = perm
    return [(int(i), pts[i]) for i in best_perm], float(best_L)


def two_opt_tour(pts, start):
    """开路 2-opt：从 start 走完 pts（不回到起点）。"""
    left = [np.asarray(p, float) for p in pts]
    cur = np.asarray(start, float)
    order = []
    while left:
        k = min(range(len(left)), key=lambda i: float(np.linalg.norm(left[i] - cur)))
        order.append(left.pop(k))
        cur = order[-1]

    def length(seq):
        s, c = 0.0, np.asarray(start, float)
        for p in seq:
            s += float(np.linalg.norm(p - c))
            c = p
        return s

    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            for j in range(i + 1, len(order)):
                new = order[:i] + order[i : j + 1][::-1] + order[j + 1 :]
                if length(new) < length(order) - 1e-6:
                    order, improved = new, True
                    break
            if improved:
                    break
    return order, length(order)


class TaskNode:
    __slots__ = ("kind", "key", "xy", "C", "r")

    def __init__(self, kind, key, xy, C=None, r=None):
        self.kind = kind  # "ring" | "chase" | "clear"
        self.key = key
        self.xy = np.asarray(xy, float)
        self.C = None if C is None else np.asarray(C, float)
        self.r = r

    def label(self) -> str:
        if self.kind == "ring":
            return f"ring:{self.key}"
        return f"{self.kind}:ch{int(self.key):02d}"


def _open_tour_length(start, seq) -> float:
    s, c = 0.0, np.asarray(start, float)
    for n in seq:
        s += float(np.linalg.norm(n.xy - c))
        c = n.xy
    return s


def solve_open_tour(start, nodes, max_rounds: int = 5):
    """开路巡线：最近邻起步，再 2-opt 与 relocate，起点固定、不回到起点。"""
    nodes = list(nodes)
    if not nodes:
        return []
    start = np.asarray(start, float)
    left = list(nodes)
    order = []
    cur = start
    while left:
        k = min(range(len(left)), key=lambda i: float(np.linalg.norm(left[i].xy - cur)))
        order.append(left.pop(k))
        cur = order[-1].xy
    if len(order) <= 2:
        return order

    def L(seq):
        return _open_tour_length(start, seq)

    for _ in range(max_rounds):
        improved = False
        while True:
            found = False
            for i in range(len(order) - 1):
                for j in range(i + 1, len(order)):
                    new = order[:i] + order[i : j + 1][::-1] + order[j + 1 :]
                    if L(new) < L(order) - 1e-6:
                        order = new
                        found = True
                        improved = True
                        break
                if found:
                    break
            if not found:
                break
        while True:
            found = False
            n = len(order)
            for i in range(n):
                node = order[i]
                tmp = order[:i] + order[i + 1 :]
                for ins in range(len(tmp) + 1):
                    if ins == i:
                        continue
                    new = tmp[:ins] + [node] + tmp[ins:]
                    if L(new) < L(order) - 1e-6:
                        order = new
                        found = True
                        improved = True
                        break
                if found:
                    break
            if not found:
                break
        if not improved:
            break
    return order


def path_from_log(log_rows):
    pts = []
    for r in log_rows:
        pl = r.get("payload") or {}
        pos = pl.get("position")
        if not pos:
            continue
        p = np.array([float(pos["x"]), float(pos["y"])], dtype=float)
        if pts and float(np.linalg.norm(p - pts[-1])) < 1e-6:
            continue
        pts.append(p)
    return pts


def unique_stops(pts, tol: float = 1.0):
    uniq = []
    for p in pts:
        p = np.asarray(p, float)
        if not uniq or min(float(np.linalg.norm(p - q)) for q in uniq) > tol:
            uniq.append(p)
    return uniq


def actual_path_length(pts):
    if len(pts) < 2:
        return 0.0
    return float(sum(np.linalg.norm(pts[i] - pts[i - 1]) for i in range(1, len(pts))))


def second_point(S: np.ndarray, theta_deg: float, t: float = BET_T, h: float = BET_H) -> np.ndarray:
    e = unit(theta_deg)
    n = np.array([-e[1], e[0]])
    return clamp_xy(np.asarray(S, float) + t * e + h * n)


def region_center(vertices: np.ndarray):
    v = np.asarray(vertices, dtype=float)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    if len(v) == 0:
        return float("inf"), np.zeros(2)
    if len(v) == 1:
        return 0.0, v[0].copy()
    if len(v) == 2:
        C = (v[0] + v[1]) / 2.0
        r = 0.5 * float(np.linalg.norm(v[0] - v[1]))
        return r, C
    r, C = mec(v)
    if C is None or not np.isfinite(r):
        C = v.mean(axis=0)
        r = float("inf")
    return float(r), np.asarray(C, float)


def long_axis_point(vertices: np.ndarray, robot: np.ndarray) -> np.ndarray:
    v = np.asarray(vertices, dtype=float)
    r, C = region_center(v)
    if len(v) < 2:
        return clamp_xy(C + np.array([80.0, 0.0]))
    L, ax = diam_and_axis(v)
    if ax is None or L <= 1e-9:
        ax = np.array([1.0, 0.0])
        L = 40.0
    rho = max(50.0, 0.5 * float(L))
    nhat = np.array([-ax[1], ax[0]], dtype=float)
    if float(nhat @ (np.asarray(robot, float) - C)) < 0:
        nhat = -nhat
    return clamp_xy(C + rho * nhat)


class ChannelBook:
    def __init__(self):
        self.cleared = set()
        self.empty = set()
        self.failed = set()
        self.hits = {ch: [] for ch in range(1, 21)}  # (xy, svd_deg)
        self.miss_pts = {ch: [] for ch in range(1, 21)}
        self.miss_ids = {ch: set() for ch in range(1, 21)}
        self.clear_time = {}

    def remaining(self) -> list:
        done = self.cleared | self.empty | self.failed
        return [ch for ch in range(1, 21) if ch not in done]

    def heard(self) -> list:
        return [ch for ch in self.remaining() if self.hits[ch]]


class Dog:
    def __init__(
        self,
        sim: Simulator,
        remain_s: float = 1200.0,
        bet_t: float = BET_T,
        bet_h: float = BET_H,
        reuse_stops: bool = True,
        hang: bool = True,
        origin_hang: bool = True,
        miss_constraint: bool = True,
        replan: bool = REPLAN_DEFAULT,
        onroute_clear: bool = True,
        adaptive_step: bool = False,
    ):
        self.sim = sim
        self.xy = np.zeros(2)
        self._heading = np.array([1.0, 0.0], dtype=float)
        self._next_xy = None
        self.last_ch = 1
        self.book = ChannelBook()
        self.n_clear_ok = 0
        self.remain_s = remain_s
        self.wall0 = time.time()
        self.bet_t = float(bet_t)
        self.bet_h = float(bet_h)
        self.reuse_stops = bool(reuse_stops)
        self.hang = bool(hang)
        self.origin_hang = bool(origin_hang)
        self.miss_constraint = bool(miss_constraint)
        self.replan = bool(replan)
        self.onroute_clear = bool(onroute_clear)
        self.adaptive_step = bool(adaptive_step)
        self.hung = set()
        self.n_replan_decisions = 0
        self.n_replan_diverge = 0

    def wall_low(self, need: float = 50.0) -> bool:
        return (time.time() - self.wall0) > (self.remain_s - need)

    def _note_move(self, p) -> np.ndarray:
        p = clamp_xy(np.asarray(p, float))
        d = p - self.xy
        n = float(np.linalg.norm(d))
        if n > 1e-6:
            self._heading = d / n
        return p

    def measure_at(self, p, ch: int, cover_id: str | None = None) -> dict:
        p = self._note_move(p)
        body = self.sim.measure(float(p[0]), float(p[1]), ch)
        self.xy = p
        self.last_ch = int(ch)
        kind = body.get("measure_result")
        print(
            f"  meas ch{ch:02d} ({p[0]:.1f},{p[1]:.1f}) -> {kind}"
            + (f"  svd={body.get('svd_deg')}" if kind == "direction" else "")
            + f"  t={self.sim.virtual_time_s:.1f}s"
        )
        if kind == "direction":
            self.book.hits[ch].append((p.copy(), float(body["svd_deg"])))
        elif kind == "no_signal":
            self.book.miss_pts[ch].append(p.copy())
            if cover_id:
                self.book.miss_ids[ch].add(cover_id)
        return body

    def clear_at(self, p, ch: int) -> bool:
        p = self._note_move(p)
        body = self.sim.clear(float(p[0]), float(p[1]), ch)
        self.xy = p
        self.last_ch = int(ch)
        ok = body.get("clear_result") == "success"
        print(
            f"  CLEAR ch{ch:02d} ({p[0]:.1f},{p[1]:.1f}) -> {body.get('clear_result')}"
            f"  t={self.sim.virtual_time_s:.1f}s"
        )
        if ok:
            self.book.cleared.add(ch)
            self.book.failed.discard(ch)
            self.book.clear_time[ch] = self.sim.virtual_time_s
            self.n_clear_ok += 1
        return ok

    def try_localize(self, ch: int):
        hits = self.book.hits[ch]
        if len(hits) < 2:
            return None
        sensors = [h[0] for h in hits]
        thetas = [h[1] for h in hits]
        return locate(sensors, thetas, 1.0)

    def ready_mec(self, ch: int):
        loc = self.try_localize(ch)
        if loc is None or loc.status != "OK" or loc.vertices is None:
            return None
        r, C = region_center(loc.vertices)
        if r <= CLEAR_R + 1e-6:
            return float(r), np.asarray(C, float)
        return None

    def ready_center(self, ch: int):
        info = self.ready_mec(ch)
        return None if info is None else info[1]

    def choose_clear_point(self, C, r, next_xy=None):
        """顺路清点：在 |q−C|+r ≤ 19.5 内朝下一节点平移，近距就地清不走这里。"""
        C = np.asarray(C, float)
        r = float(r)
        if (not self.onroute_clear) or r > CLEAR_R + 1e-6:
            return C
        delta = max(0.0, 19.5 - r)
        cur = np.asarray(self.xy, float)
        nxt = None if next_xy is None else np.asarray(next_xy, float)
        u = None
        if nxt is not None:
            v = nxt - C
            n = float(np.linalg.norm(v))
            if n > 1e-9:
                u = v / n
        if u is None:
            h = np.asarray(self._heading, float)
            hn = float(np.linalg.norm(h))
            if hn > 1e-9:
                u = h / hn
            else:
                v = C - cur
                n = float(np.linalg.norm(v))
                u = v / n if n > 1e-9 else np.array([1.0, 0.0])
        cands = [C, C + 0.5 * delta * u, C + delta * u]

        def feasible(q):
            return float(np.linalg.norm(np.asarray(q, float) - C)) + r <= 19.5 + 1e-9

        def cost(q):
            q = np.asarray(q, float)
            if nxt is not None:
                return float(np.linalg.norm(q - cur)) + float(np.linalg.norm(q - nxt))
            return float(np.linalg.norm(q - cur))

        feas = [q for q in cands if feasible(q)]
        if not feas:
            return C
        return min(feas, key=cost)

    def piggyback(self, p, exclude_ch: int | None = None, max_n: int = 2) -> None:
        """停点顺便测：只测仍为一站、且相对第一站接近时间赌注 (500,250) 的已听频道。"""
        if self.wall_low(55.0):
            return
        p = np.asarray(p, float)
        scored = []
        for ch in self.book.heard():
            if ch == exclude_ch:
                continue
            hits = self.book.hits[ch]
            if len(hits) >= 2:
                continue
            S, th = hits[-1]
            if float(np.linalg.norm(p - S)) < 60.0:
                continue
            t_along, hlat = ray_coords(S, th, p)
            if t_along < 200.0 or t_along > 1000.0:
                continue
            if hlat < 80.0 or hlat > 600.0:
                continue
            geom = abs(t_along - self.bet_t) + abs(hlat - self.bet_h)
            scored.append((geom, abs(ch - self.last_ch), ch))
        scored.sort()
        for _, _, ch in scored[:max_n]:
            if self.wall_low(45.0):
                break
            body = self.measure_at(p, ch)
            if body.get("measure_result") == "near":
                self.clear_at(p, ch)

    def harvest_ready(self, max_dist: float = 80.0, exclude_ch: int | None = None) -> None:
        """已经交会进 20 m 的，若离当前位置不远就顺路清（不打断当前追击频道）。"""
        ready = []
        for ch in list(self.book.heard()):
            if ch == exclude_ch:
                continue
            info = self.ready_mec(ch)
            if info is None:
                continue
            r, C = info
            q = self.choose_clear_point(C, r, self._next_xy)
            ready.append((float(np.linalg.norm(q - self.xy)), ch, q))
        ready.sort()
        for d, ch, q in ready:
            if ch in self.book.cleared:
                continue
            if d > max_dist:
                continue
            self.clear_at(q, ch)

    def after_stop(self, exclude_ch: int | None = None) -> None:
        if not self.reuse_stops:
            return
        self.piggyback(self.xy, exclude_ch=exclude_ch)
        self.harvest_ready(max_dist=80.0, exclude_ch=exclude_ch)

    def miss_bet_window(self, ch: int, S0, th0):
        """无信号点给出 |G−M|>1000 m。沿末听射线 1 m 扫描 ρ∈[5,1500]，
        与 [200,1400] 相交得到 [ρ_lo, ρ_hi]；空集则 None（调用方回退 BET_T）。"""
        if not self.miss_constraint:
            return None
        misses = self.book.miss_pts.get(ch) or []
        if not misses:
            return None
        S0 = np.asarray(S0, float)
        u = unit(th0)
        rho = np.arange(5.0, 1500.0 + 0.5, 1.0)
        pts = S0[None, :] + rho[:, None] * u[None, :]
        r2 = 1005.0 ** 2
        allowed = np.ones(rho.shape, dtype=bool)
        for M in misses:
            M = np.asarray(M, float)
            d2 = np.sum((pts - M) ** 2, axis=1)
            allowed &= d2 > r2
        mask = allowed & (rho >= 200.0) & (rho <= 1400.0)
        if not np.any(mask):
            return None
        surv = rho[mask]
        return float(surv.min()), float(surv.max())

    def hunt(self, ch: int) -> bool:
        print(f"-- hunt channel {ch}")
        if ch in self.book.cleared:
            return True
        logged_miss = False
        for _step in range(MAX_HUNT_MEAS):
            if ch in self.book.cleared:
                return True
            loc = self.try_localize(ch)
            if loc is not None and loc.status == "OK" and loc.vertices is not None:
                r, C = region_center(loc.vertices)
                print(f"   localize OK m={len(loc.vertices)} MEC={r:.2f}m  D={loc.diameter:.2f}m")
                if r <= CLEAR_R + 1e-6:
                    q = self.choose_clear_point(C, r, self._next_xy)
                    if self.clear_at(q, ch):
                        self.after_stop(exclude_ch=ch)
                        return True
                    if ch in self.book.cleared:
                        return True
                    self.book.failed.add(ch)
                    return False
                nxt = long_axis_point(loc.vertices, self.xy)
                body = self.measure_at(nxt, ch)
                self.after_stop(exclude_ch=ch)
                if ch in self.book.cleared:
                    return True
                if body.get("measure_result") == "near":
                    if self.clear_at(self.xy, ch):
                        self.after_stop(exclude_ch=ch)
                        return True
                continue

            hits = self.book.hits[ch]
            if not hits:
                return False
            if loc is not None and loc.status == "EMPTY" and len(hits) >= 2:
                self.book.hits[ch] = hits[-1:]
                hits = self.book.hits[ch]

            S0, th0 = hits[-1]
            if _step == 0 and float(np.linalg.norm(self.xy - S0)) > 300.0:
                if lateral_offset(S0, th0, self.xy) > 70.0:
                    body = self.measure_at(self.xy, ch)
                    self.after_stop(exclude_ch=ch)
                    if ch in self.book.cleared:
                        return True
                    kind = body.get("measure_result")
                    if kind == "near":
                        if self.clear_at(self.xy, ch):
                            self.after_stop(exclude_ch=ch)
                            return True
                    if kind == "direction":
                        continue
            t_base = self.bet_t
            if self.adaptive_step:
                loc_ad = self.try_localize(ch)
                if loc_ad is not None and loc_ad.status == "OK" and loc_ad.vertices is not None:
                    _, C_ad = region_center(loc_ad.vertices)
                    t_base = float(
                        np.clip(0.85 * float(np.linalg.norm(np.asarray(C_ad) - S0)), 20.0, 1100.0)
                    )
            t_raw = t_base + 120.0 * (_step // 2)
            win = self.miss_bet_window(ch, S0, th0)
            t = t_raw
            if win is not None:
                rho_lo, rho_hi = win
                t = float(np.clip(t_raw, rho_lo, rho_hi))
                if not logged_miss and abs(t - self.bet_t) > 1e-9:
                    print(
                        f"   miss-constraint ch={ch} t_eff={t:.0f} lo={rho_lo:.0f} hi={rho_hi:.0f} "
                        f"n_miss={len(self.book.miss_pts[ch])}"
                    )
                    logged_miss = True
            nxt = closer_offset(S0, th0, self.xy, t, self.bet_h)
            if float(np.linalg.norm(nxt - self.xy)) < 20.0:
                t2 = t + 200.0
                if win is not None:
                    t2 = float(np.clip(t2, rho_lo, rho_hi))
                nxt = closer_offset(S0, th0, self.xy, t2, self.bet_h)
            body = self.measure_at(nxt, ch)
            self.after_stop(exclude_ch=ch)
            if ch in self.book.cleared:
                return True
            kind = body.get("measure_result")
            if kind == "near":
                if self.clear_at(self.xy, ch):
                    self.after_stop(exclude_ch=ch)
                    return True
            if kind == "no_signal":
                nxt2 = second_point(S0, th0, t=t + 220.0, h=-self.bet_h)
                body2 = self.measure_at(nxt2, ch)
                self.after_stop(exclude_ch=ch)
                if ch in self.book.cleared:
                    return True
                if body2.get("measure_result") == "near":
                    if self.clear_at(self.xy, ch):
                        self.after_stop(exclude_ch=ch)
                        return True
                continue

        if ch in self.book.cleared:
            return True
        loc = self.try_localize(ch)
        if loc is not None and loc.status == "OK" and loc.vertices is not None:
            r, C = region_center(loc.vertices)
            if r <= CLEAR_R + 1e-6:
                q = self.choose_clear_point(C, r, self._next_xy)
                if self.clear_at(q, ch):
                    self.after_stop(exclude_ch=ch)
                    return True
        print(f"   hunt failed ch{ch}")
        if ch not in self.book.cleared:
            self.book.failed.add(ch)
        return False

    def listen_point(self, p, channels: list, cover_id: str | None = None) -> list:
        """只在 p 扫描，不追击。near 就地清除。cover_id 用于覆盖网判空。"""
        heard_now = []
        p = np.asarray(p, float)
        ordered = sorted(list(channels), key=lambda c: abs(int(c) - self.last_ch))
        for ch in ordered:
            if ch not in self.book.remaining():
                continue
            if self.book.hits[ch]:
                continue
            body = self.measure_at(p, ch, cover_id=cover_id)
            kind = body.get("measure_result")
            if kind == "near":
                self.clear_at(p, ch)
            elif kind == "direction":
                heard_now.append(ch)
        if self.reuse_stops:
            self.piggyback(p, max_n=3)
            self.harvest_ready(max_dist=160.0)
        return heard_now

    def hunt_key(self, ch: int):
        loc = self.try_localize(ch)
        if loc is not None and loc.status == "OK" and loc.vertices is not None:
            r, C = region_center(loc.vertices)
            tier = 0 if r <= CLEAR_R + 1e-6 else 1
            return (tier, float(np.linalg.norm(C - self.xy)))
        return (2, float(np.linalg.norm(estimate_pos(self.book.hits[ch]) - self.xy)))

    def chase_extra(self, ch: int, a_next) -> float:
        """P→S2(500,±250)→C(480 m along bearing)→A_next 相对直行 P→A_next 的额外路程。"""
        if a_next is None or not self.book.hits[ch]:
            return 0.0
        S, th = self.book.hits[ch][-1]
        P = np.asarray(self.xy, float)
        S2 = closer_offset(S, th, P, self.bet_t, self.bet_h)
        C = estimate_pos(self.book.hits[ch], dist=CLEAR_EST)
        A = np.asarray(a_next, float)
        extra = (
            float(np.linalg.norm(S2 - P))
            + float(np.linalg.norm(C - S2))
            + float(np.linalg.norm(A - C))
            - float(np.linalg.norm(A - P))
        )
        return extra

    def _chase_t_eff(self, ch: int, S0, th0) -> float:
        t = self.bet_t
        if self.adaptive_step:
            loc = self.try_localize(ch)
            if loc is not None and loc.status == "OK" and loc.vertices is not None:
                _, C = region_center(loc.vertices)
                t = float(np.clip(0.85 * float(np.linalg.norm(np.asarray(C) - S0)), 20.0, 1100.0))
        win = self.miss_bet_window(ch, S0, th0)
        if win is not None:
            t = float(np.clip(t, win[0], win[1]))
        return t

    def _channel_entry_xy(self, ch: int, next_xy=None):
        info = self.ready_mec(ch)
        if info is not None:
            r, C = info
            return self.choose_clear_point(C, r, next_xy)
        S0, th0 = self.book.hits[ch][0]
        t = self._chase_t_eff(ch, S0, th0)
        return closer_offset(S0, th0, self.xy, t, self.bet_h)

    def _next_ring_xy(self, rings, visited):
        rem = [rings[k] for k in range(len(rings)) if k not in visited]
        if not rem:
            return None
        tour, _ = ring_tour_bruteforce(self.xy, rem)
        return tour[0][1] if tour else None

    def _task_pool(self, rings, visited):
        nodes = []
        for k, p in enumerate(rings):
            if k not in visited:
                nodes.append(TaskNode("ring", k, p))
        for ch in self.book.heard():
            if ch in self.hung:
                continue
            info = self.ready_mec(ch)
            if info is not None:
                r, C = info
                q = self.choose_clear_point(C, r, None)
                nodes.append(TaskNode("clear", ch, q, C=C, r=r))
            else:
                S0, th0 = self.book.hits[ch][0]
                t = self._chase_t_eff(ch, S0, th0)
                xy = closer_offset(S0, th0, self.xy, t, self.bet_h)
                nodes.append(TaskNode("chase", ch, xy))
        return nodes

    def _replan_dispatch(self, t0, remain):
        print("== replan dispatch ==")
        rings = ring_points()
        visited = set()
        while True:
            if len(self.book.cleared) >= 16:
                print("16 sources cleared, skip remaining ring")
                break
            if not self.book.remaining():
                break
            if time.time() - t0 > remain - 45:
                print("wall-clock budget low, abort remaining ring, drain pending")
                break
            pool = self._task_pool(rings, visited)
            if not pool:
                break
            tour = solve_open_tour(self.xy, pool)
            first = tour[0]
            nearest = min(pool, key=lambda n: float(np.linalg.norm(n.xy - self.xy)))
            self.n_replan_decisions += 1
            if first.kind != nearest.kind or first.key != nearest.key:
                self.n_replan_diverge += 1
                print(f"replan first={first.label()} nearest={nearest.label()}")
            next_xy = tour[1].xy if len(tour) > 1 else None
            self._next_xy = next_xy
            if first.kind == "ring":
                k = int(first.key)
                p = first.xy
                left = self.book.remaining()
                print(f"-- ring visit R{k} left={left} pool={len(pool)}")
                heard = self.listen_point(p, left, cover_id=f"R{k}")
                visited.add(k)
                self.mark_empty_by_cover()
                a_next = self._next_ring_xy(rings, visited)
                if self.hang:
                    for ch in heard:
                        extra = self.chase_extra(ch, a_next)
                        if a_next is not None and extra > HANG_M:
                            self.hung.add(ch)
                            print(f"   hang ch{ch} extra={extra:.0f}m > {HANG_M:.0f}m")
                        else:
                            self.hung.discard(ch)
            elif first.kind == "chase":
                ch = int(first.key)
                if ch not in self.book.heard() or ch in self.book.cleared:
                    continue
                self.hunt(ch)
                self.hung.discard(ch)
                self.harvest_ready(max_dist=200.0)
            elif first.kind == "clear":
                ch = int(first.key)
                if ch not in self.book.heard() or ch in self.book.cleared:
                    continue
                info = self.ready_mec(ch)
                if info is None:
                    self.hunt(ch)
                    self.hung.discard(ch)
                    continue
                r, C = info
                q = self.choose_clear_point(C, r, next_xy)
                if self.clear_at(q, ch):
                    self.after_stop(exclude_ch=ch)
                elif ch not in self.book.cleared:
                    self.book.failed.add(ch)
                self.hung.discard(ch)

    def hunt_heard(self, skip=None, include_hung: bool = False) -> None:
        skip = set() if skip is None else set(skip)
        if not include_hung:
            skip |= set(self.hung)
        tried = set()
        while True:
            hs = [c for c in self.book.heard() if c not in tried and c not in skip]
            if not hs:
                break
            ch = min(hs, key=self.hunt_key)
            others = [c for c in hs if c != ch]
            if others:
                ch2 = min(others, key=self.hunt_key)
                self._next_xy = self._channel_entry_xy(ch2)
            tried.add(ch)
            self.hunt(ch)
            self.hung.discard(ch)
            self.harvest_ready(max_dist=200.0)

    def mark_empty_by_cover(self):
        """七点覆盖网按监听点编号记账：全部无信号则判空。"""
        need = {"O"} | {f"R{k}" for k in range(N_RING)}
        for ch in list(self.book.remaining()):
            if self.book.hits[ch]:
                continue
            if need <= self.book.miss_ids[ch]:
                self.book.empty.add(ch)
                print(f"-- channel {ch} empty (covered, all no_signal)")

    def mark_empty_by_cap(self):
        """题面源数至多 16：清满后剩余频道必为空。"""
        if len(self.book.cleared) < 16:
            return
        for ch in list(self.book.remaining()):
            self.book.empty.add(ch)
            print(f"-- channel {ch} empty (at most 16 sources)")

    def run(self, do_enter: bool = True):
        t0 = time.time()
        if do_enter:
            enter = self.sim.enter()
            remain = float(enter.get("remaining_real_duration_s", 1200))
            self.remain_s = remain
            self.wall0 = t0
            print(f"entered remaining_real={remain}s  wall={time.strftime('%H:%M:%S')}")
        else:
            remain = float(self.remain_s)
            self.wall0 = t0
            print(f"resume remaining_real~{remain}s  wall={time.strftime('%H:%M:%S')}")
        gap = covering_worst_gap()
        print(f"cover net 1+6 R=1150 worst-gap={gap:.1f}m (need <1000)")
        print(
            f"flags hang(ring)={self.hang} origin_hang={self.origin_hang} "
            f"miss_constraint={self.miss_constraint} replan={self.replan} "
            f"onroute_clear={self.onroute_clear} adaptive_step={self.adaptive_step}"
        )

        # 1) 原点扫频后再追，保证 20 个频道的覆盖票都在原点
        print("== phase 1 origin scan ==")
        self.listen_point(np.zeros(2), list(range(1, 21)), cover_id="O")
        print(f"after origin listen: cleared={sorted(self.book.cleared)} heard={self.book.heard()}")
        if self.origin_hang:
            tour0, tour0_L = ring_tour_bruteforce(self.xy, ring_points())
            a0 = tour0[0][1] if tour0 else None
            print(
                f"origin hang: planned ring L={tour0_L:.1f}m "
                f"first=({a0[0]:.0f},{a0[1]:.0f}) thresh={HANG_M:.0f}m"
            )
            for ch in list(self.book.heard()):
                extra = self.chase_extra(ch, a0)
                if a0 is not None and extra > HANG_M:
                    self.hung.add(ch)
                    print(f"   origin-hang ch{ch} extra={extra:.0f}m > {HANG_M:.0f}m")

        if self.replan:
            self._replan_dispatch(t0, remain)
        else:
            tour0, _ = ring_tour_bruteforce(self.xy, ring_points())
            self._next_xy = tour0[0][1] if tour0 else None
            self.hunt_heard()
            print(
                f"after origin hunts: cleared={sorted(self.book.cleared)} "
                f"left={self.book.remaining()} hung={sorted(self.hung)}"
            )

            # 2) 补网：最短环序；新听源绕路 >300 m 则挂起，整圈后再就近追
            print("== phase 2 ring cover ==")
            rings = ring_points()
            tour, tour_L = ring_tour_bruteforce(self.xy, rings)
            print(f"ring tour brute-force L={tour_L:.1f}m hang={self.hang} thresh={HANG_M:.0f}m")
            for step_i, (k, p) in enumerate(tour):
                if len(self.book.cleared) >= 16:
                    print("16 sources cleared, skip remaining ring")
                    break
                left = self.book.remaining()
                if not left:
                    break
                if time.time() - t0 > remain - 45:
                    print("wall-clock budget low, abort remaining ring, drain pending")
                    break
                a_next = tour[step_i + 1][1] if step_i + 1 < len(tour) else None
                self._next_xy = a_next
                print(f"-- ring {step_i + 1}/{N_RING} left={left}")
                heard = self.listen_point(p, left, cover_id=f"R{k}")
                self.mark_empty_by_cover()
                if self.hang:
                    for ch in heard:
                        extra = self.chase_extra(ch, a_next)
                        if a_next is not None and extra > HANG_M:
                            self.hung.add(ch)
                            print(f"   hang ch{ch} extra={extra:.0f}m > {HANG_M:.0f}m")
                        else:
                            self.hung.discard(ch)
                            self.hunt(ch)
                            self.hung.discard(ch)
                    self.harvest_ready(max_dist=200.0)
                else:
                    self.hunt_heard()

        # 环结束或墙钟 abort 后立即 drain：原点挂起 + 环上挂起均在此就近追
        self._next_xy = None
        if self.hung or (self.hang and self.book.heard()):
            hung_left = sorted(c for c in self.book.heard() if c in self.hung)
            if hung_left:
                print(f"== drain hung nearest-first {hung_left} ==")
                others = set(self.book.heard()) - set(self.hung)
                self.hunt_heard(skip=others, include_hung=True)
            leftover = sorted(c for c in self.book.heard())
            if leftover:
                print(f"== drain pending nearest-first {leftover} ==")
                self.hunt_heard(include_hung=True)

        self.mark_empty_by_cover()
        self.mark_empty_by_cap()
        print(
            f"done cleared={len(self.book.cleared)} empty={len(self.book.empty)} "
            f"failed={sorted(self.book.failed)} left={self.book.remaining()} "
            f"virtual={self.sim.virtual_time_s:.1f}s wall={time.time()-t0:.1f}s"
        )
        try:
            ex = self.sim.exit()
            print("exit", {k: ex.get(k) for k in ("accepted", "exit_reason", "virtual_time_s")})
        except Exception as e:
            print("exit failed:", e)
        return self.summary(time.time() - t0)

    def summary(self, wall_s: float) -> dict:
        return {
            "cleared": sorted(self.book.cleared),
            "n_cleared": len(self.book.cleared),
            "empty": sorted(self.book.empty),
            "failed": sorted(self.book.failed),
            "left": self.book.remaining(),
            "clear_virtual_times": self.book.clear_time,
            "virtual_time_s": self.sim.virtual_time_s,
            "avg_s": (float(self.sim.virtual_time_s) / len(self.book.cleared))
            if self.book.cleared
            else None,
            "wall_s": wall_s,
            "n_replan_decisions": self.n_replan_decisions,
            "n_replan_diverge": self.n_replan_diverge,
        }


def save_log(sim, summary: dict, tag: str = "q3"):
    LOG_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = LOG_DIR / f"{tag}_{stamp}.json"
    payload = {"summary": summary, "log": sim.log_rows[-800:]}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("log written", out)
    return out


def run_official(
    resume: bool = False,
    remain_s: float = 1180.0,
    hang: bool = True,
    origin_hang: bool = True,
    miss_constraint: bool = True,
    replan: bool = REPLAN_DEFAULT,
    onroute_clear: bool = True,
    adaptive_step: bool = False,
):
    gap = covering_worst_gap()
    if gap >= 1000.0:
        print(f"WARNING covering gap {gap:.1f} >= 1000")
    sim = Simulator(ROBOT_ID)
    dog = Dog(
        sim,
        remain_s=remain_s,
        hang=hang,
        origin_hang=origin_hang,
        miss_constraint=miss_constraint,
        replan=replan,
        onroute_clear=onroute_clear,
        adaptive_step=adaptive_step,
    )
    try:
        summary = dog.run(do_enter=not resume)
    except ApiError as e:
        print("API error:", e)
        try:
            sim.exit()
        except Exception:
            pass
        summary = {"error": str(e), "cleared": sorted(dog.book.cleared), "n_cleared": len(dog.book.cleared)}
    save_log(sim, summary)
    return summary


def watch_loop(
    hang: bool = True,
    origin_hang: bool = True,
    miss_constraint: bool = True,
    replan: bool = REPLAN_DEFAULT,
    onroute_clear: bool = True,
    adaptive_step: bool = False,
):
    print("watch: waiting for simulator tests, Ctrl+C to stop")
    n_fail = 0
    while True:
        sim = Simulator(ROBOT_ID)
        try:
            body = sim._post(
                "/enter",
                {"arena_id": "default", "robot_id": sim.robot_id, "request_id": sim._new_id("enter")},
                retries=1,
            )
            if body.get("accepted") is not True:
                n_fail = 0
                time.sleep(2.0)
                continue
        except Exception as e:
            n_fail += 1
            if n_fail <= 2 or n_fail % 8 == 0:
                print(f"enter wait ({n_fail}): {e}")
            time.sleep(3.0)
            continue
        n_fail = 0
        remain = float(body.get("remaining_real_duration_s", 1200))
        print(f"\n==== new test remaining={remain}s {time.strftime('%H:%M:%S')} ====")
        dog = Dog(
            sim,
            remain_s=remain,
            hang=hang,
            origin_hang=origin_hang,
            miss_constraint=miss_constraint,
            replan=replan,
            onroute_clear=onroute_clear,
            adaptive_step=adaptive_step,
        )
        try:
            summary = dog.run(do_enter=False)
        except ApiError as e:
            print("API error:", e)
            try:
                sim.exit()
            except Exception:
                pass
            summary = {"error": str(e), "cleared": sorted(dog.book.cleared), "n_cleared": len(dog.book.cleared)}
        save_log(sim, summary)
        print("waiting for next test...")
        time.sleep(1.5)


def print_local_summary(rows) -> dict:
    avgs = [float(r["avg_s"]) for r in rows if r.get("avg_s") is not None]
    Ts = [float(r["virtual_time_s"]) for r in rows]
    full = sum(1 for r in rows if r.get("full_clear"))
    ratios = [float(r["path_vs_2opt"]) for r in rows if r.get("path_vs_2opt") is not None]
    n_dec = sum(int(r.get("n_replan_decisions") or 0) for r in rows)
    n_div = sum(int(r.get("n_replan_diverge") or 0) for r in rows)
    mean_avg = float(np.mean(avgs)) if avgs else float("nan")
    std_avg = float(np.std(avgs, ddof=1)) if len(avgs) > 1 else 0.0
    mean_T = float(np.mean(Ts)) if Ts else float("nan")
    mean_ratio = float(np.mean(ratios)) if ratios else None
    rec = {
        "n": len(rows),
        "avg_s": mean_avg,
        "std_s": std_avg,
        "mean_T": mean_T,
        "full_clears": full,
        "path_vs_2opt": mean_ratio,
        "n_replan_decisions": n_dec,
        "n_replan_diverge": n_div,
        "replan_diverge_rate": (n_div / n_dec) if n_dec else None,
    }
    ratio_txt = f"{mean_ratio:.3f}" if mean_ratio is not None else "n/a"
    div_txt = f"{n_div}/{n_dec}" if n_dec else "n/a"
    print("=" * 64)
    print(
        f"LOCAL SUMMARY n={len(rows)}  avg={mean_avg:.1f}±{std_avg:.1f} s/each  "
        f"meanT={mean_T:.0f}s  full={full}/{len(rows)}  path/2opt={ratio_txt}  "
        f"replan≠NN={div_txt}"
    )
    print("=" * 64)
    return rec


def run_local(
    n_ep: int = 4,
    hang: bool = True,
    origin_hang: bool = True,
    miss_constraint: bool = True,
    replan: bool = REPLAN_DEFAULT,
    onroute_clear: bool = True,
    adaptive_step: bool = False,
):
    from local_world import LocalSimulator

    rows = []
    for i in range(n_ep):
        rng = np.random.default_rng(20260911 + i)
        sim = LocalSimulator(rng)
        print(
            f"\n==== local ep {i+1} n={sim.n_true} hang={hang} "
            f"origin_hang={origin_hang} miss={miss_constraint} "
            f"replan={replan} onroute={onroute_clear} adaptive={adaptive_step} ===="
        )
        dog = Dog(
            sim,
            hang=hang,
            origin_hang=origin_hang,
            miss_constraint=miss_constraint,
            replan=replan,
            onroute_clear=onroute_clear,
            adaptive_step=adaptive_step,
        )
        summary = dog.run()
        alive = [ch for ch, s in sim.sources.items() if s["alive"]]
        n = max(int(summary["n_cleared"]), 1)
        avg = float(summary["virtual_time_s"]) / n
        summary["alive_left"] = alive
        summary["n_true"] = sim.n_true
        summary["avg_s"] = avg
        summary["full_clear"] = (not alive) and (int(summary["n_cleared"]) == int(sim.n_true))
        traj = path_from_log(sim.log_rows)
        actual = actual_path_length(traj)
        stops = unique_stops(traj)
        if len(stops) >= 3 and actual > 1.0:
            _ord, opt_L = two_opt_tour(stops, np.zeros(2))
            ratio = actual / opt_L if opt_L > 1e-9 else None
            summary["path_actual_m"] = actual
            summary["path_2opt_m"] = opt_L
            summary["path_vs_2opt"] = ratio
            print(
                f"path vs 2-opt: actual={actual:.0f}m  2opt={opt_L:.0f}m  "
                f"ratio={ratio:.3f}  n_stops={len(stops)}"
            )
        else:
            print("path vs 2-opt: skipped (no trajectory / too few stops)")
        rows.append(summary)
        print(
            f"alive_left {alive} cleared {summary['n_cleared']}/{sim.n_true} "
            f"T={summary['virtual_time_s']:.0f}s avg={avg:.0f}s "
            f"full={int(summary['full_clear'])}"
        )
    print_local_summary(rows)
    return rows


def dog_kwargs(args) -> dict:
    return {
        "hang": not args.no_hang,
        "origin_hang": not args.no_origin_hang,
        "miss_constraint": not args.no_miss_constraint,
        "replan": args.replan,
        "onroute_clear": not args.no_onroute_clear,
        "adaptive_step": args.adaptive_step,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument(
        "--no-hang",
        action="store_true",
        help="disable ring-phase hang (chase immediately on ring); origin hang still ON unless --no-origin-hang",
    )
    ap.add_argument(
        "--no-origin-hang",
        action="store_true",
        help="disable origin-phase hang (chase all heard after origin scan); ring hang still ON unless --no-hang",
    )
    ap.add_argument(
        "--no-miss-constraint",
        action="store_true",
        help="disable miss-based |G-M|>1000 longitudinal bet clipping (always t=BET_T)",
    )
    replan_g = ap.add_mutually_exclusive_group()
    replan_g.add_argument(
        "--replan",
        dest="replan",
        action="store_true",
        help="enable rolling replan dispatcher (open tour on pool; execute first node only; default off)",
    )
    replan_g.add_argument(
        "--no-replan",
        dest="replan",
        action="store_false",
        help="disable rolling replan; sequential origin-hunt then ring (current default)",
    )
    ap.set_defaults(replan=REPLAN_DEFAULT)
    ap.add_argument(
        "--no-onroute-clear",
        action="store_true",
        help="clear at C_MEC instead of shifting toward the next planned node",
    )
    ap.add_argument(
        "--adaptive-step",
        action="store_true",
        help="if MEC center C exists, second-point t=clip(0.85*|C-S0|,20,1100) instead of BET_T",
    )
    args = ap.parse_args()
    kw = dog_kwargs(args)
    if args.local:
        run_local(n_ep=args.episodes, **kw)
        return
    if args.watch:
        watch_loop(**kw)
        return
    run_official(resume=args.resume, **kw)


if __name__ == "__main__":
    main()
