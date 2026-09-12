# -*- coding: utf-8 -*-
"""问题4：全向+定向混合源。无信号不再等于远处无源。

策略：
1. 默认环抱证书：原点扫频后巡回 ring(400,6)+ring(1300,8)+ring(1900,12)；
   26 点均无信号则可判空（含定向）。
2. 对照：hybrid = 七点覆盖 + 850 m 六角格 ∪ 1750 m 外圈，空转早停；
   hybrid_full 走满；cover7 仅七点。
3. 听到示向度后沿射线小侧偏交会。`near` 就地清除。
   无信号且已贴到上一次有效测站附近才试 `/clear`（定向背向近距）；走过源掉进阴影时在「有效测站→失联点」线段上搜，不在两个阴影点之间打转。
4. 交会 MEC<=20 则清除。已听到的源当轮追完，不挂起等整张网走完。

用法：
  python q4/run_q4.py              # 默认 cert26
  python q4/run_q4.py --mode hybrid
  python q4/run_q4.py --watch
  python q4/run_q4.py --local [--episodes N]
  python tools/practice_loop.py --problem 4 --n 8
环境变量 ROBOT_ID 可覆盖默认参赛队号。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from functools import partial
from pathlib import Path

print = partial(print, flush=True)

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "q1"))
sys.path.insert(0, str(ROOT / "q2"))
sys.path.insert(0, str(ROOT / "q3"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from api import Simulator, ApiError  # noqa: E402
from geometry import diam_and_axis, mec  # noqa: E402
from solver import locate  # noqa: E402
from tactics import closer_offset, diverse_order, estimate_pos, greedy_order, lateral_offset  # noqa: E402

ROBOT_ID = os.environ.get("ROBOT_ID", "202601006115")
ARENA_R = 1800.0
COVER_R = 1150.0
N_COVER = 6
HEX_SPACING = 850.0
RIM_R = 1750.0
N_RIM = 12
EXTRA_SKIP = 280.0
MIN_EXTRAS = 12
IDLE_STOP = 5
CLEAR_R = 20.0
MAX_HUNT_MEAS = 10
LOG_DIR = Path(__file__).resolve().parent / "logs"


class Sim(Simulator):
    def _new_id(self, prefix: str) -> str:
        self.seq += 1
        return f"{prefix}-{os.getpid()}-{self.seq:04d}"


def unit(deg: float) -> np.ndarray:
    t = math.radians(deg)
    return np.array([math.cos(t), math.sin(t)], dtype=float)


def clamp_arena(p: np.ndarray, radius: float = 1790.0) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = float(np.linalg.norm(p))
    if n > radius:
        p = p * (radius / n)
    return p


def hex_lattice(spacing: float = HEX_SPACING, radius: float = 1790.0) -> list:
    dy = spacing * math.sqrt(3) / 2.0
    pts = []
    ny = int(radius / dy) + 2
    nx = int(radius / spacing) + 2
    for r in range(-ny, ny + 1):
        y = r * dy
        x0 = (spacing / 2.0) if (r % 2) else 0.0
        for c in range(-nx, nx + 1):
            x = x0 + c * spacing
            if x * x + y * y <= radius * radius:
                pts.append(np.array([x, y], dtype=float))
    if min(float(np.linalg.norm(p)) for p in pts) > 1.0:
        pts.append(np.zeros(2))
    return pts


def rim_points(radius: float = RIM_R, n: int = N_RIM) -> list:
    return [
        radius * np.array([math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n)])
        for k in range(n)
    ]


def ring_points(radius: float, n: int, phase: float = 0.0) -> list:
    return [
        radius
        * np.array(
            [
                math.cos(2.0 * math.pi * k / n + phase),
                math.sin(2.0 * math.pi * k / n + phase),
            ]
        )
        for k in range(n)
    ]


def encircle_skeleton() -> list:
    """环抱证书骨架：内环 400 m×6、中环 1300 m×8、圆外 1900 m×12，共 26 点。"""
    return ring_points(400.0, 6) + ring_points(1300.0, 8) + ring_points(1900.0, 12)


def merge_points(groups: list, min_sep: float = 220.0) -> list:
    out = []
    for g in groups:
        for p in g:
            p = np.asarray(p, float)
            if not out or min(float(np.linalg.norm(p - q)) for q in out) >= min_sep:
                out.append(p)
    return out


def tsp_order(pts: list, start: np.ndarray) -> list:
    left = [np.asarray(p, float) for p in pts]
    cur = np.asarray(start, float)
    order = []
    while left:
        k = min(range(len(left)), key=lambda i: float(np.linalg.norm(left[i] - cur)))
        order.append(left.pop(k))
        cur = order[-1]
    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            for j in range(i + 1, len(order)):
                new = order[:i] + order[i : j + 1][::-1] + order[j + 1 :]
                def L(seq):
                    s, c = 0.0, np.asarray(start, float)
                    for p in seq:
                        s += float(np.linalg.norm(p - c))
                        c = p
                    return s
                if L(new) < L(order) - 1e-6:
                    order, improved = new, True
                    break
            if improved:
                break
    return order


def cover_points(radius: float = COVER_R, n: int = N_COVER) -> list:
    pts = [np.zeros(2)]
    for k in range(n):
        a = 2.0 * math.pi * k / n
        pts.append(radius * np.array([math.cos(a), math.sin(a)], dtype=float))
    return pts


def extra_listen_points(visited, start) -> list:
    """外缘 12 点按最近邻走一圈，再补六角格剩余。"""
    refs = [np.asarray(v, float) for v in visited] or [np.zeros(2)]
    start = np.asarray(start, float)

    def far(pts):
        out = []
        for p in pts:
            p = np.asarray(p, float)
            if min(float(np.linalg.norm(p - q)) for q in refs) >= EXTRA_SKIP:
                out.append(p)
        return out

    rims = far(rim_points())
    rims = greedy_order(rims, start)
    hex_left = []
    seen = refs + rims
    for p in hex_lattice():
        p = np.asarray(p, float)
        if min(float(np.linalg.norm(p - q)) for q in seen) >= EXTRA_SKIP:
            hex_left.append(p)
    hex_left = diverse_order(hex_left, seen if seen else [np.zeros(2)])
    return rims + hex_left


def listen_net() -> list:
    """25 点环绕网，供作图；实际巡线见 cover_points + extra_listen_points。"""
    pts = merge_points([hex_lattice(), rim_points()], min_sep=200.0)
    return tsp_order(pts, np.zeros(2))


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


class ChannelBook:
    def __init__(self):
        self.cleared = set()
        self.empty = set()
        self.failed = set()
        self.hits = {ch: [] for ch in range(1, 21)}
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
        sim,
        wall0: float | None = None,
        remain_s: float = 1200.0,
        mode: str = "cert26",
    ):
        self.sim = sim
        self.xy = np.zeros(2)
        self.last_ch = 1
        self.book = ChannelBook()
        self.n_clear_ok = 0
        self.wall0 = time.time() if wall0 is None else wall0
        self.remain_s = remain_s
        self.listen_visits = []
        self.n_listen = 0
        self.idle_stopped = False
        self.deferred = set()
        self.mode = str(mode)
        # 证书外环在 1900 m，不能再把坐标压回 1790 m。
        self.clamp_r = 1950.0 if self.mode == "cert26" else 1790.0
        self.cert_ids = {f"C{k}" for k in range(26)}

    def time_low(self, need: float = 40.0) -> bool:
        return (time.time() - self.wall0) > (self.remain_s - need)

    def measure_at(self, p, ch: int, cover_id: str | None = None) -> dict:
        p = clamp_arena(np.asarray(p, float), radius=self.clamp_r)
        body = self.sim.measure(float(p[0]), float(p[1]), ch)
        self.xy = p
        self.last_ch = int(ch)
        kind = body.get("measure_result")
        extra = f"  svd={body.get('svd_deg')}" if kind == "direction" else ""
        print(f"  meas ch{ch:02d} ({p[0]:.1f},{p[1]:.1f}) -> {kind}{extra}  t={self.sim.virtual_time_s:.1f}s")
        if kind == "direction":
            self.book.hits[ch].append((p.copy(), float(body["svd_deg"])))
        elif kind == "no_signal":
            self.book.miss_pts[ch].append(p.copy())
            if cover_id:
                self.book.miss_ids[ch].add(cover_id)
        return body

    def clear_at(self, p, ch: int) -> bool:
        p = clamp_arena(np.asarray(p, float))
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

    def maybe_clear_no_bearing(self, ch: int, kind) -> bool:
        """near 必清。无信号只在已经贴着上一次有效测站时试清除（定向背向、20 m 圈还可能套住）。"""
        if kind == "near":
            return self.clear_at(self.xy, ch)
        if kind != "no_signal" or not self.book.hits[ch]:
            return False
        S = np.asarray(self.book.hits[ch][-1][0], float)
        if float(np.linalg.norm(self.xy - S)) <= 45.0:
            return self.clear_at(self.xy, ch)
        return False

    def _fresh_point(self, p, used, eps: float = 30.0) -> bool:
        p = np.asarray(p, float)
        if float(np.linalg.norm(p - self.xy)) < 15.0:
            return False
        for q in used:
            if float(np.linalg.norm(p - np.asarray(q, float))) < eps:
                return False
        return True

    def recover_shadow(self, ch: int) -> bool:
        """走过源或掉进背向：源多半在最近一次有效测站与当前失联点之间。"""
        if not self.book.hits[ch]:
            return False
        S = np.asarray(self.book.hits[ch][-1][0], float)
        P = np.asarray(self.xy, float)
        span = float(np.linalg.norm(P - S))
        if span < 18.0 or span > 450.0:
            return False
        print(f"   shadow-search ch{ch} span={span:.0f}m")
        n = max(4, min(8, int(span / 22.0) + 1))
        e = (P - S) / span
        nvec = np.array([-e[1], e[0]])
        # 从靠近失联点往回搜，成功点通常就在阴影边界内侧
        for k in range(n, 0, -1):
            if self.time_low(25.0):
                return False
            p = S + (k / float(n)) * (P - S)
            if self.clear_at(p, ch):
                return True
        for side in (12.0, -12.0):
            for frac in (0.85, 0.6, 0.4):
                if self.time_low(25.0):
                    return False
                p = S + frac * (P - S) + side * nvec
                if self.clear_at(p, ch):
                    return True
        return False

    def try_localize(self, ch: int):
        hits = self.book.hits[ch]
        if len(hits) < 2:
            return None
        return locate([h[0] for h in hits], [h[1] for h in hits], 1.0)

    def beam_point(self, S, th, step: int, prefer_short: bool = False) -> np.ndarray:
        abs_plans = [
            (260.0, 80.0),
            (160.0, 55.0),
            (360.0, 45.0),
            (120.0, 35.0),
            (420.0, 35.0),
            (80.0, 25.0),
            (520.0, 30.0),
        ]
        if prefer_short:
            abs_plans = [
                (160.0, 55.0),
                (120.0, 35.0),
                (200.0, 50.0),
                (90.0, 28.0),
                (80.0, 25.0),
            ]
        if step < len(abs_plans):
            t, h = abs_plans[step]
        else:
            t = 180.0 + 160.0 * (step - len(abs_plans))
            h = 28.0
        if step % 2 == 0:
            return closer_offset(S, th, self.xy, t, h)
        e = unit(th)
        n = np.array([-e[1], e[0]])
        close = closer_offset(S, th, self.xy, t, h)
        p_pos = clamp_arena(np.asarray(S, float) + t * e + h * n)
        p_neg = clamp_arena(np.asarray(S, float) + t * e - h * n)
        if float(np.linalg.norm(close - p_pos)) <= float(np.linalg.norm(close - p_neg)):
            return p_neg
        return p_pos

    def along_bearing(self, ch: int, loc=None) -> np.ndarray:
        S, th = self.book.hits[ch][-1]
        e = unit(th)
        n = np.array([-e[1], e[0]])
        sgn = 1.0 if len(self.book.hits[ch]) % 2 == 0 else -1.0
        t = 120.0
        if loc is not None and loc.vertices is not None:
            r, C = region_center(loc.vertices)
            d = float(np.linalg.norm(C - S))
            if r <= 60.0:
                t = min(160.0, max(40.0, 0.6 * d))
            else:
                t = min(110.0, max(50.0, 0.25 * d))
        return clamp_arena(np.asarray(S, float) + t * e + sgn * 40.0 * n)

    def ray_clear(self, ch: int) -> bool:
        if not self.book.hits[ch]:
            return False
        S, th = self.book.hits[ch][-1]
        e = unit(th)
        n = np.array([-e[1], e[0]])
        t_now = float((self.xy - S) @ e)
        t0 = max(40.0, t_now - 80.0) if 80.0 < t_now < 1100.0 else 40.0
        t_end = 1100.0
        print(f"   ray-clear ch{ch} from ({S[0]:.0f},{S[1]:.0f}) th={th:.1f} t={t0:.0f}..{t_end:.0f}")
        ts = list(np.arange(t0, t_end + 1e-6, 20.0))
        if abs(t_now - ts[0]) > abs(t_now - ts[-1]):
            ts = ts[::-1]
        side = 1.0 if float((self.xy - S) @ n) >= 0.0 else -1.0
        n_try = 0
        for sgn in (side, -side):
            for t in ts:
                if self.time_low(25.0) or n_try >= 48:
                    print("   ray-clear stop: budget")
                    return False
                n_try += 1
                p = clamp_arena(S + t * e + sgn * 12.0 * n)
                if self.clear_at(p, ch):
                    return True
        return False

    def hunt(self, ch: int, last_ditch: bool = False) -> bool:
        print(f"-- hunt channel {ch} last_ditch={last_ditch}")
        if ch in self.book.cleared:
            return True
        used = [self.xy.copy()]
        n_miss = 0
        did_shadow = False

        def try_shadow() -> bool:
            nonlocal did_shadow
            if did_shadow:
                return False
            did_shadow = True
            return self.recover_shadow(ch)
        for step in range(MAX_HUNT_MEAS):
            if self.time_low(30.0):
                print("   hunt stop: wall clock")
                break
            loc = self.try_localize(ch)
            if loc is not None and loc.status == "OK" and loc.vertices is not None:
                r, C = region_center(loc.vertices)
                print(f"   localize OK m={len(loc.vertices)} MEC={r:.2f}m  D={loc.diameter:.2f}m")
                if r <= CLEAR_R + 1e-6:
                    if self.clear_at(C, ch):
                        return True
                    for v in loc.vertices:
                        if self.clear_at(v, ch):
                            return True
                nxt = self.along_bearing(ch, loc)
                if not self._fresh_point(nxt, used):
                    L, ax = diam_and_axis(loc.vertices)
                    if ax is None:
                        ax = np.array([1.0, 0.0])
                    nh = np.array([-ax[1], ax[0]])
                    nxt = clamp_arena(C + max(40.0, 0.25 * float(L)) * nh)
                if not self._fresh_point(nxt, used):
                    if try_shadow() or (last_ditch and self.ray_clear(ch)):
                        return True
                    break
                body = self.measure_at(nxt, ch)
                used.append(self.xy.copy())
                kind = body.get("measure_result")
                if self.maybe_clear_no_bearing(ch, kind):
                    return True
                if kind == "no_signal":
                    n_miss += 1
                    if try_shadow():
                        return True
                    if n_miss >= 2:
                        break
                else:
                    n_miss = 0
                continue

            hits = self.book.hits[ch]
            if not hits:
                return False
            if loc is not None and loc.status == "EMPTY" and len(hits) >= 2:
                self.book.hits[ch] = hits[-1:]
                hits = self.book.hits[ch]

            S0, th0 = hits[-1]
            if step == 0 and float(np.linalg.norm(self.xy - S0)) > 150.0:
                if lateral_offset(S0, th0, self.xy) > 40.0:
                    body = self.measure_at(self.xy, ch)
                    used.append(self.xy.copy())
                    kind = body.get("measure_result")
                    if self.maybe_clear_no_bearing(ch, kind):
                        return True
                    if kind == "direction":
                        continue
                    if kind == "no_signal" and try_shadow():
                        return True
            nxt = self.beam_point(S0, th0, step, prefer_short=n_miss > 0)
            if not self._fresh_point(nxt, used):
                nxt = self.beam_point(S0, th0, step + 2, prefer_short=False)
            if not self._fresh_point(nxt, used):
                if try_shadow() or (last_ditch and self.ray_clear(ch)):
                    return True
                break
            body = self.measure_at(nxt, ch)
            used.append(self.xy.copy())
            kind = body.get("measure_result")
            if self.maybe_clear_no_bearing(ch, kind):
                return True
            if kind == "no_signal":
                n_miss += 1
                if try_shadow():
                    return True
                if n_miss >= 2:
                    break
            else:
                n_miss = 0

        loc = self.try_localize(ch)
        if loc is not None and loc.status == "OK" and loc.vertices is not None:
            r, C = region_center(loc.vertices)
            if r <= CLEAR_R + 1e-6 and self.clear_at(C, ch):
                return True
        if last_ditch and self.book.hits[ch]:
            if self.ray_clear(ch):
                return True
            print(f"   hunt failed ch{ch}")
            self.book.failed.add(ch)
        else:
            print(f"   hunt defer ch{ch}")
            self.deferred.add(ch)
        return ch in self.book.cleared

    def listen_point(self, p, channels: list, cover_id: str | None = None) -> list:
        heard_now = []
        p = clamp_arena(np.asarray(p, float), radius=self.clamp_r)
        ordered = sorted(list(channels), key=lambda c: abs(int(c) - self.last_ch))
        for ch in ordered:
            if ch not in self.book.remaining():
                continue
            if self.book.hits[ch]:
                continue
            if self.time_low(25.0):
                break
            body = self.measure_at(p, ch, cover_id=cover_id)
            kind = body.get("measure_result")
            if kind == "near":
                self.clear_at(p, ch)
            elif kind == "direction":
                heard_now.append(ch)
        return heard_now

    def hunt_heard(self, last_ditch: bool = False) -> None:
        tried = set()
        while True:
            hs = [c for c in self.book.heard() if c not in tried]
            if not last_ditch:
                hs = [c for c in hs if c not in self.deferred]
            if not hs:
                break
            if self.time_low(28.0):
                break
            ch = min(
                hs,
                key=lambda c: float(np.linalg.norm(estimate_pos(self.book.hits[c]) - self.xy)),
            )
            tried.add(ch)
            if self.hunt(ch, last_ditch=last_ditch):
                self.deferred.discard(ch)

    def mark_empty_by_cap(self):
        if len(self.book.cleared) < 16:
            return
        for ch in list(self.book.remaining()):
            self.book.empty.add(ch)
            print(f"-- channel {ch} empty (at most 16 sources)")

    def mark_empty_by_cert(self):
        """26 点环抱证书：各探针均无信号则该频道必空（含定向）。"""
        if self.mode != "cert26":
            return
        need = self.cert_ids
        for ch in list(self.book.remaining()):
            if self.book.hits[ch]:
                continue
            if need <= self.book.miss_ids[ch]:
                self.book.empty.add(ch)
                print(f"-- channel {ch} empty (encircle certificate)")

    def _scan_and_hunt(self, p, tag: str, cover_id: str | None = None) -> bool:
        """返回本点是否出现新的 direction/near（用于空转计数）。"""
        left = self.book.remaining()
        n0 = len(self.book.cleared)
        print(f"-- {tag} left={left}")
        heard = self.listen_point(p, left, cover_id=cover_id)
        self.listen_visits.append(np.asarray(p, float).copy())
        self.n_listen += 1
        new_hit = bool(heard) or (len(self.book.cleared) > n0)
        self.hunt_heard(last_ditch=False)
        self.mark_empty_by_cert()
        self.mark_empty_by_cap()
        return new_hit

    def _walk_points(self, pts, tag: str, cover_ids=None, idle_stop: bool = False, nearest: bool = True):
        pending = list(pts)
        ids = list(cover_ids) if cover_ids is not None else [None] * len(pending)
        idle = 0
        step = 0
        n_tot = len(pending)
        min_before_idle = MIN_EXTRAS if idle_stop else 10**9
        idle_lim = IDLE_STOP if idle_stop else 10**9
        while pending:
            if len(self.book.cleared) >= 16:
                print("16 sources cleared, stop walk")
                break
            if not self.book.remaining():
                break
            if self.time_low(40.0):
                print("wall-clock budget low, stop walk")
                break
            if idle_stop and step >= min_before_idle and idle >= idle_lim:
                self.idle_stopped = True
                print(f"idle stop: {idle} extras with no new hit, skip {len(pending)} pts")
                break
            if nearest:
                k = min(range(len(pending)), key=lambda i: float(np.linalg.norm(self.xy - pending[i])))
            else:
                k = 0
            p = pending.pop(k)
            cid = ids.pop(k)
            step += 1
            new_hit = self._scan_and_hunt(p, f"{tag} {step}/{n_tot}", cover_id=cid)
            if new_hit:
                idle = 0
            else:
                idle += 1

    def run(self, do_enter: bool = True):
        if do_enter:
            enter = self.sim.enter()
            self.remain_s = float(enter.get("remaining_real_duration_s", 1200))
            self.wall0 = time.time()
            print(f"entered remaining_real={self.remain_s}s  wall={time.strftime('%H:%M:%S')}")
        else:
            print(f"resume remaining_real~{self.remain_s}s")
        print(f"mode={self.mode}")

        if self.mode == "cert26":
            print("== cert26: origin scan then 26-point encircle ==")
            self._scan_and_hunt(np.zeros(2), "origin")
            skel = encircle_skeleton()
            ids = [f"C{k}" for k in range(len(skel))]
            order = tsp_order(skel, self.xy)
            # tsp permutes points; recover ids by nearest original index
            used = set()
            ordered_ids = []
            for p in order:
                k = min(
                    (i for i in range(len(skel)) if i not in used),
                    key=lambda i: float(np.linalg.norm(skel[i] - p)),
                )
                used.add(k)
                ordered_ids.append(ids[k])
            self._walk_points(order, "cert", cover_ids=ordered_ids, idle_stop=False)
        elif self.mode == "cover7":
            print(f"== cover7 only: origin + {N_COVER}×{COVER_R:.0f}m ==")
            self._walk_points(cover_points(), "cover", idle_stop=False)
        else:
            cover = cover_points()
            print(f"phase-1 cover {len(cover)} points (origin + {N_COVER}×{COVER_R:.0f}m)")
            print("== phase 1 cover scan + hunt ==")
            self._walk_points(cover, "cover", idle_stop=False)
            print("== phase 2 extra surround ==")
            extras = extra_listen_points(self.listen_visits, self.xy)
            idle_stop = self.mode == "hybrid"
            print(
                f"extra candidates {len(extras)} "
                f"(skip<{EXTRA_SKIP:.0f}m, min_extras={MIN_EXTRAS}, idle-stop={idle_stop})"
            )
            self._walk_points(extras, "extra", idle_stop=idle_stop, nearest=False)

        print("== last-ditch hunts ==")
        self.hunt_heard(last_ditch=True)
        self.mark_empty_by_cert()
        self.mark_empty_by_cap()
        print(
            f"done cleared={len(self.book.cleared)} empty={len(self.book.empty)} "
            f"failed={sorted(self.book.failed)} left={self.book.remaining()} "
            f"listens={self.n_listen} idle_stop={self.idle_stopped} "
            f"virtual={self.sim.virtual_time_s:.1f}s wall={time.time()-self.wall0:.1f}s"
        )
        try:
            ex = self.sim.exit()
            print("exit", ex)
        except Exception as e:
            print("exit failed:", e)
            ex = {}
        return self.summary(time.time() - self.wall0, ex)

    def summary(self, wall_s: float, ex=None) -> dict:
        n = max(len(self.book.cleared), 1)
        vt = float(self.sim.virtual_time_s)
        return {
            "cleared": sorted(self.book.cleared),
            "n_cleared": len(self.book.cleared),
            "empty": sorted(self.book.empty),
            "failed": sorted(self.book.failed),
            "left": self.book.remaining(),
            "clear_virtual_times": {str(k): v for k, v in self.book.clear_time.items()},
            "virtual_time_s": vt,
            "avg_s": vt / n if self.book.cleared else None,
            "wall_s": wall_s,
            "n_listen": self.n_listen,
            "idle_stopped": self.idle_stopped,
            "mode": self.mode,
            "exit": ex,
        }


def save_log(sim, summary: dict, tag: str):
    LOG_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = LOG_DIR / f"{tag}_{stamp}.json"
    payload = {"summary": summary, "log": sim.log_rows[-1200:]}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("log written", out)
    return out


def run_official(resume: bool = False, remain_s: float = 1180.0):
    sim = Sim(ROBOT_ID)
    dog = Dog(sim, remain_s=remain_s)
    try:
        summary = dog.run(do_enter=not resume)
    except ApiError as e:
        print("API error:", e)
        try:
            sim.exit()
        except Exception:
            pass
        summary = {"error": str(e), "cleared": sorted(dog.book.cleared), "n_cleared": len(dog.book.cleared)}
    save_log(sim, summary, "q4")
    return summary


def run_local(n_ep: int = 3, p_dir: float = 0.5, mode: str = "cert26"):
    from local_world import LocalSimulator

    rows = []
    for i in range(n_ep):
        rng = np.random.default_rng(1000 + i)
        sim = LocalSimulator(rng, p_dir=p_dir)
        print(f"\n==== local ep {i+1} n={sim.n_true} dir={sim.n_dir} mode={mode} ====")
        dog = Dog(sim, mode=mode)
        summary = dog.run(do_enter=True)
        alive = sum(1 for s in sim.sources.values() if s["alive"])
        summary["alive_left"] = alive
        summary["n_true"] = sim.n_true
        summary["n_dir"] = sim.n_dir
        rows.append(summary)
        print("alive_left", alive, "cleared", summary["n_cleared"], "avg", summary["avg_s"])
    return rows


def watch_loop():
    print("watch: waiting for simulator tests, Ctrl+C to stop")
    n_fail = 0
    while True:
        sim = Sim(ROBOT_ID)
        try:
            body = sim._post(
                "/enter",
                {"arena_id": "default", "robot_id": sim.robot_id, "request_id": sim._new_id("enter")},
                retries=1,
            )
            if body.get("accepted") is not True:
                n_fail = 0
                time.sleep(2.5)
                continue
        except Exception as e:
            n_fail += 1
            if n_fail <= 2 or n_fail % 8 == 0:
                print(f"enter wait ({n_fail}): {e}")
            time.sleep(3.5)
            continue
        n_fail = 0
        remain = float(body.get("remaining_real_duration_s", 1200))
        print(f"\n==== new test remaining={remain}s {time.strftime('%H:%M:%S')} ====")
        dog = Dog(sim, remain_s=remain)
        try:
            summary = dog.run(do_enter=False)
        except ApiError as e:
            print("API error:", e)
            try:
                sim.exit()
            except Exception:
                pass
            summary = {"error": str(e), "cleared": sorted(dog.book.cleared), "n_cleared": len(dog.book.cleared)}
        save_log(sim, summary, "q4")
        print("waiting for next test...")
        time.sleep(2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--mode", default="cert26", choices=["hybrid", "hybrid_full", "cert26", "cover7"])
    args = ap.parse_args()
    if args.local:
        run_local(n_ep=args.episodes, mode=args.mode)
        return
    if args.watch:
        watch_loop()
        return
    run_official(resume=args.resume)


if __name__ == "__main__":
    main()
