# -*- coding: utf-8 -*-
"""问题3：全向干扰源搜索、交会定位与清除。

策略：
1. 原点扫 1–20 频道（不移动，约 119 s 虚拟时间）；
2. 对已听到的频道按示向度排序，用第二问时间赌注 (t,h)=(500,250) 交会，
   MEC 半径 >20 m 则按长轴再切，然后 /clear；
3. 半径 1150 m 上 6 个监听点补网（与原点一起用 1000 m 覆盖半径 1800 m 圆盘），
   每点先扫完剩余频道再搬家；6 点都无信号则判该频道为空。

用法：python q3/run_q3.py
环境变量 ROBOT_ID 可覆盖默认参赛队号。
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "q1"))
sys.path.insert(0, str(ROOT / "q2"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from api import Simulator, ApiError  # noqa: E402
from geometry import diam_and_axis, mec  # noqa: E402
from solver import locate  # noqa: E402

ROBOT_ID = os.environ.get("ROBOT_ID", "202601006115")
ARENA_R = 1800.0
RING_R = 1150.0
N_RING = 6
BET_T = 500.0
BET_H = 250.0
CLEAR_R = 20.0
MAX_HUNT_MEAS = 7
MAX_COORD = 1.8e6

LOG_DIR = Path(__file__).resolve().parent / "logs"


def unit(deg: float) -> np.ndarray:
    t = math.radians(deg)
    return np.array([math.cos(t), math.sin(t)], dtype=float)


def clamp_xy(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = float(np.linalg.norm(p))
    if n > MAX_COORD:
        p = p * (MAX_COORD / n)
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
        self.clear_time = {}

    def remaining(self) -> list:
        done = self.cleared | self.empty | self.failed
        return [ch for ch in range(1, 21) if ch not in done]

    def heard(self) -> list:
        return [ch for ch in self.remaining() if self.hits[ch]]


class Dog:
    def __init__(self, sim: Simulator):
        self.sim = sim
        self.xy = np.zeros(2)
        self.book = ChannelBook()
        self.n_clear_ok = 0

    def measure_at(self, p, ch: int) -> dict:
        p = clamp_xy(np.asarray(p, float))
        body = self.sim.measure(float(p[0]), float(p[1]), ch)
        self.xy = p
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
        return body

    def clear_at(self, p, ch: int) -> bool:
        p = clamp_xy(np.asarray(p, float))
        body = self.sim.clear(float(p[0]), float(p[1]), ch)
        self.xy = p
        ok = body.get("clear_result") == "success"
        print(
            f"  CLEAR ch{ch:02d} ({p[0]:.1f},{p[1]:.1f}) -> {body.get('clear_result')}"
            f"  t={self.sim.virtual_time_s:.1f}s"
        )
        if ok:
            self.book.cleared.add(ch)
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

    def hunt(self, ch: int) -> bool:
        print(f"-- hunt channel {ch}")
        if ch in self.book.cleared:
            return True
        # 已有 near 以外的命中；若尚未测第二点则从最近命中点出发
        for _step in range(MAX_HUNT_MEAS):
            loc = self.try_localize(ch)
            if loc is not None and loc.status == "OK" and loc.vertices is not None:
                r, C = region_center(loc.vertices)
                print(f"   localize OK m={len(loc.vertices)} MEC={r:.2f}m  D={loc.diameter:.2f}m")
                if r <= CLEAR_R + 1e-6:
                    if self.clear_at(C, ch):
                        return True
                    # 中心未中，试顶点
                    for v in loc.vertices:
                        if self.clear_at(v, ch):
                            return True
                    self.book.failed.add(ch)
                    return False
                nxt = long_axis_point(loc.vertices, self.xy)
                body = self.measure_at(nxt, ch)
                if body.get("measure_result") == "near":
                    return self.clear_at(self.xy, ch)
                continue

            hits = self.book.hits[ch]
            if not hits:
                return False
            if loc is not None and loc.status == "EMPTY" and len(hits) >= 2:
                # 误差导致空交：丢掉最早的一站再测
                self.book.hits[ch] = hits[-1:]
                hits = self.book.hits[ch]

            S0, th0 = hits[-1]
            side = BET_H if (_step % 2 == 0) else -BET_H
            t = BET_T + 150.0 * (_step // 2)
            nxt = second_point(S0, th0, t=t, h=side)
            if float(np.linalg.norm(nxt - self.xy)) < 20.0:
                nxt = second_point(S0, th0, t=t + 200.0, h=side)
            body = self.measure_at(nxt, ch)
            kind = body.get("measure_result")
            if kind == "near":
                return self.clear_at(self.xy, ch)
            if kind == "no_signal":
                # 换侧再试
                nxt2 = second_point(S0, th0, t=t, h=-side)
                body2 = self.measure_at(nxt2, ch)
                if body2.get("measure_result") == "near":
                    return self.clear_at(self.xy, ch)
                continue

        loc = self.try_localize(ch)
        if loc is not None and loc.status == "OK" and loc.vertices is not None:
            _, C = region_center(loc.vertices)
            if self.clear_at(C, ch):
                return True
        print(f"   hunt failed ch{ch}")
        self.book.failed.add(ch)
        return False

    def listen_point(self, p, channels: list) -> list:
        """只在 p 扫描，不追击。near 就地清除。"""
        heard_now = []
        p = np.asarray(p, float)
        for ch in list(channels):
            if ch not in self.book.remaining():
                continue
            body = self.measure_at(p, ch)
            kind = body.get("measure_result")
            if kind == "near":
                self.clear_at(p, ch)
            elif kind == "direction":
                heard_now.append(ch)
        return heard_now

    def hunt_heard(self) -> None:
        def ang(ch: int) -> float:
            return self.book.hits[ch][-1][1]

        for ch in sorted(self.book.heard(), key=ang):
            if ch in self.book.cleared:
                continue
            self.hunt(ch)

    def mark_empty_by_cover(self, cover_pts: list, tol: float = 5.0):
        for ch in list(self.book.remaining()):
            if self.book.hits[ch]:
                continue
            misses = self.book.miss_pts[ch]
            if len(misses) < len(cover_pts):
                continue
            ok = True
            for cp in cover_pts:
                if not any(float(np.linalg.norm(np.asarray(m) - cp)) < tol for m in misses):
                    ok = False
                    break
            if ok:
                self.book.empty.add(ch)
                print(f"-- channel {ch} empty (covered, all no_signal)")

    def run(self):
        t0 = time.time()
        enter = self.sim.enter()
        remain = float(enter.get("remaining_real_duration_s", 1200))
        print(f"entered remaining_real={remain}s  wall={time.strftime('%H:%M:%S')}")
        cover = [np.zeros(2)] + ring_points()
        gap = covering_worst_gap()
        print(f"cover net 1+6 R=1150 worst-gap={gap:.1f}m (need <1000)")

        # 1) 原点扫频后再追，保证 20 个频道的覆盖票都在原点
        print("== phase 1 origin scan ==")
        self.listen_point(np.zeros(2), list(range(1, 21)))
        print(f"after origin listen: cleared={sorted(self.book.cleared)} heard={self.book.heard()}")
        self.hunt_heard()
        print(f"after origin hunts: cleared={sorted(self.book.cleared)} left={self.book.remaining()}")

        # 2) 补网：每点先扫完剩余频道，再追本点新听到的
        print("== phase 2 ring cover ==")
        rings = ring_points()
        start = int(np.argmin([np.linalg.norm(self.xy - p) for p in rings]))
        order = [rings[(start + i) % len(rings)] for i in range(len(rings))]
        for i, p in enumerate(order):
            if len(self.book.cleared) >= 16:
                print("16 sources cleared, skip remaining ring")
                break
            left = self.book.remaining()
            if not left:
                break
            if time.time() - t0 > remain - 45:
                print("wall-clock budget low, stop ring")
                break
            print(f"-- ring {i+1}/{len(order)} left={left}")
            self.listen_point(p, left)
            self.mark_empty_by_cover(cover)
            self.hunt_heard()

        self.mark_empty_by_cover(cover)
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
            "wall_s": wall_s,
        }


def main():
    LOG_DIR.mkdir(exist_ok=True)
    gap = covering_worst_gap()
    if gap >= 1000.0:
        print(f"WARNING covering gap {gap:.1f} >= 1000")
    sim = Simulator(ROBOT_ID)
    dog = Dog(sim)
    try:
        summary = dog.run()
    except ApiError as e:
        print("API error:", e)
        try:
            sim.exit()
        except Exception:
            pass
        summary = {"error": str(e), "cleared": sorted(dog.book.cleared)}
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = LOG_DIR / f"q3_{stamp}.json"
    payload = {
        "summary": summary,
        "log": sim.log_rows[-800:],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("log written", out)


if __name__ == "__main__":
    main()
