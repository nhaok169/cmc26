# -*- coding: utf-8 -*-
"""问题3：全向干扰源搜索、交会定位与清除。

策略：
1. 原点扫 1–20 频道（不移动，约 119 s 虚拟时间）；
2. 已听频道：能交会且 MEC≤20 m 的先就近清除；其余按估计位置追击。
   第二点取 (t,h)=(500,±250) 中离狗更近的一侧；若当前位置已有足够侧向间隔则就地再测；
   MEC>20 m 再按长轴切。每个停点顺便测仍只有一站、且相对第一站接近 (500,250) 的已听频道；
   交会已进 20 m 且顺路的源就地收割，避免专程再跑；
3. 半径 1150 m 上 6 个监听点按最近邻补网（与原点一起用 1000 m 覆盖 1800 m 圆盘），
   每点先扫完剩余频道再搬家；7 点都无信号则判该频道为空。

用法：
  python q3/run_q3.py
  python q3/run_q3.py --watch
  python q3/run_q3.py --local [--episodes N]
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

ROBOT_ID = os.environ.get("ROBOT_ID", "202601006115")
ARENA_R = 1800.0
RING_R = 1150.0
N_RING = 6
BET_T = 500.0
BET_H = 250.0
CLEAR_R = 20.0
MAX_HUNT_MEAS = 7

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
    ):
        self.sim = sim
        self.xy = np.zeros(2)
        self.last_ch = 1
        self.book = ChannelBook()
        self.n_clear_ok = 0
        self.remain_s = remain_s
        self.wall0 = time.time()
        self.bet_t = float(bet_t)
        self.bet_h = float(bet_h)
        self.reuse_stops = bool(reuse_stops)

    def wall_low(self, need: float = 50.0) -> bool:
        return (time.time() - self.wall0) > (self.remain_s - need)

    def measure_at(self, p, ch: int, cover_id: str | None = None) -> dict:
        p = clamp_xy(np.asarray(p, float))
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
        p = clamp_xy(np.asarray(p, float))
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

    def ready_center(self, ch: int):
        loc = self.try_localize(ch)
        if loc is None or loc.status != "OK" or loc.vertices is None:
            return None
        r, C = region_center(loc.vertices)
        if r <= CLEAR_R + 1e-6:
            return C
        return None

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
            C = self.ready_center(ch)
            if C is None:
                continue
            ready.append((float(np.linalg.norm(C - self.xy)), ch, C))
        ready.sort()
        for d, ch, C in ready:
            if ch in self.book.cleared:
                continue
            if d > max_dist:
                continue
            self.clear_at(C, ch)

    def after_stop(self, exclude_ch: int | None = None) -> None:
        if not self.reuse_stops:
            return
        self.piggyback(self.xy, exclude_ch=exclude_ch)
        self.harvest_ready(max_dist=80.0, exclude_ch=exclude_ch)

    def hunt(self, ch: int) -> bool:
        print(f"-- hunt channel {ch}")
        if ch in self.book.cleared:
            return True
        for _step in range(MAX_HUNT_MEAS):
            if ch in self.book.cleared:
                return True
            loc = self.try_localize(ch)
            if loc is not None and loc.status == "OK" and loc.vertices is not None:
                r, C = region_center(loc.vertices)
                print(f"   localize OK m={len(loc.vertices)} MEC={r:.2f}m  D={loc.diameter:.2f}m")
                if r <= CLEAR_R + 1e-6:
                    if self.clear_at(C, ch):
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
            t = self.bet_t + 120.0 * (_step // 2)
            nxt = closer_offset(S0, th0, self.xy, t, self.bet_h)
            if float(np.linalg.norm(nxt - self.xy)) < 20.0:
                nxt = closer_offset(S0, th0, self.xy, t + 200.0, self.bet_h)
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
            if r <= CLEAR_R + 1e-6 and self.clear_at(C, ch):
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

    def hunt_heard(self) -> None:
        tried = set()
        while True:
            hs = [c for c in self.book.heard() if c not in tried]
            if not hs:
                break
            ch = min(hs, key=self.hunt_key)
            tried.add(ch)
            self.hunt(ch)
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

        # 1) 原点扫频后再追，保证 20 个频道的覆盖票都在原点
        print("== phase 1 origin scan ==")
        self.listen_point(np.zeros(2), list(range(1, 21)), cover_id="O")
        print(f"after origin listen: cleared={sorted(self.book.cleared)} heard={self.book.heard()}")
        self.hunt_heard()
        print(f"after origin hunts: cleared={sorted(self.book.cleared)} left={self.book.remaining()}")

        # 2) 补网：每点先扫完剩余频道，再追本点新听到的
        print("== phase 2 ring cover ==")
        rings = ring_points()
        pending = list(enumerate(rings))
        step_i = 0
        while pending:
            if len(self.book.cleared) >= 16:
                print("16 sources cleared, skip remaining ring")
                break
            left = self.book.remaining()
            if not left:
                break
            if time.time() - t0 > remain - 45:
                print("wall-clock budget low, stop ring")
                break
            k, p = min(pending, key=lambda kp: float(np.linalg.norm(self.xy - kp[1])))
            pending.remove((k, p))
            step_i += 1
            print(f"-- ring {step_i}/{N_RING} left={left}")
            self.listen_point(p, left, cover_id=f"R{k}")
            self.mark_empty_by_cover()
            self.hunt_heard()

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
        }


def save_log(sim, summary: dict, tag: str = "q3"):
    LOG_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = LOG_DIR / f"{tag}_{stamp}.json"
    payload = {"summary": summary, "log": sim.log_rows[-800:]}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("log written", out)
    return out


def run_official(resume: bool = False, remain_s: float = 1180.0):
    gap = covering_worst_gap()
    if gap >= 1000.0:
        print(f"WARNING covering gap {gap:.1f} >= 1000")
    sim = Simulator(ROBOT_ID)
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
    save_log(sim, summary)
    return summary


def watch_loop():
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
        save_log(sim, summary)
        print("waiting for next test...")
        time.sleep(1.5)


def run_local(n_ep: int = 4):
    from local_world import LocalSimulator

    rows = []
    for i in range(n_ep):
        rng = np.random.default_rng(20260911 + i)
        sim = LocalSimulator(rng)
        print(f"\n==== local ep {i+1} n={sim.n_true} ====")
        dog = Dog(sim)
        summary = dog.run()
        alive = [ch for ch, s in sim.sources.items() if s["alive"]]
        n = max(int(summary["n_cleared"]), 1)
        avg = float(summary["virtual_time_s"]) / n
        summary["alive_left"] = alive
        summary["n_true"] = sim.n_true
        summary["avg_s"] = avg
        rows.append(summary)
        print(
            f"alive_left {alive} cleared {summary['n_cleared']}/{sim.n_true} "
            f"T={summary['virtual_time_s']:.0f}s avg={avg:.0f}s"
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--episodes", type=int, default=4)
    args = ap.parse_args()
    if args.local:
        run_local(n_ep=args.episodes)
        return
    if args.watch:
        watch_loop()
        return
    run_official(resume=args.resume)


if __name__ == "__main__":
    main()
