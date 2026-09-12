# -*- coding: utf-8 -*-
"""
mock_test.py — G0-G4 × 30 seeds 批量实验 + 统计报表

G0: 全覆盖基线 (7 骨架点全扫 + LSQ + 螺旋清除)
G1: v1 (13 扫描点 + LSQ + 螺旋) = 旧 robot.py
G2: v2-core (四模块+双账, 单步贪心)
G3: v2+rollout (+ 1步 rollout)
G4: 离线下界 (已知源位 TSP + 直接清除)

用法:
    python mock_test.py G2 1       # 单组单种子
    python mock_test.py all         # 全组 × 30 种子
    python mock_test.py 3 1         # 兼容旧格式 (= G1)
"""
import hashlib
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "q3"))
import robot as R


# ===================== FakeServer =====================
class FakeServer:
    ARENA_R = 1800.0
    VT_LIMIT = 360000.0

    def __init__(self, problem, seed=1):
        self.problem = problem
        rnd = random.Random(seed)
        n = rnd.randint(10, 16)
        pts = []
        while len(pts) < n:
            while True:
                x, y = rnd.uniform(-1750, 1750), rnd.uniform(-1750, 1750)
                if math.hypot(x, y) <= 1700:
                    break
            if all(math.hypot(x - p[0], y - p[1]) >= 150 for p in pts):
                pts.append((x, y))
        chs = rnd.sample(range(1, 21), n)
        self.jammers = {}
        for p, ch in zip(pts, chs):
            j = {"pos": p, "cleared": False,
                 "omni": True if problem == 3 else rnd.random() < 0.5,
                 "recv_r": rnd.uniform(1000, 1500)}
            if not j["omni"]:
                j["heading"] = rnd.uniform(0, 360)
            self.jammers[ch] = j
        if problem == 4:
            adv_chs = rnd.sample(list(self.jammers), 2)
            base_ang = rnd.uniform(0, 360)
            for k, ch in enumerate(adv_chs):
                ang = base_ang + k * 180 + rnd.uniform(-20, 20)
                rr = rnd.uniform(1550, 1780)
                j = self.jammers[ch]
                j["pos"] = (rr * math.cos(math.radians(ang)),
                            rr * math.sin(math.radians(ang)))
                j["omni"] = False
                j["heading"] = ang + rnd.uniform(-25, 25)
                j["recv_r"] = rnd.uniform(1000, 1500)
        self.entered = False
        self.exited = False
        self.pos = (0.0, 0.0)
        self.channel = 1
        self.vt = 0.0

    def _err(self, pos, ch):
        key = f"{round(pos[0],3)},{round(pos[1],3)},{ch}".encode()
        h = int(hashlib.md5(key).hexdigest(), 16)
        return (h % 20001 - 10000) / 10000.0

    def _covered(self, j, p):
        d = math.hypot(j["pos"][0] - p[0], j["pos"][1] - p[1])
        if d > j["recv_r"]:
            return False
        if j["omni"]:
            return True
        bear = math.degrees(math.atan2(j["pos"][1] - p[1], j["pos"][0] - p[0])) % 360
        diff = abs((bear - j["heading"] + 180) % 360 - 180)
        return diff <= 90.0

    def handle(self, path, body):
        if path == "/enter":
            if self.entered or self.exited:
                return 200, {"accepted": False, "real_timestamp_ms": 0, "virtual_time_s": 0}
            self.entered = True
            return 200, {"accepted": True, "real_timestamp_ms": 0, "virtual_time_s": 0,
                         "max_virtual_duration_s": self.VT_LIMIT,
                         "max_real_duration_s": 1200, "remaining_real_duration_s": 1200}
        if not self.entered or self.exited:
            return 200, {"accepted": False, "real_timestamp_ms": 0, "virtual_time_s": 0}
        if path == "/exit":
            self.exited = True
            return 200, {"accepted": True, "real_timestamp_ms": 0, "virtual_time_s": self.vt,
                         "exit_reason": "user_exit"}
        ch = body["channel"]
        p = (body["position"]["x"], body["position"]["y"])
        moved = math.hypot(p[0] - self.pos[0], p[1] - self.pos[1])
        self.vt += moved / R.SPEED
        if path == "/measure":
            if ch != self.channel:
                self.vt += R.SWITCH_COST
                self.channel = ch
            self.vt += R.MEASURE_COST
            self.pos = p
            j = self.jammers.get(ch)
            resp = {"accepted": True, "real_timestamp_ms": 0, "virtual_time_s": self.vt}
            if j is None or j["cleared"] or not self._covered(j, p):
                resp["measure_result"] = "no_signal"
            else:
                d = math.hypot(j["pos"][0] - p[0], j["pos"][1] - p[1])
                if d <= R.NEAR_R:
                    resp["measure_result"] = "near"
                else:
                    bear = math.degrees(math.atan2(j["pos"][1] - p[1], j["pos"][0] - p[0]))
                    resp["measure_result"] = "direction"
                    resp["svd_deg"] = round((bear + self._err(p, ch)) % 360, 2)
            return 200, resp
        if path == "/clear":
            self.vt += R.CLEAR_COST
            self.pos = p
            j = self.jammers.get(ch)
            resp = {"accepted": True, "real_timestamp_ms": 0, "virtual_time_s": self.vt}
            if j is None or j["cleared"]:
                resp["clear_result"] = "no_target_in_range"
            else:
                d = math.hypot(j["pos"][0] - p[0], j["pos"][1] - p[1])
                if d <= R.CLEAR_R:
                    j["cleared"] = True
                    self.vt += 2.0
                    resp["virtual_time_s"] = self.vt
                    resp["clear_result"] = "success"
                else:
                    resp["clear_result"] = "no_target_in_range"
            return 200, resp
        return 404, {"accepted": False, "real_timestamp_ms": 0, "virtual_time_s": 0}


# ===================== Helpers =====================
class _DummyLog:
    def write(self, s): pass
    def flush(self): pass


def _setup_fake(srv):
    class FakeResp:
        def __init__(self, code, obj):
            self.status_code = code
            self._obj = obj
        def json(self):
            return self._obj

    def fake_post(url, json=None, timeout=None, headers=None):
        path = url.split("127.0.0.1:2026")[-1]
        code, obj = srv.handle(path, json)
        return FakeResp(code, obj)

    R.requests.post = fake_post


def _skeleton_points():
    pts = [(0.0, 0.0)]
    for i in range(6):
        a = i * math.pi / 3
        pts.append((1559.0 * math.cos(a), 1559.0 * math.sin(a)))
    return pts


# ===================== G0: 全覆盖基线 =====================
def run_g0(seed):
    srv = FakeServer(3, seed)
    n_all = len(srv.jammers)
    _setup_fake(srv)

    original_sp = R.scan_points
    R.scan_points = lambda rot=0.0: _skeleton_points()

    sim = R.Sim("http://127.0.0.1:2026", "TEST", _DummyLog())
    sim.enter(wait_s=5)
    R.Robot(sim, 3).run()

    R.scan_points = original_sp
    cleared = sum(1 for j in srv.jammers.values() if j["cleared"])
    return cleared == n_all, sim.vt, cleared, n_all


# ===================== G1: v1 (旧 robot.py) =====================
def run_g1(seed):
    srv = FakeServer(3, seed)
    n_all = len(srv.jammers)
    _setup_fake(srv)

    sim = R.Sim("http://127.0.0.1:2026", "TEST", _DummyLog())
    sim.enter(wait_s=5)
    R.Robot(sim, 3).run()

    cleared = sum(1 for j in srv.jammers.values() if j["cleared"])
    return cleared == n_all, sim.vt, cleared, n_all


# ===================== G2/G3: v2 策略 =====================
def run_g2(seed, rollout=False):
    from policy import Policy
    srv = FakeServer(3, seed)
    n_all = len(srv.jammers)
    _setup_fake(srv)

    sim = R.Sim("http://127.0.0.1:2026", "TEST", _DummyLog())
    sim.enter(wait_s=5)
    policy = Policy(sim, alpha=2.0, theta=2.05, rollout=rollout)
    n, vt = policy.run()

    cleared = sum(1 for j in srv.jammers.values() if j["cleared"])
    return cleared == n_all, vt, cleared, n_all


# ===================== G4: 离线下界 =====================
def run_g4(seed):
    from scheduler import two_opt_tour
    srv = FakeServer(3, seed)
    n_all = len(srv.jammers)
    positions = [np.asarray(j["pos"], float) for j in srv.jammers.values()]
    if n_all == 0:
        return True, 0.0, 0, 0
    order, dist = two_opt_tour(positions, np.array([0., 0.]))
    vt = dist / R.SPEED + n_all * (R.CLEAR_COST + 2.0)
    return True, vt, n_all, n_all


# ===================== 调度 =====================
def run_single(group, seed):
    if group == "G0":
        return run_g0(seed)
    elif group == "G1":
        return run_g1(seed)
    elif group == "G2":
        return run_g2(seed, rollout=False)
    elif group == "G3":
        return run_g2(seed, rollout=True)
    elif group == "G4":
        return run_g4(seed)
    else:
        raise ValueError(f"Unknown group: {group}")


# ===================== 报表 =====================
def report(results):
    print("\n" + "=" * 70)
    print(f"{'Group':<7} {'Seeds':>6} {'Clear%':>8} {'T_avg':>12} {'Std':>8} {'Fail':>5}")
    print("-" * 70)
    for group in ["G0", "G1", "G2", "G3", "G4"]:
        data = results.get(group, [])
        if not data:
            continue
        cleared = np.array([d[2] for d in data])
        total = np.array([d[3] for d in data])
        ratios = cleared / np.maximum(total, 1)
        avg_times = np.array([d[1] / d[2] if d[2] > 0 else np.inf for d in data])
        failures = int(np.sum(ratios < 1.0))
        mean_t = np.mean(avg_times)
        std_t = np.std(avg_times)
        print(f"{group:<7} {len(data):>6} {np.mean(ratios) * 100:>7.1f}% "
              f"{mean_t:>10.1f}  {std_t:>7.1f} {failures:>5}")
    print("=" * 70)


# ===================== Main =====================
def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "G2"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    # 兼容旧格式: python mock_test.py 3 [seed] -> G1
    if arg.isdigit():
        problem = int(arg)
        if problem == 3:
            arg = "G1"
        else:
            print(f"Problem {problem} not supported in mock test. Use G0-G4.")
            return

    if arg.lower() == "all":
        n_seeds = 30
        groups = ["G0", "G1", "G2", "G3", "G4"]
        results = {}
        for group in groups:
            results[group] = []
            for seed in range(1, n_seeds + 1):
                try:
                    ok, vt, c, n = run_single(group, seed)
                    avg = vt / c if c > 0 else float('inf')
                    results[group].append((ok, vt, c, n))
                    print(f"  {group} seed={seed:2d}: cleared {c}/{n} "
                          f"({c/n*100:.0f}%) vt={vt:.0f}s avg={avg:.1f}s"
                          f" {'OK' if ok else 'FAIL'}")
                except Exception as e:
                    results[group].append((False, 0, 0, 0))
                    print(f"  {group} seed={seed:2d}: ERROR {e}")
        report(results)
    else:
        group = arg.upper()
        try:
            ok, vt, c, n = run_single(group, seed)
            avg = vt / c if c > 0 else float('inf')
            print(f"\n########## {group} seed={seed}: cleared {c}/{n} "
                  f"({c/n*100:.0f}%) | vt={vt:.0f}s | avg={avg:.1f}s "
                  f"{'OK' if ok else 'FAIL'} ##########")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
