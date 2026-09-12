# -*- coding: utf-8 -*-
"""Q4 mock 测试: 24 种子批量实验

用法:
    python q4/mock_test.py 1        # 单种子
    python q4/mock_test.py 24       # 24 种子
"""
import hashlib
import math
import os
import random
import sys
import time

import numpy as np

# q4 模块路径
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "q4"))
sys.path.insert(0, os.path.join(_root, "q2"))
import robot as R


# ===================== FakeServer (Q4 定向源) =====================
class FakeServer:
    ARENA_R = 1800.0
    VT_LIMIT = 360000.0

    def __init__(self, problem=4, seed=1):
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


# ===================== Q4 策略测试 =====================
def run_q4(seed, rollout=False):
    from policy import Policy
    srv = FakeServer(problem=4, seed=seed)
    n_all = len(srv.jammers)
    _setup_fake(srv)

    sim = R.Sim("http://127.0.0.1:2026", "TEST", _DummyLog())
    sim.enter(wait_s=5)
    policy = Policy(sim, alpha=2.0, theta=2.05, rollout=rollout)
    n, vt = policy.run()

    cleared = sum(1 for j in srv.jammers.values() if j["cleared"])
    n_omni = sum(1 for j in srv.jammers.values() if j.get("omni", True))
    n_dir = n_all - n_omni
    return cleared == n_all, vt, cleared, n_all, n_omni, n_dir


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    print(f"Q4 v1 mock 测试: {n_seeds} 种子\n")

    results = []
    for seed in range(1, n_seeds + 1):
        t0 = time.time()
        try:
            ok, vt, c, n, n_omni, n_dir = run_q4(seed)
            avg = vt / c if c > 0 else float('inf')
            results.append((ok, vt, c, n, n_omni, n_dir))
            elapsed = time.time() - t0
            print(f"  seed={seed:2d}: cleared {c}/{n} ({c/n*100:.0f}%) "
                  f"omni={n_omni} dir={n_dir} "
                  f"vt={vt:.0f}s avg={avg:.1f}s "
                  f"real={elapsed:.1f}s {'OK' if ok else 'FAIL'}")
        except Exception as e:
            results.append((False, 0, 0, 0, 0, 0))
            print(f"  seed={seed:2d}: ERROR {e}")
            import traceback; traceback.print_exc()

    # 汇总
    print("\n" + "=" * 70)
    ok_list = [r[0] for r in results]
    avg_times = [r[1] / r[2] if r[2] > 0 else float('inf') for r in results]
    cleared_ratios = [r[2] / r[3] if r[3] > 0 else 0 for r in results]
    n_fail = sum(1 for r in ok_list if not r)

    print(f"Q4 v1  24 seeds  Clear={sum(ok_list)}/{len(ok_list)}  "
          f"T_avg={np.mean(avg_times):.1f}  Std={np.std(avg_times):.1f}  "
          f"Fail={n_fail}")
    print(f"清除率均值: {np.mean(cleared_ratios)*100:.1f}%")

    # 源类型统计
    total_omni = sum(r[4] for r in results)
    total_dir = sum(r[5] for r in results)
    total_src = sum(r[3] for r in results)
    print(f"源类型: 全向={total_omni}/{total_src} ({total_omni/total_src*100:.0f}%)  "
          f"定向={total_dir}/{total_src} ({total_dir/total_src*100:.0f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
