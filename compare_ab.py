# -*- coding: utf-8 -*-
"""路线A (Q2赌注点) vs 路线B (自适应追踪) 同种子对比."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "q3"))
import numpy as np

def run_bet(seed):
    """路线A: Q2 赌注点策略."""
    from policy_bet import PolicyBet
    from mock_test import FakeServer, _setup_fake, _DummyLog
    import robot as R

    srv = FakeServer(3, seed)
    n_all = len(srv.jammers)
    _setup_fake(srv)
    sim = R.Sim("http://127.0.0.1:2026", "TEST", _DummyLog())
    sim.enter(wait_s=5)
    policy = PolicyBet(sim, alpha=2.0, theta=2.05)
    n, vt = policy.run()
    cleared = sum(1 for j in srv.jammers.values() if j["cleared"])
    return cleared == n_all, vt, cleared, n_all

def run_adaptive(seed):
    """路线B: 自适应追踪策略."""
    from policy import Policy
    from mock_test import FakeServer, _setup_fake, _DummyLog
    import robot as R

    srv = FakeServer(3, seed)
    n_all = len(srv.jammers)
    _setup_fake(srv)
    sim = R.Sim("http://127.0.0.1:2026", "TEST", _DummyLog())
    sim.enter(wait_s=5)
    policy = Policy(sim, alpha=2.0, theta=2.05)
    n, vt = policy.run()
    cleared = sum(1 for j in srv.jammers.values() if j["cleared"])
    return cleared == n_all, vt, cleared, n_all

print("=" * 68)
print("路线A (Q2赌注点) vs 路线B (自适应追踪) — 12 种子对比")
print("=" * 68)

results_a = []
results_b = []

for seed in range(1, 13):
    ok_a, vt_a, c_a, n_a = run_bet(seed)
    avg_a = vt_a / c_a if c_a > 0 else float('inf')

    ok_b, vt_b, c_b, n_b = run_adaptive(seed)
    avg_b = vt_b / c_b if c_b > 0 else float('inf')

    delta = avg_b - avg_a
    winner = "A" if avg_a < avg_b else "B"

    results_a.append((ok_a, avg_a, c_a, n_a))
    results_b.append((ok_b, avg_b, c_b, n_b))

    sa = "OK" if ok_a else "FAIL"
    sb = "OK" if ok_b else "FAIL"
    print(f"  seed={seed:2d}  A: {avg_a:6.1f}s ({c_a}/{n_a}) {sa}  |  B: {avg_b:6.1f}s ({c_b}/{n_b}) {sb}  |  {winner} wins ({delta:+.1f}s)")

avgs_a = [r[1] for r in results_a]
avgs_b = [r[1] for r in results_b]
clears_a = sum(r[0] for r in results_a)
clears_b = sum(r[0] for r in results_b)

print("-" * 68)
print(f"  路线A (赌注点):   均值 {np.mean(avgs_a):.1f}s  std {np.std(avgs_a):.1f}s  清除 {clears_a}/12")
print(f"  路线B (自适应):   均值 {np.mean(avgs_b):.1f}s  std {np.std(avgs_b):.1f}s  清除 {clears_b}/12")
print(f"  差异:             {np.mean(avgs_b) - np.mean(avgs_a):+.1f}s ({(np.mean(avgs_b) - np.mean(avgs_a))/np.mean(avgs_b)*100:+.1f}%)")
print("=" * 68)
